"""End-to-end orchestration for a bounded Drive-to-Drive corpus run."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from attention_maps.common.google_drive import (
    DriveDownloadSummary,
    DriveUploadSummary,
    download_google_drive_path,
    upload_path_to_google_drive,
)
from attention_maps.common.pipeline_logging import pipeline_logger
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec, SurveyPlan, SurveyRun
from attention_maps.eda.deduplication import (
    DeduplicationConfig as EdaDeduplicationConfig,
    MultiStageDeduplicator,
    normalize_for_deduplication,
    strip_boilerplate_paragraphs,
)
from attention_maps.eda.pipeline import analyze_records
from attention_maps.eda.reporting import write_survey_report
from attention_maps.eda.text import devanagari_ratio, tokenize
from attention_maps.pruning.pipeline import (
    D2PipelineResult,
    run_d2_pruning_on_parquet,
)

from .artifacts import (
    artifact_inventory,
    create_zip,
    file_sha256,
    runtime_metadata,
    write_checksums,
    write_manifest,
    write_pdf_report,
)
from .contracts import PipelineConfig
from .processing import (
    AuditWriter,
    ParquetShardWriter,
    PreparedRecord,
    RejectedRecord,
    iter_parquet_records,
    ordered_prepared_batches,
    prepared_from_mapping,
    selected_by_sampling,
)
from .sources import SourceRecord, iter_record_batches


LOGGER = pipeline_logger("batch_preprocessing")
ProgressCallback = Callable[[str, int, int | None], None]


@dataclass(frozen=True)
class DriveAuth:
    credentials_file: Path | None = None
    oauth_client_secrets_file: Path | None = None
    oauth_token_file: Path = Path(".google-drive-token.json")
    full_drive_access: bool = True
    impersonate_user: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    clean_documents: int
    package_path: Path
    manifest_path: Path
    upload: DriveUploadSummary | None = None
    d2_documents: int | None = None


def run_pipeline(
    config: PipelineConfig,
    *,
    drive_auth: DriveAuth = DriveAuth(),
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    """Execute every stage and return only after validation and optional upload."""

    started = time.monotonic()
    run_dir = (config.output_root / config.run_name).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        prior_manifest = run_dir / "output" / "run_manifest.json"
        try:
            prior_status = json.loads(prior_manifest.read_text(encoding="utf-8")).get(
                "status"
            )
        except (OSError, json.JSONDecodeError):
            prior_status = None
        if prior_status != "running":
            raise ValueError(
                f"Run directory is not an incomplete pipeline run: {run_dir}. "
                "Use a new run_name to preserve provenance."
            )
        _reset_incomplete_run(run_dir)
    output_dir = run_dir / "output"
    work_dir = run_dir / "work"
    packages_dir = run_dir / "packages"
    for directory in (output_dir, work_dir, packages_dir, run_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        {
            "schema_version": 1,
            "status": "running",
            "run_name": config.run_name,
            "started_at": _utc_now(),
            "config": asdict(config),
        },
    )

    download_summary = None
    input_path = config.input.local_path
    if config.input.google_drive:
        _progress(progress, "download", 0, None)
        input_path = run_dir / "input" / "downloaded"
        download_summary = download_google_drive_path(
            config.input.google_drive,
            input_path,
            credentials_file=drive_auth.credentials_file,
            oauth_client_secrets_file=drive_auth.oauth_client_secrets_file,
            oauth_token_file=drive_auth.oauth_token_file,
            full_drive_access=drive_auth.full_drive_access,
            impersonate_user=drive_auth.impersonate_user,
            chunk_size=config.execution.drive_chunk_mib * 1024**2,
        )
    assert input_path is not None
    input_inventory = artifact_inventory(
        input_path, workers=min(4, config.execution.workers)
    )

    counters: dict[str, int] = {
        "scanned": 0,
        "sampled": 0,
        "cleaning_rejected": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "boilerplate_paragraphs_removed": 0,
        "empty_after_boilerplate": 0,
    }
    sampled_batches = _sampled_batches(config, input_path, work_dir, counters, progress)
    prepared_batches = ordered_prepared_batches(
        sampled_batches,
        config.input,
        config.cleaning,
        workers=config.execution.workers,
        max_pending=config.execution.pending_batches,
    )

    candidate_writer = ParquetShardWriter(
        work_dir / "candidate-shards", config.execution.shard_rows
    )
    audit_path = output_dir / "audit" / "preprocessing_audit.jsonl"
    deduplicator = _deduplicator(config)
    exact_hashes: set[str] = set()
    with AuditWriter(audit_path) as audit:
        for prepared in prepared_batches:
            for rejection in prepared.rejections:
                counters["cleaning_rejected"] += 1
                audit.add(rejection)
            for record in prepared.records:
                reason = _duplicate_reason(config, record, exact_hashes, deduplicator)
                if reason:
                    counters[reason] += 1
                    audit.add(
                        RejectedRecord(
                            record.doc_id,
                            record.source_file,
                            record.source_row,
                            "deduplication",
                            reason,
                        )
                    )
                else:
                    candidate_writer.add(record)
            _progress(progress, "preprocessing", counters["sampled"], None)
        candidate_paths = candidate_writer.close()

        repeated_hashes = (
            deduplicator.boilerplate_hashes() if deduplicator is not None else frozenset()
        )
        clean_writer = ParquetShardWriter(
            output_dir / "clean-data", config.execution.shard_rows
        )
        for value in iter_parquet_records(candidate_paths, config.execution.batch_size):
            record = prepared_from_mapping(value)
            if repeated_hashes:
                text, removed = strip_boilerplate_paragraphs(
                    record.text, repeated_hashes, deduplicator.config
                )
            else:
                text, removed = record.text, 0
            if removed:
                counters["boilerplate_paragraphs_removed"] += removed
                audit.add(
                    RejectedRecord(
                        record.doc_id,
                        record.source_file,
                        record.source_row,
                        "boilerplate",
                        f"removed_{removed}_paragraphs",
                    )
                )
            if not text:
                counters["empty_after_boilerplate"] += 1
                continue
            clean_writer.add(_updated_record(record, text))
        clean_paths = clean_writer.close()
        audit_events = audit.events

    clean_documents = sum(1 for _ in iter_parquet_records(clean_paths, config.execution.batch_size))
    if not clean_documents:
        raise RuntimeError("Preprocessing retained zero documents; relax cleaning settings.")
    d2_result: D2PipelineResult | None = None
    if config.pruning.enabled:
        _progress(progress, "d2_pruning", 0, clean_documents)
        d2_result = run_d2_pruning_on_parquet(
            clean_paths,
            clean_documents,
            output_dir / "d2",
            config.pruning,
            shard_rows=config.execution.shard_rows,
            batch_size=config.execution.batch_size,
            progress=lambda stage, value, total: _progress(
                progress, f"d2_{stage}", value, total
            ),
        )
    _progress(progress, "eda", 0, clean_documents)
    profile = None
    d2_profile = None
    eda_written: Sequence[Path] = ()
    if config.eda.enabled:
        dataset_key = _safe_key(config.run_name)
        analysis = AnalysisConfig(
            sample_size=clean_documents,
            seed=config.sampling.seed,
            min_tokens=config.cleaning.min_tokens,
            min_devanagari_ratio=config.cleaning.min_devanagari_ratio,
            reservoir_size=config.eda.reservoir_size,
            max_vocabulary=config.eda.max_vocabulary,
            top_tokens=config.eda.top_items,
            minhash_permutations=config.deduplication.minhash_permutations,
            minhash_bands=config.deduplication.minhash_bands,
            minhash_shingle_size=config.deduplication.shingle_size,
            near_duplicate_threshold=config.deduplication.near_duplicate_threshold,
            edit_similarity_threshold=config.deduplication.edit_similarity_threshold,
            boilerplate_min_documents=config.deduplication.boilerplate_min_documents,
            boilerplate_min_characters=config.deduplication.boilerplate_min_characters,
            max_ngrams=config.eda.max_ngrams,
            top_ngrams=config.eda.top_items,
            max_pattern_tokens_per_document=config.eda.max_pattern_tokens_per_document,
        )
        spec = DatasetSpec(
            key=dataset_key,
            dataset_id=f"local/{dataset_key}",
            text_columns=("text",),
            source_columns=("source",),
            sample_size=clean_documents,
            population_rows=clean_documents,
        )
        profile = analyze_records(
            spec,
            iter_parquet_records(clean_paths, config.execution.batch_size),
            analysis,
            progress=lambda _dataset, _stage, value: _progress(
                progress, "eda", value, clean_documents
            ),
        )
        survey_specs = [spec]
        survey_profiles = [profile]
        if d2_result is not None:
            d2_spec = DatasetSpec(
                key=f"{dataset_key}-d2-coreset",
                dataset_id=f"local/{dataset_key}/d2",
                text_columns=("text",),
                source_columns=("source",),
                sample_size=d2_result.selected_documents,
                population_rows=d2_result.selected_documents,
            )
            d2_profile = analyze_records(
                d2_spec,
                iter_parquet_records(
                    d2_result.coreset_paths, config.execution.batch_size
                ),
                replace(analysis, sample_size=d2_result.selected_documents),
                progress=lambda _dataset, _stage, value: _progress(
                    progress, "d2_eda", value, d2_result.selected_documents
                ),
            )
            survey_specs.append(d2_spec)
            survey_profiles.append(d2_profile)
        survey = SurveyRun(
            SurveyPlan("Clean batch pipeline EDA", tuple(survey_specs), analysis),
            tuple(survey_profiles),
            {},
        )
        eda_written = write_survey_report(survey, output_dir / "eda", plots=True)

    report_summary = {
        "Run": config.run_name,
        "Input records scanned": counters["scanned"],
        "Records sampled": counters["sampled"],
        "Clean documents": clean_documents,
        "D2 coreset documents": (
            d2_result.selected_documents if d2_result is not None else "disabled"
        ),
        "Exact duplicates removed": counters["exact_duplicates"],
        "Near duplicates removed": counters["near_duplicates"],
        "Boilerplate paragraphs removed": counters["boilerplate_paragraphs_removed"],
        "Audit events": audit_events,
    }
    images = [path for path in eda_written if path.suffix.lower() == ".png"]
    pdf_path = write_pdf_report(
        output_dir / "report" / "eda_report.pdf",
        title=f"Clean corpus EDA — {config.run_name}",
        summary=report_summary,
        image_paths=images,
    )
    completed_at = _utc_now()
    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "run_name": config.run_name,
        "completed_at": completed_at,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "config": asdict(config),
        "input": {
            "local_path": str(input_path),
            "drive_download": asdict(download_summary) if download_summary else None,
            "artifacts": input_inventory,
        },
        "counts": {**counters, "clean_documents": clean_documents, "audit_events": audit_events},
        "deduplication": {
            "mode": config.deduplication.mode,
            "repeated_paragraph_patterns": len(repeated_hashes),
        },
        "pruning": (
            {
                "method": "D2 Pruning",
                "selected_documents": d2_result.selected_documents,
                "manifest": str(d2_result.manifest_path.relative_to(output_dir)),
                "scores": str(d2_result.scores_path.relative_to(output_dir)),
            }
            if d2_result is not None
            else {"enabled": False}
        ),
        "eda_summary": profile.summary.as_dict() if profile else None,
        "d2_eda_summary": d2_profile.summary.as_dict() if d2_profile else None,
        "runtime": runtime_metadata(),
        "report": str(pdf_path.relative_to(output_dir)),
    }
    write_manifest(manifest_path, base_manifest)
    checksum_workers = min(4, config.execution.workers)
    inventory = artifact_inventory(
        output_dir, excluded=(manifest_path,), workers=checksum_workers
    )
    base_manifest["artifacts"] = inventory
    write_manifest(manifest_path, base_manifest)
    checksum_inventory = artifact_inventory(
        output_dir,
        excluded=(output_dir / "checksums.sha256",),
        workers=checksum_workers,
    )
    write_checksums(output_dir, checksum_inventory)

    package_path = create_zip(
        output_dir, packages_dir / f"{config.run_name}-clean-eda.zip"
    )
    package_checksum = file_sha256(package_path)
    package_checksum_path = packages_dir / f"{package_path.name}.sha256"
    package_checksum_path.write_text(
        f"{package_checksum}  {package_path.name}\n", encoding="utf-8"
    )

    upload = None
    if config.destination_drive_folder_id:
        upload_bundle = run_dir / "upload-bundle"
        upload_bundle.mkdir()
        for source in (package_path, package_checksum_path, manifest_path):
            shutil.copy2(source, upload_bundle / source.name)
        _progress(progress, "upload", 0, package_path.stat().st_size)
        upload = upload_path_to_google_drive(
            upload_bundle,
            config.destination_drive_folder_id,
            credentials_file=drive_auth.credentials_file,
            oauth_client_secrets_file=drive_auth.oauth_client_secrets_file,
            oauth_token_file=drive_auth.oauth_token_file,
            full_drive_access=drive_auth.full_drive_access,
            impersonate_user=drive_auth.impersonate_user,
            chunk_size=config.execution.drive_chunk_mib * 1024**2,
        )
    _progress(progress, "complete", clean_documents, clean_documents)
    LOGGER.info("PIPELINE COMPLETE run=%s clean_documents=%d", config.run_name, clean_documents)
    return PipelineResult(
        run_dir=run_dir,
        clean_documents=clean_documents,
        package_path=package_path,
        manifest_path=manifest_path,
        upload=upload,
        d2_documents=(
            d2_result.selected_documents if d2_result is not None else None
        ),
    )


def _sampled_batches(
    config: PipelineConfig,
    input_path: Path,
    work_dir: Path,
    counters: dict[str, int],
    progress: ProgressCallback | None,
) -> Iterator[tuple[int, Sequence[SourceRecord]]]:
    batch_id = 0
    selected_total = 0
    pending: list[SourceRecord] = []
    pending_bytes = 0
    maximum_bytes = config.execution.batch_max_mib * 1024**2
    stop = False
    for rows in iter_record_batches(
        input_path,
        batch_size=config.execution.batch_size,
        extraction_dir=work_dir / "extracted",
    ):
        for row in rows:
            counters["scanned"] += 1
            if selected_by_sampling(row.row_id, config.sampling):
                row_bytes = len(
                    json.dumps(row.record, ensure_ascii=False, default=str).encode(
                        "utf-8"
                    )
                )
                if pending and (
                    len(pending) >= config.execution.batch_size
                    or pending_bytes + row_bytes > maximum_bytes
                ):
                    yield batch_id, pending
                    batch_id += 1
                    pending = []
                    pending_bytes = 0
                pending.append(row)
                pending_bytes += row_bytes
                counters["sampled"] += 1
                selected_total += 1
            if config.sampling.max_records and selected_total >= config.sampling.max_records:
                stop = True
                break
        _progress(progress, "sampling", counters["scanned"], None)
        if stop:
            break
    if pending:
        yield batch_id, pending


def _deduplicator(config: PipelineConfig) -> MultiStageDeduplicator | None:
    if config.deduplication.mode != "multistage":
        return None
    value = config.deduplication
    return MultiStageDeduplicator(
        EdaDeduplicationConfig(
            shingle_size=value.shingle_size,
            minhash_permutations=value.minhash_permutations,
            minhash_bands=value.minhash_bands,
            near_duplicate_threshold=value.near_duplicate_threshold,
            edit_similarity_threshold=value.edit_similarity_threshold,
            boilerplate_min_documents=value.boilerplate_min_documents,
            boilerplate_min_characters=value.boilerplate_min_characters,
            seed=config.sampling.seed,
        )
    )


def _duplicate_reason(
    config: PipelineConfig,
    record: PreparedRecord,
    exact_hashes: set[str],
    deduplicator: MultiStageDeduplicator | None,
) -> str | None:
    if config.deduplication.mode == "none":
        return None
    if deduplicator is not None:
        decision = deduplicator.observe(record.text)
        if decision.exact_duplicate:
            return "exact_duplicates"
        if decision.near_duplicate:
            return "near_duplicates"
        return None
    digest = hashlib.sha256(
        normalize_for_deduplication(record.text).encode("utf-8")
    ).hexdigest()
    if digest in exact_hashes:
        return "exact_duplicates"
    exact_hashes.add(digest)
    return None


def _updated_record(record: PreparedRecord, text: str) -> PreparedRecord:
    if text == record.text:
        return record
    tokens = tokenize(text)
    return replace(
        record,
        text=text,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=len(tokens),
        character_count=len(text),
        devanagari_ratio=round(devanagari_ratio(text), 8),
    )


def _safe_key(value: str) -> str:
    safe = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "._-")
        else "-"
        for character in value
    )
    return safe.strip(".-_") or "clean-corpus"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reset_incomplete_run(run_dir: Path) -> None:
    """Keep downloaded bytes, but restart generated stages deterministically."""

    for relative in (
        "work",
        "output",
        "packages",
        "upload-bundle",
    ):
        target = run_dir / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _progress(
    callback: ProgressCallback | None, stage: str, completed: int, total: int | None
) -> None:
    if callback:
        callback(stage, completed, total)
