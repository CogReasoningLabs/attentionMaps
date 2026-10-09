"""Script-owned inspection reports and optional whole-record script filtering."""

from __future__ import annotations

import csv
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import multiprocessing
import os
import random
import tempfile
import sys
from itertools import islice
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from types import SimpleNamespace

from attention_maps.eda.text import devanagari_ratio, extract_text

from .file_formats import describe_dataset_formats, validate_format_metadata
from .inspection import sample_dataset_rows
from .inspection_progress import inspection_progress
from .language_status import LANGUAGE_FIELDS, inventory_language_status
from .text import text_columns
from .classification_thresholds import resolve_thresholds
from .language_detection import detection_settings, make_language_detector
from .row_selection import with_row_filters, matches_row_filters, field_value, validate_row_filters
from .sampling_vote import (
    RepeatedSampleAnalysis, analyze_selected_batch, initialize_evidence_worker, validate_sampling_result,
)

REPORT_TYPE = "dataset_inspection"
REPORT_VERSION = 1


def iter_inspection_records(inventory: dict, *, token: str | None = None,
                            batch_size: int = 1024) -> Iterator[dict]:
    """Read selected records, applying metadata filters before any consumer samples."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    source_records = _iter_source_records(inventory, token=token, batch_size=batch_size)
    try:
        for record in source_records:
            if matches_row_filters(record, inventory.get("row_filters")):
                yield record
    finally:
        source_records.close()


def _iter_source_records(inventory: dict, *, token: str | None = None,
                         batch_size: int = 1024) -> Iterator[dict]:
    """Read exactly the chosen source files, without explorer-only row metadata."""
    kind = inventory["format"]
    if kind == "huggingface":
        from .huggingface import load_selected_huggingface_stream

        shards = inventory.get("dataset_shards")
        if not shards:
            raise ValueError("Inspection requires an explicit, non-empty resolved shard selection")
        records = load_selected_huggingface_stream(
            inventory["dataset_id"], inventory.get("dataset_config"),
            inventory["dataset_split"], inventory.get("dataset_revision"), shards,
            loader=inventory.get("dataset_loader"), token=token,
            parquet_features=inventory.get("parquet_features"),
        )
        # IterableDataset.iter is the installed datasets library's native
        # batch iterator. Test doubles and older row-only iterables still work.
        native_batches = getattr(records, "iter", None)
        if callable(native_batches):
            batches = native_batches(batch_size=batch_size)
            try:
                for batch in batches:
                    if not isinstance(batch, dict):
                        raise ValueError("Hugging Face batch must map columns to record lists")
                    lengths = {len(values) for values in batch.values()}
                    if len(lengths) > 1:
                        raise ValueError("Hugging Face batch has mismatched column lengths")
                    for index in range(next(iter(lengths), 0)):
                        record = {column: values[index] for column, values in batch.items()}
                        column = inventory.get("filter_column")
                        if not column or record.get(column) == inventory.get("filter_value"):
                            yield record
            finally:
                close = getattr(batches, "close", None)
                if close is not None:
                    close()
        else:
            for record in records:
                column = inventory.get("filter_column")
                if not column or record.get(column) == inventory.get("filter_value"):
                    yield dict(record)
    elif kind == "parquet":
        import pyarrow.parquet as pq

        paths = inventory.get("source_files") or list(dict.fromkeys(
            group["path"] for group in inventory["row_groups"]
        ))
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(batch_size=batch_size):
                    yield from batch.to_pylist()
    elif kind == "json":
        yield from json.loads(Path(inventory["path"]).read_text(encoding="utf-8"))
    elif kind in {"csv", "jsonl", "text"}:
        with Path(inventory["path"]).open(encoding="utf-8-sig", newline="") as stream:
            if kind == "csv":
                yield from csv.DictReader(stream)
            else:
                for line in stream:
                    if line.strip():
                        value = json.loads(line) if kind == "jsonl" else line.strip()
                        yield value if isinstance(value, dict) else {"text": value}
    elif kind == "xlsx":
        from attention_maps.datasets.kaggle import _open_workbook, _workbook_rows, _serializable_value

        workbook = _open_workbook(Path(inventory["path"]))
        try:
            _, columns, rows = _workbook_rows(workbook)
            for values in rows:
                values = tuple(values[:len(columns)])
                if any(value is not None for value in values):
                    yield {column: _serializable_value(values[index]) if index < len(values) else None
                           for index, column in enumerate(columns)}
        finally:
            workbook.close()
    else:
        raise ValueError("Inspection requires Hugging Face, Parquet, JSON, JSONL, CSV, TXT, or XLSX")


def _add_sample(sample: list, record: dict, count: int, size: int, rng: random.Random) -> None:
    if size == 0:
        return
    if len(sample) < size:
        sample.append(record)
    else:
        index = rng.randrange(count)
        if index < size:
            sample[index] = record


def _filter_records(inventory: dict, settings: dict, fields: list[str], token: str | None,
                    analysis: RepeatedSampleAnalysis | None = None, instance_definition=None) -> tuple:
    destination = Path(settings["filtered_output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sample, kept_sample = [], []
    source_rng = random.Random(settings["seed"])
    kept_rng = random.Random(settings["seed"])
    scanned = kept = 0
    exhausted = True
    limit = settings.get("max_records")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            total = analysis.population if analysis is not None else inventory.get("rows")
            if limit is not None and total is not None:
                total = min(total, limit)
            processor = _EvidenceBatchProcessor(analysis) if analysis is not None else nullcontext()
            with processor as evidence_processor, inspection_progress("Filter selected rows", total=total) as progress:
                for record in _instance_records(inventory, instance_definition, token=token,
                                                batch_size=settings.get("batch_size", 1024)):
                    if limit is not None and scanned >= limit:
                        exhausted = False
                        break
                    scanned += 1
                    if analysis is not None:
                        evidence_processor.observe(record)
                    else:
                        _add_sample(source_sample, record, scanned, settings["sample_size"], source_rng)
                    if instance_definition is not None:
                        from .semantic_instances import inspection_instance_text
                        text = inspection_instance_text(record, instance_definition, source_row=scanned - 1,
                                                        source_identity=inventory.get("dataset_id") or inventory.get("path") or "inspection",
                                                        source_split=inventory.get("dataset_split"))
                    else:
                        text = extract_text(record, fields)
                    ratio = devanagari_ratio(text)
                    # Empty/number-only text never qualifies, even for a zero threshold.
                    if ratio > 0 and ratio >= settings["min_devanagari_ratio"]:
                        output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                        kept += 1
                        if analysis is None:
                            _add_sample(kept_sample, record, kept, settings["sample_size"], kept_rng)
                    progress.update(1)
                    if progress.disable and scanned % 100000 == 0:
                        print(f"Inspection/filter: {scanned:,} records scanned", file=sys.stderr)
                if analysis is not None:
                    evidence_processor.finish()
            if analysis is not None:
                analysis.finish()  # Validate the population before publishing the filtered file.
            output.flush()
            os.fsync(output.fileno())
        # Publish only after the complete selected pass succeeds.
        if settings.get("overwrite"):
            temporary.replace(destination)
        else:
            os.link(temporary, destination)  # Exclusive creation; concurrent runs cannot clobber it.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return source_sample, kept_sample, {
        "mode": "keep_whole_records", "min_devanagari_ratio": settings["min_devanagari_ratio"],
        "rule": "Keep records containing Devanagari letters/marks whose share of all Unicode letters/marks meets the threshold.",
        "text_columns": fields, "rows_scanned": scanned, "rows_kept": kept,
        "rows_dropped": scanned - kept, "max_records": limit,
        "selection_exhausted": exhausted,
        "scope": "complete_selection" if exhausted else "limited_prefix",
        "output_path": str(destination), "output_bytes": destination.stat().st_size,
        "output_format": "jsonl",
        "file_formats": describe_dataset_formats([{"path": str(destination), "bytes": destination.stat().st_size}]),
    }


def _instance_records(inventory, definition, *, token=None, batch_size=1024):
    if definition is None:
        return iter_inspection_records(inventory, token=token, batch_size=batch_size)
    from .semantic_instances import iter_instance_records
    return iter_instance_records(inventory, definition, token=token, batch_size=batch_size)


def _status_records(records, definition, inventory):
    if definition is None:
        return records
    from .semantic_instances import inspection_instance_text
    identity = inventory.get("dataset_id") or inventory.get("path") or "inspection"
    return [
        {**record, "__inspection_text": inspection_instance_text(
            record, definition, source_row=index, source_identity=identity,
            source_split=inventory.get("dataset_split"))}
        for index, record in enumerate(records)
    ]


def build_inspection_report(inventory: dict, settings: dict, *, token: str | None = None) -> dict:
    """Single execution path for CLI reports; UI consumers only read this result."""
    settings = {**settings, **resolve_thresholds(settings), **detection_settings(settings)}
    detector = make_language_detector(settings)
    metadata_only = settings.get("formats_only", False)
    if settings.get("languages"):
        print("Ignoring legacy languages/--language: manual language declarations are not evidence. Use column/HF labels or enable text language detection.", file=sys.stderr)
    if settings.get("row_filters"):
        if metadata_only:
            raise ValueError("formats_only cannot apply row_filters; disable filtering for metadata-only inspection")
        inventory = with_row_filters(inventory, settings["row_filters"])
    fields = [] if metadata_only else list(settings.get("text_columns") or text_columns(inventory["schema"]))
    if not settings.get("text_columns"):
        fields = [field for field in fields if field not in inventory.get("row_filters", {})]
    instance_definition = None
    if not metadata_only and settings.get("training_schema"):
        from .semantic_instances import resolve_instance_definition
        instance_definition = resolve_instance_definition(inventory, fields, SimpleNamespace(**settings))
        fields = instance_definition["text_columns"]
        if inventory.get("row_filters") and instance_definition["text_record_unit"] == "blank_line":
            raise ValueError("blank_line instances cannot be selected with row_filters")
    roots = list(dict.fromkeys(field.split(".")[0] for field in fields))
    unknown = set(roots) - set(inventory["columns"])
    if unknown:
        raise ValueError(f"Unknown text columns: {sorted(unknown)}")
    columns = tuple(dict.fromkeys([*roots, *(field for field in LANGUAGE_FIELDS if field in inventory["columns"])]))
    settings = dict(settings) if metadata_only else {**settings, "text_columns": fields}
    if settings.get("row_filters"):
        inventory = _count_row_selection(inventory, token, settings.get("batch_size", 1024))
    instance_selection = None
    eligible_mask = None
    if settings.get("invalid_instance_policy") == "skip":
        instance_selection, eligible_mask = _count_valid_instances(
            inventory, instance_definition, token=token, batch_size=settings.get("batch_size", 1024))
    filtered = None
    status = None
    if metadata_only:
        records = []
    elif settings.get("sample_fraction") is not None:
        analysis = _prepare_repeated_analysis(
            inventory, settings, fields, token, instance_definition,
            eligible_rows=instance_selection["eligible_rows"] if instance_selection else None)
        if settings.get("min_devanagari_ratio") is not None:
            if not fields and instance_definition is None:
                raise ValueError("Filtering requires --text-column or detectable text fields")
            _, _, filtered = _filter_records(inventory, settings, fields, token, analysis, instance_definition)
            kept_inventory = {**inventory, "format": "jsonl", "path": filtered["output_path"],
                              "rows": filtered["rows_kept"]}
            kept_definition = ({**instance_definition, "text_record_unit": "source_row"}
                               if instance_definition is not None else None)
            kept_analysis = RepeatedSampleAnalysis(kept_inventory, settings, fields, filtered["rows_kept"],
                                                   population_basis="filtered output count", instance_definition=kept_definition)
            _observe_records(kept_analysis, _instance_records(
                kept_inventory, kept_definition, batch_size=settings.get("batch_size", 1024)))
            filtered["language_status"] = kept_analysis.finish()
        else:
            source_records = _instance_records(
                inventory, instance_definition, token=token, batch_size=settings.get("batch_size", 1024))
            _observe_records(analysis, _eligible_instance_records(source_records, eligible_mask)
                             if eligible_mask is not None else source_records)
        status = analysis.finish()
    elif settings.get("min_devanagari_ratio") is not None:
        if not fields and instance_definition is None:
            raise ValueError("Filtering requires --text-column or detectable text fields")
        records, kept, filtered = _filter_records(inventory, settings, fields, token, instance_definition=instance_definition)
        filtered["language_status"] = inventory_language_status(
            inventory, _status_records(kept, instance_definition, inventory),
            text_columns=["__inspection_text"] if instance_definition else fields, declared_languages=settings.get("languages") or (), thresholds=resolve_thresholds(settings), detector=detector,
        )
    elif inventory.get("row_filters") or instance_definition is not None:
        records = []
        rng = random.Random(settings["seed"])
        known_total = None if instance_definition and instance_definition["text_record_unit"] == "blank_line" else inventory.get("rows")
        with inspection_progress("Sample selected rows", total=known_total) as progress:
            for count, record in enumerate(_instance_records(
                    inventory, instance_definition, token=token, batch_size=settings.get("batch_size", 1024)), 1):
                _add_sample(records, record, count, settings["sample_size"], rng)
                progress.update(1)
    else:
        records = sample_dataset_rows(inventory, settings["sample_size"], settings["seed"], columns) if columns and settings["sample_size"] else []
    if status is None:
        status = inventory_language_status(
            inventory, _status_records(records, instance_definition, inventory),
            text_columns=["__inspection_text"] if instance_definition else fields, declared_languages=settings.get("languages") or (), thresholds=resolve_thresholds(settings), detector=detector,
        )
    sample_scope = "language_selected_sample" if inventory.get("row_filters") else "selected_source_sample"
    if metadata_only:
        sample_scope = "metadata_only"
    elif filtered:
        sample_scope = filtered["scope"]
    return {
        "report_type": REPORT_TYPE, "report_version": REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "inventory": inventory, "instance_definition": instance_definition, "language_status": status,
        "sample_scope": sample_scope,
        "instance_selection": instance_selection,
        "filter_result": filtered,
    }


def _count_row_selection(inventory: dict, token, batch_size: int = 1024) -> dict:
    print("Selecting matching language/metadata rows before sampling…", file=sys.stderr)
    scanned = matched = 0
    examples = {column: [] for column in inventory["row_filters"]}
    with inspection_progress("Match source rows", total=inventory.get("rows")) as progress:
        for record in _iter_source_records(inventory, token=token, batch_size=batch_size):
            scanned += 1
            for column, values in examples.items():
                value = str(field_value(record, column))[:120]
                if len(values) < 10 and value not in values:
                    values.append(value)
            matched += matches_row_filters(record, inventory["row_filters"])
            progress.update(1)
            if progress.disable and scanned % 100000 == 0:
                print(f"Language selection: {matched:,} matched / {scanned:,} scanned", file=sys.stderr)
    if not matched:
        raise ValueError(f"No records matched row_filters {inventory['row_filters']}. "
                         f"Observed label examples: {examples}. Use exact dataset labels or choose its language configuration.")
    print(f"Language selection: {matched:,} matched / {scanned:,} scanned; sampling uses matching rows only", file=sys.stderr)
    return {**inventory, "rows": matched, "rows_before_selection": scanned,
            "row_selection": {"filters": inventory["row_filters"], "rows_scanned": scanned,
                              "rows_matched": matched, "rows_excluded": scanned - matched,
                              "rule": "Exact, case-sensitive labels; OR within a column, AND between columns. List labels match any allowed value."}}


class _EvidenceBatchProcessor:
    """Bounded process pool for the selected records of one inspection."""

    def __init__(self, analysis: RepeatedSampleAnalysis):
        self.analysis = analysis
        self.executor = None
        self.pending = set()
        self.batch = []

    def __enter__(self):
        if self.analysis.concurrency > 1:
            if self.analysis.detector is not None and self.analysis.population:
                self.analysis.detector.ensure_model_file()
            self.executor = ProcessPoolExecutor(
                max_workers=self.analysis.concurrency,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_evidence_worker,
                initargs=(self.analysis.worker_settings,),
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=exc_type is not None)

    def _drain_one(self) -> None:
        completed, _ = wait(self.pending, return_when=FIRST_COMPLETED)
        for future in completed:
            self.pending.remove(future)
            self.analysis.merge_batch(future.result())

    def _submit_batch(self) -> None:
        if not self.batch:
            return
        self.pending.add(self.executor.submit(
            analyze_selected_batch, self.batch, self.analysis.fields,
            tuple(self.analysis.context["hf_selection"]), self.analysis.thresholds,
            len(self.analysis.evidence), self.analysis.instance_definition,
            self.analysis.source_identity, self.analysis.source_split,
            self.analysis.worker_settings["seed"],
        ))
        self.batch = []
        if len(self.pending) >= 2 * self.analysis.concurrency:
            self._drain_one()

    def observe(self, record: dict) -> None:
        if self.executor is None:
            self.analysis.observe(record)
            return
        selected = self.analysis.select_record(record)
        if selected is not None:
            self.batch.append((*selected, self.analysis.scanned - 1))
            if len(self.batch) >= self.analysis.batch_size:
                self._submit_batch()

    def finish(self) -> None:
        if self.executor is not None:
            self._submit_batch()
            while self.pending:
                self._drain_one()


def _observe_records(analysis: RepeatedSampleAnalysis, records) -> None:
    with _EvidenceBatchProcessor(analysis) as processor, \
            inspection_progress("Sample and classify", total=analysis.population) as progress:
        try:
            for record in records:
                processor.observe(record)
                progress.update(1)
                if progress.disable and analysis.scanned % 100000 == 0:
                    print(f"Inspection: {analysis.scanned:,} / {analysis.population:,} records scanned", file=sys.stderr)
            processor.finish()
        finally:
            # Release an active streaming reader before shutting down workers;
            # otherwise its background I/O can survive exception unwinding.
            close = getattr(records, "close", None)
            if close is not None:
                close()


def _count_valid_instances(inventory: dict, definition: dict, *, token=None, batch_size=1024):
    """Validate every selected source instance before drawing exact random samples."""
    from .semantic_instances import inspection_instance_text

    mask = bytearray()
    reasons = {}
    examples = []
    records = _instance_records(inventory, definition, token=token, batch_size=batch_size)
    try:
        with inspection_progress("Validate instances", total=inventory.get("rows")) as progress:
            for source_row, record in enumerate(records):
                try:
                    inspection_instance_text(
                        record, definition, source_row=source_row,
                        source_identity=inventory.get("dataset_id") or inventory.get("path") or "inspection",
                        source_split=inventory.get("dataset_split"))
                except ValueError as error:
                    reason = str(error)
                    mask.append(0)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if len(examples) < 10:
                        examples.append({"source_row": source_row, "reason": reason})
                else:
                    mask.append(1)
                progress.update(1)
    finally:
        records.close()
    source_rows = len(mask)
    if definition["text_record_unit"] != "blank_line" and inventory.get("rows") is not None and source_rows != inventory["rows"]:
        raise ValueError("Source row count changed during instance validation; report was not published")
    eligible = mask.count(1)
    return ({"policy": "skip", "source_rows": source_rows, "eligible_rows": eligible,
             "skipped_rows": source_rows - eligible, "reason_counts": reasons, "examples": examples}, mask)


def _eligible_instance_records(records, mask: bytearray):
    """Replay a pinned source, excluding only positions rejected in validation."""
    scanned = 0
    try:
        for record in records:
            if scanned >= len(mask):
                raise ValueError("Source row count changed after instance validation; report was not published")
            if mask[scanned]:
                yield record
            scanned += 1
        if scanned != len(mask):
            raise ValueError("Source row count changed after instance validation; report was not published")
    finally:
        records.close()


def _prepare_repeated_analysis(inventory: dict, settings: dict, fields: list[str], token,
                               instance_definition=None, *, eligible_rows=None) -> RepeatedSampleAnalysis:
    population = eligible_rows if eligible_rows is not None else inventory.get("rows")
    if instance_definition and instance_definition["text_record_unit"] == "blank_line":
        population = None  # Source inventory counts lines; sample complete documents.
    limit = settings.get("max_records") if settings.get("min_devanagari_ratio") is not None else None
    basis = ("valid instances after explicit skip policy" if eligible_rows is not None
             else "matching row count before sampling" if inventory.get("row_selection")
             else "source metadata")
    if instance_definition and instance_definition["text_record_unit"] == "blank_line":
        basis = "counting complete blank-line documents"
    if population is None:
        print("Counting the selected source population before exact random sampling…", file=sys.stderr)
        records = _instance_records(inventory, instance_definition, token=token,
                                    batch_size=settings.get("batch_size", 1024))
        try:
            population = 0
            with inspection_progress("Count selected rows") as progress:
                for _ in (islice(records, limit) if limit is not None else records):
                    population += 1
                    progress.update(1)
                    if progress.disable and population % 100000 == 0:
                        print(f"Counting: {population:,} records scanned", file=sys.stderr)
        finally:
            records.close()
        basis = "counting pass"
    elif limit is not None:
        population = min(population, limit)
    if limit is not None:
        basis += "; filter scan scope (possibly a limited prefix)"
    return RepeatedSampleAnalysis(inventory, settings, fields, population, population_basis=basis,
                                  instance_definition=instance_definition)


def write_inspection_report(path: Path, report: dict) -> None:
    """Atomically replace a report so the UI never reads a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_inspection_report(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv" and not content.lstrip().startswith("{"):
        from .inspection_history import read_history, load_history_report
        rows = read_history(path)
        if not rows:
            raise ValueError("Inspection CSV has no completed runs")
        return load_history_report(path, rows[-1])
    return validate_inspection_report(json.loads(content))


def validate_inspection_report(report: dict) -> dict:
    """Validate saved evidence independently of its JSON/CSV container."""
    if (not isinstance(report, dict) or report.get("report_type") != REPORT_TYPE
            or type(report.get("report_version")) is not int or report["report_version"] != REPORT_VERSION):
        raise ValueError("Choose a report produced by scripts/inspect_dataset.py (report version 1)")
    inventory = report.get("inventory")
    if not isinstance(inventory, dict) or not isinstance(report.get("settings"), dict):
        raise ValueError("Inspection report is missing inventory or settings")
    if not isinstance(inventory.get("format"), str) or any(
        inventory.get(key) is not None and (type(inventory[key]) is not int or inventory[key] < 0)
        for key in ("rows", "files", "bytes")
    ):
        raise ValueError("Inspection report has invalid source counts")
    for key in ("dataset_shards", "source_files"):
        values = inventory.get(key)
        if values is not None and (not isinstance(values, list) or any(not isinstance(item, str) for item in values)):
            raise ValueError("Inspection report has an invalid source file list")
    if inventory.get("file_formats") is not None:
        validate_format_metadata(inventory["file_formats"])
    selection = inventory.get("row_selection")
    if selection is not None:
        if (not isinstance(selection, dict)
                or not all(type(selection.get(key)) is int and selection[key] >= 0
                           for key in ("rows_scanned", "rows_matched", "rows_excluded"))
                or selection["rows_scanned"] != selection["rows_matched"] + selection["rows_excluded"]
                or selection["rows_matched"] != inventory.get("rows")
                or not isinstance(selection.get("rule"), str)
                or not selection.get("filters") or selection["filters"] != inventory.get("row_filters")):
            raise ValueError("Inspection report has invalid pre-sampling row selection evidence")
        validate_row_filters(selection["filters"])
    instance_selection = report.get("instance_selection")
    if instance_selection is not None:
        if (not isinstance(instance_selection, dict) or instance_selection.get("policy") != "skip"
                or not all(type(instance_selection.get(key)) is int and instance_selection[key] >= 0
                           for key in ("source_rows", "eligible_rows", "skipped_rows"))
                or instance_selection["source_rows"] != instance_selection["eligible_rows"] + instance_selection["skipped_rows"]
                or not isinstance(instance_selection.get("reason_counts"), dict)
                or not isinstance(instance_selection.get("examples"), list)
                or sum(instance_selection["reason_counts"].values()) != instance_selection["skipped_rows"]):
            raise ValueError("Inspection report has invalid instance-selection counts")
        definition = report.get("instance_definition") or {}
        if (definition.get("text_record_unit") != "blank_line" and inventory.get("rows") is not None
                and instance_selection["source_rows"] != inventory["rows"]):
            raise ValueError("Inspection report has inconsistent source and instance counts")
    statuses = [report.get("language_status")]
    filtered = report.get("filter_result")
    if filtered is not None:
        if (not isinstance(filtered, dict)
                or not all(type(filtered.get(key)) is int and filtered[key] >= 0
                           for key in ("rows_scanned", "rows_kept", "rows_dropped", "output_bytes"))
                or not all(isinstance(filtered.get(key), str) for key in ("output_path", "scope"))
                or type(filtered.get("selection_exhausted")) is not bool
                or type(filtered.get("min_devanagari_ratio")) not in (int, float)
                or not 0 <= filtered["min_devanagari_ratio"] <= 1
                or filtered["rows_scanned"] != filtered["rows_kept"] + filtered["rows_dropped"]):
            raise ValueError("Inspection report has an incomplete or invalid filtering result")
        if filtered.get("file_formats") is not None:
            validate_format_metadata(filtered["file_formats"])
        statuses.append(filtered.get("language_status"))
    if any(not isinstance(status, dict)
           or not all(isinstance(status.get(key), str) for key in ("language_coverage", "script", "language_basis", "note"))
           or type(status.get("sampled_records")) is not int or status["sampled_records"] < 0
           for status in statuses):
        raise ValueError("Inspection report is missing language/script evidence")
    for status in statuses:
        validate_sampling_result(status)
    if instance_selection is not None:
        sampling = report["language_status"].get("sampling", {})
        if sampling.get("population_rows") != instance_selection["eligible_rows"]:
            raise ValueError("Inspection report sampling population does not match valid instances")
    return report
