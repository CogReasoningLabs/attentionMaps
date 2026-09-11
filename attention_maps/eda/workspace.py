"""Immutable-source workspace stages that produce an in-memory clean sample."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from attention_maps.eda.deduplication import (
    DeduplicationConfig,
    MultiStageDeduplicator,
    strip_boilerplate_paragraphs,
)
from attention_maps.eda.text import extract_source, extract_text, normalize_structured_text


WorkspaceProgress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class WorkspaceDocument:
    row_id: str
    text: str
    source: str = ""

    def as_record(self) -> dict[str, str]:
        return {"row_id": self.row_id, "text": self.text, "source": self.source}


@dataclass(frozen=True)
class NormalizationResult:
    documents: tuple[WorkspaceDocument, ...]
    sampled_rows: int
    normalized_rows: int
    missing_text_rows: int
    normalization: str = "NFC"


@dataclass(frozen=True)
class WorkspaceRemoval:
    row_id: str
    stage: str


@dataclass(frozen=True)
class WorkspaceDeduplicationResult:
    documents: tuple[WorkspaceDocument, ...]
    input_documents: int
    exact_documents_removed: int
    near_documents_removed: int
    repeated_paragraph_patterns: int
    paragraphs_removed: int
    boilerplate_affected_documents: int
    empty_after_boilerplate_removed: int
    removals: tuple[WorkspaceRemoval, ...]

    @property
    def retained_documents(self) -> int:
        return len(self.documents)

    @property
    def retention_pct(self) -> float:
        if not self.input_documents:
            return 0.0
        return round(100 * self.retained_documents / self.input_documents, 4)


def normalize_workspace_sample(
    records: Sequence[Mapping[str, Any]],
    text_columns: Sequence[str],
    source_columns: Sequence[str] = (),
    *,
    progress: WorkspaceProgress | None = None,
) -> NormalizationResult:
    """Extract selected fields and create NFC-normalized workspace documents."""

    documents: list[WorkspaceDocument] = []
    missing = 0
    for index, record in enumerate(records):
        text = extract_text(record, text_columns)
        normalized = normalize_structured_text(text) if text else ""
        if not normalized:
            missing += 1
            continue
        identity = json.dumps(
            {
                "row_index": record.get(
                    "__viewer_workspace_identity",
                    record.get("__viewer_row_index", index),
                ),
                "file": record.get("__viewer_file", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        row_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        documents.append(
            WorkspaceDocument(
                row_id=row_id,
                text=normalized,
                source=extract_source(record, source_columns),
            )
        )
        if progress and (index + 1) % 250 == 0:
            progress("normalizing", index + 1, len(records))
    if progress:
        progress("normalizing", len(records), len(records))
    return NormalizationResult(
        documents=tuple(documents),
        sampled_rows=len(records),
        normalized_rows=len(documents),
        missing_text_rows=missing,
    )


def deduplicate_workspace_documents(
    documents: Sequence[WorkspaceDocument],
    config: DeduplicationConfig,
    *,
    progress: WorkspaceProgress | None = None,
) -> WorkspaceDeduplicationResult:
    """Materialize exact/near removals, then strip repeated paragraphs."""

    deduplicator = MultiStageDeduplicator(config)
    retained: list[WorkspaceDocument] = []
    removals: list[WorkspaceRemoval] = []
    exact_removed = near_removed = 0
    for index, document in enumerate(documents, start=1):
        decision = deduplicator.observe(document.text)
        if decision.exact_duplicate:
            exact_removed += 1
            removals.append(WorkspaceRemoval(document.row_id, "exact_document"))
        elif decision.near_duplicate:
            near_removed += 1
            removals.append(WorkspaceRemoval(document.row_id, "near_document"))
        else:
            retained.append(document)
        if progress and index % 100 == 0:
            progress("document_deduplication", index, len(documents))
    if progress:
        progress("document_deduplication", len(documents), len(documents))

    repeated_hashes = deduplicator.boilerplate_hashes()
    cleaned: list[WorkspaceDocument] = []
    paragraphs_removed = affected = emptied = 0
    for index, document in enumerate(retained, start=1):
        text, removed = strip_boilerplate_paragraphs(
            document.text, repeated_hashes, config
        )
        paragraphs_removed += removed
        affected += removed > 0
        if removed:
            removals.append(
                WorkspaceRemoval(document.row_id, "boilerplate_paragraph_stripped")
            )
        if not text:
            emptied += 1
            removals.append(
                WorkspaceRemoval(document.row_id, "empty_after_boilerplate")
            )
            continue
        cleaned.append(WorkspaceDocument(document.row_id, text, document.source))
        if progress and index % 250 == 0:
            progress("boilerplate_removal", index, len(retained))
    if progress:
        progress("boilerplate_removal", len(retained), len(retained))

    return WorkspaceDeduplicationResult(
        documents=tuple(cleaned),
        input_documents=len(documents),
        exact_documents_removed=exact_removed,
        near_documents_removed=near_removed,
        repeated_paragraph_patterns=len(repeated_hashes),
        paragraphs_removed=paragraphs_removed,
        boilerplate_affected_documents=affected,
        empty_after_boilerplate_removed=emptied,
        removals=tuple(removals),
    )


def workspace_documents_jsonl(documents: Sequence[WorkspaceDocument]) -> bytes:
    lines = [
        json.dumps(document.as_record(), ensure_ascii=False, sort_keys=True)
        for document in documents
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def workspace_audit_csv(result: WorkspaceDeduplicationResult) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=("row_id", "stage"))
    writer.writeheader()
    writer.writerows(
        {"row_id": removal.row_id, "stage": removal.stage}
        for removal in result.removals
    )
    return output.getvalue().encode("utf-8")


def persist_workspace_artifacts(
    result: WorkspaceDeduplicationResult,
    output_dir: Path,
    *,
    metadata: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Persist one auditable clean workspace run without touching source data."""

    output_dir.mkdir(parents=True, exist_ok=False)
    clean_path = output_dir / "clean_sample.jsonl"
    audit_path = output_dir / "deduplication_audit.csv"
    manifest_path = output_dir / "workspace_manifest.json"
    clean_path.write_bytes(workspace_documents_jsonl(result.documents))
    audit_path.write_bytes(workspace_audit_csv(result))
    manifest = {
        **dict(metadata),
        "deduplication_result": {
            "input_documents": result.input_documents,
            "exact_documents_removed": result.exact_documents_removed,
            "near_documents_removed": result.near_documents_removed,
            "repeated_paragraph_patterns": result.repeated_paragraph_patterns,
            "paragraphs_removed": result.paragraphs_removed,
            "boilerplate_affected_documents": (
                result.boilerplate_affected_documents
            ),
            "empty_after_boilerplate_removed": (
                result.empty_after_boilerplate_removed
            ),
            "retained_documents": result.retained_documents,
            "retention_pct": result.retention_pct,
            "audit_events": len(result.removals),
        },
        "source_dataset_modified": False,
        "clean_sample_contains_text": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return clean_path, audit_path, manifest_path
