"""CSV tables of observed workspace counts and reproducibility settings."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .sampling import RepeatedSamplingPlan
    from .workspace import NormalizationResult, WorkspaceDeduplicationResult


def records_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Keep numbers numeric and encode structured settings inside CSV cells."""

    output = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list, tuple)) else value
        for key, value in row.items()
    } for row in rows)
    return output.getvalue().encode("utf-8")


def _percentage(count: int, total: int) -> float | None:
    return round(100 * count / total, 6) if total else None


def sampling_summary(
    plan: RepeatedSamplingPlan,
    folds: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    returned = sum(fold["returned_rows"] for fold in folds)
    workspace_rows = sum(fold["new_workspace_rows"] for fold in folds)
    stable = all(fold["source_identity_stable"] for fold in folds)
    return {
        **metadata,
        "scope": "workspace sample before Step 5 language selection",
        "population_rows": plan.population_rows,
        "requested_percentage_per_fold": plan.requested_percentage,
        "planned_folds": plan.folds,
        "requested_rows_per_fold": plan.requested_rows_per_fold,
        "planned_rows_per_fold": plan.rows_per_fold,
        "planned_total_sampled_rows": plan.total_rows_read,
        "planned_effective_percentage_per_fold": plan.effective_percentage_per_fold,
        "planned_expected_unique_rows": plan.expected_unique_rows,
        "planned_expected_population_coverage_pct": plan.expected_population_coverage_pct,
        "sampling_capped": plan.capped,
        "disjoint_folds": plan.disjoint_folds,
        "completed_folds": len(folds),
        "returned_rows_total": returned,
        "sampled_workspace_rows": workspace_rows,
        "rows_collapsed_by_identity": returned - workspace_rows,
        "source_identity_stable": stable,
        "observed_unique_source_rows": workspace_rows if stable else None,
        "observed_population_coverage_pct": (
            _percentage(workspace_rows, plan.population_rows) if stable else None
        ),
        "notes": (
            "Returned rows count sample results, not physical source rows scanned. "
            "Observed source coverage is blank when stable source row IDs are unavailable. "
            "Planned coverage assumes uniform sampling; Hugging Face uses bounded streaming shuffle."
        ),
    }


def persist_sampling_reports(
    summary: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "sampling_summary.csv"
    folds_path = output_dir / "sampling_folds.csv"
    summary_path.write_bytes(records_csv([summary]))
    folds_path.write_bytes(records_csv(folds))
    return summary_path, folds_path


def deduplication_summary(
    result: WorkspaceDeduplicationResult,
    *,
    metadata: Mapping[str, Any],
    normalization: NormalizationResult | None = None,
) -> dict[str, Any]:
    """Document percentages use input documents; paragraphs are separate units."""

    counts = {
        "exact_documents_removed": result.exact_documents_removed,
        "near_documents_removed": result.near_documents_removed,
        "empty_after_boilerplate_removed": result.empty_after_boilerplate_removed,
        "total_documents_removed": result.input_documents - result.retained_documents,
        "retained_documents": result.retained_documents,
        "boilerplate_affected_documents": result.boilerplate_affected_documents,
    }
    return {
        **{key: value for key, value in metadata.items()
           if key not in {"sampling", "deduplication_config"}},
        "scope": "entire normalized workspace sample before D2 and Step 5 language selection",
        "sampled_workspace_rows": normalization.sampled_rows if normalization else None,
        "normalization_missing_text_rows": normalization.missing_text_rows if normalization else None,
        "input_documents": result.input_documents,
        **counts,
        **{f"{name}_pct_of_input": _percentage(count, result.input_documents)
           for name, count in counts.items()},
        "repeated_paragraph_patterns": result.repeated_paragraph_patterns,
        "paragraphs_removed": result.paragraphs_removed,
        "audit_events": len(result.removals),
        **{f"config_{key}": value
           for key, value in metadata.get("deduplication_config", {}).items()},
        "notes": (
            "All document percentages use input_documents as denominator; blank means zero denominator. "
            "Input = exact removed + near removed + empty after boilerplate + retained. "
            "Paragraph counts and affected-document counts must not be added to removed-document counts."
        ),
    }


def normalization_summary(result: NormalizationResult) -> dict[str, Any]:
    return {
        "normalization": result.normalization,
        "sampled_rows": result.sampled_rows,
        "normalized_rows": result.normalized_rows,
        "missing_text_rows": result.missing_text_rows,
        "missing_text_pct_of_sample": _percentage(result.missing_text_rows, result.sampled_rows),
        "scope": "workspace sample after source selection, before deduplication",
    }
