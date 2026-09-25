"""Self-contained inspection CSV: one row per run, including all evidence and votes."""

import csv
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import uuid

from .inspection_runs import load_inspection_report, validate_inspection_report
from .language_status import LANGUAGE_COVERAGE_OPTIONS, SCRIPT_OPTIONS

DEFAULT_HISTORY_SHEET = Path(__file__).resolve().parents[2] / "artifacts/dataset_inspection/history.csv"
FIELDS = (
    "Run ID", "Completed at", "Provider", "Dataset", "Configuration", "Split", "Revision/version",
    "Selected files", "File formats", "Source rows", "Source bytes", "Sample scope", "Text fields",
    "Language Coverage", "Script", "Language category %", "Nepali script category %",
    "Devanagari %", "Language evidence", "Sample fraction", "Sampling runs",
    "Rows per run", "Unique records analyzed", "Unique coverage %", "Seed",
    "Language votes", "Language decision", "Script votes", "Script decision", "Per-run votes",
    "Filtered Language Coverage", "Filtered Script", "Filtered Devanagari %", "Filtered rows kept",
    "Row filters", "Rows before language selection", "Rows after language selection", "Report JSON",
)

PRE_PROPORTION_FIELDS = tuple(field for field in FIELDS
                              if field not in {"Language category %", "Nepali script category %"})
PRE_SELECTION_FIELDS = (*PRE_PROPORTION_FIELDS[:-4], "Report JSON")
LEGACY_FIELDS = (*PRE_SELECTION_FIELDS[:-1], "Report")


def decision_text(status, key):
    vote = status.get("voting", {}).get(key)
    if vote is None:
        return "No repeated-sample vote; result uses the saved language/script evidence."
    threshold = vote.get("required_votes", vote["runs"] // 2 + 1)
    if vote["decision"] == "majority":
        return (f"{vote['winning_votes']}/{vote['runs']} votes selected {vote['label']}. "
                f"The vote rule needs at least {threshold} votes (including a strict majority).")
    return (f"No category assigned: no supported label met the vote threshold "
            f"(at least {threshold} of {vote['runs']} votes). Abstentions count in the total.")


def read_history(sheet):
    sheet = Path(sheet).expanduser().resolve()
    if not sheet.exists():
        return []
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        csv.field_size_limit(max(csv.field_size_limit(), 64 * 1024 * 1024))
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if reader.fieldnames not in (list(FIELDS), list(PRE_PROPORTION_FIELDS),
                                        list(PRE_SELECTION_FIELDS), list(LEGACY_FIELDS)):
            raise ValueError("History sheet has an incompatible header; choose a new CSV path")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("History sheet contains an incomplete row")
        return [{key: value.strip() for key, value in row.items()} for row in rows]


def _cell(value):
    """Keep dataset-controlled values literal when opened in spreadsheet software."""
    value = str(value) if value is not None else ""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _category_cell(value, allowed):
    """Leave unsupported summary cells empty; retain exact evidence in Report JSON."""
    return value if value in allowed else ""


def _category_summary(row):
    row = dict(row)
    for column, allowed in (("Language Coverage", LANGUAGE_COVERAGE_OPTIONS),
                            ("Filtered Language Coverage", LANGUAGE_COVERAGE_OPTIONS),
                            ("Script", SCRIPT_OPTIONS), ("Filtered Script", SCRIPT_OPTIONS)):
        row[column] = _category_cell(row.get(column), allowed)
    return row


def history_row(report, run_id):
    inventory, settings, status = report["inventory"], report["settings"], report["language_status"]
    sampling, votes = status.get("sampling", {}), status.get("voting", {})
    filtered = report.get("filter_result") or {}
    filtered_status = filtered.get("language_status", {})
    formats = inventory.get("file_formats") or {}
    row = {
        "Run ID": run_id, "Completed at": report.get("created_at", ""),
        "Provider": settings.get("provider") or inventory.get("provider", "local"),
        "Dataset": inventory.get("dataset_id") or settings.get("dataset") or settings.get("local") or inventory.get("path"),
        "Configuration": inventory.get("dataset_config"), "Split": inventory.get("dataset_split"),
        "Revision/version": inventory.get("dataset_revision") or inventory.get("dataset_version"),
        "Selected files": _json(inventory.get("dataset_shards") or inventory.get("selected_files") or inventory.get("source_files") or []),
        "File formats": _json(formats.get("groups", [])),
        "Source rows": inventory.get("rows"), "Source bytes": inventory.get("bytes"),
        "Sample scope": report.get("sample_scope"), "Text fields": _json(settings.get("text_columns")),
        "Language Coverage": _category_cell(status["language_coverage"], LANGUAGE_COVERAGE_OPTIONS),
        "Script": _category_cell(status["script"], SCRIPT_OPTIONS),
        "Language category %": _json(status.get("language_category_percentages", {})),
        "Nepali script category %": _json(status.get("script_category_percentages", {})),
        "Devanagari %": status.get("script_percentages", {}).get("Devanagari", ""),
        "Language evidence": status["language_basis"], "Sample fraction": sampling.get("fraction_requested"),
        "Sampling runs": sampling.get("runs", 0), "Rows per run": sampling.get("rows_per_run", status["sampled_records"]),
        "Unique records analyzed": sampling.get("unique_sampled_records", status["sampled_records"]),
        "Unique coverage %": sampling["unique_coverage"] * 100 if sampling else "", "Seed": settings.get("seed"),
        "Language votes": _json(votes.get("language_coverage", {}).get("counts", {})),
        "Language decision": decision_text(status, "language_coverage"),
        "Script votes": _json(votes.get("script", {}).get("counts", {})),
        "Script decision": decision_text(status, "script"),
        "Per-run votes": _json(sampling.get("run_results", [])),
        "Filtered Language Coverage": _category_cell(filtered_status.get("language_coverage"), LANGUAGE_COVERAGE_OPTIONS),
        "Filtered Script": _category_cell(filtered_status.get("script"), SCRIPT_OPTIONS),
        "Filtered Devanagari %": filtered_status.get("script_percentages", {}).get("Devanagari", ""),
        "Filtered rows kept": filtered.get("rows_kept"), "Report JSON": _json(report),
        "Row filters": _json(inventory.get("row_filters", {})),
        "Rows before language selection": inventory.get("row_selection", {}).get("rows_scanned"),
        "Rows after language selection": inventory.get("row_selection", {}).get("rows_matched"),
    }
    return {field: _cell(row.get(field)) for field in FIELDS}


@contextmanager
def history_lock(sheet):
    # Lock the directory, which survives atomic CSV replacement; no .lock file.
    sheet.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(sheet.parent, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def write_history(sheet, rows):
    """Atomically publish a CSV. Caller holds the directory lock."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=sheet.parent,
                                         prefix=f".{sheet.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(sheet)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def inline_report(report, sheet, run_id):
    return {**report, "history": {"run_id": run_id, "sheet": str(sheet), "storage": "csv"}}


def append_history(report, sheet):
    """Append one complete report; no per-run JSON files or persistent lock file."""
    sheet = Path(sheet).expanduser().resolve()
    saved = inline_report(report, sheet, uuid.uuid4().hex)
    with history_lock(sheet):
        rows = []
        for row in read_history(sheet):
            if "Report JSON" not in row or "Row filters" not in row:
                row = history_row(inline_report(load_history_report(sheet, row), sheet, row["Run ID"]), row["Run ID"])
            rows.append(_category_summary(row))
        write_history(sheet, [*rows, history_row(saved, saved["history"]["run_id"])])
    return saved


def load_history_report(sheet, row):
    if "Report JSON" in row:
        report = validate_inspection_report(json.loads(row["Report JSON"]))
    else:
        # Read older history sheets during migration, without rewriting on view.
        report = load_inspection_report(Path(sheet).expanduser().resolve().parent / row["Report"])
    if report.get("history", {}).get("run_id") != row["Run ID"]:
        raise ValueError("History row does not match its saved report")
    return report


def consolidate_inspection_reports(directory, sheet=None):
    """Migrate and verify old reports before removing redundant inspection JSONs."""
    directory = Path(directory).resolve()
    sheet = Path(sheet).resolve() if sheet else directory / "history.csv"

    def signature(report):
        return json.dumps({key: value for key, value in report.items()
                           if key not in {"history", "imported_from_report"}}, sort_keys=True, ensure_ascii=False)

    with history_lock(sheet):
        reports = [load_history_report(sheet, row) for row in read_history(sheet)]
        signatures = {signature(report) for report in reports}
        removable = []
        for path in sorted(directory.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("report_type") != "dataset_inspection":
                continue
            report = validate_inspection_report(data)
            removable.append(path)
            if signature(report) not in signatures:
                reports.append(report)
                signatures.add(signature(report))
        rows = []
        used_ids = set()
        for report in reports:
            run_id = report.get("history", {}).get("run_id") or uuid.uuid4().hex
            if run_id in used_ids:
                run_id = uuid.uuid4().hex
            used_ids.add(run_id)
            saved = inline_report(report, sheet, run_id)
            rows.append(history_row(saved, run_id))
        write_history(sheet, rows)
        recovered = [load_history_report(sheet, row) for row in read_history(sheet)]
        if len(recovered) != len(reports) or {signature(report) for report in recovered} != signatures:
            raise ValueError("CSV migration verification failed; original report files were retained")
        for path in removable:
            path.unlink()
        archive = directory / f"{sheet.stem}_reports"
        if archive.is_dir() and not any(archive.iterdir()):
            archive.rmdir()
        sheet.with_suffix(sheet.suffix + ".lock").unlink(missing_ok=True)
    return {"runs": len(rows), "removed_json_reports": len(removable), "csv": str(sheet)}
