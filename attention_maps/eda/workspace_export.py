"""Bundle the current completed workspace and EDA scope, without stale runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .workspace_reports import records_csv


def export_workspace_zip(state, destination):
    """Write an archive on demand. Large artifacts are streamed into the ZIP."""
    if not state.get("eda_profile") or not state.get("eda_output"):
        raise ValueError("Complete Step 5 before exporting this workspace")
    workspace = Path(state["workspace_dir"]).resolve()
    eda = Path(state["eda_output"]).parent.resolve()
    eda.relative_to(workspace / "eda")
    files = {}

    def add(path, name, root):
        path = Path(path)
        if path.is_symlink():
            raise ValueError(f"Cannot export a symbolic link: {path.name}")
        path.resolve().relative_to(Path(root).resolve())
        if not path.is_file():
            raise ValueError(f"Missing run artifact: {path.name}")
        if path.suffix not in {".zip", ".tmp"}:
            files[name] = path

    for path in state["workspace_artifacts"]:
        add(path, Path(path).name, workspace)
    for path in state.get("sampling_artifacts", ()):
        if Path(path).name == "sampled_rows.jsonl":
            add(path, "sampled_rows.jsonl", Path(path).parent)
    for path in sorted(eda.rglob("*")):
        if path.is_file():
            add(path, "eda/" + path.relative_to(eda).as_posix(), eda)
    d2 = state.get("d2_result")
    if d2 is not None:
        d2_dir = Path(state["d2_output"]).resolve()
        d2_dir.relative_to(workspace / "d2")
        for path in sorted(d2_dir.rglob("*")):
            if path.is_file():
                add(path, "d2/" + path.relative_to(d2_dir).as_posix(), d2_dir)
    if "eda/language_selection.json" not in files or "eda/survey_summary.csv" not in files:
        raise ValueError("EDA artifacts are incomplete; rerun Step 5")

    sampling = state["sampling_summary"]
    normalization = state["normalization"]
    dedup = state["deduplication"]
    profile = state["eda_profile"]
    selection = sampling.get("source_selection") or {}
    steps = [
        {"step": 1, "name": "Sampling", "status": "complete",
         "input_rows": sampling["population_rows"], "output_rows": sampling["sampled_workspace_rows"],
         "scope": selection.get("population_scope", "selected source"),
         "details": "See sampling_summary.csv and sampling_folds.csv for returned reads and coverage."},
        {"step": 2, "name": "NFC normalization", "status": "complete",
         "input_rows": normalization.sampled_rows, "output_rows": normalization.normalized_rows,
         "scope": "sampled workspace", "details": f"{normalization.missing_text_rows} rows without text excluded"},
        {"step": 3, "name": "Deduplication", "status": "complete",
         "input_rows": dedup.input_documents, "output_rows": dedup.retained_documents,
         "scope": "normalized workspace", "details": "See deduplication_summary.csv for removal counts and denominators."},
        {"step": 4, "name": "D2 pruning (optional)", "status": "complete" if d2 else "skipped",
         "input_rows": d2.source_documents if d2 else None,
         "output_rows": len(d2.selected_documents) if d2 else None,
         "scope": "deduplicated workspace", "details": "Optional branch; Step 5 uses Step 3 output."},
        {"step": 5, "name": "EDA", "status": "complete",
         "input_rows": profile.summary.rows_seen, "output_rows": profile.summary.usable_rows,
         "scope": ("selected language within Step 3 output" if state["eda_selection"][0]
                   else "all cleaned rows from the workspace selection"),
         "details": "See eda/language_selection.json and eda/survey_summary.csv."},
    ]
    extra = {
        "steps_summary.csv": records_csv(steps),
        "source_selection.csv": records_csv([{
            **{key: sampling.get(key) for key in (
                "dataset_key", "dataset_id", "dataset_location", "dataset_config", "dataset_split",
                "dataset_revision", "dataset_shards", "text_fields", "source_fields", "label_field", "language_field",
            )}, **selection,
        }]),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_run_id": sampling.get("run_id"), "workspace_run_id": workspace.name,
        "eda_selection": state["eda_selection"], "eda_completed_at": state.get("eda_completed_at"),
        "d2_status": "complete" if d2 else "skipped",
        "artifacts": sorted([*files, *extra, "export_manifest.json"]),
        "notes": "Contains sampled records, preprocessing results, and only the current EDA and D2 runs. "
                 "The full source and pre-sampling filtered snapshot are not duplicated in this archive.",
    }
    extra["export_manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in files.items():
                archive.write(path, name)
            for name, content in extra.items():
                archive.writestr(name, content)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
