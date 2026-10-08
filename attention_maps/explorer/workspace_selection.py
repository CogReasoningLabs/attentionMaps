"""Disk-backed source selection with an exact population for workspace sampling."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

from .row_selection import matches_row_filters, with_row_filters
from .sample_filters import _matches_source_selection, _source_records


def materialize_workspace_selection(inventory, columns, row_filters, output_dir, *, max_scan_rows=None,
                                    progress=None):
    """Stream matches to JSONL before sampling; memory does not grow with the source.

    The population is the matching rows in this snapshot, never an estimate of
    matches beyond a scan limit. Filter columns need not be projected columns.
    """
    with_row_filters(inventory, row_filters)
    if not columns or set(columns) - set(inventory["columns"]):
        raise ValueError("Select valid workspace columns")
    if max_scan_rows is not None and (type(max_scan_rows) is not int or max_scan_rows < 1):
        raise ValueError("max_scan_rows must be a positive integer or None")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "selected_rows.jsonl"
    temporary = path.with_suffix(".tmp")
    source = _source_records(inventory)
    scanned = matched = 0
    exhausted = False
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for record in islice(source, max_scan_rows):
                scanned += 1
                if _matches_source_selection(record, inventory) and matches_row_filters(record, row_filters):
                    output.write(json.dumps({column: record.get(column) for column in columns},
                                            ensure_ascii=False, default=str) + "\n")
                    matched += 1
                if progress and scanned % 5_000 == 0:
                    progress(scanned, matched)
            # An exact boundary can be conservatively marked as limited without
            # fetching one extra (potentially remote) row.
            exhausted = max_scan_rows is None or scanned < max_scan_rows
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
        close = getattr(source, "close", None)
        if close:
            close()
    selection = {
        **{key: inventory[key] for key in (
            "dataset_id", "dataset_config", "dataset_split", "dataset_revision", "dataset_shards",
        ) if key in inventory},
        "source_format": inventory["format"],
        "source_population_rows": inventory.get("rows"),
        "filter_column": inventory.get("filter_column"),
        "filter_value": inventory.get("filter_value"),
        "source_row_filters": inventory.get("row_filters") or {},
        "row_filters": row_filters,
        "columns": list(columns),
        "max_scan_rows": max_scan_rows,
        "rows_scanned": scanned,
        "matching_rows": matched,
        "scan_limit_reached": not exhausted,
        "population_scope": "matching rows in selected source" if exhausted else "matching rows in scanned prefix",
        "identity_basis": "row positions in the immutable selected_rows.jsonl snapshot",
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return {
        "format": "jsonl", "path": str(path), "rows": matched,
        "columns": list(columns),
        "schema": [field for field in inventory["schema"] if field["column"] in columns],
        "bytes": path.stat().st_size,
        "workspace_selection": selection,
    }
