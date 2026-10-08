"""Bounded, read-only row selection for the Source sample tab."""

from __future__ import annotations

import bisect
import os
import random
from collections import Counter
from itertools import islice

from .catalog import VIEWER_PREFIX
from .inspection import _load_inventory_huggingface_stream
from .row_selection import field_value, matches_row_filters, with_row_filters


def _source_records(inventory):
    """Open the same selected source for value discovery and filtered previews."""
    token = os.getenv("HF_TOKEN") or os.getenv("HF_token")
    kind = inventory["format"]
    if kind == "huggingface":
        return iter(_load_inventory_huggingface_stream(inventory, token=token))
    from .inspection_runs import iter_inspection_records

    # Count source rows before applying either the saved or UI filters.
    return iter_inspection_records(
        {**inventory, "format": {"kaggle": "xlsx", "kaggle_text": "text"}.get(kind, kind),
         "row_filters": {}}, token=token,
    )


def _matches_source_selection(record, inventory):
    column = inventory.get("filter_column")
    return ((not column or record.get(column) == inventory.get("filter_value"))
            and matches_row_filters(record, inventory.get("row_filters")))


def discover_filter_values(inventory, columns, max_scan_rows, max_values=200):
    """Count selectable text values in a bounded scan, once per value per row.

    Values stay literal (including case and surrounding spaces) so selecting
    them uses precisely the same equality/membership rule as preview filtering.
    """
    for name, value in (("max_scan_rows", max_scan_rows), ("max_values", max_values)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    unknown = [column for column in columns
               if column not in inventory["columns"] and column.split(".")[0] not in inventory["columns"]]
    if unknown:
        raise ValueError(f"Unknown filter columns: {unknown}")
    counts = {column: Counter() for column in columns}
    truncated = set()
    scanned = 0
    if columns:
        source = _source_records(inventory)
        try:
            for record in islice(source, max_scan_rows):
                scanned += 1
                if not _matches_source_selection(record, inventory):
                    continue
                for column in columns:
                    value = field_value(record, column)
                    values = value if isinstance(value, (list, tuple)) else (value,)
                    # Ignore values the exact-text filter cannot match.
                    for value in dict.fromkeys(item for item in values if isinstance(item, str) and item.strip()):
                        if value in counts[column] or len(counts[column]) < max_values:
                            counts[column][value] += 1
                        else:
                            truncated.add(column)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
    return {
        "counts": {column: dict(sorted(values.items())) for column, values in counts.items()},
        "rows_scanned": scanned,
        "scan_limit_reached": scanned == max_scan_rows,
        "truncated_columns": sorted(truncated),
    }


def sample_filtered_rows(inventory, sample_size, seed, columns, row_filters, max_scan_rows):
    """Reservoir-sample matches within a bounded prefix, then project columns.

    The scan count includes nonmatching rows. This is a preview of the scanned
    portion, not a claim of uniform sampling across an unscanned remote source.
    """
    for name, value in (("sample_size", sample_size), ("max_scan_rows", max_scan_rows)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    with_row_filters(inventory, row_filters)  # Validate against the full schema.
    if not columns or set(columns) - set(inventory["columns"]):
        raise ValueError("Select valid columns to read")

    kind = inventory["format"]
    source = _source_records(inventory)
    if kind == "huggingface":
        source_uri = f"hf://datasets/{inventory['dataset_id']}/{inventory['dataset_split']}"
    else:
        source_uri = str(inventory.get("source_uri") or inventory.get("path") or "Selected files")

    groups = inventory.get("row_groups", []) if kind == "parquet" else []
    starts = [group["start"] for group in groups]
    rng = random.Random(seed)
    sampled, scanned, matched = [], 0, 0
    try:
        for index, record in enumerate(islice(source, max_scan_rows)):
            scanned += 1
            if not (_matches_source_selection(record, inventory)
                    and matches_row_filters(record, row_filters)):
                continue
            matched += 1
            slot = len(sampled) if len(sampled) < sample_size else rng.randrange(matched)
            if slot >= sample_size:
                continue

            projected = {column: record.get(column) for column in columns}
            projected[f"{VIEWER_PREFIX}row_index"] = index
            projected[f"{VIEWER_PREFIX}file"] = source_uri
            if kind == "huggingface":
                source_id = record.get("id")
                projected[f"{VIEWER_PREFIX}row_index"] = source_id if source_id is not None else index
                projected[f"{VIEWER_PREFIX}identity_stable"] = source_id is not None
            elif groups:
                group = groups[bisect.bisect_right(starts, index) - 1]
                projected[f"{VIEWER_PREFIX}file"] = str(inventory.get("source_uri") or group["path"])
                projected[f"{VIEWER_PREFIX}row_group"] = group["row_group"]
            if slot == len(sampled):
                sampled.append((index, projected))
            else:
                sampled[slot] = (index, projected)
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()

    return {
        "records": [record for _, record in sorted(sampled, key=lambda item: item[0])],
        "rows_scanned": scanned,
        "rows_matched": matched,
        "scan_limit_reached": scanned == max_scan_rows,
    }
