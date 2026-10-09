"""Bounded inspection and sampling for explorer dataset sources."""

from __future__ import annotations

import bisect
import csv
import json
import os
import random
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.datasets.kaggle import (
    inspect_workbook_path,
    sample_kaggle_text_rows,
    sample_kaggle_workbook_rows,
    sample_workbook_rows,
)

from .catalog import VIEWER_PREFIX
from .file_formats import describe_dataset_formats
from .inspection_progress import inspection_progress
from .parallel_reads import ordered_parallel_reads


HUGGINGFACE_DATASET_VIEWER_SIZE_URL = (
    "https://datasets-server.huggingface.co/size"
)

def project_inventory(
    inventory: dict[str, Any], visible_columns: Sequence[str] | None
) -> dict[str, Any]:
    """Limit a shared physical dataset to the columns exposed by one view."""

    if visible_columns is None:
        return inventory
    available = set(inventory["columns"])
    selected = [column for column in visible_columns if column in available]
    projected = dict(inventory)
    projected["columns"] = selected
    projected["schema"] = [
        field for field in inventory["schema"] if field["column"] in selected
    ]
    return projected


def file_signatures(files: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    """Build a cache key that changes when a source file changes."""

    signatures = []
    for path in files:
        stat = path.stat()
        signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(signatures)


def inspect_parquet(
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Read Parquet metadata only; no dataset rows are materialized."""

    import pyarrow.parquet as pq

    row_groups: list[dict[str, Any]] = []
    common_columns: list[str] | None = None
    first_schema: list[dict[str, str]] = []
    schema_variants: set[tuple[tuple[str, str], ...]] = set()
    total_rows = 0

    for file_index, (path_text, _, _) in enumerate(signatures):
        parquet = pq.ParquetFile(path_text)
        arrow_schema = parquet.schema_arrow
        schema_tuple = tuple((field.name, str(field.type)) for field in arrow_schema)
        schema_variants.add(schema_tuple)
        names = list(arrow_schema.names)
        if common_columns is None:
            common_columns = names
            first_schema = [
                {
                    "column": field.name,
                    "type": str(field.type),
                    "nullable": str(field.nullable),
                }
                for field in arrow_schema
            ]
        else:
            name_set = set(names)
            common_columns = [name for name in common_columns if name in name_set]

        for group_index in range(parquet.num_row_groups):
            rows = parquet.metadata.row_group(group_index).num_rows
            row_groups.append(
                {
                    "path": path_text,
                    "file_index": file_index,
                    "row_group": group_index,
                    "start": total_rows,
                    "rows": rows,
                }
            )
            total_rows += rows

    return {
        "format": "parquet",
        "rows": total_rows,
        "files": len(signatures),
        "bytes": sum(size for _, size, _ in signatures),
        "row_groups": row_groups,
        "columns": common_columns or [],
        "schema": first_schema,
        "schema_variants": len(schema_variants),
    }


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("JSON dataset must be a top-level array of objects")
    return data


def inspect_json(
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Inspect a single JSON array dataset and infer its top-level schema."""

    if len(signatures) != 1:
        raise ValueError("JSON datasets must contain exactly one file")
    path_text, size, _ = signatures[0]
    records = _read_json_records(Path(path_text))
    columns = list(dict.fromkeys(key for record in records for key in record))
    schema = []
    for column in columns:
        types = {
            _json_value_type(record.get(column))
            for record in records
            if column in record and record.get(column) is not None
        }
        schema.append(
            {
                "column": column,
                "type": " | ".join(sorted(types)) if types else "null",
                "nullable": str(
                    any(
                        column not in record or record.get(column) is None
                        for record in records
                    )
                ),
            }
        )
    return {
        "format": "json",
        "path": path_text,
        "rows": len(records),
        "files": 1,
        "bytes": size,
        "row_groups": [],
        "columns": columns,
        "schema": schema,
        "schema_variants": 1,
    }


def inspect_dataset(
    signatures: tuple[tuple[str, int, int], ...], *, show_progress: bool = False,
) -> dict[str, Any]:
    """Inspect a supported dataset using its file extension."""

    suffixes = {Path(path_text).suffix.lower() for path_text, _, _ in signatures}
    if suffixes == {".parquet"}:
        inventory = inspect_parquet(signatures)
    elif suffixes == {".json"}:
        inventory = inspect_json(signatures)
    elif suffixes == {".csv"}:
        inventory = inspect_delimited(signatures, format_name="csv", show_progress=show_progress)
    elif suffixes <= {".jsonl", ".ndjson"} and suffixes:
        inventory = inspect_json_lines(signatures, show_progress=show_progress)
    elif suffixes == {".txt"}:
        inventory = inspect_text_lines(signatures, show_progress=show_progress)
    elif suffixes == {".xlsx"} and len(signatures) == 1:
        inventory = inspect_workbook_path(Path(signatures[0][0]))
    else:
        raise ValueError(f"Unsupported or mixed dataset formats: {sorted(suffixes)}; use --formats-only to catalog extensions without parsing")
    inventory["file_formats"] = describe_dataset_formats(
        {"path": path, "bytes": size} for path, size, _ in signatures
    )
    return inventory


def _merge_observed_schema(
    records: Sequence[dict[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    columns = list(dict.fromkeys(key for record in records for key in record))
    schema = []
    for column in columns:
        types = {
            _json_value_type(record.get(column))
            for record in records
            if record.get(column) is not None
        }
        schema.append(
            {
                "column": column,
                "type": " | ".join(sorted(types)) if types else "null",
                "nullable": str(any(record.get(column) is None for record in records)),
            }
        )
    return columns, schema


def inspect_delimited(
    signatures: tuple[tuple[str, int, int], ...], *, format_name: str = "csv",
    show_progress: bool = False,
) -> dict[str, Any]:
    """Scan CSV metadata with bounded schema inference memory."""

    if len(signatures) != 1:
        raise ValueError("CSV explorer imports must select exactly one file")
    path_text, size, _ = signatures[0]
    examples: list[dict[str, Any]] = []
    with Path(path_text).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = 0
        nullable = {column: False for column in columns}
        with inspection_progress("Read CSV", enabled=show_progress) as progress:
            for record in reader:
                rows += 1
                if len(examples) < 1_000:
                    examples.append(dict(record))
                for column in columns:
                    if record.get(column) in (None, ""):
                        nullable[column] = True
                progress.update(1)
    return {
        "format": format_name,
        "path": path_text,
        "rows": rows,
        "files": 1,
        "bytes": size,
        "row_groups": [],
        "columns": columns,
        "schema": [
            {"column": column, "type": "string", "nullable": str(nullable[column])}
            for column in columns
        ],
        "schema_variants": 1,
    }


def inspect_json_lines(
    signatures: tuple[tuple[str, int, int], ...], *, show_progress: bool = False,
) -> dict[str, Any]:
    """Scan JSONL/NDJSON metadata without retaining the corpus."""

    if len(signatures) != 1:
        raise ValueError("JSONL explorer imports must select exactly one file")
    path_text, size, _ = signatures[0]
    examples: list[dict[str, Any]] = []
    rows = 0
    with Path(path_text).open("r", encoding="utf-8-sig") as stream, \
         inspection_progress("Read JSONL", enabled=show_progress) as progress:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            record = dict(value) if isinstance(value, dict) else {"text": value}
            rows += 1
            if len(examples) < 1_000:
                examples.append(record)
            progress.update(1)
    columns, schema = _merge_observed_schema(examples)
    return {
        "format": "jsonl",
        "path": path_text,
        "rows": rows,
        "files": 1,
        "bytes": size,
        "row_groups": [],
        "columns": columns,
        "schema": schema,
        "schema_variants": 1,
    }


def inspect_text_lines(
    signatures: tuple[tuple[str, int, int], ...], *, show_progress: bool = False,
) -> dict[str, Any]:
    """Count non-empty UTF-8 lines as individual text records."""

    if len(signatures) != 1:
        raise ValueError("Text explorer imports must select exactly one file")
    path_text, size, _ = signatures[0]
    rows = 0
    with Path(path_text).open("rb") as stream, \
         inspection_progress("Read TXT", total=size, unit="B", enabled=show_progress) as progress:
        for line in stream:
            rows += bool(line.strip())
            progress.update(len(line))
    return {
        "format": "text",
        "path": path_text,
        "rows": rows,
        "files": 1,
        "bytes": size,
        "row_groups": [],
        "columns": ["text"],
        "schema": [{"column": "text", "type": "string", "nullable": "False"}],
        "schema_variants": 1,
    }


def inspect_huggingface_dataset(
    dataset_id: str,
    split: str,
    *,
    config: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> dict[str, Any]:
    """Inspect streaming metadata or a source adapter's disk-backed row index."""

    try:
        import datasets  # noqa: F401 - optional dependency check
    except ImportError as error:
        raise ValueError("Hugging Face datasets support requires `datasets`") from error
    if bool(filter_column) != bool(filter_value):
        raise ValueError("A Hugging Face filter requires both a column and value")
    try:
        load_options = (
            {"filters": [(filter_column, "==", filter_value)]}
            if filter_column and filter_value
            else {}
        )
        loaded = load_huggingface_stream(
            dataset_id,
            config,
            split=split,
            revision=revision,
            token=token or None,
            **load_options,
        )
        stream = loaded.stream
        config = loaded.config or getattr(stream.info, "config_name", None)
        split_info = (getattr(stream.info, "splits", None) or {}).get(split)
        features = stream.features or {}
    except Exception as error:
        raise ValueError(f"Could not inspect Hugging Face dataset: {error}") from error
    viewer_size = None
    if split_info is None:
        viewer_size = _huggingface_viewer_split_size(
            dataset_id,
            config=config or getattr(stream.info, "config_name", None),
            split=split,
            token=token,
        )
    shard_lengths = list(getattr(split_info, "shard_lengths", None) or [])
    shard_count = len(shard_lengths) or int(getattr(stream, "num_shards", 1))
    rows = int(
        split_info.num_examples if split_info is not None else viewer_size["num_rows"]
    )
    memory_bytes = int(
        split_info.num_bytes
        if split_info is not None
        else viewer_size.get("num_bytes_memory") or 0
    )
    hub_file_bytes = getattr(stream.info, "download_size", None)
    if not hub_file_bytes:
        checksums = getattr(stream.info, "download_checksums", None) or {}
        checksum_sizes = [
            item.get("num_bytes")
            for item in checksums.values()
            if isinstance(item, dict) and item.get("num_bytes") is not None
        ]
        hub_file_bytes = sum(int(size) for size in checksum_sizes) or None
    hub_file_size_basis = "download metadata" if hub_file_bytes is not None else "unavailable"
    hub_file_size_scope = (
        "selected configuration" if len(getattr(stream.info, "splits", None) or {}) > 1
        else "selected split"
    )
    if hub_file_bytes is not None and loaded.loading_strategy.startswith("memory_mapped_"):
        hub_file_size_basis = "original files"
    if loaded.hub_file_bytes is not None:
        hub_file_bytes = loaded.hub_file_bytes
        hub_file_size_basis = loaded.hub_file_size_basis
        hub_file_size_scope = loaded.hub_file_size_scope
    # Storage sizes can be absent even when DatasetInfo has rows and decoded
    # bytes. This optional lookup must not prevent inspection on Viewer failure.
    # Viewer metadata describes main, so do not apply it to a pinned revision.
    hub_file_size_error = None
    if not hub_file_bytes and viewer_size is None and (loaded.revision or revision) in (None, "main"):
        try:
            viewer_size = _huggingface_viewer_split_size(
                dataset_id, config=config, split=split, token=token
            )
        except ValueError as error:
            hub_file_size_error = str(error)
    if not hub_file_bytes and viewer_size is not None:
        original_bytes = viewer_size.get("num_bytes_original_files")
        parquet_bytes = viewer_size.get("num_bytes_parquet_files")
        if original_bytes is not None:
            hub_file_bytes, hub_file_size_basis = original_bytes, "original files"
        elif parquet_bytes is not None:
            hub_file_bytes, hub_file_size_basis = parquet_bytes, "converted Parquet"
        hub_file_size_scope = "selected split"
    if hub_file_bytes is not None:
        hub_file_bytes = int(hub_file_bytes)
    inventory = {
        "format": "huggingface",
        "dataset_id": dataset_id,
        "dataset_config": config,
        "dataset_split": split,
        "dataset_revision": loaded.revision or revision,
        "loading_strategy": loaded.loading_strategy,
        "dataset_parquet_files": list(loaded.parquet_files),
        "rows": rows,
        "files": shard_count,
        # Prefer physical/download bytes for storage displays. SplitInfo's
        # num_bytes is the decoded Arrow footprint, not disk usage.
        "bytes": hub_file_bytes if hub_file_bytes is not None else memory_bytes,
        "hub_file_bytes": hub_file_bytes,
        "hub_file_size_basis": hub_file_size_basis,
        "hub_file_size_scope": hub_file_size_scope,
        "hub_file_size_error": hub_file_size_error,
        "memory_bytes": memory_bytes,
        "streaming": True,
        "row_groups": [],
        "columns": list(features),
        "schema": [
            {
                "column": name,
                "type": str(feature),
                "nullable": "unknown",
            }
            for name, feature in features.items()
        ],
        "schema_variants": 1,
        "size_metadata_source": (
            "dataset_info" if split_info is not None else "dataset_viewer"
        ),
    }
    if filter_column and filter_value:
        try:
            filtered_rows = sum(1 for _ in stream)
        except Exception as error:
            raise ValueError(
                f"Could not count filtered Hugging Face rows: {error}"
            ) from error
        source_rows = inventory["rows"]
        source_memory_bytes = inventory["memory_bytes"]
        inventory.update(
            {
                "rows": filtered_rows,
                "source_rows": source_rows,
                "memory_bytes": (
                    round(source_memory_bytes * filtered_rows / source_rows)
                    if source_rows
                    else 0
                ),
                "memory_bytes_estimated": True,
                "filter_column": filter_column,
                "filter_value": filter_value,
            }
        )
        if inventory["hub_file_bytes"] is None:
            inventory["bytes"] = inventory["memory_bytes"]
            inventory["bytes_estimated"] = True
    return inventory


def _huggingface_viewer_split_size(
    dataset_id: str,
    *,
    config: str | None,
    split: str,
    token: str | None,
) -> dict[str, Any]:
    """Fetch size metadata for exactly one configuration and split."""

    query = urllib.parse.urlencode({"dataset": dataset_id})
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{HUGGINGFACE_DATASET_VIEWER_SIZE_URL}?{query}",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Hugging Face Dataset Viewer size lookup failed: {error}"
        ) from error
    split_sizes = payload.get("size", {}).get("splits", [])
    matches = [
        item
        for item in split_sizes
        if isinstance(item, dict)
        and item.get("split") == split
        and (config is None or item.get("config") == config)
    ]
    if config is None and len(matches) > 1:
        default_matches = [item for item in matches if item.get("config") == "default"]
        matches = default_matches or matches
    if len(matches) != 1 or matches[0].get("num_rows") is None:
        requested = f"{config or '<default>'}/{split}"
        raise ValueError(
            f"Hugging Face Dataset Viewer has no unambiguous size entry for {requested}. "
            "The selected source may have incomplete or failed Viewer processing."
        )
    return matches[0]


def sample_parquet_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
    *, read_workers: int = 1, token: str | None = None,
) -> list[dict[str, Any]]:
    """Uniformly sample logical rows while reading only selected row groups."""

    import pyarrow.parquet as pq

    total_rows = int(inventory["rows"])
    if total_rows <= 0 or sample_size <= 0:
        return []

    selected_columns = list(columns)
    invalid = set(selected_columns) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Columns are not shared by every shard: {sorted(invalid)}")

    rng = random.Random(seed)
    global_indices = rng.sample(range(total_rows), k=min(sample_size, total_rows))
    row_groups = inventory["row_groups"]
    group_ends = [group["start"] + group["rows"] for group in row_groups]
    requested: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)

    for global_index in global_indices:
        group_position = bisect.bisect_right(group_ends, global_index)
        group = row_groups[group_position]
        local_index = global_index - group["start"]
        requested[(group["path"], group["row_group"])].append(
            (global_index, local_index)
        )

    def read_group(item):
        import fsspec

        (path_text, group_index), positions = item
        options = {"token": token} if path_text.startswith("hf://") else {}
        if path_text.startswith("https://huggingface.co/") and token:
            options = {"headers": {"Authorization": f"Bearer {token}"}}
        filesystem, path = fsspec.core.url_to_fs(path_text, **options)
        positions = sorted(positions, key=lambda position: position[1])
        cursor = 0
        found = {}
        with filesystem.open(path, "rb", block_size=1024 * 1024) as stream:
            # Avoid Arrow's background read-ahead buffers on top of each reader's
            # file cache. A row-count batch limit is not a byte/RAM limit.
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            offset = 0
            # Each reader decodes small batches, without an additional Arrow thread pool.
            for batch in parquet.iter_batches(batch_size=1024, row_groups=[group_index],
                                               columns=selected_columns, use_threads=False):
                end = cursor
                while end < len(positions) and positions[end][1] < offset + batch.num_rows:
                    end += 1
                selected = positions[cursor:end]
                records = batch.take([local - offset for _, local in selected]).to_pylist() if selected else []
                for (global_index, local), record in zip(selected, records):
                    record[f"{VIEWER_PREFIX}row_index"] = global_index
                    record[f"{VIEWER_PREFIX}file"] = str(inventory.get("source_uri") or path_text)
                    record[f"{VIEWER_PREFIX}row_group"] = group_index
                    record[f"{VIEWER_PREFIX}identity_stable"] = True
                    found[global_index] = record
                cursor = end
                if cursor == len(positions):
                    break
                offset += batch.num_rows
        if cursor != len(positions):
            raise ValueError("Parquet rows changed since inspection. Reload the source before sampling.")
        return found

    sampled_by_index = {}
    for found in ordered_parallel_reads(read_group, list(requested.items()), read_workers):
        sampled_by_index.update(found)

    return [sampled_by_index[index] for index in global_indices]


def has_indexed_parquet_preview(inventory: dict[str, Any]) -> bool:
    """Adapters and filtered streams must retain their original row semantics."""
    return bool(
        inventory.get("row_groups") and inventory.get("format") in {"parquet", "huggingface"}
        and not inventory.get("dataset_loader") and not inventory.get("filter_column")
        and not inventory.get("row_filters")
    )


def sample_instance_rows(inventory, sample_size, seed, columns, *, read_workers=1):
    """Preview indexed Parquet instances directly; retain streaming for other sources."""
    if has_indexed_parquet_preview(inventory):
        return sample_parquet_rows(
            inventory, sample_size, seed, columns, read_workers=read_workers,
            token=os.getenv("HF_TOKEN") or os.getenv("HF_token"),
        )
    return sample_dataset_rows(inventory, sample_size, seed, columns)


def sample_json_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly sample selected fields from a JSON array dataset."""

    selected_columns = list(columns)
    invalid = set(selected_columns) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Unknown JSON columns: {sorted(invalid)}")
    records = _read_json_records(Path(inventory["path"]))
    if not records or sample_size <= 0:
        return []
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), k=min(sample_size, len(records)))
    sampled = []
    for index in indices:
        record = {column: records[index].get(column) for column in selected_columns}
        record[f"{VIEWER_PREFIX}row_index"] = index
        record[f"{VIEWER_PREFIX}file"] = str(
            inventory.get("source_uri") or inventory["path"]
        )
        sampled.append(record)
    return sampled


def _reservoir_records(
    records: Any,
    *,
    sample_size: int,
    seed: int,
    columns: Sequence[str],
    source_uri: str,
) -> list[dict[str, Any]]:
    """Uniformly sample a one-pass record iterator with bounded memory."""

    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    for index, source_record in enumerate(records):
        record = {column: source_record.get(column) for column in columns}
        record[f"{VIEWER_PREFIX}row_index"] = index
        record[f"{VIEWER_PREFIX}file"] = source_uri
        if len(reservoir) < sample_size:
            reservoir.append((index, record))
            continue
        replacement = rng.randrange(index + 1)
        if replacement < sample_size:
            reservoir[replacement] = (index, record)
    reservoir.sort(key=lambda item: item[0])
    return [record for _, record in reservoir]


def sample_csv_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly reservoir-sample a CSV file."""

    selected = list(columns)
    invalid = set(selected) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Unknown CSV columns: {sorted(invalid)}")
    if sample_size <= 0:
        return []
    path = Path(inventory["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return _reservoir_records(
            csv.DictReader(stream),
            sample_size=sample_size,
            seed=seed,
            columns=selected,
            source_uri=str(inventory.get("source_uri") or path),
        )


def sample_json_lines_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly reservoir-sample a JSONL/NDJSON file."""

    selected = list(columns)
    invalid = set(selected) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Unknown JSONL columns: {sorted(invalid)}")
    if sample_size <= 0:
        return []
    path = Path(inventory["path"])

    def records() -> Any:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                value = json.loads(line)
                yield dict(value) if isinstance(value, dict) else {"text": value}

    return _reservoir_records(
        records(),
        sample_size=sample_size,
        seed=seed,
        columns=selected,
        source_uri=str(inventory.get("source_uri") or path),
    )


def sample_text_line_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly reservoir-sample non-empty text lines."""

    invalid = set(columns) - {"text"}
    if invalid:
        raise ValueError(f"Unknown text columns: {sorted(invalid)}")
    if sample_size <= 0:
        return []
    path = Path(inventory["path"])

    def records() -> Any:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                text = line.strip()
                if text:
                    yield {"text": text}

    return _reservoir_records(
        records(),
        sample_size=sample_size,
        seed=seed,
        columns=list(columns),
        source_uri=str(inventory.get("source_uri") or path),
    )


def _load_inventory_huggingface_stream(inventory, *, token=None, filters=()):
    """Preserve explicit shard selection and legacy adapter inventories."""
    shards = inventory.get("dataset_shards")
    if shards is not None:
        if not shards:
            raise ValueError("Select at least one Hugging Face shard")
        from .huggingface import load_selected_huggingface_stream

        return load_selected_huggingface_stream(
            inventory["dataset_id"], inventory.get("dataset_config"),
            inventory["dataset_split"], inventory.get("dataset_revision"), shards,
            loader=inventory.get("dataset_loader"), token=token,
            filters=list(filters) if filters else None,
            parquet_features=inventory.get("parquet_features"),
        )
    return load_huggingface_stream(
        inventory["dataset_id"], inventory.get("dataset_config"),
        split=inventory["dataset_split"], revision=inventory.get("dataset_revision"),
        token=token or None, filters=filters,
        parquet_files=inventory.get("dataset_parquet_files", ()),
    ).stream


def sample_huggingface_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
    *,
    token: str | None = None,
    shuffle_buffer: int = 1_000,
) -> list[dict[str, Any]]:
    """Sample through a bounded streaming shuffle buffer."""

    selected_columns = list(columns)
    invalid = set(selected_columns) - set(inventory["columns"])
    if invalid:
        raise ValueError(f"Unknown Hugging Face columns: {sorted(invalid)}")
    if sample_size <= 0:
        return []
    if inventory.get("filter_column") and inventory.get("filter_value"):
        try:
            stream = _load_inventory_huggingface_stream(
                inventory, token=token,
                filters=[
                    (
                        inventory["filter_column"],
                        "==",
                        inventory["filter_value"],
                    )
                ],
            )
            stream = stream.shuffle(
                seed=seed,
                buffer_size=max(sample_size, min(int(shuffle_buffer), 200)),
            )
            sampled = []
            for index, source_record in enumerate(stream.take(sample_size)):
                if (
                    source_record.get(inventory["filter_column"])
                    != inventory["filter_value"]
                ):
                    continue
                record = {
                    column: source_record.get(column) for column in selected_columns
                }
                record[f"{VIEWER_PREFIX}row_index"] = index
                record[f"{VIEWER_PREFIX}identity_stable"] = False
                record[f"{VIEWER_PREFIX}file"] = (
                    f"hf://datasets/{inventory['dataset_id']}/"
                    f"{inventory['dataset_split']}?"
                    f"{inventory['filter_column']}={inventory['filter_value']}"
                )
                sampled.append(record)
            return sampled
        except Exception as error:
            raise ValueError(
                f"Could not sample filtered Hugging Face rows: {error}"
            ) from error

    try:
        import datasets  # noqa: F401 - optional dependency check
    except ImportError as error:
        raise ValueError("Hugging Face datasets support requires `datasets`") from error
    try:
        stream = _load_inventory_huggingface_stream(inventory, token=token)
        stream = stream.shuffle(
            seed=seed,
            buffer_size=max(sample_size, min(int(shuffle_buffer), 5_000)),
        )
        sampled = []
        for index, source_record in enumerate(stream.take(sample_size)):
            record = {column: source_record.get(column) for column in selected_columns}
            source_id = source_record.get("id")
            record[f"{VIEWER_PREFIX}row_index"] = (
                source_id if source_id is not None else index
            )
            record[f"{VIEWER_PREFIX}identity_stable"] = source_id is not None
            record[f"{VIEWER_PREFIX}file"] = (
                f"hf://datasets/{inventory['dataset_id']}/{inventory['dataset_split']}"
            )
            sampled.append(record)
        return sampled
    except Exception as error:
        raise ValueError(
            f"Could not stream Hugging Face dataset rows: {error}"
        ) from error


def sample_dataset_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Uniformly sample a supported dataset without changing the UI contract."""

    if inventory.get("format") == "huggingface":
        token = os.getenv("HF_TOKEN") or os.getenv("HF_token")
        return sample_huggingface_rows(
            inventory, sample_size, seed, columns, token=token
        )
    if inventory.get("format") == "kaggle":
        return sample_kaggle_workbook_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "kaggle_text":
        return sample_kaggle_text_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "json":
        return sample_json_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "csv":
        return sample_csv_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "jsonl":
        return sample_json_lines_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "text":
        return sample_text_line_rows(inventory, sample_size, seed, columns)
    if inventory.get("format") == "xlsx":
        return sample_workbook_rows(
            inventory,
            sample_size,
            seed,
            columns,
            source_uri=inventory.get("source_uri"),
        )
    return sample_parquet_rows(inventory, sample_size, seed, columns)
