"""Bounded inspection and sampling for explorer dataset sources."""

from __future__ import annotations

import bisect
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from attention_maps.datasets.kaggle import (
    sample_kaggle_text_rows,
    sample_kaggle_workbook_rows,
)

from .catalog import VIEWER_PREFIX

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
    signatures: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    """Inspect a supported dataset using its file extension."""

    suffixes = {Path(path_text).suffix.lower() for path_text, _, _ in signatures}
    if suffixes == {".parquet"}:
        return inspect_parquet(signatures)
    if suffixes == {".json"}:
        return inspect_json(signatures)
    raise ValueError(f"Unsupported or mixed dataset formats: {sorted(suffixes)}")


def inspect_huggingface_dataset(
    dataset_id: str,
    split: str,
    *,
    config: str | None = None,
    token: str | None = None,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> dict[str, Any]:
    """Read remote streaming metadata without materializing dataset rows."""

    try:
        from datasets import load_dataset
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
        stream = load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
            token=token or None,
            **load_options,
        )
        split_info = stream.info.splits.get(split)
        features = stream.features or {}
    except Exception as error:
        raise ValueError(f"Could not inspect Hugging Face dataset: {error}") from error
    if split_info is None:
        raise ValueError(f"Hugging Face dataset has no {split!r} split metadata")
    shard_lengths = list(getattr(split_info, "shard_lengths", None) or [])
    shard_count = len(shard_lengths) or int(getattr(stream, "num_shards", 1))
    inventory = {
        "format": "huggingface",
        "dataset_id": dataset_id,
        "dataset_config": config,
        "dataset_split": split,
        "rows": int(split_info.num_examples),
        "files": shard_count,
        "bytes": int(split_info.num_bytes),
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
    }
    if filter_column and filter_value:
        try:
            filtered_rows = sum(1 for _ in stream)
        except Exception as error:
            raise ValueError(
                f"Could not count filtered Hugging Face rows: {error}"
            ) from error
        source_rows = inventory["rows"]
        inventory.update(
            {
                "rows": filtered_rows,
                "source_rows": source_rows,
                "bytes": (
                    round(inventory["bytes"] * filtered_rows / source_rows)
                    if source_rows
                    else 0
                ),
                "bytes_estimated": True,
                "filter_column": filter_column,
                "filter_value": filter_value,
            }
        )
    return inventory


def sample_parquet_rows(
    inventory: dict[str, Any],
    sample_size: int,
    seed: int,
    columns: Sequence[str],
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

    sampled_by_index: dict[int, dict[str, Any]] = {}
    for (path_text, group_index), positions in requested.items():
        parquet = pq.ParquetFile(path_text)
        table = parquet.read_row_group(group_index, columns=selected_columns)
        for global_index, local_index in positions:
            record = table.slice(local_index, 1).to_pylist()[0]
            record[f"{VIEWER_PREFIX}row_index"] = global_index
            record[f"{VIEWER_PREFIX}file"] = path_text
            record[f"{VIEWER_PREFIX}row_group"] = group_index
            sampled_by_index[global_index] = record

    return [sampled_by_index[index] for index in global_indices]


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
        record[f"{VIEWER_PREFIX}file"] = inventory["path"]
        sampled.append(record)
    return sampled


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
            from datasets import load_dataset

            stream = load_dataset(
                inventory["dataset_id"],
                inventory.get("dataset_config"),
                split=inventory["dataset_split"],
                streaming=True,
                token=token or None,
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
        from datasets import load_dataset
    except ImportError as error:
        raise ValueError("Hugging Face datasets support requires `datasets`") from error
    try:
        stream = load_dataset(
            inventory["dataset_id"],
            inventory.get("dataset_config"),
            split=inventory["dataset_split"],
            streaming=True,
            token=token or None,
        )
        stream = stream.shuffle(
            seed=seed,
            buffer_size=max(sample_size, min(int(shuffle_buffer), 5_000)),
        )
        sampled = []
        for index, source_record in enumerate(stream.take(sample_size)):
            record = {column: source_record.get(column) for column in selected_columns}
            record[f"{VIEWER_PREFIX}row_index"] = source_record.get("id", index)
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
    return sample_parquet_rows(inventory, sample_size, seed, columns)
