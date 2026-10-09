"""Revision-aware configuration, split, and physical-shard inspection."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import urllib.parse
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from .catalog import DatasetSpec
from .file_formats import describe_dataset_formats
from .source_imports import normalize_huggingface_id
from .parallel_reads import ordered_parallel_reads
from .huggingface_adapters import (
    ADAPTER_DATASETS, adapter_configuration, adapter_configurations, load_selected_adapter,
)


def discover_huggingface_dataset(
    dataset_id: str, *, revision: str | None = None, token: str | None = None
) -> dict[str, Any]:
    """Discover configurations and file sizes without downloading the corpus."""

    from datasets import get_dataset_config_names
    from huggingface_hub import HfApi

    dataset_id = normalize_huggingface_id(dataset_id)
    try:
        info = HfApi(token=token).dataset_info(
            dataset_id, revision=revision or "main", files_metadata=True
        )
        resolved_revision = info.sha
        adapter_configs = None
        try:
            if dataset_id in ADAPTER_DATASETS:
                adapter_configs = adapter_configurations(
                    dataset_id, {item.rfilename: item.size for item in info.siblings or []}
                )
                configs = list(adapter_configs)
            else:
                configs = get_dataset_config_names(
                    dataset_id, revision=resolved_revision, token=token
                )
            raw_data_files = None
        except Exception as error:
            if (dataset_id != "MBZUAI/Bactrian-X"
                    or "Dataset scripts are no longer supported" not in str(error)):
                raise
            # Bactrian-X's builder maps data/<config>.json.gz to train.
            # Read those files at the pinned revision without executing it.
            raw_data_files = {}
            for item in info.siblings or []:
                match = re.fullmatch(r"data/([A-Za-z0-9_-]+)\.json\.gz", item.rfilename)
                if match:
                    raw_data_files[match.group(1)] = item.rfilename
            if not raw_data_files:
                raise ValueError("Legacy dataset script is unsupported and no data/<config>.json.gz files were found") from error
            configs = sorted(raw_data_files)
    except Exception as error:
        raise ValueError(f"Could not discover Hugging Face dataset: {error}") from error
    card = info.card_data.to_dict() if info.card_data else {}
    languages = card.get("language") or []
    if isinstance(languages, str):
        languages = [languages]
    catalog = {
        "dataset_id": dataset_id,
        "requested_revision": revision or "main",
        "revision": resolved_revision,
        "configs": list(configs),
        "languages": languages,
        "file_sizes": {item.rfilename: item.size for item in info.siblings or []},
    }
    if raw_data_files is not None:
        catalog.update(dataset_loader="json", raw_data_files=raw_data_files)
    if adapter_configs is not None:
        catalog.update(dataset_loader="source_adapter", adapter_configs=adapter_configs)
    return catalog


def _repository_path(url: str, dataset_id: str) -> str | None:
    prefix = f"hf://datasets/{dataset_id}"
    if url.startswith(prefix + "@"):
        return url[len(prefix) + 1:].split("/", 1)[1]
    if url.startswith(prefix + "/"):
        return url[len(prefix) + 1:]
    parsed = urlparse(url)
    prefix = f"/datasets/{dataset_id}/resolve/"
    if parsed.netloc == "huggingface.co" and parsed.path.startswith(prefix):
        parts = parsed.path[len(prefix):].split("/", 1)
        return unquote(parts[1]) if len(parts) == 2 else None
    return None


def inspect_huggingface_configuration(
    catalog: dict[str, Any], config: str, *, token: str | None = None
) -> dict[str, Any]:
    """Resolve the actual files belonging to every split of one configuration."""

    from datasets import load_dataset_builder

    if config not in catalog["configs"]:
        raise ValueError(f"Unknown dataset configuration: {config}")
    if catalog.get("dataset_loader") == "source_adapter":
        return adapter_configuration(catalog, config)
    if catalog.get("dataset_loader") == "json":
        relative = catalog["raw_data_files"][config]
        shard = {"path": f"hf://datasets/{catalog['dataset_id']}@{catalog['revision']}/{relative}",
                 "name": relative, "bytes": catalog["file_sizes"].get(relative)}
        return {"dataset_id": catalog["dataset_id"], "revision": catalog["revision"],
                "config": config, "languages": catalog.get("languages", []),
                "dataset_loader": "json", "schema": [],
                "splits": {"train": {"rows": None, "memory_bytes": None,
                                     "shards": [shard], "file_formats": describe_dataset_formats([shard])}}}
    try:
        builder = load_dataset_builder(
            catalog["dataset_id"], name=config, revision=catalog["revision"], token=token
        )
    except Exception as error:
        raise ValueError(f"Could not inspect configuration {config!r}: {error}") from error
    splits = {}
    for split, paths in (builder.config.data_files or {}).items():
        info = (builder.info.splits or {}).get(split)
        shards = []
        for path in dict.fromkeys(map(str, paths)):
            relative = _repository_path(path, catalog["dataset_id"])
            shards.append({
                "path": path,
                "name": relative or path,
                "bytes": catalog["file_sizes"].get(relative),
            })
        reported_rows = getattr(info, "num_examples", None)
        # A zero-valued SplitInfo for nonempty data files often means the
        # builder has no prepared statistics, not that the files have no rows.
        if reported_rows == 0 and shards:
            reported_rows = None
        splits[str(split)] = {
            "rows": reported_rows,
            "memory_bytes": getattr(info, "num_bytes", None) if reported_rows is not None else None,
            "shards": shards,
            "file_formats": describe_dataset_formats(shards),
        }
    if not splits:
        raise ValueError("This configuration exposes no selectable data files/splits.")
    features = builder.info.features or {}
    return {
        "dataset_id": catalog["dataset_id"],
        "revision": catalog["revision"],
        "config": config,
        "languages": catalog.get("languages", []),
        "splits": splits,
        "schema": [
            {"column": name, "type": str(feature), "nullable": "unknown"}
            for name, feature in features.items()
        ],
    }


def load_selected_huggingface_stream(
    dataset_id: str, config: str, split: str, revision: str, shards: Sequence[str],
    *, loader: str | None = None, token: str | None = None, filters=None,
    parquet_features: dict | None = None,
):
    """Load pinned source files, bypassing obsolete repository builder scripts."""
    if loader == "source_adapter":
        return load_selected_adapter(
            dataset_id, config, split, revision, shards, token=token, filters=filters,
        )
    from datasets import load_dataset

    kwargs = {"split": split, "data_files": {split: list(shards)}, "streaming": True, "token": token}
    if filters is not None:
        kwargs["filters"] = filters
    if parquet_features is not None:
        from datasets import Features

        # Physical schemas take precedence over a first-shard or dataset-card
        # null type, which cannot accept strings present in other selected files.
        kwargs["features"] = Features.from_dict(parquet_features)
    if loader == "json":
        return load_dataset("json", **kwargs)
    return load_dataset(dataset_id, config, revision=revision, **kwargs)


def select_huggingface_shards(
    configuration: dict[str, Any], split: str, shards: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    if split not in configuration["splits"]:
        raise ValueError(f"Unknown split {split!r} for {configuration['config']!r}")
    available = configuration["splits"][split]["shards"]
    if shards is None:
        return list(available)
    requested = set(shards)
    unknown = requested - {item["path"] for item in available}
    if unknown:
        raise ValueError(f"Shards do not belong to this configuration/split: {sorted(unknown)}")
    if not requested:
        raise ValueError("Select at least one shard.")
    if configuration.get("requires_complete_split") and requested != {item["path"] for item in available}:
        raise ValueError("This source adapter requires all files for the selected split")
    return [item for item in available if item["path"] in requested]


def selected_source_spec(
    spec: DatasetSpec, configuration: dict[str, Any], split: str,
    shards: Sequence[str] | None = None,
) -> DatasetSpec:
    """Include every selection dimension in workspace/cache identity."""

    selected = select_huggingface_shards(configuration, split, shards)
    identity = {
        "dataset": configuration["dataset_id"], "config": configuration["config"],
        "split": split, "revision": configuration["revision"],
        "shards": [item["path"] for item in selected],
        "filter": [spec.filter_column, spec.filter_value],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return replace(
        spec, key=f"{spec.key}:selection:{digest}",
        label=f"{configuration['dataset_id']} · {configuration['config']} · {split}",
        dataset_config=configuration["config"], dataset_split=split,
        dataset_revision=configuration["revision"],
        dataset_shards=tuple(item["path"] for item in selected),
    )


def _parquet_metadata(path: str, *, token: str | None) -> dict[str, Any]:
    """Read range-addressed Parquet footers, never materialize the data table."""

    import fsspec
    import pyarrow.parquet as pq

    options = {"token": token} if path.startswith("hf://") else {}
    if path.startswith("https://huggingface.co/") and token:
        options = {"headers": {"Authorization": f"Bearer {token}"}}
    filesystem, remote_path = fsspec.core.url_to_fs(path, **options)
    with filesystem.open(remote_path, "rb", block_size=64 * 1024) as stream:
        parquet = pq.ParquetFile(stream)
        metadata = parquet.metadata
        return {
            "rows": metadata.num_rows,
            "arrow_schema": parquet.schema_arrow,
            "row_groups": [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)],
            "memory_bytes": sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)),
            "schema": [
                {"column": field.name, "type": str(field.type), "nullable": str(field.nullable)}
                for field in parquet.schema_arrow
            ],
        }


def _viewer_split_metadata(configuration: dict, split: str, token: str | None) -> dict | None:
    """Use cached Viewer counts only when its revision matches the selected commit."""

    url = "https://datasets-server.huggingface.co/size?" + urllib.parse.urlencode(
        {"dataset": configuration["dataset_id"]}
    )
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            if response.headers.get("X-Revision") != configuration["revision"]:
                return None
            payload = json.load(response)
    except (OSError, ValueError):
        return None
    if payload.get("partial") or payload.get("pending") or payload.get("failed"):
        return None
    matches = [item for item in payload.get("size", {}).get("splits", [])
               if item.get("config") == configuration["config"] and item.get("split") == split]
    return matches[0] if len(matches) == 1 else None


def inspect_huggingface_selection(
    configuration: dict[str, Any], split: str, *,
    shards: Sequence[str] | None = None, token: str | None = None,
    filter_column: str | None = None, filter_value: str | None = None,
    metadata_only: bool = False,
    read_workers: int = 1,
) -> dict[str, Any]:
    """Return inventory for precisely the selected physical files."""

    selected = select_huggingface_shards(configuration, split, shards)
    split_info = configuration["splits"][split]
    if bool(filter_column) != bool(filter_value):
        raise ValueError("A Hugging Face filter requires both a column and value")
    if filter_column and metadata_only:
        raise ValueError("Metadata-only inspection cannot count filtered records")
    if configuration.get("dataset_loader") == "source_adapter" and not metadata_only:
        from .inspection import inspect_huggingface_dataset

        inventory = inspect_huggingface_dataset(
            configuration["dataset_id"], split, config=configuration["config"],
            revision=configuration["revision"], token=token,
            filter_column=filter_column, filter_value=filter_value,
        )
        sizes = [item["bytes"] for item in selected]
        storage = sum(sizes) if all(size is not None for size in sizes) else inventory["hub_file_bytes"]
        inventory.update(
            provider="huggingface", dataset_loader="source_adapter",
            dataset_shards=[item["path"] for item in selected],
            shards=selected, total_split_shards=len(selected), files=len(selected),
            file_formats=describe_dataset_formats(selected), metadata_only=False,
            selection_scope="split", declared_languages=configuration.get("languages", []),
            bytes=storage, hub_file_bytes=storage,
            hub_file_size_basis=configuration["hub_file_size_basis"],
            hub_file_size_scope=configuration["hub_file_size_scope"],
        )
        return inventory
    whole_split = len(selected) == len(split_info["shards"])
    rows = split_info["rows"] if whole_split else None
    memory = split_info["memory_bytes"] if whole_split else None
    schema = configuration["schema"]
    source = "split metadata"
    variants = 1
    row_groups = []
    parquet_features = None
    parquet_only = all(PurePosixPath(urlparse(item["path"]).path).suffix.lower() == ".parquet" for item in selected)
    if not metadata_only and whole_split and rows is None and not parquet_only:
        viewer = _viewer_split_metadata(configuration, split, token)
        if viewer is not None and viewer.get("num_rows") is not None:
            rows = int(viewer["num_rows"])
            memory = viewer.get("num_bytes_memory")
            source = "Dataset Viewer (matching revision and split)"
    # Builder split counts can be missing, zero, or stale for data-files-based
    # datasets. Exact random sampling needs the selected files' actual row count.
    if not metadata_only and (parquet_only or rows is None or not schema):
        if parquet_only:
            metadata = ordered_parallel_reads(
                lambda item: _parquet_metadata(item["path"], token=token), selected, read_workers,
            )
            rows = sum(item["rows"] for item in metadata)
            memory = sum(item["memory_bytes"] for item in metadata)
            schemas = [item["schema"] for item in metadata]
            common = set.intersection(*(set(field["column"] for field in value) for value in schemas))
            schema = [field for field in schemas[0] if field["column"] in common]
            variants = len({json.dumps(value, sort_keys=True) for value in schemas})
            source = "Parquet footers (selected shards)"
            if all("arrow_schema" in item for item in metadata):
                import pyarrow as pa
                from datasets import Features

                try:
                    unified = pa.unify_schemas([item["arrow_schema"] for item in metadata])
                except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
                    raise ValueError(f"Selected Parquet shards have incompatible column types: {error}") from error
                # Nulls can adopt a concrete type; incompatible non-null types
                # are reported instead of discarding records or stringifying them.
                parquet_features = Features.from_arrow_schema(unified).to_dict()
                schema = [{"column": field.name, "type": str(field.type), "nullable": str(field.nullable)}
                          for field in unified if field.name in common]
            if all("row_groups" in item for item in metadata):
                start = 0
                for shard, item in zip(selected, metadata):
                    for index, count in enumerate(item["row_groups"]):
                        if count:
                            row_groups.append({"path": shard["path"], "row_group": index,
                                               "start": start, "rows": count})
                        start += count
        elif not schema:
            stream = load_selected_huggingface_stream(
                configuration["dataset_id"], configuration["config"], split,
                configuration["revision"], [item["path"] for item in selected],
                loader=configuration.get("dataset_loader"), token=token,
            )
            # Resolving features reads a bounded first batch, not the whole split.
            features = stream.features
            if not features:
                first = next(iter(stream), {})
                from .inspection import _json_value_type

                schema = [{"column": name, "type": _json_value_type(value), "nullable": "unknown"}
                          for name, value in first.items()]
            else:
                schema = [{"column": name, "type": str(value), "nullable": "unknown"}
                          for name, value in features.items()]
    sizes = [item["bytes"] for item in selected]
    storage = sum(sizes) if all(size is not None for size in sizes) else None
    inventory = {
        "format": "huggingface", "provider": "huggingface", "dataset_id": configuration["dataset_id"],
        "dataset_config": configuration["config"], "dataset_split": split,
        "dataset_revision": configuration["revision"],
        "dataset_loader": configuration.get("dataset_loader"),
        "dataset_shards": [item["path"] for item in selected],
        "file_formats": describe_dataset_formats(selected), "metadata_only": metadata_only,
        "shards": selected, "total_split_shards": len(split_info["shards"]),
        "selection_scope": "split" if whole_split else "selected shards",
        "rows": rows, "files": len(selected), "bytes": storage,
        "hub_file_bytes": storage, "memory_bytes": memory,
        "hub_file_size_basis": configuration.get("hub_file_size_basis", "original files"),
        "hub_file_size_scope": configuration.get("hub_file_size_scope", "selected split" if whole_split else "selected shards"),
        "memory_bytes_estimated": source.startswith("Parquet"),
        "streaming": True, "row_groups": row_groups, "schema": schema,
        "columns": [field["column"] for field in schema], "schema_variants": variants,
        "size_metadata_source": source, "declared_languages": configuration.get("languages", []),
    }
    if parquet_features is not None:
        inventory["parquet_features"] = parquet_features
    if bool(filter_column) != bool(filter_value):
        raise ValueError("A Hugging Face filter requires both a column and value")
    if filter_column and metadata_only:
        raise ValueError("Metadata-only inspection cannot count filtered records")
    if filter_column:
        stream = load_selected_huggingface_stream(
            configuration["dataset_id"], configuration["config"], split,
            configuration["revision"], inventory["dataset_shards"],
            loader=configuration.get("dataset_loader"), token=token,
            filters=[(filter_column, "==", filter_value)],
            parquet_features=parquet_features,
        )
        count = sum(record.get(filter_column) == filter_value for record in stream)
        inventory.update(
            source_rows=rows, rows=count, filter_column=filter_column, filter_value=filter_value,
            memory_bytes=round(memory * count / rows) if memory is not None and rows else None,
            memory_bytes_estimated=True,
        )
    return inventory
