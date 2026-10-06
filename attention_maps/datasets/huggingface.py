"""Shared streaming loader for Hub datasets, including legacy script repos."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from .crosssum import CROSSSUM_DATASET_ID, load_crosssum
from .flores import FACEBOOK_FLORES_DATASET_ID, load_facebook_flores
from .indicgenbench import FLORES_IN_DATASET_ID, load_flores_in
from .nepfake import NEPFAKE_DATASET_ID, load_nepfake


@dataclass(frozen=True)
class HuggingFaceStream:
    stream: Any
    config: str | None
    parquet_files: tuple[str, ...] = ()
    revision: str | None = None
    loading_strategy: str = "streaming"
    hub_file_bytes: int | None = None
    hub_file_size_basis: str = "original files"
    hub_file_size_scope: str = "selected split"


def load_huggingface_stream(
    dataset_id: str,
    config: str | None = None,
    *,
    split: str,
    revision: str | None = None,
    token: str | None = None,
    filters: Sequence[tuple[str, str, Any]] = (),
    parquet_files: Sequence[str] = (),
) -> HuggingFaceStream:
    """Apply a source adapter or stream normally with a legacy-script fallback.

    Dataset Viewer conversions track the default source revision. Never silently
    substitute them for a requested source commit, tag, or non-default branch.
    Resolved files are returned so inspection and sampling use the same subset.
    """

    if dataset_id == CROSSSUM_DATASET_ID:
        stream, resolved_revision, file_bytes = load_crosssum(
            config, split=split, revision=revision, token=token, filters=filters
        )
        return HuggingFaceStream(
            stream, config, revision=resolved_revision,
            loading_strategy="memory_mapped_archive", hub_file_bytes=file_bytes,
            hub_file_size_basis="compressed archive",
            hub_file_size_scope="selected language pair (all splits)",
        )

    if dataset_id == FACEBOOK_FLORES_DATASET_ID:
        stream, resolved_revision, file_bytes = load_facebook_flores(
            config, split=split, revision=revision, token=token, filters=filters
        )
        return HuggingFaceStream(
            stream, config, revision=resolved_revision, hub_file_bytes=file_bytes,
        )

    if dataset_id == FLORES_IN_DATASET_ID:
        stream, resolved_revision = load_flores_in(
            config, split=split, revision=revision, token=token, filters=filters
        )
        return HuggingFaceStream(
            stream, config, revision=resolved_revision,
            loading_strategy="memory_mapped_json",
        )

    if dataset_id == NEPFAKE_DATASET_ID:
        stream, resolved_revision = load_nepfake(
            config, split=split, revision=revision, token=token, filters=filters
        )
        return HuggingFaceStream(
            stream, "default", revision=resolved_revision,
            loading_strategy="memory_mapped_csv",
        )

    from datasets import load_dataset

    options = {"filters": list(filters)} if filters else {}
    if not parquet_files:
        try:
            stream = load_dataset(
                dataset_id,
                config,
                split=split,
                streaming=True,
                revision=revision,
                token=token or None,
                **options,
            )
            return HuggingFaceStream(stream, config, revision=revision)
        except RuntimeError as error:
            if "Dataset scripts are no longer supported" not in str(error):
                raise
        _validate_conversion_revision(revision)
        config, parquet_files = _converted_parquet_files(
            dataset_id, config=config, split=split, token=token
        )

    _validate_conversion_revision(revision)
    stream = load_dataset(
        "parquet",
        data_files={split: list(parquet_files)},
        split=split,
        streaming=True,
        token=token or None,
        **options,
    )
    return HuggingFaceStream(stream, config, tuple(parquet_files), revision=revision)


def _validate_conversion_revision(revision: str | None) -> None:
    if revision not in (None, "main"):
        raise ValueError(
            "This dataset uses an unsupported legacy loading script. Its "
            "converted Parquet files cannot guarantee the requested source "
            f"revision {revision!r}. Use a script-free dataset export at that "
            "revision, or leave Revision empty to use the default conversion."
        )


def _converted_parquet_files(
    dataset_id: str,
    *,
    config: str | None,
    split: str,
    token: str | None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve a complete, unambiguous configuration/split without running code."""

    from huggingface_hub import get_token

    query = urllib.parse.urlencode({"dataset": dataset_id})
    headers = {"Accept": "application/json"}
    effective_token = token or get_token()
    if effective_token:
        headers["Authorization"] = f"Bearer {effective_token}"
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/parquet?{query}", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise ValueError(
            "This dataset uses an unsupported legacy loading script, and its "
            "converted Parquet files could not be resolved. Use a JSON/Parquet "
            "export of the dataset or retry the Hugging Face service."
        ) from error
    if not isinstance(payload, dict) or payload.get("partial"):
        raise ValueError("Hugging Face returned an incomplete Parquet conversion.")
    files = [
        item
        for item in payload.get("parquet_files", [])
        if isinstance(item, dict) and item.get("dataset") == dataset_id
    ]
    configurations = sorted({item["config"] for item in files if item.get("config")})
    if config is None:
        if "default" in configurations:
            config = "default"
        elif len(configurations) == 1:
            config = configurations[0]
        else:
            hint = " For Nepali, use configuration 'ne'." if "ne" in configurations else ""
            raise ValueError(
                "Select a Dataset configuration for this legacy dataset."
                f"{hint} Available configurations: {', '.join(configurations) or 'none'}."
            )
    selected = [
        item for item in files
        if item.get("config") == config and item.get("split") == split
    ]
    if not selected:
        raise ValueError(
            f"No converted Parquet files for {dataset_id} configuration "
            f"{config!r}, split {split!r}. Check the configuration and split."
        )
    prefix = f"https://huggingface.co/datasets/{dataset_id}/resolve/"
    urls = tuple(sorted(item.get("url", "") for item in selected))
    if any(not url.startswith(prefix) for url in urls):
        raise ValueError("Hugging Face returned an unexpected Parquet file URL.")
    return config, urls
