"""Consistent size buckets for inspected datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class DatasetSizeBucket:
    """The effective size and operational bucket for an inspected dataset."""

    label: str
    size_bytes: int | None
    size_basis: str


def bucket_dataset_size(size_bytes: int | None) -> str:
    """Classify bytes into stable, compute-oriented size buckets."""

    if size_bytes is None:
        return "Unknown"
    if size_bytes < 100 * MIB:
        return "Tiny (<100 MiB)"
    if size_bytes < GIB:
        return "Small (100 MiB–1 GiB)"
    if size_bytes < 10 * GIB:
        return "Medium (1–10 GiB)"
    if size_bytes < 100 * GIB:
        return "Large (10–100 GiB)"
    return "Very large (≥100 GiB)"


def dataset_size_bucket(inventory: dict[str, Any]) -> DatasetSizeBucket:
    """Bucket an inventory, preferring decoded size over compressed source size."""

    raw_size = inventory.get("memory_bytes")
    basis = "decoded dataset size"
    if raw_size is None:
        raw_size = inventory.get("bytes")
        basis = (
            "estimated source size"
            if inventory.get("bytes_estimated")
            else "source/disk size"
        )
    size = None if raw_size is None else max(0, int(raw_size))
    return DatasetSizeBucket(bucket_dataset_size(size), size, basis)
