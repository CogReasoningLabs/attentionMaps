"""Expose disk-backed source adapters through configuration and shard discovery."""

from __future__ import annotations

import re

from attention_maps.datasets.crosssum import CROSSSUM_DATASET_ID
from attention_maps.datasets.indicgenbench import FLORES_IN_DATASET_ID
from attention_maps.datasets.nepfake import NEPFAKE_DATASET_ID, NEPFAKE_CSV

from .file_formats import describe_dataset_formats


ADAPTER_DATASETS = {CROSSSUM_DATASET_ID, FLORES_IN_DATASET_ID, NEPFAKE_DATASET_ID}


def adapter_files(dataset_id: str, config: str, split: str) -> list[str]:
    """Return the indivisible source files consumed by one adapter selection."""
    if dataset_id == CROSSSUM_DATASET_ID:
        if split not in {"train", "validation", "test"}:
            raise ValueError("CrossSum split must be train, validation, or test")
        return [f"data/{config}_CrossSum.tar.bz2"]
    if dataset_id == FLORES_IN_DATASET_ID:
        suffix = {"validation": "dev", "test": "test"}.get(split)
        if suffix is None:
            raise ValueError("Flores-IN split must be validation or test")
        return [f"flores_en_{config}_{suffix}.json", f"flores_{config}_en_{suffix}.json"]
    if dataset_id == NEPFAKE_DATASET_ID and config == "default" and split == "train":
        return [NEPFAKE_CSV]
    raise ValueError("Unsupported source adapter configuration or split")


def adapter_configurations(dataset_id: str, file_sizes: dict) -> dict:
    """Discover selections from repository filenames without reading records."""
    if dataset_id == CROSSSUM_DATASET_ID:
        configs = sorted({
            match.group(1) for name in file_sizes
            if (match := re.fullmatch(r"data/([a-z_]+-[a-z_]+)_CrossSum\.tar\.bz2", name))
        })
        splits = ("train", "validation", "test")
    elif dataset_id == FLORES_IN_DATASET_ID:
        configs = sorted({
            match.group(1) for name in file_sizes
            if (match := re.fullmatch(r"flores_en_([a-z]{2,3})_(dev|test)\.json", name))
        })
        splits = ("validation", "test")
    elif dataset_id == NEPFAKE_DATASET_ID:
        configs, splits = ["default"], ("train",)
    else:
        raise ValueError("Unsupported source adapter")
    result = {}
    for config in configs:
        available = {}
        for split in splits:
            files = adapter_files(dataset_id, config, split)
            if all(name in file_sizes for name in files):
                available[split] = files
        if available:
            result[config] = available
    if not result:
        raise ValueError("No complete source adapter files were found at this revision")
    return result


def adapter_configuration(catalog: dict, config: str) -> dict:
    splits = {}
    for split, files in catalog["adapter_configs"][config].items():
        shards = [{
            "path": f"hf://datasets/{catalog['dataset_id']}@{catalog['revision']}/{name}",
            "name": name, "bytes": catalog["file_sizes"][name],
        } for name in files]
        splits[split] = {"rows": None, "memory_bytes": None, "shards": shards,
                         "file_formats": describe_dataset_formats(shards)}
    archive = catalog["dataset_id"] == CROSSSUM_DATASET_ID
    return {
        "dataset_id": catalog["dataset_id"], "revision": catalog["revision"],
        "config": config, "languages": catalog.get("languages", []),
        "dataset_loader": "source_adapter", "requires_complete_split": True,
        "schema": [], "splits": splits,
        "hub_file_size_basis": "compressed archive" if archive else "original files",
        "hub_file_size_scope": "selected language pair (all splits)" if archive else "selected split",
    }


def load_selected_adapter(dataset_id, config, split, revision, shards, *, token=None, filters=None):
    """Reject subsets or stale file references instead of silently broadening them."""
    expected = {
        f"hf://datasets/{dataset_id}@{revision}/{name}"
        for name in adapter_files(dataset_id, config, split)
    }
    if not revision or set(shards) != expected:
        raise ValueError("This source adapter requires all files for the selected split at its pinned revision")
    from attention_maps.datasets.huggingface import load_huggingface_stream

    return load_huggingface_stream(
        dataset_id, config, split=split, revision=revision, token=token,
        filters=filters or (),
    ).stream
