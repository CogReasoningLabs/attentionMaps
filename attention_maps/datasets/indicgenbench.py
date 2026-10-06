"""Disk-backed reader for the nested JSON in Google's Flores-IN benchmark."""

from __future__ import annotations

import re
from typing import Any, Sequence


FLORES_IN_DATASET_ID = "google/IndicGenBench_flores_in"
_FILE_PATTERN = re.compile(r"flores_(?:en_([a-z]+)|([a-z]+)_en)_(dev|test)\.json")
_SPLIT_FILES = {"validation": "dev", "test": "test"}


def load_flores_in(
    config: str | None,
    *,
    split: str,
    revision: str | None,
    token: str | None,
    filters: Sequence[tuple[str, str, Any]] = (),
) -> tuple[Any, str]:
    """Cache one language's two translation directions as memory-mapped Arrow.

    The Hub viewer is disabled and provides neither sizes nor row counts. The
    JSON reader extracts only ``examples`` and indexes the selected files once
    on disk. No guessed split counts or canary-envelope records are emitted.
    ``config`` is this adapter's language selector, not a native Hub config.
    """

    if split not in _SPLIT_FILES:
        raise ValueError(
            "IndicGenBench Flores-IN has no train split. Choose split "
            "'validation' (dev files) or 'test', and configuration 'ne' for Nepali. "
            "This is an evaluation benchmark."
        )
    if not config or not re.fullmatch(r"[a-z]{2,3}", config):
        raise ValueError(
            "Select a language in Dataset configuration for IndicGenBench "
            "Flores-IN; use 'ne' for Nepali. Splits are 'validation' and 'test'."
        )
    if any(operator != "==" for _, operator, _ in filters):
        raise ValueError("Flores-IN row filters support equality only")

    from datasets import Features, SplitDict, SplitInfo, Value, load_dataset
    from huggingface_hub import HfApi, hf_hub_url

    info = HfApi().dataset_info(
        FLORES_IN_DATASET_ID, revision=revision or "main", token=token or None
    )
    if not info.sha:
        raise ValueError("Could not resolve the Flores-IN source revision")
    available = {item.rfilename for item in info.siblings or []}
    languages = sorted({
        match.group(1) or match.group(2)
        for name in available
        if (match := _FILE_PATTERN.fullmatch(name))
    })
    names = [
        f"flores_{direction}_{_SPLIT_FILES[split]}.json"
        for direction in (f"en_{config}", f"{config}_en")
    ]
    if not all(name in available for name in names):
        raise ValueError(
            f"Flores-IN has no complete {config!r}/{split} file pair at this "
            f"revision. Available languages: {', '.join(languages)}."
        )
    urls = [
        hf_hub_url(FLORES_IN_DATASET_ID, name, repo_type="dataset", revision=info.sha)
        for name in names
    ]
    data = load_dataset(
        "json",
        data_files={split: urls},
        field="examples",
        features=Features({
            column: Value("string")
            for column in ("source", "target", "lang", "translation_direction")
        }),
        split=split,
        keep_in_memory=False,
        token=token or None,
    )
    if not len(data):
        raise ValueError("The selected Flores-IN files contain no examples")
    stream = data.to_iterable_dataset(num_shards=min(len(names), len(data)))
    # to_iterable_dataset does not retain split metadata. Restore measured sizes
    # from the Arrow cache so inspection does not call the disabled Hub viewer.
    stream.info.splits = SplitDict({
        split: SplitInfo(name=split, num_examples=len(data), num_bytes=data.data.nbytes)
    })
    stream.info.download_size = data.info.download_size
    stream.info.config_name = config
    if filters:
        stream = stream.filter(
            lambda row: all(row.get(column) == value for column, _, value in filters)
        )
    return stream, info.sha
