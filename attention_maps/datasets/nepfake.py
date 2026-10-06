"""Explicit CSV reader for NepFakeV2, excluding duplicate exports and stats."""

from __future__ import annotations

from typing import Any, Sequence


NEPFAKE_DATASET_ID = "Nandan007/NepFakeV2"
NEPFAKE_CSV = "data/nepfakev2.csv"
CSV_BATCH_ROWS = 1_000


def load_nepfake(
    config: str | None,
    *,
    split: str,
    revision: str | None,
    token: str | None,
    filters: Sequence[tuple[str, str, Any]] = (),
) -> tuple[Any, str]:
    """Cache the CSV in batches, then sample from memory-mapped Arrow.

    Automatic Hub discovery also selects stats.json as a training shard. Use
    only the documented CSV export and measure its rows rather than trusting
    the separately published statistics. Pin the resolved source commit.
    """

    if config not in (None, "default") or split != "train":
        raise ValueError(
            "NepFakeV2 supports configuration 'default' (or leave it empty) "
            "and split 'train'."
        )
    if any(operator != "==" for _, operator, _ in filters):
        raise ValueError("NepFakeV2 row filters support equality only")

    from datasets import Features, SplitDict, SplitInfo, Value, load_dataset
    from huggingface_hub import HfApi, hf_hub_url

    info = HfApi().dataset_info(
        NEPFAKE_DATASET_ID, revision=revision or "main", token=token or None
    )
    if not info.sha or NEPFAKE_CSV not in {item.rfilename for item in info.siblings or []}:
        raise ValueError("The selected NepFakeV2 revision has no data/nepfakev2.csv export")
    url = hf_hub_url(
        NEPFAKE_DATASET_ID, NEPFAKE_CSV, repo_type="dataset", revision=info.sha
    )
    features = Features({
        column: Value("string")
        for column in (
            "example_id", "claim_text", "verdict_label_text", "evidence_text",
            "source_name", "source_url", "date_published", "topic_category",
            "annotator_notes", "schema_version",
        )
    })
    features.update({"verdict_label": Value("int64"), "is_native_nepali": Value("bool")})
    data = load_dataset(
        "csv", data_files={split: [url]}, split=split, features=features,
        chunksize=CSV_BATCH_ROWS, keep_in_memory=False, token=token or None,
    )
    if not len(data):
        raise ValueError("The selected NepFakeV2 CSV contains no examples")
    stream = data.to_iterable_dataset(num_shards=1)
    stream.info.splits = SplitDict({
        split: SplitInfo(name=split, num_examples=len(data), num_bytes=data.data.nbytes)
    })
    stream.info.download_size = data.info.download_size
    stream.info.config_name = "default"
    if filters:
        stream = stream.filter(
            lambda row: all(row.get(column) == value for column, _, value in filters)
        )
    return stream, info.sha
