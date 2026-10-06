"""Read CrossSum's official language-pair archives without remote Python code."""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterator, Sequence


CROSSSUM_DATASET_ID = "csebuetnlp/CrossSum"
CROSSSUM_COLUMNS = ("source_url", "target_url", "summary", "text")
CROSSSUM_BATCH_ROWS = 1_000
_SPLIT_SUFFIX = {"train": "train", "validation": "val", "test": "test"}


def _selected_member(archive: tarfile.TarFile, member_name: str) -> tarfile.TarInfo:
    matches = [item for item in archive.getmembers() if item.name.removeprefix("./") == member_name]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError(f"CrossSum archive must contain exactly one regular file named {member_name!r}")
    return matches[0]


def _crosssum_rows(archive_path: str, member_name: str) -> Iterator[dict[str, str]]:
    # Read an exact archive member without extracting files onto the filesystem.
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = _selected_member(archive, member_name)
        with archive.extractfile(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid CrossSum JSON at {member_name}:{line_number}") from error
                    if not isinstance(row, dict) or any(not isinstance(row.get(key), str) for key in CROSSSUM_COLUMNS):
                        raise ValueError(f"Invalid CrossSum record schema at {member_name}:{line_number}")
                    yield {key: row[key] for key in CROSSSUM_COLUMNS}


def load_crosssum(
    config: str | None,
    *,
    split: str,
    revision: str | None,
    token: str | None,
    filters: Sequence[tuple[str, str, Any]] = (),
) -> tuple[Any, str, int]:
    """Cache one pair's archive and index only the requested JSONL split.

    The archive includes all three splits, so its compressed byte count must
    be labelled as language-pair storage. Arrow rows/bytes describe the selected
    split. Empty source splits remain empty, without synthetic placeholder rows.
    """

    if not config or not re.fullmatch(r"[a-z_]+-[a-z_]+", config):
        raise ValueError(
            "CrossSum requires a source-target language pair. Use 'nepali-nepali' "
            "for Nepali articles and summaries, 'english-nepali' for English "
            "articles with Nepali summaries, or 'nepali-english' for the reverse. "
            "Splits are 'train', 'validation', and 'test'."
        )
    if split not in _SPLIT_SUFFIX:
        raise ValueError("CrossSum split must be 'train', 'validation', or 'test'.")
    if any(operator != "==" for _, operator, _ in filters):
        raise ValueError("CrossSum row filters support equality only")

    from datasets import Dataset, Features, IterableDataset, SplitDict, SplitInfo, Value
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(CROSSSUM_DATASET_ID, revision=revision or "main", token=token or None)
    filename = f"data/{config}_CrossSum.tar.bz2"
    if not info.sha or filename not in {item.rfilename for item in info.siblings or []}:
        raise ValueError(f"CrossSum has no archive for language pair {config!r} at the selected revision.")
    archive_path = hf_hub_download(
        CROSSSUM_DATASET_ID, filename, repo_type="dataset", revision=info.sha, token=token or None
    )
    member_name = f"{config}_{_SPLIT_SUFFIX[split]}.jsonl"
    # Validate before calling the builder so missing/empty splits stay explicit.
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = _selected_member(archive, member_name)
        empty = member.size == 0
    features = Features({key: Value("string") for key in CROSSSUM_COLUMNS})
    if empty:
        data = Dataset.from_dict({key: [] for key in CROSSSUM_COLUMNS}, features=features)
    else:
        data = Dataset.from_generator(
            _crosssum_rows, gen_kwargs={"archive_path": str(archive_path), "member_name": member_name},
            features=features, keep_in_memory=False, writer_batch_size=CROSSSUM_BATCH_ROWS,
        )
    stream = (
        data.to_iterable_dataset(num_shards=1) if len(data) else
        IterableDataset.from_generator(lambda: iter(()), features=features)
    )
    archive_bytes = Path(archive_path).stat().st_size
    stream.info.splits = SplitDict({
        split: SplitInfo(name=split, num_examples=len(data), num_bytes=data.data.nbytes)
    })
    stream.info.download_size = archive_bytes
    stream.info.config_name = config
    if filters:
        stream = stream.filter(lambda row: all(row.get(key) == value for key, _, value in filters))
    return stream, info.sha, archive_bytes
