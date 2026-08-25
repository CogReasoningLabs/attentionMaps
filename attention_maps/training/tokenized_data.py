"""Load saved token IDs and expose fixed-length causal language-model blocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from attention_maps.tokenization.models import HuggingFaceBPETokenizer
from attention_maps.training.data import EncodedDataset


SPLITS = ("train", "validation", "test")
REQUIRED_COLUMNS = {"doc_id", "source", "text_sha256", "input_ids", "num_tokens"}


class PackedSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Non-overlapping fixed-length next-token blocks over a packed token stream."""

    def __init__(self, token_ids: torch.Tensor, sequence_length: int):
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a one-dimensional torch.long tensor")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.block_count = max(0, (token_ids.numel() - 1) // sequence_length)
        if self.block_count == 0:
            raise ValueError(
                f"Need at least {sequence_length + 1} tokens to create one block"
            )

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += self.block_count
        if not 0 <= index < self.block_count:
            raise IndexError(index)
        start = index * self.sequence_length
        window = self.token_ids[start : start + self.sequence_length + 1]
        return window[:-1], window[1:]


@dataclass(frozen=True)
class PackedDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    blocks: dict[str, int]


def _load_manifest(tokenized_dir: Path) -> dict:
    path = tokenized_dir / "tokenization_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load tokenization manifest {path}: {exc}") from exc
    if manifest.get("status") != "complete":
        raise ValueError(f"Tokenization manifest is not complete: {path}")
    if manifest.get("output_stage") != "tokenized":
        raise ValueError(f"Expected output_stage='tokenized' in {path}")
    return manifest


def _parquet_files(split_dir: Path) -> list[Path]:
    files = sorted(split_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No tokenized Parquet shards found under {split_dir}")
    return files


def _load_split_ids(
    tokenized_dir: Path,
    split: str,
    expected_tokens: int,
    expected_documents: int,
    batch_size: int = 512,
) -> torch.Tensor:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise RuntimeError("PyArrow is required; install requirements.txt") from exc

    dataset = ds.dataset(
        [str(path) for path in _parquet_files(tokenized_dir / split)],
        format="parquet",
    )
    missing = REQUIRED_COLUMNS - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Tokenized {split} data is missing columns: {sorted(missing)}")

    token_buffer = np.empty(expected_tokens, dtype=np.int64)
    offset = 0
    documents = 0
    scanner = dataset.scanner(columns=["input_ids", "num_tokens"], batch_size=batch_size)
    for batch in scanner.to_batches():
        input_ids = batch.column(0)
        declared_lengths = batch.column(1)
        if input_ids.null_count or declared_lengths.null_count:
            raise ValueError(f"Null token data found in {split}")
        actual_lengths = pc.list_value_length(input_ids)
        if not pc.all(pc.equal(actual_lengths, declared_lengths)).as_py():
            raise ValueError(f"num_tokens mismatch found in {split}")
        values = pc.list_flatten(input_ids)
        if not pa.types.is_integer(values.type):
            raise ValueError(f"input_ids must contain integers, found {values.type}")
        array = values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        end = offset + len(array)
        if end > expected_tokens:
            raise ValueError(f"{split} contains more tokens than its manifest records")
        token_buffer[offset:end] = array
        offset = end
        documents += batch.num_rows

    if offset != expected_tokens or documents != expected_documents:
        raise ValueError(
            f"{split} manifest mismatch: documents={documents}/{expected_documents}, "
            f"tokens={offset}/{expected_tokens}"
        )
    return torch.from_numpy(token_buffer)


def load_materialized_tokenized_corpus(
    tokenized_dir: str | Path,
    batch_size: int = 512,
) -> EncodedDataset:
    """Load all tokenized splits into contiguous int64 tensors with validation."""

    root = Path(tokenized_dir).resolve()
    manifest = _load_manifest(root)
    tokenizer = HuggingFaceBPETokenizer.from_directory(root / "tokenizer")
    expected_vocab = manifest["tokenizer"]["vocab_size"]
    if tokenizer.vocab_size != expected_vocab:
        raise ValueError(
            f"Tokenizer vocabulary mismatch: {tokenizer.vocab_size} != {expected_vocab}"
        )
    split_tensors = {}
    for split in SPLITS:
        stats = manifest["split_stats"][split]
        split_tensors[split] = _load_split_ids(
            root,
            split,
            expected_tokens=stats["tokens"],
            expected_documents=stats["documents"],
            batch_size=batch_size,
        )
    return EncodedDataset(
        train_ids=split_tensors["train"],
        valid_ids=split_tensors["validation"],
        test_ids=split_tensors["test"],
        tokenizer=tokenizer,
        metadata={
            "data_mode": "materialized_token_ids",
            "tokenized_dir": str(root),
            "tokenization_manifest": manifest,
            "tokens": {
                "train": split_tensors["train"].numel(),
                "validation": split_tensors["validation"].numel(),
                "test": split_tensors["test"].numel(),
            },
        },
    )


def create_packed_dataloaders(
    encoded: EncodedDataset,
    sequence_length: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_batch: bool = True,
) -> PackedDataLoaders:
    """Create shuffled train and deterministic evaluation DataLoaders."""

    datasets = {
        "train": PackedSequenceDataset(encoded.train_ids, sequence_length),
        "validation": PackedSequenceDataset(encoded.valid_ids, sequence_length),
        "test": PackedSequenceDataset(encoded.test_ids, sequence_length),
    }
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return PackedDataLoaders(
        train=DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=drop_last_batch,
            generator=generator,
            **common,
        ),
        validation=DataLoader(
            datasets["validation"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
        test=DataLoader(
            datasets["test"],
            shuffle=False,
            drop_last=False,
            **common,
        ),
        blocks={name: len(dataset) for name, dataset in datasets.items()},
    )

