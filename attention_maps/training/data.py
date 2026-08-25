from __future__ import annotations

import glob
import hashlib
import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset

from attention_maps.config import ExperimentProfile, load_experiment_profile
from attention_maps.tokenization.models import (
    HuggingFaceBPETokenizer,
    SentencePieceTokenizer,
    SimpleTokenizer,
    build_tokenizer,
    tokenizer_from_state,
    train_sentencepiece_tokenizer,
)


@dataclass
class EncodedDataset:
    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    test_ids: torch.Tensor
    tokenizer: SimpleTokenizer | SentencePieceTokenizer | HuggingFaceBPETokenizer
    metadata: dict | None = None


def _clean_texts(texts: list[str]) -> list[str]:
    cleaned = []
    for t in texts:
        t = t.strip()
        if t:
            cleaned.append(t)
    return cleaned


def load_wikitext2(max_vocab_size: int = 20000, min_freq: int = 2) -> EncodedDataset:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = _clean_texts(ds["train"]["text"])
    valid_texts = _clean_texts(ds["validation"]["text"])
    test_texts = _clean_texts(ds["test"]["text"])
    tokenizer = build_tokenizer(train_texts, max_vocab_size=max_vocab_size, min_freq=min_freq)

    def encode_split(texts: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for text in texts:
            ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        t = torch.tensor(ids, dtype=torch.long)
        return t.pin_memory() if torch.cuda.is_available() else t

    return EncodedDataset(
        train_ids=encode_split(train_texts),
        valid_ids=encode_split(valid_texts),
        test_ids=encode_split(test_texts),
        tokenizer=tokenizer,
        metadata={"dataset": "wikitext", "config": "wikitext-103-raw-v1"},
    )


def _resolve_local_data_files(
    profile: ExperimentProfile,
    profile_path: Path,
) -> dict[str, str | list[str]] | None:
    data_files = profile.dataset.data_files
    if not data_files or profile.dataset.source != "local":
        return data_files

    resolved: dict[str, str | list[str]] = {}
    for split, paths in data_files.items():
        path_list = [paths] if isinstance(paths, str) else paths
        expanded: list[str] = []
        for value in path_list:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = profile_path.parent / candidate
            matches = sorted(glob.glob(str(candidate)))
            if not matches:
                raise FileNotFoundError(
                    f"No files matched dataset.data_files[{split!r}]: {candidate}"
                )
            expanded.extend(str(Path(match).resolve()) for match in matches)
        resolved[split] = expanded[0] if len(expanded) == 1 else expanded
    return resolved


def _load_text_splits(
    profile: ExperimentProfile,
    profile_path: Path,
    hf_cache_dir: Path,
) -> tuple[list[str], list[str], list[str], dict]:
    if profile.dataset.streaming:
        raise ValueError(
            "Profile-driven tokenizer training currently requires streaming=false"
        )
    data_files = _resolve_local_data_files(profile, profile_path)
    kwargs = {
        "path": profile.dataset.loader,
        "name": profile.dataset.dataset_config,
        "data_files": data_files,
        "streaming": False,
        "cache_dir": str(hf_cache_dir),
    }
    if profile.dataset.source == "huggingface" and profile.dataset.revision:
        kwargs["revision"] = profile.dataset.revision
    dataset = load_dataset(**{key: value for key, value in kwargs.items() if value is not None})

    split_cfg = profile.dataset.splits
    if split_cfg.train not in dataset:
        raise ValueError(
            f"Training split {split_cfg.train!r} not found; available: {list(dataset)}"
        )
    train_data = dataset[split_cfg.train]
    validation_data = dataset.get(split_cfg.validation) if split_cfg.validation else None
    test_data = dataset.get(split_cfg.test) if split_cfg.test else None

    holdout_fraction = split_cfg.validation_fraction + split_cfg.test_fraction
    if holdout_fraction:
        split = train_data.train_test_split(
            test_size=holdout_fraction,
            seed=split_cfg.seed,
        )
        train_data = split["train"]
        holdout = split["test"]
        if split_cfg.validation_fraction and split_cfg.test_fraction:
            test_share = split_cfg.test_fraction / holdout_fraction
            held_out = holdout.train_test_split(
                test_size=test_share,
                seed=split_cfg.seed,
            )
            validation_data = held_out["train"]
            test_data = held_out["test"]
        elif split_cfg.validation_fraction:
            validation_data = holdout
        else:
            test_data = holdout

    if validation_data is None or test_data is None:
        raise ValueError(
            "The profile must provide or derive both validation and test splits"
        )

    canonical = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
    }
    texts: dict[str, list[str]] = {}
    for split_name, split_data in canonical.items():
        limit = profile.dataset.max_documents.get(split_name)
        if limit is not None:
            split_data = split_data.select(range(min(limit, len(split_data))))
        if profile.dataset.text_column not in split_data.column_names:
            raise ValueError(
                f"Text column {profile.dataset.text_column!r} missing from {split_name}"
            )
        texts[split_name] = _clean_texts(split_data[profile.dataset.text_column])

    metadata = {
        "profile": profile.to_dict(),
        "documents": {name: len(values) for name, values in texts.items()},
        "data_files": data_files,
    }
    return texts["train"], texts["validation"], texts["test"], metadata


def _tokenizer_training_sentences(
    documents: Iterable[str],
    limit: int | None,
    max_chars: int = 12_000,
) -> Iterable[str]:
    """Yield bounded pieces so giant PDF records are not skipped by the trainer."""

    yielded = 0
    for document in documents:
        for paragraph in document.splitlines() or [document]:
            paragraph = paragraph.strip()
            while paragraph:
                if limit is not None and yielded >= limit:
                    return
                if len(paragraph) <= max_chars:
                    piece, paragraph = paragraph, ""
                else:
                    boundary = paragraph.rfind(" ", max_chars // 2, max_chars)
                    boundary = boundary if boundary > 0 else max_chars
                    piece, paragraph = paragraph[:boundary], paragraph[boundary:]
                piece = piece.strip()
                if piece:
                    yielded += 1
                    yield piece


def _encode_documents(
    documents: Iterable[str],
    tokenizer: SimpleTokenizer | SentencePieceTokenizer,
) -> torch.Tensor:
    ids = array("q")
    for document in documents:
        ids.extend(tokenizer.encode(document, add_bos=True, add_eos=True))
    if not ids:
        raise ValueError("Cannot encode an empty corpus split")
    tensor = torch.frombuffer(ids, dtype=torch.int64).clone()
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


def _profile_cache_key(
    profile_path: Path,
    data_files: dict[str, str | list[str]] | None,
) -> str:
    digest = hashlib.sha256(profile_path.read_bytes())
    for value in (data_files or {}).values():
        paths = [value] if isinstance(value, str) else value
        for path_value in paths:
            path = Path(path_value)
            if path.is_file():
                stat = path.stat()
                digest.update(str(path.resolve()).encode())
                digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def load_profile_dataset(
    profile_path: str | Path,
    cache_root: str | Path = "data/cache",
    use_cache: bool = True,
) -> EncodedDataset:
    """Load, tokenize, and encode a profile with a reusable on-disk cache."""

    profile_path = Path(profile_path).resolve()
    profile = load_experiment_profile(profile_path)
    resolved_files = _resolve_local_data_files(profile, profile_path)
    cache_key = _profile_cache_key(profile_path, resolved_files)
    cache_dir = Path(cache_root) / f"{profile.name}-{cache_key}"
    cache_files = {
        "train": cache_dir / "train.pt",
        "validation": cache_dir / "validation.pt",
        "test": cache_dir / "test.pt",
        "tokenizer": cache_dir / "tokenizer.pt",
        "metadata": cache_dir / "metadata.json",
    }
    if use_cache and all(path.is_file() for path in cache_files.values()):
        tokenizer_state = torch.load(cache_files["tokenizer"], weights_only=False)
        return EncodedDataset(
            train_ids=torch.load(cache_files["train"], weights_only=True),
            valid_ids=torch.load(cache_files["validation"], weights_only=True),
            test_ids=torch.load(cache_files["test"], weights_only=True),
            tokenizer=tokenizer_from_state(tokenizer_state),
            metadata=json.loads(cache_files["metadata"].read_text(encoding="utf-8")),
        )

    train_texts, valid_texts, test_texts, metadata = _load_text_splits(
        profile,
        profile_path,
        hf_cache_dir=Path(cache_root) / "huggingface",
    )
    if profile.tokenizer.backend == "sentencepiece":
        tokenizer = train_sentencepiece_tokenizer(
            _tokenizer_training_sentences(
                train_texts,
                limit=profile.tokenizer.training_sentence_limit,
            ),
            profile.tokenizer,
        )
    else:
        tokenizer = build_tokenizer(
            train_texts,
            max_vocab_size=profile.tokenizer.vocab_size,
        )

    encoded = EncodedDataset(
        train_ids=_encode_documents(train_texts, tokenizer),
        valid_ids=_encode_documents(valid_texts, tokenizer),
        test_ids=_encode_documents(test_texts, tokenizer),
        tokenizer=tokenizer,
        metadata=metadata,
    )
    encoded.metadata = {
        **(encoded.metadata or {}),
        "tokens": {
            "train": len(encoded.train_ids),
            "validation": len(encoded.valid_ids),
            "test": len(encoded.test_ids),
        },
        "tokenizer": profile.tokenizer.__dict__,
        "cache_key": cache_key,
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(encoded.train_ids, cache_files["train"])
        torch.save(encoded.valid_ids, cache_files["validation"])
        torch.save(encoded.test_ids, cache_files["test"])
        torch.save(encoded.tokenizer.to_state(), cache_files["tokenizer"])
        cache_files["metadata"].write_text(
            json.dumps(encoded.metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return encoded


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError("Dataset too small for the configured sequence length.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)
