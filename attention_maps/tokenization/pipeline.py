"""Reusable BPE training and Parquet tokenization utilities.

The functions in this module are intentionally independent of the command-line
entry point.  They can be imported by tests, notebooks, and future training
code without executing a script or parsing process arguments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SUPPORTED_NORMALIZERS = {"none", "NFC", "NFKC"}


@dataclass(frozen=True)
class InputConfig:
    dataset_dir: Path
    splits: tuple[str, ...] = ("train", "validation", "test")
    training_split: str = "train"
    text_column: str = "text"
    id_column: str = "doc_id"
    source_column: str = "source"
    hash_column: str = "text_sha256"

    def validate(self) -> None:
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Processed dataset directory not found: {self.dataset_dir}"
            )
        if not self.splits:
            raise ValueError("input.splits must contain at least one split")
        if any(not isinstance(split, str) or not split.strip() for split in self.splits):
            raise ValueError("Every input split must be a non-empty string")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError("input.splits must not contain duplicates")
        if self.training_split not in self.splits:
            raise ValueError("input.training_split must be included in input.splits")
        for split in self.splits:
            path = self.dataset_dir / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing input split: {path}")


@dataclass(frozen=True)
class SpecialTokenConfig:
    pad: str | None = "<|pad|>"
    unk: str | None = "<|unk|>"
    bos: str | None = "<|begin_of_text|>"
    eos: str | None = "<|end_of_text|>"
    additional: tuple[str, ...] = ()

    def ordered(self) -> list[str]:
        tokens = [
            token
            for token in (self.pad, self.unk, self.bos, self.eos, *self.additional)
            if token is not None
        ]
        if any(not isinstance(token, str) or not token for token in tokens):
            raise ValueError("Every configured special token must be a non-empty string")
        if len(set(tokens)) != len(tokens):
            raise ValueError("Configured special tokens must be unique")
        return tokens

    def roles(self) -> dict[str, str | None]:
        return {
            "pad": self.pad,
            "unk": self.unk,
            "bos": self.bos,
            "eos": self.eos,
        }


@dataclass(frozen=True)
class TokenizerSourceConfig:
    """Where tokenizer artifacts come from; never changes their vocabulary."""

    type: str = "train"
    path: Path | None = None
    repository: str | None = None
    revision: str | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False

    def validate(self) -> None:
        if self.type not in {"train", "local", "huggingface"}:
            raise ValueError(
                "tokenizer.source.type must be train, local, or huggingface"
            )
        if self.type == "local":
            if self.path is None:
                raise ValueError(
                    "A local tokenizer source requires tokenizer.source.path"
                )
            if not (self.path / "tokenizer.json").is_file():
                raise FileNotFoundError(
                    f"Local tokenizer.json not found: {self.path / 'tokenizer.json'}"
                )
        if self.type == "huggingface" and not self.repository:
            raise ValueError(
                "A Hugging Face tokenizer source requires tokenizer.source.repository"
            )
        if self.type == "train" and any(
            value is not None for value in (self.path, self.repository, self.revision)
        ):
            raise ValueError(
                "A train tokenizer source cannot set path, repository, or revision"
            )


@dataclass(frozen=True)
class BPEConfig:
    source: TokenizerSourceConfig = field(default_factory=TokenizerSourceConfig)
    vocab_size: int = 50_006
    min_frequency: int = 2
    normalization: str = "NFC"
    pre_tokenizer: str = "whitespace"
    continuing_subword_prefix: str = ""
    end_of_word_suffix: str = "</w>"
    max_token_length: int = 64
    max_training_segments: int | None = None
    max_chars_per_segment: int = 16_384
    add_bos: bool = True
    add_eos: bool = True
    special_tokens: SpecialTokenConfig = field(default_factory=SpecialTokenConfig)

    def validate(self) -> None:
        self.source.validate()
        reserved = self.special_tokens.ordered()
        if self.source.type == "train" and self.vocab_size <= len(reserved):
            raise ValueError(
                "tokenizer.vocab_size must exceed the number of special tokens"
            )
        if self.source.type == "train" and self.min_frequency <= 0:
            raise ValueError("tokenizer.min_frequency must be positive")
        if (
            self.source.type == "train"
            and self.normalization not in SUPPORTED_NORMALIZERS
        ):
            raise ValueError(
                f"tokenizer.normalization must be one of {sorted(SUPPORTED_NORMALIZERS)}"
            )
        if self.source.type == "train" and self.pre_tokenizer not in {
            "whitespace",
            "whitespace_split",
        }:
            raise ValueError(
                "tokenizer.pre_tokenizer must be 'whitespace' or 'whitespace_split'"
            )
        if self.max_token_length <= 0:
            raise ValueError("tokenizer.max_token_length must be positive")
        if self.max_chars_per_segment <= 0:
            raise ValueError("tokenizer.max_chars_per_segment must be positive")
        if (
            self.max_training_segments is not None
            and self.max_training_segments <= 0
        ):
            raise ValueError(
                "tokenizer.max_training_segments must be positive when set"
            )
        if self.add_bos and self.special_tokens.bos is None:
            raise ValueError("add_bos requires tokenizer.special_tokens.bos")
        if self.add_eos and self.special_tokens.eos is None:
            raise ValueError("add_eos requires tokenizer.special_tokens.eos")
        if self.source.type == "train" and any(
            token is None
            for token in (
                self.special_tokens.pad,
                self.special_tokens.unk,
                self.special_tokens.bos,
                self.special_tokens.eos,
            )
        ):
            raise ValueError("A trained tokenizer requires PAD/UNK/BOS/EOS tokens")


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path
    rows_per_shard: int = 100_000
    batch_size: int = 512
    compression: str = "zstd"
    model_max_length: int = 2_048

    def validate(self) -> None:
        if self.rows_per_shard <= 0:
            raise ValueError("output.rows_per_shard must be positive")
        if self.batch_size <= 0:
            raise ValueError("output.batch_size must be positive")
        if self.compression not in {"zstd", "snappy"}:
            raise ValueError("output.compression must be 'zstd' or 'snappy'")
        if self.model_max_length <= 0:
            raise ValueError("output.model_max_length must be positive")


@dataclass(frozen=True)
class TokenizationConfig:
    input: InputConfig
    tokenizer: BPEConfig
    output: OutputConfig
    schema_version: int = 1
    reference: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only tokenization config schema_version=1 is supported")
        self.input.validate()
        self.tokenizer.validate()
        self.output.validate()
        try:
            self.output.output_dir.resolve().relative_to(self.input.dataset_dir.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("output.output_dir must not be inside input.dataset_dir")


@dataclass
class SplitStats:
    documents: int = 0
    tokens: int = 0
    unknown_tokens: int = 0
    empty_documents: int = 0
    min_tokens: int | None = None
    max_tokens: int = 0
    source_documents: Counter[str] = field(default_factory=Counter)

    def observe(self, source: str, token_ids: list[int], unk_id: int | None) -> None:
        count = len(token_ids)
        self.documents += 1
        self.tokens += count
        if unk_id is not None:
            self.unknown_tokens += sum(token_id == unk_id for token_id in token_ids)
        self.min_tokens = (
            count if self.min_tokens is None else min(self.min_tokens, count)
        )
        self.max_tokens = max(self.max_tokens, count)
        self.source_documents[source] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "tokens": self.tokens,
            "unknown_tokens": self.unknown_tokens,
            "unknown_token_rate": (
                self.unknown_tokens / self.tokens if self.tokens else 0.0
            ),
            "empty_documents": self.empty_documents,
            "min_tokens_per_document": self.min_tokens or 0,
            "max_tokens_per_document": self.max_tokens,
            "mean_tokens_per_document": (
                self.tokens / self.documents if self.documents else 0.0
            ),
            "documents_by_source": dict(sorted(self.source_documents.items())),
        }
def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _resolve_path(value: Any, config_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def load_tokenization_config(path: str | Path) -> TokenizationConfig:
    """Load a YAML config and resolve its paths relative to the YAML file."""

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements.txt") from exc

    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read tokenization config {config_path}: {exc}") from exc
    root = _mapping(payload, "config")
    config_dir = config_path.parent

    try:
        input_data = _mapping(root.pop("input"), "input")
        tokenizer_data = _mapping(root.pop("tokenizer"), "tokenizer")
        output_data = _mapping(root.pop("output"), "output")
        source_data = _mapping(
            tokenizer_data.pop("source", {}), "tokenizer.source"
        )
        special_data = _mapping(
            tokenizer_data.pop("special_tokens", {}), "tokenizer.special_tokens"
        )

        input_data["dataset_dir"] = _resolve_path(
            input_data["dataset_dir"], config_dir, "input.dataset_dir"
        )
        raw_splits = input_data.get("splits", ("train", "validation", "test"))
        if not isinstance(raw_splits, (list, tuple)):
            raise ValueError("input.splits must be a YAML list")
        input_data["splits"] = tuple(raw_splits)
        for key in ("path", "cache_dir"):
            if source_data.get(key) is not None:
                source_data[key] = _resolve_path(
                    source_data[key], config_dir, f"tokenizer.source.{key}"
                )
        tokenizer_data["source"] = TokenizerSourceConfig(**source_data)
        tokenizer_data["special_tokens"] = SpecialTokenConfig(
            additional=tuple(special_data.pop("additional", ())),
            **special_data,
        )
        output_data["output_dir"] = _resolve_path(
            output_data["output_dir"], config_dir, "output.output_dir"
        )
        config = TokenizationConfig(
            input=InputConfig(**input_data),
            tokenizer=BPEConfig(**tokenizer_data),
            output=OutputConfig(**output_data),
            **root,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tokenization config {config_path}: {exc}") from exc
    config.validate()
    return config


def with_path_overrides(
    config: TokenizationConfig,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TokenizationConfig:
    """Return a validated config with optional CLI path overrides."""

    input_config = InputConfig(
        **{
            **asdict(config.input),
            "dataset_dir": (
                Path(input_dir).resolve() if input_dir is not None else config.input.dataset_dir
            ),
            "splits": config.input.splits,
        }
    )
    output_config = OutputConfig(
        **{
            **asdict(config.output),
            "output_dir": (
                Path(output_dir).resolve() if output_dir is not None else config.output.output_dir
            ),
        }
    )
    updated = TokenizationConfig(
        input=input_config,
        tokenizer=config.tokenizer,
        output=output_config,
        schema_version=config.schema_version,
        reference=config.reference,
    )
    updated.validate()
    return updated


def _input_file(config: TokenizationConfig, split: str) -> Path:
    return config.input.dataset_dir / f"{split}.parquet"


def _bounded_segments(text: str, max_chars: int) -> Iterator[str]:
    for paragraph in text.splitlines() or [text]:
        paragraph = paragraph.strip()
        while paragraph:
            if len(paragraph) <= max_chars:
                segment, paragraph = paragraph, ""
            else:
                boundary = paragraph.rfind(" ", max_chars // 2, max_chars)
                boundary = boundary if boundary > 0 else max_chars
                segment, paragraph = paragraph[:boundary], paragraph[boundary:]
            segment = segment.strip()
            if segment:
                yield segment


def iter_training_segments(config: TokenizationConfig) -> Iterator[str]:
    """Stream bounded text segments from the training split only."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required; install requirements.txt") from exc

    path = _input_file(config, config.input.training_split)
    parquet = pq.ParquetFile(path)
    if config.input.text_column not in parquet.schema_arrow.names:
        raise ValueError(f"Missing text column {config.input.text_column!r} in {path}")

    emitted = 0
    limit = config.tokenizer.max_training_segments
    for batch in parquet.iter_batches(
        columns=[config.input.text_column],
        batch_size=config.output.batch_size,
    ):
        values = batch.column(0).to_pylist()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                continue
            for segment in _bounded_segments(
                unicodedata.normalize(config.tokenizer.normalization, value)
                if config.tokenizer.normalization != "none"
                else value,
                config.tokenizer.max_chars_per_segment,
            ):
                if limit is not None and emitted >= limit:
                    return
                emitted += 1
                yield segment


def train_bpe_tokenizer(config: TokenizationConfig, show_progress: bool = True):
    """Train a Hugging Face Tokenizers BPE from the configured train split."""

    try:
        from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required; install requirements.txt") from exc

    settings = config.tokenizer
    boundary_options = {}
    if settings.continuing_subword_prefix:
        boundary_options["continuing_subword_prefix"] = (
            settings.continuing_subword_prefix
        )
    if settings.end_of_word_suffix:
        boundary_options["end_of_word_suffix"] = settings.end_of_word_suffix
    model = BPE(
        unk_token=settings.special_tokens.unk,
        fuse_unk=True,
        **boundary_options,
    )
    tokenizer = Tokenizer(model)
    if settings.normalization == "NFC":
        tokenizer.normalizer = normalizers.NFC()
    elif settings.normalization == "NFKC":
        tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = (
        pre_tokenizers.Whitespace()
        if settings.pre_tokenizer == "whitespace"
        else pre_tokenizers.WhitespaceSplit()
    )
    if settings.end_of_word_suffix:
        tokenizer.decoder = decoders.BPEDecoder(suffix=settings.end_of_word_suffix)

    trainer = BpeTrainer(
        vocab_size=settings.vocab_size,
        min_frequency=settings.min_frequency,
        show_progress=show_progress,
        special_tokens=settings.special_tokens.ordered(),
        max_token_length=settings.max_token_length,
        **boundary_options,
    )
    tokenizer.train_from_iterator(iter_training_segments(config), trainer=trainer)

    bos_id = tokenizer.token_to_id(settings.special_tokens.bos)
    eos_id = tokenizer.token_to_id(settings.special_tokens.eos)
    if bos_id is None or eos_id is None:
        raise RuntimeError("Trained tokenizer is missing configured BOS/EOS tokens")
    if settings.add_bos and settings.add_eos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{settings.special_tokens.bos} $A {settings.special_tokens.eos}",
            special_tokens=[
                (settings.special_tokens.bos, bos_id),
                (settings.special_tokens.eos, eos_id),
            ],
        )
    elif settings.add_bos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{settings.special_tokens.bos} $A",
            special_tokens=[(settings.special_tokens.bos, bos_id)],
        )
    elif settings.add_eos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A {settings.special_tokens.eos}",
            special_tokens=[(settings.special_tokens.eos, eos_id)],
        )
    return tokenizer


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_tokenizer_artifacts(
    tokenizer,
    config: TokenizationConfig,
    directory: Path,
    source_directory: Path | None = None,
) -> None:
    """Export exact tokenizer data plus pipeline-specific token-role metadata."""

    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(directory / "tokenizer.json"))
    special = config.tokenizer.special_tokens
    if source_directory is not None:
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            source_path = source_directory / name
            if source_path.is_file():
                shutil.copy2(source_path, directory / name)
            else:
                _write_json(directory / name, {})
    else:
        special_map = {
            f"{role}_token": token
            for role, token in special.roles().items()
            if token is not None
        }
        special_map["additional_special_tokens"] = list(special.additional)
        _write_json(directory / "special_tokens_map.json", special_map)
        _write_json(
            directory / "tokenizer_config.json",
            {
                **special_map,
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": config.output.model_max_length,
                "clean_up_tokenization_spaces": False,
            },
        )
    _write_json(
        directory / "attention_maps_tokenizer_config.json",
        {
            "schema_version": 1,
            "token_roles": special.roles(),
            "additional_special_tokens": list(special.additional),
            "add_bos": config.tokenizer.add_bos,
            "add_eos": config.tokenizer.add_eos,
        },
    )


def load_existing_tokenizer(
    directory: str | Path,
    config: TokenizationConfig,
):
    """Load and validate a previously saved tokenizer without training."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required; install requirements.txt") from exc

    tokenizer_dir = Path(directory).resolve()
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Existing tokenizer file not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    missing = [
        token
        for token in config.tokenizer.special_tokens.ordered()
        if tokenizer.token_to_id(token) is None
    ]
    if missing:
        raise ValueError(
            f"Existing tokenizer is missing configured special tokens: {missing}"
        )
    return tokenizer, tokenizer_dir


def load_huggingface_tokenizer(config: TokenizationConfig):
    """Download/load one immutable Hub snapshot and return its exact tokenizer."""

    try:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub and tokenizers are required; install requirements.txt"
        ) from exc

    source = config.tokenizer.source
    snapshot = Path(
        snapshot_download(
            repo_id=source.repository,
            repo_type="model",
            revision=source.revision,
            cache_dir=str(source.cache_dir) if source.cache_dir else None,
            local_files_only=source.local_files_only,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
    ).resolve()
    tokenizer_path = snapshot / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face repository has no tokenizer.json: {source.repository}"
        )
    return Tokenizer.from_file(str(tokenizer_path)), snapshot


def validate_tokenizer_roles(tokenizer, config: TokenizationConfig) -> None:
    missing = [
        f"{role}={token!r}"
        for role, token in config.tokenizer.special_tokens.roles().items()
        if token is not None and tokenizer.token_to_id(token) is None
    ]
    missing.extend(
        repr(token)
        for token in config.tokenizer.special_tokens.additional
        if tokenizer.token_to_id(token) is None
    )
    if missing:
        raise ValueError(
            "Tokenizer is missing configured tokens; existing vocabularies are never "
            f"mutated automatically: {missing}"
        )


class _ParquetShardWriter:
    def __init__(self, directory: Path, schema, rows_per_shard: int, compression: str):
        import pyarrow.parquet as pq

        self.directory = directory
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.pq = pq
        self.writer = None
        self.shard_index = -1
        self.rows_in_shard = 0
        self.paths: list[Path] = []
        directory.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        self.shard_index += 1
        path = self.directory / f"part-{self.shard_index:05d}.parquet"
        self.writer = self.pq.ParquetWriter(
            path,
            self.schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.paths.append(path)
        self.rows_in_shard = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        import pyarrow as pa

        offset = 0
        while offset < len(rows):
            if self.writer is None or self.rows_in_shard >= self.rows_per_shard:
                self.close_active()
                self._open()
            capacity = self.rows_per_shard - self.rows_in_shard
            chunk = rows[offset : offset + capacity]
            self.writer.write_table(pa.Table.from_pylist(chunk, schema=self.schema))
            self.rows_in_shard += len(chunk)
            offset += len(chunk)

    def close_active(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self) -> None:
        self.close_active()


def tokenize_split(
    tokenizer,
    config: TokenizationConfig,
    split: str,
) -> tuple[SplitStats, list[Path]]:
    """Encode one processed split into document-preserving Parquet shards."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required; install requirements.txt") from exc

    input_path = _input_file(config, split)
    parquet = pq.ParquetFile(input_path)
    columns = [
        config.input.id_column,
        config.input.text_column,
        config.input.source_column,
        config.input.hash_column,
    ]
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Input {input_path} is missing columns: {sorted(missing)}")

    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("source", pa.string()),
            ("text_sha256", pa.string()),
            ("input_ids", pa.list_(pa.int32())),
            ("num_tokens", pa.int32()),
        ]
    )
    writer = _ParquetShardWriter(
        config.output.output_dir / split,
        schema,
        config.output.rows_per_shard,
        config.output.compression,
    )
    stats = SplitStats()
    unk_token = config.tokenizer.special_tokens.unk
    unk_id = tokenizer.token_to_id(unk_token) if unk_token is not None else None
    try:
        for batch in parquet.iter_batches(
            columns=columns,
            batch_size=config.output.batch_size,
        ):
            values = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(columns)
            }
            valid_indices = [
                index
                for index, text in enumerate(values[config.input.text_column])
                if isinstance(text, str) and text.strip()
            ]
            stats.empty_documents += batch.num_rows - len(valid_indices)
            if not valid_indices:
                continue
            encodings = tokenizer.encode_batch(
                [values[config.input.text_column][index] for index in valid_indices],
                add_special_tokens=False,
            )
            rows: list[dict[str, Any]] = []
            for index, encoding in zip(valid_indices, encodings):
                token_ids = list(encoding.ids)
                if config.tokenizer.add_bos:
                    token_ids.insert(
                        0,
                        tokenizer.token_to_id(config.tokenizer.special_tokens.bos),
                    )
                if config.tokenizer.add_eos:
                    token_ids.append(
                        tokenizer.token_to_id(config.tokenizer.special_tokens.eos)
                    )
                source = str(values[config.input.source_column][index] or "")
                stats.observe(source, token_ids, unk_id)
                rows.append(
                    {
                        "doc_id": values[config.input.id_column][index],
                        "source": source,
                        "text_sha256": values[config.input.hash_column][index],
                        "input_ids": token_ids,
                        "num_tokens": len(token_ids),
                    }
                )
            writer.write(rows)
    finally:
        writer.close()
    if stats.documents == 0:
        raise ValueError(f"No non-empty documents found in {input_path}")
    return stats, writer.paths


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "sha256": digest.hexdigest()}


def _serializable_config(config: TokenizationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["input"]["dataset_dir"] = str(config.input.dataset_dir.resolve())
    payload["input"]["splits"] = list(config.input.splits)
    payload["output"]["output_dir"] = str(config.output.output_dir.resolve())
    for key in ("path", "cache_dir"):
        value = getattr(config.tokenizer.source, key)
        if value is not None:
            payload["tokenizer"]["source"][key] = str(value.resolve())
    payload["tokenizer"]["special_tokens"]["additional"] = list(
        config.tokenizer.special_tokens.additional
    )
    return payload


def run_tokenization(
    config: TokenizationConfig,
    show_progress: bool = True,
    existing_tokenizer_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train or load a tokenizer, encode all splits, and write a manifest."""

    config.validate()
    output_dir = config.output.output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory "
            "to keep tokenizer artifacts reproducible."
        )

    source = config.tokenizer.source
    if existing_tokenizer_dir is not None:
        tokenizer, source_tokenizer_dir = load_existing_tokenizer(
            existing_tokenizer_dir,
            config,
        )
        tokenizer_origin = {
            "mode": "local",
            "source_directory": str(source_tokenizer_dir),
            "source_tokenizer_json": _file_fingerprint(
                source_tokenizer_dir / "tokenizer.json"
            ),
            "configured_by": "cli_override",
        }
    elif source.type == "train":
        tokenizer = train_bpe_tokenizer(config, show_progress=show_progress)
        tokenizer_origin = {
            "mode": "trained",
            "training_split": config.input.training_split,
        }
        source_tokenizer_dir = None
    elif source.type == "local":
        tokenizer, source_tokenizer_dir = load_existing_tokenizer(
            source.path,
            config,
        )
        tokenizer_origin = {
            "mode": "local",
            "source_directory": str(source_tokenizer_dir),
            "source_tokenizer_json": _file_fingerprint(
                source_tokenizer_dir / "tokenizer.json"
            ),
            "configured_by": "yaml",
        }
    else:
        tokenizer, source_tokenizer_dir = load_huggingface_tokenizer(config)
        resolved_revision = (
            source_tokenizer_dir.name
            if source_tokenizer_dir.parent.name == "snapshots"
            else None
        )
        tokenizer_origin = {
            "mode": "huggingface",
            "repository": source.repository,
            "requested_revision": source.revision,
            "resolved_revision": resolved_revision,
            "snapshot_directory": str(source_tokenizer_dir),
            "source_tokenizer_json": _file_fingerprint(
                source_tokenizer_dir / "tokenizer.json"
            ),
        }
    validate_tokenizer_roles(tokenizer, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    save_tokenizer_artifacts(
        tokenizer,
        config,
        tokenizer_dir,
        source_directory=(
            source_tokenizer_dir if tokenizer_origin["mode"] != "trained" else None
        ),
    )

    split_stats: dict[str, Any] = {}
    output_files: dict[str, list[str]] = {}
    for split in config.input.splits:
        stats, paths = tokenize_split(tokenizer, config, split)
        split_stats[split] = stats.to_dict()
        output_files[split] = [str(path.resolve()) for path in paths]

    input_files = {
        split: _file_fingerprint(_input_file(config, split))
        for split in config.input.splits
    }
    source_manifest_path = config.input.dataset_dir / "build_manifest.json"
    source_manifest = (
        _file_fingerprint(source_manifest_path)
        if source_manifest_path.is_file()
        else None
    )
    special_ids = {
        token: tokenizer.token_to_id(token)
        for token in config.tokenizer.special_tokens.ordered()
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_stage": "processed",
        "output_stage": "tokenized",
        "training_split_only": (
            config.input.training_split
            if tokenizer_origin["mode"] == "trained"
            else None
        ),
        "config": _serializable_config(config),
        "input_files": input_files,
        "source_build_manifest": source_manifest,
        "tokenizer": {
            "algorithm": tokenizer.model.__class__.__name__,
            "implementation": "huggingface-tokenizers",
            "vocab_size": tokenizer.get_vocab_size(),
            "origin": tokenizer_origin,
            "special_token_ids": special_ids,
            "artifacts": {
                name: _file_fingerprint(tokenizer_dir / name)
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "attention_maps_tokenizer_config.json",
                )
            },
        },
        "split_stats": split_stats,
        "output_files": output_files,
        "tokenized_columns": [
            "doc_id",
            "source",
            "text_sha256",
            "input_ids",
            "num_tokens",
        ],
    }
    _write_json(output_dir / "tokenization_manifest.json", manifest)
    return manifest
