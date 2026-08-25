from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 64
    dropout: float = 0.1
    attention_type: str = "softmax"
    # MoE parameters
    use_moe: bool = True
    num_experts: int = 4
    top_k: int = 2
    moe_hidden_dim: int | None = None  # If None, uses d_ff


@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 3e-4
    epochs: int = 3
    eval_every: int = 200
    max_steps: int | None = None
    grad_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    artifact_dir: str = "artifacts"


@dataclass(frozen=True)
class SplitConfig:
    """Map canonical splits to source splits or derive held-out data from train."""

    train: str = "train"
    validation: str | None = "validation"
    test: str | None = "test"
    validation_fraction: float = 0.0
    test_fraction: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.train.strip():
            raise ValueError("splits.train must be a non-empty split name")
        for name, value in (
            ("validation_fraction", self.validation_fraction),
            ("test_fraction", self.test_fraction),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"splits.{name} must be in [0, 1)")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        if self.validation is not None and self.validation_fraction:
            raise ValueError(
                "Choose either an explicit validation split or validation_fraction"
            )
        if self.test is not None and self.test_fraction:
            raise ValueError("Choose either an explicit test split or test_fraction")


@dataclass(frozen=True)
class CorpusConfig:
    """Language-neutral description of a Hugging Face or local text corpus."""

    source: str
    loader: str
    text_column: str = "text"
    language: str | None = None
    dataset_config: str | None = None
    revision: str | None = None
    data_files: dict[str, str | list[str]] | None = None
    streaming: bool = False
    max_documents: dict[str, int] = field(default_factory=dict)
    splits: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        if self.source not in {"huggingface", "local"}:
            raise ValueError("dataset.source must be 'huggingface' or 'local'")
        if not self.loader.strip():
            raise ValueError("dataset.loader must be non-empty")
        if not self.text_column.strip():
            raise ValueError("dataset.text_column must be non-empty")
        if self.language is not None and (
            not self.language.strip() or any(ch.isspace() for ch in self.language)
        ):
            raise ValueError("dataset.language must be a non-empty language tag")
        if self.source == "local" and not self.data_files:
            raise ValueError("Local datasets require dataset.data_files")
        for split, limit in self.max_documents.items():
            if not split.strip() or limit <= 0:
                raise ValueError(
                    "dataset.max_documents must map split names to positive integers"
                )


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizer training settings independent of any particular language."""

    backend: str = "sentencepiece"
    vocab_size: int = 8_000
    model_type: str = "unigram"
    normalization: str = "nmt_nfkc"
    character_coverage: float = 1.0
    byte_fallback: bool = True
    hard_vocab_limit: bool = False
    training_sentence_limit: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.backend not in {"sentencepiece", "legacy_word"}:
            raise ValueError(
                "tokenizer.backend must be 'sentencepiece' or 'legacy_word'"
            )
        if self.vocab_size <= 4:
            raise ValueError("tokenizer.vocab_size must leave room for normal tokens")
        if self.model_type not in {"unigram", "bpe", "char", "word"}:
            raise ValueError(
                "tokenizer.model_type must be unigram, bpe, char, or word"
            )
        if not 0.0 < self.character_coverage <= 1.0:
            raise ValueError("tokenizer.character_coverage must be in (0, 1]")
        if (
            self.training_sentence_limit is not None
            and self.training_sentence_limit <= 0
        ):
            raise ValueError("tokenizer.training_sentence_limit must be positive")


@dataclass(frozen=True)
class ExperimentProfile:
    """Versioned data-pipeline configuration stored with every future run."""

    name: str
    dataset: CorpusConfig
    tokenizer: TokenizerConfig
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported profile schema_version={self.schema_version}; expected 1"
            )
        if not self.name.strip():
            raise ValueError("profile.name must be non-empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentProfile:
        data = dict(payload)
        try:
            dataset_data = dict(data.pop("dataset"))
            tokenizer_data = dict(data.pop("tokenizer"))
            splits_data = dict(dataset_data.pop("splits", {}))
            dataset = CorpusConfig(
                **dataset_data,
                splits=SplitConfig(**splits_data),
            )
            tokenizer = TokenizerConfig(**tokenizer_data)
            return cls(dataset=dataset, tokenizer=tokenizer, **data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid experiment profile: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_profile(path: str | Path) -> ExperimentProfile:
    """Load and validate a UTF-8 JSON experiment profile."""

    profile_path = Path(path)
    try:
        with profile_path.open(encoding="utf-8") as profile_file:
            payload = json.load(profile_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load profile {profile_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Profile {profile_path} must contain a JSON object")
    return ExperimentProfile.from_dict(payload)
