"""Configuration and result contracts for the corpus EDA pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DatasetSpec:
    """One streaming dataset input and its optional schema overrides."""

    key: str
    dataset_id: str
    config_name: str | None = None
    split: str = "train"
    revision: str | None = None
    text_columns: tuple[str, ...] = ()
    source_columns: tuple[str, ...] = ()
    sample_size: int | None = None

    def __post_init__(self) -> None:
        if not _SAFE_KEY.fullmatch(self.key):
            raise ValueError(
                "dataset key must contain only letters, digits, '.', '_', or '-'"
            )
        if "/" not in self.dataset_id or self.dataset_id.startswith("/"):
            raise ValueError("dataset_id must look like 'organization/name'")
        if not self.split.strip():
            raise ValueError("dataset split cannot be empty")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("dataset sample_size must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            key=str(value["key"]),
            dataset_id=str(value["dataset_id"]),
            config_name=_optional_string(value.get("config_name")),
            split=str(value.get("split", "train")),
            revision=_optional_string(value.get("revision")),
            text_columns=tuple(map(str, value.get("text_columns", ()))),
            source_columns=tuple(map(str, value.get("source_columns", ()))),
            sample_size=(
                int(value["sample_size"])
                if value.get("sample_size") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class AnalysisConfig:
    """Language-quality metrics and bounded-memory controls."""

    sample_size: int = 50_000
    seed: int = 42
    shuffle_buffer_size: int = 0
    min_tokens: int = 5
    min_devanagari_ratio: float = 0.70
    reservoir_size: int = 10_000
    max_vocabulary: int = 500_000
    top_tokens: int = 30
    near_duplicate_distance: int = 3
    simhash_bands: int = 8
    ngram_orders: tuple[int, ...] = (2, 3, 4)
    max_ngrams: int = 250_000
    top_ngrams: int = 25
    max_pattern_tokens_per_document: int = 5_000
    cooccurrence_terms: tuple[str, ...] = ()
    cooccurrence_window: int = 6
    max_cooccurrence_edges: int = 100_000
    top_cooccurrence_edges: int = 50
    cooccurrence_stopwords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        positive = {
            "sample_size": self.sample_size,
            "min_tokens": self.min_tokens,
            "reservoir_size": self.reservoir_size,
            "max_vocabulary": self.max_vocabulary,
            "top_tokens": self.top_tokens,
            "simhash_bands": self.simhash_bands,
            "max_ngrams": self.max_ngrams,
            "top_ngrams": self.top_ngrams,
            "max_pattern_tokens_per_document": self.max_pattern_tokens_per_document,
            "cooccurrence_window": self.cooccurrence_window,
            "max_cooccurrence_edges": self.max_cooccurrence_edges,
            "top_cooccurrence_edges": self.top_cooccurrence_edges,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size cannot be negative")
        if not 0 <= self.min_devanagari_ratio <= 1:
            raise ValueError("min_devanagari_ratio must be in [0, 1]")
        if not 0 <= self.near_duplicate_distance <= 16:
            raise ValueError("near_duplicate_distance must be in [0, 16]")
        if 64 % self.simhash_bands:
            raise ValueError("simhash_bands must divide 64")
        if not self.ngram_orders:
            raise ValueError("ngram_orders cannot be empty")
        if len(set(self.ngram_orders)) != len(self.ngram_orders):
            raise ValueError("ngram_orders must be unique")
        if any(order < 2 or order > 5 for order in self.ngram_orders):
            raise ValueError("ngram_orders must be between 2 and 5")
        if any(not term.strip() for term in self.cooccurrence_terms):
            raise ValueError("cooccurrence_terms cannot contain empty values")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "AnalysisConfig":
        arguments = dict(value or {})
        for key in ("ngram_orders", "cooccurrence_terms", "cooccurrence_stopwords"):
            if key in arguments:
                arguments[key] = tuple(arguments[key])
        return cls(**arguments)


@dataclass(frozen=True)
class SurveyPlan:
    """A named, repeatable multi-dataset study."""

    name: str
    datasets: tuple[DatasetSpec, ...]
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("survey name cannot be empty")
        if not self.datasets:
            raise ValueError("survey must contain at least one dataset")
        keys = [dataset.key for dataset in self.datasets]
        if len(keys) != len(set(keys)):
            raise ValueError("survey dataset keys must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SurveyPlan":
        return cls(
            name=str(value["name"]),
            datasets=tuple(
                DatasetSpec.from_dict(dataset) for dataset in value["datasets"]
            ),
            analysis=AnalysisConfig.from_dict(value.get("analysis")),
        )


@dataclass(frozen=True)
class ProfileSample:
    characters: int
    tokens: int
    devanagari_ratio: float


@dataclass(frozen=True)
class DatasetSummary:
    dataset_key: str
    dataset_id: str
    config_name: str | None
    split: str
    requested_revision: str | None
    sample_limit: int
    seed: int
    rows_seen: int
    usable_rows: int
    missing_text_rows: int
    average_characters: float
    median_characters: float
    p95_characters: float
    maximum_characters: int
    average_tokens: float
    median_tokens: float
    p95_tokens: float
    maximum_tokens: int
    sentences_observed: int
    average_sentence_tokens: float
    median_sentence_tokens: float
    p95_sentence_tokens: float
    maximum_sentence_tokens: int
    lines_observed: int
    average_line_characters: float
    median_line_characters: float
    p95_line_characters: float
    maximum_line_characters: int
    average_devanagari_ratio: float
    devanagari_clean_rows: int
    devanagari_clean_ratio_pct: float
    quality_pass_rows: int
    quality_pass_ratio_pct: float
    exact_duplicate_rows: int
    near_duplicate_rows: int
    duplicate_ratio_pct: float
    total_tokens: int
    tracked_vocabulary: int
    type_token_ratio: float
    vocabulary_truncated: bool
    ngram_tracking_truncated: bool
    cooccurrence_tracking_truncated: bool
    pattern_truncated_rows: int
    source_metadata_rows: int
    source_categories_observed: int
    dominant_source: str | None
    dominant_source_share_pct: float
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetProfile:
    summary: DatasetSummary
    samples: tuple[ProfileSample, ...]
    top_tokens: tuple[tuple[str, int], ...]
    top_sources: tuple[tuple[str, int], ...]
    sentence_length_samples: tuple[int, ...]
    line_length_samples: tuple[int, ...]
    top_ngrams: Mapping[int, tuple[tuple[str, int], ...]]
    cooccurrence_edges: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class SurveyRun:
    plan: SurveyPlan
    profiles: tuple[DatasetProfile, ...]
    failures: Mapping[str, str]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
