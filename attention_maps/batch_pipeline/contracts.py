"""Validated configuration for the batch preprocessing pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class InputConfig:
    local_path: Path | None = None
    google_drive: str | None = None
    text_columns: tuple[str, ...] = ()
    source_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.local_path is None) == (self.google_drive is None):
            raise ValueError("input must define exactly one of local_path or google_drive")


@dataclass(frozen=True)
class SamplingConfig:
    fraction: float = 1.0
    max_records: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("sampling.fraction must be in (0, 1]")
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("sampling.max_records must be positive")


@dataclass(frozen=True)
class CleaningConfig:
    min_tokens: int = 1
    min_devanagari_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.min_tokens <= 0:
            raise ValueError("cleaning.min_tokens must be positive")
        if not 0 <= self.min_devanagari_ratio <= 1:
            raise ValueError("cleaning.min_devanagari_ratio must be in [0, 1]")


@dataclass(frozen=True)
class DeduplicationConfig:
    mode: str = "multistage"
    near_duplicate_threshold: float = 0.80
    minhash_permutations: int = 128
    minhash_bands: int = 16
    shingle_size: int = 5
    edit_similarity_threshold: float = 0.80
    boilerplate_min_documents: int = 3
    boilerplate_min_characters: int = 40

    def __post_init__(self) -> None:
        if self.mode not in {"none", "exact", "multistage"}:
            raise ValueError("deduplication.mode must be none, exact, or multistage")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if not 0 < self.edit_similarity_threshold <= 1:
            raise ValueError("edit_similarity_threshold must be in (0, 1]")
        if self.minhash_permutations % self.minhash_bands:
            raise ValueError("minhash_bands must divide minhash_permutations")
        if self.shingle_size <= 0:
            raise ValueError("deduplication.shingle_size must be positive")
        if self.boilerplate_min_documents < 2:
            raise ValueError("boilerplate_min_documents must be at least 2")
        if self.boilerplate_min_characters <= 0:
            raise ValueError("boilerplate_min_characters must be positive")


@dataclass(frozen=True)
class ExecutionConfig:
    batch_size: int = 512
    batch_max_mib: int = 64
    workers: int = field(default_factory=lambda: max(1, min(8, (os.cpu_count() or 2) - 1)))
    max_pending_batches: int = 0
    shard_rows: int = 50_000
    drive_chunk_mib: int = 8

    def __post_init__(self) -> None:
        values = {
            "batch_size": self.batch_size,
            "batch_max_mib": self.batch_max_mib,
            "workers": self.workers,
            "shard_rows": self.shard_rows,
            "drive_chunk_mib": self.drive_chunk_mib,
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("execution numeric values must be positive")
        if self.max_pending_batches < 0:
            raise ValueError("execution.max_pending_batches cannot be negative")
        if self.drive_chunk_mib % 1:
            raise ValueError("execution.drive_chunk_mib must be a whole number")

    @property
    def pending_batches(self) -> int:
        return self.max_pending_batches or self.workers * 2


@dataclass(frozen=True)
class EdaConfig:
    enabled: bool = True
    top_items: int = 100
    max_vocabulary: int = 500_000
    max_ngrams: int = 250_000
    reservoir_size: int = 10_000
    max_pattern_tokens_per_document: int = 5_000

    def __post_init__(self) -> None:
        values = (
            self.top_items,
            self.max_vocabulary,
            self.max_ngrams,
            self.reservoir_size,
            self.max_pattern_tokens_per_document,
        )
        if any(value <= 0 for value in values):
            raise ValueError("eda numeric limits must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str
    input: InputConfig
    output_root: Path = Path("data/pipeline-runs")
    destination_drive_folder_id: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    eda: EdaConfig = field(default_factory=EdaConfig)

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name cannot be empty")
        if any(part in self.run_name for part in ("/", "\\", "..")):
            raise ValueError("run_name must be a safe directory name")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base_dir: Path) -> "PipelineConfig":
        input_value = dict(value.get("input", {}))
        local = input_value.get("local_path")
        output = Path(value.get("output_root", "data/pipeline-runs"))
        if not output.is_absolute():
            output = (base_dir / output).resolve()
        if local is not None:
            local_path = Path(str(local))
            if not local_path.is_absolute():
                local_path = (base_dir / local_path).resolve()
        else:
            local_path = None
        input_config = InputConfig(
            local_path=local_path,
            google_drive=_optional(input_value.get("google_drive")),
            text_columns=tuple(map(str, input_value.get("text_columns", ()))),
            source_columns=tuple(map(str, input_value.get("source_columns", ()))),
        )
        return cls(
            run_name=str(value["run_name"]),
            input=input_config,
            output_root=output,
            destination_drive_folder_id=_optional(
                value.get("destination_drive_folder_id")
            ),
            sampling=SamplingConfig(**dict(value.get("sampling", {}))),
            cleaning=CleaningConfig(**dict(value.get("cleaning", {}))),
            deduplication=DeduplicationConfig(**dict(value.get("deduplication", {}))),
            execution=ExecutionConfig(**dict(value.get("execution", {}))),
            eda=EdaConfig(**dict(value.get("eda", {}))),
        )


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load JSON or YAML, resolving local paths relative to the config file."""

    config_path = path.expanduser().resolve()
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read pipeline config {config_path}: {error}") from error
    try:
        if config_path.suffix.lower() == ".json":
            value = json.loads(content)
        else:
            import yaml

            value = yaml.safe_load(content)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid pipeline configuration: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("pipeline configuration must be an object")
    return PipelineConfig.from_dict(value, base_dir=config_path.parent)


def _optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
