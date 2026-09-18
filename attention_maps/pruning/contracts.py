"""Validated configuration and result contracts for D2 data pruning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from attention_maps.datasets.schemas import D2_SUPPORTED_USE_CASES


@dataclass(frozen=True)
class D2PruningConfig:
    """Configuration for one-shot D2 coreset selection.

    ``retention_fraction`` is the fraction retained in the coreset, rather than
    the fraction removed. The paper calls the removed fraction the pruning
    rate; using a retention fraction here avoids ambiguity in pipeline code.
    """

    enabled: bool = False
    use_case: str = "traditional_nlp"
    retention_fraction: float = 0.5
    n_neighbors: int = 10
    gamma_forward: float = 1.0
    gamma_reverse: float = 0.8
    graph_backend: str = "auto"
    exact_block_size: int = 1_024
    max_exact_records: int = 10_000
    label_balanced: bool = False
    embedding_model_id: str = "FacebookAI/xlm-roberta-base"
    embedding_revision: str = "main"
    embedding_pooling: str = "cls"
    embedding_batch_size: int = 32
    embedding_max_length: int = 256
    embedding_device: str = "auto"
    normalize_embeddings: bool = False
    local_files_only: bool = False
    embeddings_path: Path | None = None
    difficulty_mode: str = "uniform"
    difficulty_scores_path: Path | None = None

    def __post_init__(self) -> None:
        if self.use_case not in D2_SUPPORTED_USE_CASES:
            supported = ", ".join(D2_SUPPORTED_USE_CASES)
            raise ValueError(f"pruning.use_case must be one of: {supported}")
        if not 0 < self.retention_fraction <= 1:
            raise ValueError("pruning.retention_fraction must be in (0, 1]")
        if self.n_neighbors <= 0:
            raise ValueError("pruning.n_neighbors must be positive")
        if self.gamma_forward < 0 or self.gamma_reverse < 0:
            raise ValueError("D2 gamma values cannot be negative")
        if self.graph_backend not in {"auto", "exact", "faiss"}:
            raise ValueError("pruning.graph_backend must be auto, exact, or faiss")
        if self.exact_block_size <= 0 or self.max_exact_records <= 1:
            raise ValueError("D2 exact-graph limits must be positive")
        if not self.embedding_model_id.strip() and self.embeddings_path is None:
            raise ValueError("D2 requires embedding_model_id or embeddings_path")
        if not self.embedding_revision.strip():
            raise ValueError("pruning.embedding_revision cannot be empty")
        if self.embedding_pooling not in {"cls", "mean"}:
            raise ValueError("pruning.embedding_pooling must be cls or mean")
        if self.embedding_batch_size <= 0 or self.embedding_max_length <= 0:
            raise ValueError("D2 embedding limits must be positive")
        if self.embedding_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("pruning.embedding_device must be auto, cpu, or cuda")
        if self.difficulty_mode not in {"uniform", "external"}:
            raise ValueError("pruning.difficulty_mode must be uniform or external")
        if self.difficulty_mode == "external" and self.difficulty_scores_path is None:
            raise ValueError(
                "external D2 difficulty requires difficulty_scores_path"
            )

    @property
    def pruning_rate(self) -> float:
        return 1.0 - self.retention_fraction

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, base_dir: Path
    ) -> "D2PruningConfig":
        values = dict(value)
        for key in ("embeddings_path", "difficulty_scores_path"):
            raw = values.get(key)
            if raw in (None, ""):
                values[key] = None
                continue
            path = Path(str(raw)).expanduser()
            values[key] = path if path.is_absolute() else (base_dir / path).resolve()
        return cls(**values)


@dataclass(frozen=True)
class D2SelectionResult:
    """Complete deterministic output of D2 graph scoring and selection."""

    selected_indices: tuple[int, ...]
    selection_order: tuple[int, ...]
    selection_scores: tuple[float, ...]
    forward_scores: tuple[float, ...]
    final_scores: tuple[float, ...]
    backend: str
    directed_neighbor_edges: int
    undirected_edges: int

    @property
    def selected_count(self) -> int:
        return len(self.selected_indices)
