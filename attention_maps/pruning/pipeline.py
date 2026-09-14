"""Artifact-producing D2 stage for the immutable batch preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from attention_maps.batch_pipeline.artifacts import write_manifest
from attention_maps.batch_pipeline.processing import (
    ParquetShardWriter,
    iter_parquet_records,
    prepared_from_mapping,
)

from .contracts import D2PruningConfig, D2SelectionResult
from .embeddings import generate_transformer_embeddings
from .selection import select_d2_coreset


D2Progress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class D2PipelineResult:
    selected_documents: int
    coreset_paths: tuple[Path, ...]
    scores_path: Path
    manifest_path: Path
    selection: D2SelectionResult


def run_d2_pruning_on_parquet(
    clean_paths: Sequence[Path],
    total_documents: int,
    output_dir: Path,
    config: D2PruningConfig,
    *,
    shard_rows: int,
    batch_size: int,
    progress: D2Progress | None = None,
) -> D2PipelineResult:
    """Select and materialize a D2 coreset from ordered clean Parquet shards."""

    if not config.enabled:
        raise ValueError("D2 pipeline stage requires pruning.enabled=true")
    if total_documents < 2:
        raise ValueError("D2 pruning requires at least two clean documents")
    output_dir.mkdir(parents=True, exist_ok=True)

    def documents() -> Iterator[tuple[str, str]]:
        for value in iter_parquet_records(clean_paths, batch_size):
            yield str(value["doc_id"]), str(value["text"])

    if config.embeddings_path is not None:
        embeddings, document_ids = _load_external_values(
            config.embeddings_path, "embeddings", expected_rows=total_documents
        )
        _validate_document_order(documents(), document_ids)
        if config.normalize_embeddings:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
    else:
        embeddings, document_ids = generate_transformer_embeddings(
            documents(),
            total_documents,
            output_dir / "embeddings.npy",
            config,
            progress=(
                (lambda completed, total: progress("embeddings", completed, total))
                if progress
                else None
            ),
        )

    if config.difficulty_mode == "uniform":
        difficulty = np.ones(total_documents, dtype=np.float64)
    else:
        assert config.difficulty_scores_path is not None
        difficulty, difficulty_ids = _load_external_values(
            config.difficulty_scores_path,
            "scores",
            expected_rows=total_documents,
        )
        if tuple(difficulty_ids) != tuple(document_ids):
            raise ValueError(
                "external D2 difficulty doc_ids do not match embedding doc_ids"
            )
        difficulty = np.asarray(difficulty, dtype=np.float64)

    if progress:
        progress("graph_selection", 0, total_documents)
    labels = None
    if config.label_balanced:
        labels = tuple(
            str(value.get("label", ""))
            for value in iter_parquet_records(clean_paths, batch_size)
        )
        missing_labels = sum(not label for label in labels)
        if missing_labels:
            raise ValueError(
                "label_balanced D2 requires input.label_column on every clean row; "
                f"{missing_labels:,}/{len(labels):,} labels are empty"
            )
    selection = select_d2_coreset(
        embeddings, difficulty, config, labels=labels
    )
    if progress:
        progress("graph_selection", total_documents, total_documents)
    coreset_paths, scores_path = _write_selection_artifacts(
        clean_paths,
        output_dir,
        difficulty,
        selection,
        shard_rows=shard_rows,
        batch_size=batch_size,
        progress=progress,
    )
    manifest_path = write_manifest(
        output_dir / "d2_manifest.json",
        {
            "schema_version": 1,
            "method": "D2 Pruning",
            "source_documents": total_documents,
            "selected_documents": selection.selected_count,
            "retention_fraction_requested": config.retention_fraction,
            "retention_fraction_actual": round(
                selection.selected_count / total_documents, 8
            ),
            "pruning_rate_actual": round(
                1 - selection.selected_count / total_documents, 8
            ),
            "config": asdict(config),
            "graph": {
                "backend": selection.backend,
                "directed_neighbor_edges": selection.directed_neighbor_edges,
                "undirected_edges": selection.undirected_edges,
                "distance": "squared_euclidean",
                "symmetric": True,
            },
            "difficulty": {
                "mode": config.difficulty_mode,
                "minimum": float(np.min(difficulty)),
                "maximum": float(np.max(difficulty)),
                "mean": float(np.mean(difficulty)),
            },
            "artifacts": {
                "scores": scores_path.name,
                "coreset": [str(path.relative_to(output_dir)) for path in coreset_paths],
            },
            "source_dataset_modified": False,
        },
    )
    return D2PipelineResult(
        selection.selected_count,
        tuple(coreset_paths),
        scores_path,
        manifest_path,
        selection,
    )


def _load_external_values(
    path: Path, key: str, *, expected_rows: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"external D2 {key} must be an .npz with '{key}' and 'doc_ids' arrays"
        )
    try:
        with np.load(path, allow_pickle=False) as artifact:
            values = np.array(artifact[key])
            document_ids = tuple(map(str, artifact["doc_ids"].tolist()))
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"could not load D2 artifact {path}: {error}") from error
    if len(values) != expected_rows or len(document_ids) != expected_rows:
        raise ValueError(
            f"D2 artifact {path} has {len(values):,} rows; expected {expected_rows:,}"
        )
    return values, document_ids


def _validate_document_order(
    documents: Iterable[tuple[str, str]], expected_ids: Sequence[str]
) -> None:
    actual = tuple(row_id for row_id, _ in documents)
    if actual != tuple(expected_ids):
        raise ValueError(
            "external D2 embeddings are not aligned with clean-data doc_id order"
        )


def _write_selection_artifacts(
    clean_paths: Sequence[Path],
    output_dir: Path,
    difficulty: np.ndarray,
    selection: D2SelectionResult,
    *,
    shard_rows: int,
    batch_size: int,
    progress: D2Progress | None,
) -> tuple[tuple[Path, ...], Path]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - base pipeline requires pyarrow
        raise RuntimeError("D2 artifacts require pyarrow") from error

    selected = set(selection.selected_indices)
    ranks = {index: rank for rank, index in enumerate(selection.selection_order, 1)}
    selected_scores = {
        index: score
        for index, score in zip(selection.selection_order, selection.selection_scores)
    }
    coreset_writer = ParquetShardWriter(output_dir / "coreset", shard_rows)
    scores_path = output_dir / "d2_scores.parquet"
    temporary_scores_path = scores_path.with_suffix(".parquet.tmp")
    score_schema = pa.schema(
        [
            ("document_index", pa.int64()),
            ("doc_id", pa.string()),
            ("source", pa.string()),
            ("difficulty_score", pa.float64()),
            ("forward_score", pa.float64()),
            ("final_score", pa.float64()),
            ("selected", pa.bool_()),
            ("selection_rank", pa.int64()),
            ("selection_score", pa.float64()),
        ]
    )
    score_writer = pq.ParquetWriter(
        temporary_scores_path, score_schema, compression="zstd"
    )
    score_buffer: list[dict[str, object]] = []

    def flush_scores() -> None:
        if not score_buffer:
            return
        table = pa.Table.from_pylist(score_buffer, schema=score_schema)
        score_writer.write_table(table)
        score_buffer.clear()

    try:
        for index, value in enumerate(iter_parquet_records(clean_paths, batch_size)):
            record = prepared_from_mapping(value)
            is_selected = index in selected
            if is_selected:
                coreset_writer.add(record)
            final_score = selection.final_scores[index]
            score_buffer.append(
                {
                    "document_index": index,
                    "doc_id": record.doc_id,
                    "source": record.source,
                    "difficulty_score": float(difficulty[index]),
                    "forward_score": float(selection.forward_scores[index]),
                    "final_score": (
                        None if not np.isfinite(final_score) else float(final_score)
                    ),
                    "selected": is_selected,
                    "selection_rank": ranks.get(index),
                    "selection_score": selected_scores.get(index),
                }
            )
            if len(score_buffer) >= 4_096:
                flush_scores()
            if progress and (index + 1) % 1_000 == 0:
                progress("artifacts", index + 1, len(difficulty))
        flush_scores()
    finally:
        score_writer.close()
    temporary_scores_path.replace(scores_path)
    coreset_paths = tuple(coreset_writer.close())
    if not coreset_paths or not scores_path.is_file():
        raise RuntimeError("D2 did not materialize its expected artifacts")
    if progress:
        progress("artifacts", len(difficulty), len(difficulty))
    return coreset_paths, scores_path
