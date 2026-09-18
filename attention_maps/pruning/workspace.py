"""D2 execution and auditable artifacts for the interactive clean workspace."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from attention_maps.datasets.schemas import TASK_SPECIFIC_SUPERVISED_SCHEMA
from attention_maps.eda.workspace import WorkspaceDocument

from .contracts import D2PruningConfig, D2SelectionResult
from .embeddings import generate_transformer_embeddings
from .selection import confidence_variability, select_d2_coreset


WorkspaceD2Progress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class D2ProjectionPoint:
    document_index: int
    row_id: str
    source: str
    label: str
    x: float
    y: float
    difficulty: float
    forward_score: float
    selected: bool


@dataclass(frozen=True)
class WorkspaceD2Result:
    source_documents: int
    selected_documents: tuple[WorkspaceDocument, ...]
    selection: D2SelectionResult
    projection: tuple[D2ProjectionPoint, ...]
    output_dir: Path
    coreset_path: Path
    scores_path: Path
    manifest_path: Path

    @property
    def retention_pct(self) -> float:
        return round(100 * len(self.selected_documents) / self.source_documents, 4)


def load_aligned_npz(
    payload: bytes,
    key: str,
    expected_ids: Sequence[str],
    *,
    record_axis: int = 0,
) -> np.ndarray:
    """Load one array and require exact row-ID alignment with clean documents."""

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
            values = np.array(artifact[key])
            document_ids = tuple(map(str, artifact["doc_ids"].tolist()))
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(
            f"uploaded artifact must contain '{key}' and 'doc_ids': {error}"
        ) from error
    if values.ndim <= record_axis or values.shape[record_axis] != len(expected_ids):
        raise ValueError(
            f"'{key}' has {values.shape}; axis {record_axis} must contain "
            f"{len(expected_ids):,} documents"
        )
    if document_ids != tuple(expected_ids):
        raise ValueError("uploaded doc_ids do not match clean workspace row order")
    return values


def difficulty_from_confidence_artifact(
    payload: bytes, expected_ids: Sequence[str]
) -> np.ndarray:
    confidences = load_aligned_npz(
        payload, "confidences", expected_ids, record_axis=1
    )
    return confidence_variability(confidences)


def run_workspace_d2(
    documents: Sequence[WorkspaceDocument],
    config: D2PruningConfig,
    output_dir: Path,
    difficulty_scores: np.ndarray,
    *,
    embeddings: np.ndarray | None = None,
    difficulty_artifact: bytes | None = None,
    embeddings_artifact: bytes | None = None,
    progress: WorkspaceD2Progress | None = None,
) -> WorkspaceD2Result:
    """Run D2 on a clean workspace copy and persist an immutable result folder."""

    if not config.enabled:
        raise ValueError("workspace D2 requires pruning.enabled=true")
    if len(documents) < 2:
        raise ValueError("D2 requires at least two clean workspace documents")
    output_dir.mkdir(parents=True, exist_ok=False)
    if difficulty_artifact is not None:
        (output_dir / "difficulty_input.npz").write_bytes(difficulty_artifact)
    if embeddings_artifact is not None:
        (output_dir / "embeddings_input.npz").write_bytes(embeddings_artifact)
    document_ids = tuple(document.row_id for document in documents)
    if embeddings is None:
        embeddings, generated_ids = generate_transformer_embeddings(
            ((document.row_id, document.text) for document in documents),
            len(documents),
            output_dir / "embeddings.npy",
            config,
            progress=(
                (lambda done, total: progress("embeddings", done, total))
                if progress
                else None
            ),
        )
        if generated_ids != document_ids:  # pragma: no cover - defensive
            raise RuntimeError("generated embedding order changed unexpectedly")
    values = np.asarray(embeddings, dtype=np.float32)
    if config.normalize_embeddings:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.maximum(norms, 1e-12)
    difficulty = np.asarray(difficulty_scores, dtype=np.float64)
    labels = None
    if config.label_balanced:
        labels = tuple(document.label for document in documents)
        missing = sum(not label for label in labels)
        if missing:
            raise ValueError(
                f"class-balanced D2 requires every label; {missing:,} are empty"
            )
    selection = select_d2_coreset(
        values,
        difficulty,
        config,
        labels=labels,
        progress=progress,
    )
    if progress:
        progress("artifacts", 0, len(documents))
    coordinates = project_embeddings_2d(values)
    selected_indices = set(selection.selected_indices)
    projection = tuple(
        D2ProjectionPoint(
            document_index=index,
            row_id=document.row_id,
            source=document.source,
            label=document.label,
            x=float(coordinates[index, 0]),
            y=float(coordinates[index, 1]),
            difficulty=float(difficulty[index]),
            forward_score=float(selection.forward_scores[index]),
            selected=index in selected_indices,
        )
        for index, document in enumerate(documents)
    )
    selected_documents = tuple(
        document
        for index, document in enumerate(documents)
        if index in selected_indices
    )
    coreset_path = output_dir / "d2_coreset.jsonl"
    scores_path = output_dir / "d2_scores.csv"
    manifest_path = output_dir / "d2_manifest.json"
    _write_coreset(coreset_path, selected_documents)
    _write_scores(scores_path, projection, selection)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "D2 Pruning",
                "training_schema": TASK_SPECIFIC_SUPERVISED_SCHEMA,
                "use_case": config.use_case,
                "source_documents": len(documents),
                "selected_documents": len(selected_documents),
                "retention_fraction_actual": (
                    len(selected_documents) / len(documents)
                ),
                "config": asdict(config),
                "graph": {
                    "backend": selection.backend,
                    "directed_neighbor_edges": selection.directed_neighbor_edges,
                    "undirected_edges": selection.undirected_edges,
                },
                "artifacts": {
                    "coreset": coreset_path.name,
                    "scores": scores_path.name,
                    "embeddings": (
                        "embeddings.npy" if (output_dir / "embeddings.npy").is_file()
                        else None
                    ),
                    "difficulty_input": (
                        "difficulty_input.npz"
                        if (output_dir / "difficulty_input.npz").is_file()
                        else None
                    ),
                    "embeddings_input": (
                        "embeddings_input.npz"
                        if (output_dir / "embeddings_input.npz").is_file()
                        else None
                    ),
                },
                "embedding_source": (
                    "generated"
                    if (output_dir / "embeddings.npy").is_file()
                    else "streamlit_upload"
                ),
                "source_dataset_modified": False,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    if progress:
        progress("artifacts", len(documents), len(documents))
    return WorkspaceD2Result(
        source_documents=len(documents),
        selected_documents=selected_documents,
        selection=selection,
        projection=projection,
        output_dir=output_dir,
        coreset_path=coreset_path,
        scores_path=scores_path,
        manifest_path=manifest_path,
    )


def project_embeddings_2d(embeddings: np.ndarray) -> np.ndarray:
    """Return a deterministic randomized-PCA view without an sklearn dependency."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("projection requires at least two embedding rows")
    centered = values - np.mean(values, axis=0, keepdims=True)
    if values.shape[1] == 1:
        return np.column_stack((centered[:, 0], np.zeros(values.shape[0])))
    rank = min(8, values.shape[0], values.shape[1])
    randomizer = np.random.default_rng(42)
    basis = randomizer.standard_normal((values.shape[1], rank))
    sketch = centered @ basis
    for _ in range(2):
        sketch = centered @ (centered.T @ sketch)
    orthogonal, _ = np.linalg.qr(sketch, mode="reduced")
    reduced = orthogonal.T @ centered
    _, _, components = np.linalg.svd(reduced, full_matrices=False)
    projected = centered @ components[:2].T
    if projected.shape[1] == 1:
        projected = np.column_stack((projected[:, 0], np.zeros(values.shape[0])))
    return projected


def _write_coreset(
    path: Path, documents: Sequence[WorkspaceDocument]
) -> None:
    path.write_text(
        "".join(
            json.dumps(document.as_record(), ensure_ascii=False, sort_keys=True)
            + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def _write_scores(
    path: Path,
    projection: Sequence[D2ProjectionPoint],
    selection: D2SelectionResult,
) -> None:
    ranks = {
        index: rank for rank, index in enumerate(selection.selection_order, start=1)
    }
    selection_scores = dict(
        zip(selection.selection_order, selection.selection_scores)
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "document_index",
                "row_id",
                "source",
                "label",
                "difficulty",
                "forward_score",
                "final_score",
                "selected",
                "selection_rank",
                "selection_score",
                "projection_x",
                "projection_y",
            ),
        )
        writer.writeheader()
        for point in projection:
            final_score = selection.final_scores[point.document_index]
            writer.writerow(
                {
                    "document_index": point.document_index,
                    "row_id": point.row_id,
                    "source": point.source,
                    "label": point.label,
                    "difficulty": point.difficulty,
                    "forward_score": point.forward_score,
                    "final_score": final_score if np.isfinite(final_score) else "",
                    "selected": point.selected,
                    "selection_rank": ranks.get(point.document_index, ""),
                    "selection_score": selection_scores.get(
                        point.document_index, ""
                    ),
                    "projection_x": point.x,
                    "projection_y": point.y,
                }
            )
