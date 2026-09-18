"""Sparse k-nearest-neighbor graph construction for D2 pruning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


GraphProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class KNNGraph:
    """Symmetric sparse graph stored in compressed-row form."""

    indptr: np.ndarray
    indices: np.ndarray
    squared_distances: np.ndarray
    backend: str
    directed_neighbor_edges: int

    @property
    def nodes(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def undirected_edges(self) -> int:
        return int(self.indices.size // 2)

    def neighbors(self, node: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(self.indptr[node]), int(self.indptr[node + 1])
        return self.indices[start:end], self.squared_distances[start:end]


def build_knn_graph(
    embeddings: np.ndarray,
    n_neighbors: int,
    *,
    backend: str = "auto",
    block_size: int = 1_024,
    max_exact_records: int = 10_000,
    progress: GraphProgress | None = None,
) -> KNNGraph:
    """Build and symmetrize a Euclidean k-NN graph.

    FAISS and NumPy both return squared L2 distances. The graph stores those
    values directly so the paper's RBF equation is applied consistently.
    """

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("embeddings must have shape (at least 2, dimensions)")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain non-finite values")
    neighbors = min(int(n_neighbors), values.shape[0] - 1)
    if neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    selected_backend = backend
    if backend == "auto":
        selected_backend = "exact" if values.shape[0] <= max_exact_records else "faiss"
    if selected_backend == "exact":
        if values.shape[0] > max_exact_records:
            raise ValueError(
                f"exact D2 graph is limited to {max_exact_records:,} records; "
                "install faiss-cpu or select graph_backend=faiss"
            )
        distances, indices = _exact_neighbors(
            values, neighbors, block_size, progress=progress
        )
    elif selected_backend == "faiss":
        if progress:
            progress(0, values.shape[0])
        distances, indices = _faiss_neighbors(values, neighbors)
        if progress:
            progress(values.shape[0], values.shape[0])
    else:
        raise ValueError(f"unsupported D2 graph backend: {selected_backend}")
    return _symmetrize(indices, distances, selected_backend)


def _exact_neighbors(
    values: np.ndarray,
    n_neighbors: int,
    block_size: int,
    *,
    progress: GraphProgress | None,
) -> tuple[np.ndarray, np.ndarray]:
    count = values.shape[0]
    norms = np.einsum("ij,ij->i", values, values)
    all_indices = np.arange(count)
    result_indices = np.empty((count, n_neighbors), dtype=np.int64)
    result_distances = np.empty((count, n_neighbors), dtype=np.float32)
    if progress:
        progress(0, count)
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        block = values[start:end]
        squared = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + norms[None, :]
            - 2.0 * block @ values.T
        )
        np.maximum(squared, 0.0, out=squared)
        squared[np.arange(end - start), np.arange(start, end)] = np.inf
        for offset, row in enumerate(squared):
            chosen = _deterministic_smallest(row, n_neighbors, all_indices)
            result_indices[start + offset] = chosen
            result_distances[start + offset] = row[chosen]
        if progress:
            progress(end, count)
    return result_distances, result_indices


def _deterministic_smallest(
    distances: np.ndarray, count: int, all_indices: np.ndarray
) -> np.ndarray:
    partition = np.argpartition(distances, count - 1)[:count]
    boundary = float(np.max(distances[partition]))
    below = all_indices[distances < boundary]
    tied = all_indices[distances == boundary]
    needed = count - below.size
    candidates = np.concatenate((below, tied[:needed]))
    order = np.lexsort((candidates, distances[candidates]))
    return candidates[order]


def _faiss_neighbors(
    values: np.ndarray, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "FAISS D2 graphs require the optional faiss-cpu package"
        ) from error
    index = faiss.IndexFlatL2(values.shape[1])
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    index.add(contiguous)
    raw_distances, raw_indices = index.search(contiguous, n_neighbors + 1)
    distances = np.empty((len(values), n_neighbors), dtype=np.float32)
    indices = np.empty((len(values), n_neighbors), dtype=np.int64)
    for row, (row_distances, row_indices) in enumerate(
        zip(raw_distances, raw_indices)
    ):
        keep = row_indices != row
        filtered_indices = row_indices[keep][:n_neighbors]
        filtered_distances = row_distances[keep][:n_neighbors]
        if len(filtered_indices) != n_neighbors:
            raise RuntimeError("FAISS did not return enough non-self neighbors")
        indices[row] = filtered_indices
        distances[row] = filtered_distances
    return distances, indices


def _symmetrize(
    directed_indices: np.ndarray,
    directed_distances: np.ndarray,
    backend: str,
) -> KNNGraph:
    adjacency: list[dict[int, float]] = [
        {} for _ in range(int(directed_indices.shape[0]))
    ]
    for node, (neighbors, distances) in enumerate(
        zip(directed_indices, directed_distances)
    ):
        for neighbor, distance in zip(neighbors, distances):
            other = int(neighbor)
            if other < 0 or other == node:
                continue
            value = float(distance)
            previous = adjacency[node].get(other)
            if previous is None or value < previous:
                adjacency[node][other] = value
                adjacency[other][node] = value

    indptr = [0]
    indices: list[int] = []
    distances: list[float] = []
    for row in adjacency:
        for neighbor in sorted(row):
            indices.append(neighbor)
            distances.append(row[neighbor])
        indptr.append(len(indices))
    return KNNGraph(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        squared_distances=np.asarray(distances, dtype=np.float32),
        backend=backend,
        directed_neighbor_edges=int(directed_indices.size),
    )
