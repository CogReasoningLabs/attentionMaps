"""Paper-faithful forward and reverse message passing for D2 pruning."""

from __future__ import annotations

import heapq
import math
from dataclasses import replace
from typing import Sequence

import numpy as np

from .contracts import D2PruningConfig, D2SelectionResult
from .graph import KNNGraph, build_knn_graph


def confidence_variability(confidences: np.ndarray) -> np.ndarray:
    """Return per-example training-dynamics variability for supervised NLP.

    Rows are checkpoints/epochs and columns are examples. This is the standard
    deviation of the probability assigned to the correct label, matching the
    dataset-cartography difficulty signal used by the D2 paper for NLP.
    """

    values = np.asarray(confidences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("confidences must have shape (at least 2 epochs, examples)")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("confidence values must be finite probabilities in [0, 1]")
    return np.std(values, axis=0, ddof=0)


def select_d2_coreset(
    embeddings: np.ndarray,
    difficulty_scores: np.ndarray,
    config: D2PruningConfig,
    *,
    labels: Sequence[object] | np.ndarray | None = None,
) -> D2SelectionResult:
    """Select a coreset by balancing local density and example difficulty."""

    values = np.asarray(embeddings, dtype=np.float32)
    difficulty = np.asarray(difficulty_scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if difficulty.shape != (values.shape[0],):
        raise ValueError("difficulty_scores must contain one value per embedding")
    if not np.isfinite(difficulty).all() or np.any(difficulty < 0):
        raise ValueError("difficulty_scores must be finite and non-negative")
    if values.shape[0] < 2:
        raise ValueError("D2 pruning requires at least two examples")

    target = max(
        1,
        min(
            values.shape[0],
            math.ceil(values.shape[0] * config.retention_fraction),
        ),
    )
    if config.label_balanced:
        if labels is None:
            raise ValueError("label_balanced D2 selection requires labels")
        label_values = np.asarray(labels, dtype=object)
        if label_values.shape != (values.shape[0],):
            raise ValueError("labels must contain one value per embedding")
        return _select_label_balanced(values, difficulty, label_values, config, target)
    return _select_unbalanced(values, difficulty, config, target)


def _select_unbalanced(
    embeddings: np.ndarray,
    difficulty: np.ndarray,
    config: D2PruningConfig,
    target: int,
) -> D2SelectionResult:
    graph = build_knn_graph(
        embeddings,
        config.n_neighbors,
        backend=config.graph_backend,
        block_size=config.exact_block_size,
        max_exact_records=config.max_exact_records,
    )
    forward = _forward_message_passing(graph, difficulty, config.gamma_forward)
    order, chosen_scores, final = _reverse_message_selection(
        graph, forward, target, config.gamma_reverse
    )
    return D2SelectionResult(
        selected_indices=tuple(sorted(order)),
        selection_order=tuple(order),
        selection_scores=tuple(map(float, chosen_scores)),
        forward_scores=tuple(map(float, forward)),
        final_scores=tuple(map(float, final)),
        backend=graph.backend,
        directed_neighbor_edges=graph.directed_neighbor_edges,
        undirected_edges=graph.undirected_edges,
    )


def _forward_message_passing(
    graph: KNNGraph, difficulty: np.ndarray, gamma_forward: float
) -> np.ndarray:
    scores = difficulty.astype(np.float64, copy=True)
    for node in range(graph.nodes):
        neighbors, distances = graph.neighbors(node)
        weights = np.exp(-gamma_forward * distances.astype(np.float64))
        scores[node] += float(np.dot(weights, difficulty[neighbors]))
    return scores


def _reverse_message_selection(
    graph: KNNGraph,
    forward_scores: np.ndarray,
    target: int,
    gamma_reverse: float,
) -> tuple[list[int], list[float], np.ndarray]:
    scores = forward_scores.astype(np.float64, copy=True)
    versions = np.zeros(graph.nodes, dtype=np.int64)
    selected = np.zeros(graph.nodes, dtype=bool)
    heap = [(-float(score), node, 0) for node, score in enumerate(scores)]
    heapq.heapify(heap)
    order: list[int] = []
    chosen_scores: list[float] = []
    while len(order) < target:
        while heap:
            negative, node, version = heapq.heappop(heap)
            if not selected[node] and version == versions[node]:
                break
        else:  # pragma: no cover - defensive guard for corrupted heap state
            raise RuntimeError("D2 selection heap was exhausted")
        selected_score = -negative
        selected[node] = True
        order.append(node)
        chosen_scores.append(float(selected_score))
        neighbors, distances = graph.neighbors(node)
        weights = np.exp(-gamma_reverse * distances.astype(np.float64))
        for neighbor, weight in zip(neighbors, weights):
            other = int(neighbor)
            if selected[other]:
                continue
            scores[other] -= float(weight) * selected_score
            versions[other] += 1
            heapq.heappush(
                heap, (-float(scores[other]), other, int(versions[other]))
            )
    scores[selected] = -np.inf
    return order, chosen_scores, scores


def _select_label_balanced(
    embeddings: np.ndarray,
    difficulty: np.ndarray,
    labels: np.ndarray,
    config: D2PruningConfig,
    target: int,
) -> D2SelectionResult:
    label_strings = np.asarray([str(value) for value in labels], dtype=object)
    groups: dict[str, np.ndarray] = {}
    for label in sorted(set(label_strings)):
        groups[label] = np.flatnonzero(label_strings == label)
    budgets = _balanced_budgets(groups, target)
    forward = np.zeros(embeddings.shape[0], dtype=np.float64)
    final = np.zeros(embeddings.shape[0], dtype=np.float64)
    local_results: list[tuple[str, D2SelectionResult, np.ndarray]] = []
    directed_edges = undirected_edges = 0
    backends: set[str] = set()
    local_config = replace(config, label_balanced=False)
    for label, indices in groups.items():
        budget = budgets[label]
        if not budget:
            continue
        if len(indices) == 1:
            forward[indices[0]] = difficulty[indices[0]]
            final[indices[0]] = -np.inf
            result = D2SelectionResult(
                (0,), (0,), (float(difficulty[indices[0]]),),
                (float(difficulty[indices[0]]),), (-np.inf,), "singleton", 0, 0,
            )
        else:
            fraction = budget / len(indices)
            result = _select_unbalanced(
                embeddings[indices], difficulty[indices],
                replace(local_config, retention_fraction=fraction), budget,
            )
            directed_edges += result.directed_neighbor_edges
            undirected_edges += result.undirected_edges
            backends.add(result.backend)
            forward[indices] = result.forward_scores
            final[indices] = result.final_scores
        local_results.append((label, result, indices))

    order: list[int] = []
    chosen_scores: list[float] = []
    max_rank = max(len(result.selection_order) for _, result, _ in local_results)
    for rank in range(max_rank):
        for _, result, indices in local_results:
            if rank < len(result.selection_order):
                order.append(int(indices[result.selection_order[rank]]))
                chosen_scores.append(float(result.selection_scores[rank]))
    return D2SelectionResult(
        selected_indices=tuple(sorted(order)),
        selection_order=tuple(order),
        selection_scores=tuple(chosen_scores),
        forward_scores=tuple(map(float, forward)),
        final_scores=tuple(map(float, final)),
        backend="+".join(sorted(backends)) if backends else "singleton",
        directed_neighbor_edges=directed_edges,
        undirected_edges=undirected_edges,
    )


def _balanced_budgets(groups: dict[str, np.ndarray], target: int) -> dict[str, int]:
    total = sum(len(indices) for indices in groups.values())
    exact = {label: target * len(indices) / total for label, indices in groups.items()}
    budgets = {
        label: min(len(groups[label]), int(value)) for label, value in exact.items()
    }
    remaining = target - sum(budgets.values())
    candidates = sorted(
        groups,
        key=lambda label: (-(exact[label] - int(exact[label])), label),
    )
    while remaining:
        progressed = False
        for label in candidates:
            if budgets[label] < len(groups[label]):
                budgets[label] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:  # pragma: no cover - target is bounded by total
            break
    return budgets
