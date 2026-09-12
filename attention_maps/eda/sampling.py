"""Population-aware plans for repeatable, resource-bounded EDA samples."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RepeatedSamplingPlan:
    population_rows: int
    requested_percentage: float
    folds: int
    requested_rows_per_fold: int
    rows_per_fold: int
    effective_percentage_per_fold: float
    expected_unique_rows: int
    expected_population_coverage_pct: float
    total_rows_read: int
    capped: bool
    full_population: bool
    disjoint_folds: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "population_rows": self.population_rows,
            "requested_percentage": self.requested_percentage,
            "folds": self.folds,
            "requested_rows_per_fold": self.requested_rows_per_fold,
            "rows_per_fold": self.rows_per_fold,
            "effective_percentage_per_fold": self.effective_percentage_per_fold,
            "expected_unique_rows": self.expected_unique_rows,
            "expected_population_coverage_pct": self.expected_population_coverage_pct,
            "total_rows_read": self.total_rows_read,
            "capped": self.capped,
            "full_population": self.full_population,
            "disjoint_folds": self.disjoint_folds,
        }


def plan_repeated_sampling(
    population_rows: int,
    percentage: float,
    *,
    folds: int = 5,
    max_rows_per_fold: int = 50_000,
    disjoint_folds: bool = False,
) -> RepeatedSamplingPlan:
    """Derive fold sizes and expected overlap from a population percentage."""

    if population_rows <= 0:
        raise ValueError("population_rows must be positive")
    if not 0 < percentage <= 100:
        raise ValueError("percentage must be in (0, 100]")
    if folds <= 0:
        raise ValueError("folds must be positive")
    if max_rows_per_fold <= 0:
        raise ValueError("max_rows_per_fold must be positive")

    requested = min(
        population_rows,
        max(1, math.ceil(population_rows * percentage / 100)),
    )
    rows_per_fold = min(requested, max_rows_per_fold)
    fraction = rows_per_fold / population_rows
    expected_unique_rows = (
        min(population_rows, rows_per_fold * folds)
        if disjoint_folds
        else round(population_rows * (1 - (1 - fraction) ** folds))
    )
    total_rows_read = (
        expected_unique_rows if disjoint_folds else rows_per_fold * folds
    )
    return RepeatedSamplingPlan(
        population_rows=population_rows,
        requested_percentage=round(percentage, 6),
        folds=folds,
        requested_rows_per_fold=requested,
        rows_per_fold=rows_per_fold,
        effective_percentage_per_fold=round(100 * fraction, 6),
        expected_unique_rows=expected_unique_rows,
        expected_population_coverage_pct=round(
            100 * expected_unique_rows / population_rows, 6
        ),
        total_rows_read=total_rows_read,
        capped=rows_per_fold < requested,
        full_population=expected_unique_rows == population_rows,
        disjoint_folds=disjoint_folds,
    )


def fold_seed(base_seed: int, fold_index: int) -> int:
    """Return a stable, well-separated seed for a one-based fold index."""

    if fold_index <= 0:
        raise ValueError("fold_index must be positive")
    return int(base_seed) + (fold_index - 1) * 1_000_003
