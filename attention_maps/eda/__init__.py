"""Bounded, reproducible exploratory analysis for text dataset surveys."""

from attention_maps.eda.contracts import (
    AnalysisConfig,
    DatasetProfile,
    DatasetSpec,
    SurveyPlan,
    SurveyRun,
)
from attention_maps.eda.pipeline import analyze_records, run_survey

__all__ = [
    "AnalysisConfig",
    "DatasetProfile",
    "DatasetSpec",
    "SurveyPlan",
    "SurveyRun",
    "analyze_records",
    "run_survey",
]
