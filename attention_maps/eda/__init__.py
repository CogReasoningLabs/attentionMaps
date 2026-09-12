"""Bounded, reproducible exploratory analysis for text dataset surveys."""

from attention_maps.eda.contracts import (
    AnalysisConfig,
    DatasetProfile,
    DatasetSpec,
    SurveyPlan,
    SurveyRun,
)
from attention_maps.eda.pipeline import analyze_records, run_survey
from attention_maps.eda.deduplication import (
    DeduplicationConfig,
    MultiStageDeduplicator,
    normalize_for_deduplication,
)
from attention_maps.eda.sampling import (
    RepeatedSamplingPlan,
    fold_seed,
    plan_repeated_sampling,
)
from attention_maps.eda.script import (
    SCRIPT_CATEGORIES,
    ScriptEvidence,
    identify_script_category,
)
from attention_maps.eda.workspace import (
    NormalizationResult,
    WorkspaceDeduplicationResult,
    WorkspaceDocument,
    WorkspaceRemoval,
    deduplicate_workspace_documents,
    normalize_workspace_sample,
    persist_workspace_artifacts,
)

__all__ = [
    "AnalysisConfig",
    "DatasetProfile",
    "DatasetSpec",
    "DeduplicationConfig",
    "MultiStageDeduplicator",
    "NormalizationResult",
    "RepeatedSamplingPlan",
    "SurveyPlan",
    "SurveyRun",
    "SCRIPT_CATEGORIES",
    "ScriptEvidence",
    "WorkspaceDeduplicationResult",
    "WorkspaceDocument",
    "WorkspaceRemoval",
    "analyze_records",
    "deduplicate_workspace_documents",
    "identify_script_category",
    "fold_seed",
    "normalize_for_deduplication",
    "normalize_workspace_sample",
    "persist_workspace_artifacts",
    "plan_repeated_sampling",
    "run_survey",
]
