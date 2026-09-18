"""Canonical training-data schema taxonomy shared by UI and documentation."""

from __future__ import annotations

from dataclasses import dataclass


PRETRAINING_SCHEMA = "pretraining"
INSTRUCTION_FINETUNING_SCHEMA = "instruction_finetuning"
TASK_SPECIFIC_SUPERVISED_SCHEMA = "task_specific_supervised"
PREFERENCE_TUNING_SCHEMA = "preference_tuning"


@dataclass(frozen=True)
class TrainingDataSchema:
    key: str
    label: str
    required_fields: tuple[str, ...]
    description: str


STANDARD_TRAINING_SCHEMAS = (
    TrainingDataSchema(
        PRETRAINING_SCHEMA,
        "Pretraining corpus",
        ("id", "text", "split"),
        "Unlabelled documents used for language-model pretraining.",
    ),
    TrainingDataSchema(
        INSTRUCTION_FINETUNING_SCHEMA,
        "Instruction fine-tuning",
        ("id", "messages", "split"),
        "Ordered user/system/assistant conversations used for SFT.",
    ),
    TrainingDataSchema(
        TASK_SPECIFIC_SUPERVISED_SCHEMA,
        "Task-specific supervised",
        ("id", "text", "label", "task", "split"),
        "Labelled traditional-NLP or domain-specific fine-tuning examples.",
    ),
    TrainingDataSchema(
        PREFERENCE_TUNING_SCHEMA,
        "Preference tuning",
        ("id", "prompt", "chosen", "rejected", "split"),
        "Atomic preference pairs used for reward modelling or alignment.",
    ),
)


D2_SUPPORTED_USE_CASES = (
    "traditional_nlp",
    "domain_specific_finetuning",
)


def infer_training_schema(primary_purpose: str) -> str | None:
    """Map the curated catalog purpose to one of the four standard schemas."""

    return {
        "Pretraining corpus": PRETRAINING_SCHEMA,
        "Instruction fine-tuning": INSTRUCTION_FINETUNING_SCHEMA,
        "Task-specific fine-tuning": TASK_SPECIFIC_SUPERVISED_SCHEMA,
        "Preference tuning": PREFERENCE_TUNING_SCHEMA,
    }.get(primary_purpose)
