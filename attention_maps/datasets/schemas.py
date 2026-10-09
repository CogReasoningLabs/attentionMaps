"""Canonical instance schemas for training and reference-based evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import re


PRETRAINING_SCHEMA = "pretraining"
INSTRUCTION_FINETUNING_SCHEMA = "instruction_finetuning"
TASK_SPECIFIC_SUPERVISED_SCHEMA = "task_specific_supervised"
PREFERENCE_TUNING_SCHEMA = "preference_tuning"
EVALUATION_SCHEMA = "evaluation"


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

# Evaluation is a dataset role and a separate input/reference instance contract;
# it is not a fifth training objective.
EVALUATION_DATASET_SCHEMA = TrainingDataSchema(
    EVALUATION_SCHEMA,
    "Evaluation",
    ("id", "input", "reference", "split"),
    "One evaluation input paired with its expected reference answer.",
)
STANDARD_DATASET_SCHEMAS = (*STANDARD_TRAINING_SCHEMAS, EVALUATION_DATASET_SCHEMA)


D2_SUPPORTED_USE_CASES = (
    "traditional_nlp",
    "domain_specific_finetuning",
)


@dataclass(frozen=True)
class DatasetRole:
    key: str
    label: str
    training_schema: str
    description: str
    d2_use_case: str | None = None


# Roles describe the dataset's purpose; both supervised subcategories share
# the existing text/label/task instance contract.
_SUPERVISED_ROLES = (
    DatasetRole(
        "traditional_nlp", "Traditional NLP", TASK_SPECIFIC_SUPERVISED_SCHEMA,
        "Labelled examples for NLP tasks such as sentiment analysis, classification, or named-entity recognition.",
        "traditional_nlp",
    ),
    DatasetRole(
        "domain_specific_supervised", "Domain-specific fine-tuning", TASK_SPECIFIC_SUPERVISED_SCHEMA,
        "Labelled examples for supervised fine-tuning in a specific domain, such as medicine, law, or finance.",
        "domain_specific_finetuning",
    ),
)
_ROLE_LABELS = {
    PRETRAINING_SCHEMA: "Pretraining",
    INSTRUCTION_FINETUNING_SCHEMA: "Instruction SFT",
    TASK_SPECIFIC_SUPERVISED_SCHEMA: "Task-specific fine-tuning",
    PREFERENCE_TUNING_SCHEMA: "Preference tuning",
    EVALUATION_SCHEMA: "Evaluation",
}
DATASET_ROLES = tuple(
    role
    for schema in STANDARD_DATASET_SCHEMAS
    for role in (
        DatasetRole(schema.key, _ROLE_LABELS[schema.key], schema.key, schema.description),
        *(_SUPERVISED_ROLES if schema.key == TASK_SPECIFIC_SUPERVISED_SCHEMA else ()),
    )
)


def resolve_dataset_role(value: str) -> str | None:
    """Accept existing tracker/catalog spelling while storing one role vocabulary."""
    normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.casefold())
    aliases = {
        "Pre-training": PRETRAINING_SCHEMA,
        "Domain-specific supervised": "domain_specific_supervised",
        "Domain-specific finetuning": "domain_specific_supervised",
        "Task-specific finetuning": TASK_SPECIFIC_SUPERVISED_SCHEMA,
        "Evaluation / benchmark": EVALUATION_SCHEMA,
        **{schema.label: schema.key for schema in STANDARD_DATASET_SCHEMAS},
        **{role.label: role.key for role in DATASET_ROLES},
        **{role.key: role.key for role in DATASET_ROLES},
    }
    return {normalize(label): key for label, key in aliases.items()}.get(normalize(value))


def parse_dataset_roles(value: str) -> tuple[str, ...]:
    """Keep unrecognized labels visible for manual review instead of dropping them."""
    return tuple(dict.fromkeys(resolve_dataset_role(item.strip()) or item.strip()
                               for item in re.split(r"[,;|\n]+", value) if item.strip()))


def infer_training_schema(primary_purpose: str) -> str | None:
    """Map current and legacy dataset-role labels to the instance contract."""
    key = resolve_dataset_role(primary_purpose)
    return next((role.training_schema for role in DATASET_ROLES if role.key == key), None)
