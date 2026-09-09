"""Registry of synthetic dataset families and their materialized variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from attention_maps.explorer import (
    DEFAULT_LIMA_TRANSLATIONS_PATH,
    DatasetSpec,
    lima_synthetic_dataset_specs,
)


@dataclass(frozen=True)
class SyntheticDatasetFamily:
    """One generation pipeline and the dataset variants it materializes."""

    key: str
    label: str
    description: str
    input_label: str
    transformation_label: str
    output_label: str
    expected_location: Path
    variants: tuple[DatasetSpec, ...]


def synthetic_dataset_families() -> tuple[SyntheticDatasetFamily, ...]:
    """Return the registry consumed by Explorer and Dataset Generator."""

    return (
        SyntheticDatasetFamily(
            key="lima-translation",
            label="LIMA translation",
            description=(
                "English instruction data translated into Nepali for supervised "
                "finetuning, with source and generated views retained together."
            ),
            input_label="LIMA English",
            transformation_label="Gemini / Gemma",
            output_label="Nepali SFT JSON",
            expected_location=DEFAULT_LIMA_TRANSLATIONS_PATH,
            variants=lima_synthetic_dataset_specs(),
        ),
    )
