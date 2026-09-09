"""Reusable Streamlit card for registered synthetic dataset families."""

from __future__ import annotations

from typing import Any

from attention_maps.explorer import DatasetSpec
from attention_maps.generation import synthetic_dataset_families


DATASET_CATALOG_AREA = "Dataset catalog"
SYNTHETIC_DATA_AREA = "Synthetic data generation"


def render_synthetic_dataset_card(
    st: Any,
    *,
    key: str,
) -> DatasetSpec | None:
    """Select a registered synthetic family and one materialized variant."""

    families = synthetic_dataset_families()
    card = st.container(border=True)
    card.markdown("### Synthetic data generation")
    card.caption(
        "Generated datasets are organized by pipeline family and variant. "
        "Registering another family makes it available here and in every app "
        "that uses this shared component."
    )
    family = card.selectbox(
        "Synthetic dataset family",
        families,
        format_func=lambda item: item.label,
        key=f"{key}:family",
    )
    card.write(family.description)
    stages = card.columns(3)
    stages[0].metric("Input", family.input_label)
    stages[1].metric("Transformation", family.transformation_label)
    stages[2].metric("Output", family.output_label)
    card.code(str(family.expected_location), language=None)

    if not family.variants:
        card.warning(
            f"No materialized variants were found for {family.label}. "
            "Run its generation pipeline first."
        )
        return None

    card.success(
        f"{len(family.variants)} materialized variant(s) detected for "
        f"{family.label}"
    )
    return card.selectbox(
        "Dataset variant",
        family.variants,
        format_func=lambda spec: spec.label,
        key=f"{key}:variant",
        help=(
            "A variant can be generated output, source/reference data, or a future "
            "task-specific rendering produced by the selected family."
        ),
    )
