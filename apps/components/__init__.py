"""Reusable Streamlit components shared by application entrypoints."""

from .synthetic_data import (
    DATASET_CATALOG_AREA,
    SYNTHETIC_DATA_AREA,
    render_synthetic_dataset_card,
)

__all__ = [
    "DATASET_CATALOG_AREA",
    "SYNTHETIC_DATA_AREA",
    "render_synthetic_dataset_card",
]
