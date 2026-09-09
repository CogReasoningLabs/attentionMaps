"""Manifest tab renderers for the dataset explorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import find_manifest, json

def render_manifest_tab(*, st: Any, spec: Any, data_root: Path) -> None:
    remote_dataset = spec.format in {
        "huggingface",
        "kaggle",
        "kaggle_text",
    }
    manifest_path = None if remote_dataset else find_manifest(spec, data_root)
    if manifest_path is None:
        st.info(
            "Remote dataset metadata is shown in the Schema tab."
            if remote_dataset
            else "No associated manifest was found."
        )
    else:
        st.caption(str(manifest_path))
        try:
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read manifest: {error}")
