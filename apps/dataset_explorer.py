"""Focused dataset preprocessing and post-cleaning EDA workspace.

Run with:

    python scripts/run_dataset_explorer.py

The supervisor launcher provides bounded Ctrl+C shutdown for long-running
preprocessing, EDA, upload, and inference work.

The app discovers the repository's raw, cleaned, processed, and tokenized
Parquet datasets. The repository's LIMA source and translation JSON is exposed
as separate original-English and translated-Nepali datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from apps.components import (
    DATASET_CATALOG_AREA,
    SYNTHETIC_DATA_AREA,
    render_synthetic_dataset_card,
)
from attention_maps.datasets.kaggle import (
    inspect_kaggle_text,
    inspect_kaggle_workbook,
)
from attention_maps.explorer import (
    ALL_PROVIDERS,
    ALL_PURPOSES,
    DEFAULT_DATA_ROOT,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGES,
    TRACKED_PROVIDERS,
    DatasetSpec,
    available_dataset_purposes,
    aya_nepali_dataset_specs,
    custom_dataset,
    dataset_size_bucket,
    discover_datasets,
    file_signatures,
    filter_dataset_specs,
    format_bytes,
    format_decimal_bytes,
    himalaya_ai_dataset_specs,
    inspect_dataset,
    inspect_huggingface_dataset,
    iriis_nepali_text_corpus_specs,
    kaggle_dataset_specs,
    project_inventory,
    sample_dataset_rows,
    secret_fingerprint,
)
from apps.explorer_tabs import (
    render_details_tab,
    render_inference_hub,
    render_manifest_tab,
    render_sample_tab,
    render_workspace_tab,
)


def run_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -r requirements.txt`."
        ) from error

    st.set_page_config(
        page_title="Dataset Preprocessing",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Dataset Preprocessing & EDA Workspace")
    st.caption(
        "Select a dataset, verify its source and schema, then run Sampling → "
        "NFC normalization → Deduplication → EDA on the preprocessed output."
    )

    @st.cache_data(show_spinner=False)
    def cached_inventory(
        signatures: tuple[tuple[str, int, int], ...],
    ) -> dict[str, Any]:
        return inspect_dataset(signatures)

    @st.cache_data(show_spinner=False)
    def cached_huggingface_inventory(
        dataset_id: str,
        config: str | None,
        split: str,
        filter_column: str | None,
        filter_value: str | None,
        credential_fingerprint: str,
        _token: str,
    ) -> dict[str, Any]:
        del credential_fingerprint
        return inspect_huggingface_dataset(
            dataset_id,
            split,
            config=config,
            token=_token or None,
            filter_column=filter_column,
            filter_value=filter_value,
        )

    @st.cache_data(show_spinner=False)
    def cached_kaggle_inventory(
        dataset_id: str,
        dataset_file: str,
    ) -> dict[str, Any]:
        return inspect_kaggle_workbook(dataset_id, dataset_file)

    @st.cache_data(show_spinner=False)
    def cached_kaggle_text_inventory(
        dataset_id: str,
        dataset_file: str,
    ) -> dict[str, Any]:
        return inspect_kaggle_text(dataset_id, dataset_file)

    @st.cache_data(show_spinner=False)
    def cached_sample(
        inventory: dict[str, Any],
        sample_size: int,
        seed: int,
        columns: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        return sample_dataset_rows(inventory, sample_size, seed, columns)

    st.sidebar.header("Dataset")
    source_settings = st.sidebar.expander("Local source settings")
    data_root_text = source_settings.text_input("Data root", str(DEFAULT_DATA_ROOT))
    data_root = Path(data_root_text).expanduser()
    use_custom = source_settings.checkbox("Use custom Parquet path")

    spec: DatasetSpec | None = None
    if use_custom:
        custom_path_text = source_settings.text_input("Parquet file or directory")
        if custom_path_text:
            spec = custom_dataset(Path(custom_path_text))
            if spec is None:
                st.sidebar.error("No Parquet files were found at that path.")
    else:
        source_area = st.sidebar.radio(
            "Dataset workspace",
            (DATASET_CATALOG_AREA, SYNTHETIC_DATA_AREA),
            help=(
                "Keep curated corpus browsing separate from artifacts produced "
                "by the synthetic data generation pipeline."
            ),
        )
        if source_area == SYNTHETIC_DATA_AREA:
            spec = render_synthetic_dataset_card(st, key="explorer-synthetic")
        else:
            pipeline_specs = discover_datasets(data_root)
            external_specs = []
            external_specs.extend(himalaya_ai_dataset_specs())
            external_specs.extend(aya_nepali_dataset_specs())
            external_specs.extend(iriis_nepali_text_corpus_specs())
            external_specs.extend(kaggle_dataset_specs())
            if not pipeline_specs and not external_specs:
                st.warning(f"No supported datasets found under `{data_root}`.")
                st.stop()
            catalog_filters = st.sidebar.expander("Optional catalog filters")
            stage = catalog_filters.selectbox(
                "Pipeline stage",
                PIPELINE_STAGES,
                format_func=lambda item: PIPELINE_STAGE_LABELS[item],
                help="Applies to local pipeline datasets; remote catalog entries are stage-independent.",
            )
            # Prefer an available local stage by default; remote datasets can
            # otherwise trigger a network request as soon as the app starts.
            all_specs = [*pipeline_specs, *external_specs]
            provider_options = list(
                dict.fromkeys(
                    (
                        ALL_PROVIDERS,
                        *TRACKED_PROVIDERS,
                        *(facet for item in all_specs for facet in item.provider_facets),
                    )
                )
            )
            selected_provider = catalog_filters.selectbox(
                "Provider / lineage",
                provider_options,
                help=(
                    "Matches the dataset publisher, source provider, or adapting "
                    "provider without conflating those roles."
                ),
            )
            purpose_options = available_dataset_purposes(
                all_specs,
                provider=selected_provider,
                stage=stage,
            )
            purpose_state_key = "dataset-purpose-filter"
            if st.session_state.get(purpose_state_key) not in purpose_options:
                st.session_state[purpose_state_key] = ALL_PURPOSES
            selected_purpose = catalog_filters.selectbox(
                "Dataset purpose",
                purpose_options,
                key=purpose_state_key,
                help=(
                    "Shows only purposes assigned to datasets available for the "
                    "selected provider/lineage and pipeline stage. These mappings "
                    "are curated by humans in the dataset catalog."
                ),
            )
            dataset_specs = filter_dataset_specs(
                all_specs,
                provider=selected_provider,
                purpose=selected_purpose,
                stage=stage,
            )
            if not dataset_specs:
                st.info(
                    "No dataset is cataloged for this provider/purpose combination yet. "
                    "The taxonomy bucket remains available for future entries."
                )
                st.stop()
            spec = st.sidebar.selectbox(
                "Dataset / split",
                dataset_specs,
                format_func=lambda item: f"{item.label} · {item.purpose_label}",
            )

    if spec is None:
        st.info("Select a valid dataset to begin.")
        st.stop()

    heading = (
        f"{PIPELINE_STAGE_LABELS[spec.stage]} · {spec.label}"
        if spec.stage in PIPELINE_STAGE_LABELS
        else spec.label
    )
    large_download_key = f"large-download-confirmed:{spec.key}"
    if spec.format == "kaggle_text" and not st.session_state.get(
        large_download_key, False
    ):
        st.subheader(heading)
        st.code(str(spec.location), language=None)
        st.warning(
            "This selection uses the deduplicated OSCAR file. Kaggle must download "
            f"approximately {format_bytes(spec.download_bytes or 0)} once before "
            "records can be inspected. The larger duplicate-containing file is not "
            "downloaded."
        )
        summary_columns = st.columns(3)
        summary_columns[0].metric("Rows", "Not scanned")
        summary_columns[1].metric("Files", "1")
        summary_columns[2].metric(
            "Approx. download", format_bytes(spec.download_bytes or 0)
        )
        if st.button(
            "Download and index deduplicated OSCAR Nepali",
            type="primary",
            key=f"confirm-large-download:{spec.key}",
        ):
            st.session_state[large_download_key] = True
            st.rerun()
        st.info(
            "After confirmation, the file is cached by KaggleHub. Line counting "
            "and sampling are streaming and do not load the corpus into RAM."
        )
        st.stop()

    try:
        with st.spinner("Reading dataset metadata…"):
            if spec.format == "huggingface":
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
                inventory = cached_huggingface_inventory(
                    spec.dataset_id or "",
                    spec.dataset_config,
                    spec.dataset_split or "train",
                    spec.filter_column,
                    spec.filter_value,
                    secret_fingerprint(hf_token),
                    hf_token,
                )
            elif spec.format == "kaggle":
                inventory = cached_kaggle_inventory(
                    spec.dataset_id or "",
                    spec.dataset_file or "",
                )
            elif spec.format == "kaggle_text":
                inventory = cached_kaggle_text_inventory(
                    spec.dataset_id or "",
                    spec.dataset_file or "",
                )
            else:
                signatures = file_signatures(spec.files)
                inventory = project_inventory(
                    cached_inventory(signatures), spec.visible_columns
                )
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Could not inspect this dataset: {error}")
        st.stop()

    st.subheader(heading)
    st.code(str(spec.location), language=None)
    taxonomy_parts = [
        f"**Provider:** {spec.provider}",
        f"**Purpose:** {spec.purpose_label}",
    ]
    if spec.source_provider:
        taxonomy_parts.append(f"**Source provider:** {spec.source_provider}")
    if spec.adapted_by:
        taxonomy_parts.append(f"**Adapted by:** {', '.join(spec.adapted_by)}")
    st.markdown(" · ".join(taxonomy_parts))
    metric_columns = st.columns(5)
    metric_columns[0].metric("Rows", f"{inventory['rows']:,}")
    if inventory["format"] == "huggingface":
        metric_columns[1].metric("Remote shards", f"{inventory['files']:,}")
        metric_columns[2].metric("Columns", f"{len(inventory['columns']):,}")
        hub_file_bytes = inventory.get("hub_file_bytes")
        metric_columns[3].metric(
            "Hub file size",
            (
                format_decimal_bytes(hub_file_bytes)
                if hub_file_bytes is not None
                else "Unknown"
            ),
            help="Compressed source files stored/downloaded from Hugging Face.",
        )
        metric_columns[4].metric(
            "Estimated memory size",
            format_bytes(inventory["memory_bytes"]),
            help=(
                "Decoded Arrow data size for the selected split; this is not "
                "the download or disk size."
            ),
        )
        loading_message = (
            "Large remote dataset: bounded streaming mode is active. "
            if hub_file_bytes is not None and hub_file_bytes >= 1024**3
            else "Bounded streaming mode is active. "
        )
        st.info(
            loading_message
            + "Sampling is bounded and the full dataset is not held in memory."
        )
    else:
        metric_columns[1].metric("Files", f"{inventory['files']:,}")
        if inventory["format"] == "json":
            grouping_label, grouping_value = "JSON documents", inventory["files"]
        elif inventory["format"] == "kaggle":
            grouping_label, grouping_value = "Worksheets", 1
        elif inventory["format"] == "kaggle_text":
            grouping_label, grouping_value = "Text files", inventory["files"]
        else:
            grouping_label = "Row groups"
            grouping_value = len(inventory["row_groups"])
        metric_columns[2].metric(grouping_label, f"{grouping_value:,}")
        metric_columns[3].metric("Columns", f"{len(inventory['columns']):,}")
        metric_columns[4].metric(
            "Estimated size" if inventory.get("bytes_estimated") else "Disk size",
            format_bytes(inventory["bytes"]),
        )

    size_bucket = dataset_size_bucket(inventory)
    st.caption(
        f"Size bucket: **{size_bucket.label}** ({size_bucket.size_basis})."
    )

    if inventory["schema_variants"] > 1:
        st.warning(
            f"The files contain {inventory['schema_variants']} schema variants. "
            "Sampling is limited to columns shared by every shard."
        )

    workspace_tab, sample_tab, inference_tab, metadata_tab = st.tabs(
        ["WORKSPACE", "Source sample", "Inference", "Metadata"]
    )
    with workspace_tab:
        render_workspace_tab(st=st, inventory=inventory, spec=spec)

    with sample_tab:
        render_sample_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
        )

    with inference_tab:
        render_inference_hub(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
        )

    with metadata_tab:
        render_details_tab(st=st, inventory=inventory, spec=spec)
        st.divider()
        render_manifest_tab(st=st, spec=spec, data_root=data_root)


if __name__ == "__main__":
    run_app()
