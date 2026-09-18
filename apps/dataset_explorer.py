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
    SOURCE_TYPES,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGES,
    TRACKED_PROVIDERS,
    DatasetSpec,
    available_dataset_purposes,
    aya_nepali_dataset_specs,
    custom_dataset,
    dataset_size_bucket,
    discover_datasets,
    discover_import_files,
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
    stage_google_drive_source,
    stage_kaggle_source,
    stage_s3_object,
    staged_source_spec,
    huggingface_source_spec,
)
from attention_maps.datasets.schemas import STANDARD_TRAINING_SCHEMAS
from apps.explorer_tabs import (
    render_details_tab,
    render_inference_hub,
    render_manifest_tab,
    render_overlap_tab,
    render_sample_tab,
    render_workspace_tab,
)


def _render_remote_source(st: Any, source_type: str) -> DatasetSpec | None:
    """Render identifier-driven source controls and return a registered source."""

    schema = st.sidebar.selectbox(
        "Standard dataset schema",
        STANDARD_TRAINING_SCHEMAS,
        format_func=lambda item: item.label,
        key=f"source-schema:{source_type}",
        help="Controls atomic records and whether supervised D2 is available.",
    ).key
    state_key = f"registered-source:{source_type}"

    if source_type == "Hugging Face":
        st.sidebar.caption(
            "Loading strategy: inspect metadata and stream a bounded sample; the "
            "complete split is not downloaded."
        )
        dataset_id = st.sidebar.text_input(
            "Hugging Face dataset ID",
            placeholder="owner/dataset",
            key="source-hf-id",
        )
        config = st.sidebar.text_input(
            "Dataset configuration (optional)", key="source-hf-config"
        )
        split = st.sidebar.text_input(
            "Dataset split", value="train", key="source-hf-split"
        )
        revision = st.sidebar.text_input(
            "Revision (recommended)",
            placeholder="commit, tag, or branch",
            key="source-hf-revision",
        )
        signature = (dataset_id.strip(), config.strip(), split.strip(), revision.strip(), schema)
        if st.sidebar.button("Load Hugging Face source", type="primary"):
            try:
                spec = huggingface_source_spec(
                    dataset_id,
                    schema=schema,
                    config=config,
                    split=split,
                    revision=revision,
                )
            except ValueError as error:
                st.sidebar.error(str(error))
            else:
                st.session_state[state_key] = {"signature": signature, "spec": spec}
        saved = st.session_state.get(state_key, {})
        return saved.get("spec") if saved.get("signature") == signature else None

    if source_type == "Kaggle":
        st.sidebar.caption(
            "Loading strategy: cache only the requested internal path when supplied; "
            "otherwise cache the dataset and choose a file."
        )
        identifier = st.sidebar.text_input(
            "Kaggle dataset handle",
            placeholder="owner/dataset",
            key="source-kaggle-id",
        )
        requested_path = st.sidebar.text_input(
            "File or folder in dataset (optional)",
            help="Leave blank to cache the dataset and select one supported file.",
            key="source-kaggle-path",
        )
        signature = (identifier.strip(), requested_path.strip(), schema)
        if st.sidebar.button("Load Kaggle source", type="primary"):
            try:
                root = stage_kaggle_source(identifier, requested_path)
            except ValueError as error:
                st.sidebar.error(str(error))
            else:
                st.session_state[state_key] = {"signature": signature, "root": str(root)}
        return _staged_spec_from_state(
            st,
            state_key=state_key,
            signature=signature,
            source_type=source_type,
            schema=schema,
            source_prefix=f"kaggle://datasets/{identifier.strip()}",
        )

    if source_type == "Google Drive":
        st.sidebar.caption(
            "Loading strategy: resumably stage the selected file/folder, then sample "
            "the chosen supported file."
        )
        identifier = st.sidebar.text_input(
            "Drive file/folder ID or URL", key="source-drive-id"
        )
        signature = (identifier.strip(), schema)
        if st.sidebar.button("Load Drive source", type="primary"):
            try:
                root = stage_google_drive_source(identifier)
            except (RuntimeError, ValueError) as error:
                st.sidebar.error(str(error))
            else:
                st.session_state[state_key] = {"signature": signature, "root": str(root)}
        return _staged_spec_from_state(
            st,
            state_key=state_key,
            signature=signature,
            source_type=source_type,
            schema=schema,
            source_prefix=f"gdrive://{identifier.strip()}",
        )

    if source_type == "S3":
        st.sidebar.caption(
            "Loading strategy: validate and stage one exact object, then sample it "
            "with the format-specific bounded reader."
        )
        uri = st.sidebar.text_input(
            "S3 object URI",
            placeholder="s3://bucket/path/dataset.parquet",
            key="source-s3-uri",
        )
        signature = (uri.strip(), schema)
        if st.sidebar.button("Load S3 object", type="primary"):
            try:
                root = stage_s3_object(uri)
            except ValueError as error:
                st.sidebar.error(str(error))
            else:
                st.session_state[state_key] = {"signature": signature, "root": str(root)}
        return _staged_spec_from_state(
            st,
            state_key=state_key,
            signature=signature,
            source_type=source_type,
            schema=schema,
            source_prefix=uri.strip(),
        )

    raise ValueError(f"Unknown source type: {source_type}")


def _staged_spec_from_state(
    st: Any,
    *,
    state_key: str,
    signature: tuple[str, ...],
    source_type: str,
    schema: str,
    source_prefix: str,
) -> DatasetSpec | None:
    saved = st.session_state.get(state_key, {})
    if saved.get("signature") != signature or not saved.get("root"):
        return None
    root = Path(saved["root"])
    try:
        files = discover_import_files(root)
    except ValueError as error:
        st.sidebar.error(str(error))
        return None
    if not files:
        st.sidebar.error(
            "No supported Parquet, JSON, JSONL, CSV, TXT, or XLSX file was found."
        )
        return None
    selected = st.sidebar.selectbox(
        "Dataset file",
        files,
        format_func=lambda path: str(path.relative_to(root)) if root.is_dir() else path.name,
        key=f"source-file:{source_type}:{hash(signature)}",
    )
    relative = str(selected.relative_to(root)) if root.is_dir() else selected.name
    if source_prefix.startswith("s3://"):
        uri = source_prefix
    elif source_type == "Kaggle" and root.is_file() and len(signature) > 1 and signature[1]:
        uri = f"{source_prefix}/{signature[1]}"
    else:
        uri = f"{source_prefix}/{relative}"
    try:
        return staged_source_spec(
            selected,
            source_type=source_type,
            schema=schema,
            source_uri=uri,
        )
    except ValueError as error:
        st.sidebar.error(str(error))
        return None


def run_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -r requirements.txt`."
        ) from error

    # Provider SDKs read credentials from the process environment. Loading the
    # project-local file here keeps secrets out of widgets and session state.
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        pass
    else:
        from attention_maps.explorer import PROJECT_ROOT

        load_dotenv(PROJECT_ROOT / ".env")

    st.set_page_config(
        page_title="Dataset Preprocessing",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Dataset Preprocessing & EDA Workspace")
    st.caption(
        "Select a dataset, verify its source and schema, then run Sampling → "
        "NFC normalization → Deduplication → optional supervised D2 → EDA, "
        "or compare report-only overlap across catalog datasets."
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
        revision: str | None,
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
            revision=revision,
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

    st.sidebar.header("Dataset source")
    source_type = st.sidebar.selectbox(
        "Data source",
        SOURCE_TYPES,
        index=SOURCE_TYPES.index("Local"),
        help=(
            "Remote sources are registered by their standard identifier. Provider "
            "credentials are read from .env and are never stored in UI state."
        ),
    )
    if source_type == "Local":
        source_settings = st.sidebar.expander("Local source settings")
        data_root_text = source_settings.text_input("Data root", str(DEFAULT_DATA_ROOT))
        data_root = Path(data_root_text).expanduser()
        use_custom = source_settings.checkbox(
            "Use custom Parquet path",
            help=(
                "Also accepts one JSON, JSONL, CSV, TXT, or XLSX file. The label is "
                "retained for backward compatibility."
            ),
        )
    else:
        data_root = DEFAULT_DATA_ROOT
        use_custom = False

    pipeline_specs = discover_datasets(data_root)
    external_specs = []
    external_specs.extend(himalaya_ai_dataset_specs())
    external_specs.extend(aya_nepali_dataset_specs())
    external_specs.extend(iriis_nepali_text_corpus_specs())
    external_specs.extend(kaggle_dataset_specs())
    catalog_specs = [*pipeline_specs, *external_specs]

    spec: DatasetSpec | None = None
    if source_type != "Local":
        spec = _render_remote_source(st, source_type)
    elif use_custom:
        custom_path_text = source_settings.text_input("Parquet file or directory")
        if custom_path_text:
            spec = custom_dataset(Path(custom_path_text))
            if spec is None:
                st.sidebar.error("No supported dataset file was found at that path.")
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
            if not pipeline_specs and not external_specs:
                st.warning(f"No supported datasets found under `{data_root}`.")
                st.stop()
            catalog_filters = st.sidebar.expander("Optional catalog filters")
            stage = catalog_filters.selectbox(
                "Pipeline stage",
                PIPELINE_STAGES,
                format_func=lambda item: PIPELINE_STAGE_LABELS[item],
                help=(
                    "Applies to local pipeline datasets; remote catalog entries "
                    "are stage-independent."
                ),
            )
            # Prefer an available local stage by default; remote datasets can
            # otherwise trigger a network request as soon as the app starts.
            all_specs = catalog_specs
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
                    spec.dataset_revision,
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
                if spec.source_uri:
                    inventory = dict(inventory)
                    inventory["source_uri"] = spec.source_uri
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
        if inventory["format"] in {"json", "jsonl"}:
            grouping_label, grouping_value = "JSON files", inventory["files"]
        elif inventory["format"] in {"csv", "xlsx"}:
            grouping_label, grouping_value = "Tables", inventory["files"]
        elif inventory["format"] == "text":
            grouping_label, grouping_value = "Text files", inventory["files"]
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

    overlap_specs = list(
        {
            item.key: item
            for item in (spec, *catalog_specs)
            if item.format != "kaggle_text"
        }.values()
    )

    def inventory_for_spec(selected_spec: DatasetSpec) -> dict[str, Any]:
        if selected_spec.key == spec.key:
            return inventory
        if selected_spec.format == "huggingface":
            selected_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
            return cached_huggingface_inventory(
                selected_spec.dataset_id or "",
                selected_spec.dataset_config,
                selected_spec.dataset_split or "train",
                selected_spec.dataset_revision,
                selected_spec.filter_column,
                selected_spec.filter_value,
                secret_fingerprint(selected_hf_token),
                selected_hf_token,
            )
        if selected_spec.format == "kaggle":
            return cached_kaggle_inventory(
                selected_spec.dataset_id or "", selected_spec.dataset_file or ""
            )
        signatures = file_signatures(selected_spec.files)
        selected_inventory = project_inventory(
            cached_inventory(signatures), selected_spec.visible_columns
        )
        if selected_spec.source_uri:
            selected_inventory = dict(selected_inventory)
            selected_inventory["source_uri"] = selected_spec.source_uri
        return selected_inventory

    workspace_tab, overlap_tab, sample_tab, inference_tab, metadata_tab = st.tabs(
        ["WORKSPACE", "Dataset overlap", "Source sample", "Inference", "Metadata"]
    )
    with workspace_tab:
        render_workspace_tab(st=st, inventory=inventory, spec=spec)

    with overlap_tab:
        render_overlap_tab(
            st=st,
            specs=overlap_specs,
            current_spec=spec,
            inventory_for_spec=inventory_for_spec,
            cached_sample=cached_sample,
        )

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
