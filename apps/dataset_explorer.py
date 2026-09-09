"""Interactive explorer and survey EDA for pretraining and finetuning datasets.

Run with:

    streamlit run apps/dataset_explorer.py

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
from attention_maps.evaluation.flores import (
    load_flores_examples,
)
from attention_maps.evaluation.nlue import (
    load_nlue_examples,
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
    configured_nepali_stopwords,
    custom_dataset,
    discover_datasets,
    file_signatures,
    filter_dataset_specs,
    format_bytes,
    himalaya_ai_dataset_specs,
    inspect_dataset,
    inspect_huggingface_dataset,
    iriis_nepali_text_corpus_specs,
    kaggle_dataset_specs,
    project_inventory,
    sample_dataset_rows,
    secret_fingerprint,
)
from attention_maps.inference.comparison import (
    GoogleGenAIBackend,
    HuggingFaceBackend,
)
from attention_maps.inference.arkios import (
    load_arkios,
)
from attention_maps.inference.himalayagpt import (
    load_himalayagpt,
)
from attention_maps.inference.iriis_gpt2 import (
    iriis_gpt2_spec,
    load_iriis_gpt2,
)
from attention_maps.inference.gemini_translation import GeminiTranslationBackend
from attention_maps.inference.gemma4_base import load_gemma4_base
from attention_maps.inference.local_comparison import (
    LocalAdapterSpec,
    load_local_model_pair,
)
from attention_maps.tokenization.analysis import TokenizerSpec, load_tokenizer
from apps.explorer_tabs import (
    render_comparison_tab,
    render_details_tab,
    render_eda_tab,
    render_evaluation_tab,
    render_local_inference_tab,
    render_manifest_tab,
    render_nlue_tab,
    render_notes_tab,
    render_sample_tab,
    render_tokenizer_tab,
    render_wordcloud_tab,
)


def run_app() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Streamlit is not installed. Run `pip install -r requirements.txt`."
        ) from error

    st.set_page_config(
        page_title="Dataset Explorer",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Pretraining & Finetuning Dataset Explorer")
    st.caption(
        "Inspect full Nepali records, schemas, manifests, and uniform random "
        "samples, then run bounded survey EDA with detailed metrics and plots. "
        "Future English and finetuning Parquet data use the same workflow."
    )
    nepali_stopwords = configured_nepali_stopwords()

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

    @st.cache_data(show_spinner=False)
    def cached_flores_examples(
        split: str,
        offset: int,
        limit: int,
        credential_fingerprint: str,
        _token: str,
    ) -> list[Any]:
        del credential_fingerprint
        return load_flores_examples(
            split=split,
            offset=offset,
            limit=limit,
            token=_token or None,
        )

    @st.cache_data(show_spinner=False)
    def cached_nlue_examples(
        task_key: str,
        offset: int,
        limit: int,
        credential_fingerprint: str,
        _token: str,
    ) -> list[Any]:
        del credential_fingerprint
        return load_nlue_examples(
            task_key,
            offset=offset,
            limit=limit,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_google_backend(
        model_id: str,
        credential_fingerprint: str,
        _api_key: str,
    ) -> GoogleGenAIBackend:
        del credential_fingerprint
        return GoogleGenAIBackend(model_id, _api_key)

    @st.cache_resource(show_spinner=False)
    def cached_lima_teacher(
        model_id: str,
        credential_fingerprint: str,
        _api_key: str,
    ) -> GeminiTranslationBackend:
        del credential_fingerprint
        return GeminiTranslationBackend(_api_key, model_id)

    @st.cache_resource(show_spinner=False)
    def cached_huggingface_backend(
        model_id: str,
        provider: str,
        credential_fingerprint: str,
        _token: str,
    ) -> HuggingFaceBackend:
        del credential_fingerprint
        return HuggingFaceBackend(model_id, _token, provider)

    @st.cache_resource(show_spinner=False)
    def cached_local_model_pair(
        adapter_key: str,
        adapter_label: str,
        adapter_path: str,
        base_model_id: str,
        device: str,
        dtype: str,
        quantization: str,
        load_adapter: bool,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        adapter_spec = LocalAdapterSpec(
            key=adapter_key,
            label=adapter_label,
            path=Path(adapter_path),
            base_model_id=base_model_id,
        )
        return load_local_model_pair(
            adapter_spec,
            device=device,
            dtype=dtype,
            quantization=quantization,
            load_adapter=load_adapter,
            local_files_only=local_files_only,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_arkios(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        return load_arkios(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_himalayagpt(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        return load_himalayagpt(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_iriis_gpt2(
        model_key: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        return load_iriis_gpt2(
            iriis_gpt2_spec(model_key),
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_gemma4_base(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        return load_gemma4_base(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_analysis_tokenizer(
        key: str,
        label: str,
        source: str,
        revision: str,
        trust_remote_code: bool,
        local: bool,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        tokenizer_spec = TokenizerSpec(
            key=key,
            label=label,
            source=source,
            revision=revision,
            trust_remote_code=trust_remote_code,
            local=local,
        )
        return load_tokenizer(
            tokenizer_spec,
            token=_token or None,
            local_files_only=local_files_only,
        )

    st.sidebar.header("Dataset")
    data_root_text = st.sidebar.text_input("Data root", str(DEFAULT_DATA_ROOT))
    data_root = Path(data_root_text).expanduser()
    use_custom = st.sidebar.checkbox("Use custom Parquet path")

    spec: DatasetSpec | None = None
    if use_custom:
        custom_path_text = st.sidebar.text_input("Parquet file or directory")
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
            stage = st.sidebar.selectbox(
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
            selected_provider = st.sidebar.selectbox(
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
            selected_purpose = st.sidebar.selectbox(
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
    st.caption(
        "Tags: " + ", ".join(spec.tags)
        if spec.tags
        else "Tags: pending taxonomy review"
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("Rows", f"{inventory['rows']:,}")
    metric_columns[1].metric("Files", f"{inventory['files']:,}")
    if inventory["format"] == "json":
        grouping_label, grouping_value = "JSON documents", inventory["files"]
    elif inventory["format"] == "huggingface":
        grouping_label, grouping_value = "Remote shards", inventory["files"]
    elif inventory["format"] == "kaggle":
        grouping_label, grouping_value = "Worksheets", 1
    elif inventory["format"] == "kaggle_text":
        grouping_label, grouping_value = "Text files", inventory["files"]
    else:
        grouping_label, grouping_value = "Row groups", len(inventory["row_groups"])
    metric_columns[2].metric(grouping_label, f"{grouping_value:,}")
    metric_columns[3].metric("Columns", f"{len(inventory['columns']):,}")
    metric_columns[4].metric(
        "Estimated size" if inventory.get("bytes_estimated") else "Disk size",
        format_bytes(inventory["bytes"]),
    )

    if inventory["schema_variants"] > 1:
        st.warning(
            f"The files contain {inventory['schema_variants']} schema variants. "
            "Sampling is limited to columns shared by every shard."
        )

    (
        details_tab,
        sample_tab,
        eda_tab,
        notes_tab,
        wordcloud_tab,
        tokenizer_tab,
        local_inference_tab,
        inference_tab,
        evaluation_tab,
        nlue_tab,
        manifest_tab,
    ) = st.tabs(
        [
            "Schema",
            "Random records",
            "Survey EDA",
            "EDA & cleaning notes",
            "Word cloud",
            "Tokenizer analysis",
            "Local base vs finetuned",
            "Model comparison",
            "Evaluation",
            "Decoder benchmarks",
            "Manifest",
        ]
    )

    with details_tab:
        render_details_tab(st=st, inventory=inventory, spec=spec)

    with sample_tab:
        base_seed = render_sample_tab(
            st=st, inventory=inventory, spec=spec, cached_sample=cached_sample
        )

    with eda_tab:
        render_eda_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            nepali_stopwords=nepali_stopwords,
        )

    with notes_tab:
        render_notes_tab(st=st)

    with wordcloud_tab:
        render_wordcloud_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            nepali_stopwords=nepali_stopwords,
            base_seed=base_seed,
        )

    with tokenizer_tab:
        render_tokenizer_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            cached_analysis_tokenizer=cached_analysis_tokenizer,
        )

    with local_inference_tab:
        render_local_inference_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            cached_local_model_pair=cached_local_model_pair,
        )

    with inference_tab:
        render_comparison_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            cached_arkios=cached_arkios,
            cached_gemma4_base=cached_gemma4_base,
            cached_google_backend=cached_google_backend,
            cached_himalayagpt=cached_himalayagpt,
            cached_huggingface_backend=cached_huggingface_backend,
            cached_iriis_gpt2=cached_iriis_gpt2,
            cached_local_model_pair=cached_local_model_pair,
        )

    with evaluation_tab:
        render_evaluation_tab(
            st=st,
            cached_arkios=cached_arkios,
            cached_flores_examples=cached_flores_examples,
            cached_gemma4_base=cached_gemma4_base,
            cached_google_backend=cached_google_backend,
            cached_himalayagpt=cached_himalayagpt,
            cached_iriis_gpt2=cached_iriis_gpt2,
            cached_lima_teacher=cached_lima_teacher,
            cached_local_model_pair=cached_local_model_pair,
        )

    with nlue_tab:
        render_nlue_tab(
            st=st,
            cached_arkios=cached_arkios,
            cached_gemma4_base=cached_gemma4_base,
            cached_google_backend=cached_google_backend,
            cached_himalayagpt=cached_himalayagpt,
            cached_iriis_gpt2=cached_iriis_gpt2,
            cached_local_model_pair=cached_local_model_pair,
            cached_nlue_examples=cached_nlue_examples,
        )

    with manifest_tab:
        render_manifest_tab(st=st, spec=spec, data_root=data_root)


if __name__ == "__main__":
    run_app()
