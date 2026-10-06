"""Interactive Hugging Face configuration, split, and shard selection controls."""

from __future__ import annotations

from typing import Any

from attention_maps.explorer.huggingface import selected_source_spec
from attention_maps.explorer.source_imports import huggingface_source_spec
from attention_maps.explorer.text import format_decimal_bytes


def render_huggingface_source(
    st: Any, *, catalog_loader: Any, configuration_loader: Any,
) -> tuple[Any, dict | None]:
    st.sidebar.caption("Discover configurations and splits, then choose all or specific shards.")
    dataset_id = st.sidebar.text_input(
        "Hugging Face dataset ID", placeholder="owner/dataset", key="source-hf-id"
    ).strip()
    guidance = {
        "csebuetnlp/CrossSum": "Use 'nepali-nepali' or 'english-nepali'. The cached language-pair archive includes all splits; only your selected split is indexed.",
        "facebook/flores": "Evaluation benchmark: use 'npi_Deva' or 'eng_Latn-npi_Deva', split 'dev' or 'devtest'. Accept access conditions on Hugging Face and set HF_TOKEN for that account.",
        "google/IndicGenBench_flores_in": "Evaluation benchmark: use 'ne', split 'validation' or 'test'. Both translation directions are cached together. Keep benchmark examples out of pretraining.",
        "Nandan007/NepFakeV2": "Use 'default', split 'train'. Only the records CSV is cached; statistics and duplicate JSON exports are excluded.",
    }
    if dataset_id in guidance:
        st.sidebar.caption(guidance[dataset_id])
    revision = st.sidebar.text_input(
        "Revision (recommended)", placeholder="commit, tag, or branch", key="source-hf-revision"
    ).strip()
    signature = (dataset_id, revision)
    state_key = "hf-discovered-source"
    if st.sidebar.button("Load Hugging Face source", type="primary"):
        try:
            with st.spinner("Discovering dataset configurations…"):
                catalog = catalog_loader(dataset_id, revision or None)
            st.session_state[state_key] = {"signature": signature, "catalog": catalog}
        except (ValueError, OSError, ImportError) as error:
            st.sidebar.error(str(error))
            st.session_state.pop(state_key, None)
    saved = st.session_state.get(state_key, {})
    if saved.get("signature") != signature:
        return None, None
    spec = huggingface_source_spec(dataset_id, revision=revision)
    return render_huggingface_selection(
        st, spec, saved["catalog"], configuration_loader=configuration_loader
    )


def render_huggingface_selection(
    st: Any, spec: Any, catalog: dict, *, configuration_loader: Any,
) -> tuple[Any, dict | None]:
    prefix = f"hf-selection:{spec.key}:{catalog['revision']}"
    configs = catalog["configs"]
    if not configs:
        st.sidebar.error("No dataset configurations were found.")
        return None, None
    preferred = spec.dataset_config if spec.dataset_config in configs else configs[0]
    config = st.sidebar.selectbox(
        "Dataset configuration", configs, index=configs.index(preferred), key=f"{prefix}:config"
    )
    try:
        configuration = configuration_loader(catalog, config)
    except (ValueError, OSError, ImportError) as error:
        st.sidebar.error(str(error))
        return None, None
    splits = list(configuration["splits"])
    preferred_split = spec.dataset_split if spec.dataset_split in splits else splits[0]
    split = st.sidebar.selectbox(
        "Dataset split", splits, index=splits.index(preferred_split), key=f"{prefix}:{config}:split"
    )
    shard_key = f"{prefix}:{config}:{split}"
    available = configuration["splits"][split]["shards"]
    if configuration.get("requires_complete_split"):
        st.sidebar.caption("This source reads all files for the selected split together.")
        mode = "All shards"
    else:
        mode = st.sidebar.radio("Dataset shards", ("All shards", "Choose shards"), key=f"{shard_key}:mode")
    selected = None
    if mode == "Choose shards":
        by_path = {item["path"]: item for item in available}
        selected = st.sidebar.multiselect(
            "Shards to use", list(by_path), default=[], key=f"{shard_key}:files",
            format_func=lambda path: (
                f"{by_path[path]['name']} · "
                + (format_decimal_bytes(by_path[path]["bytes"]) if by_path[path]["bytes"] is not None else "size unknown")
            ),
            help="Only these files feed Source sample, WORKSPACE, and EDA. Select any shard or combine several.",
        )
        if not selected:
            st.sidebar.info("Select at least one shard to inspect.")
            return None, None
    st.sidebar.caption(
        f"{len(available) if selected is None else len(selected)} / {len(available)} shards · "
        f"revision {catalog['revision'][:12]}"
    )
    format_by_path = {
        item["path"]: item for item in configuration["splits"][split].get("file_formats", {}).get("files", [])
    }
    with st.sidebar.expander("Available shard sizes and formats"):
        st.dataframe(
            [{"Shard": item["name"], "Bytes": item["bytes"],
              "Extension": format_by_path.get(item["path"], {}).get("extension", "Unknown"),
              "Format": format_by_path.get(item["path"], {}).get("format", "Unknown")} for item in available],
            hide_index=True, width="stretch",
        )
    return selected_source_spec(spec, configuration, split, selected), configuration
