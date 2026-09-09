"""Local model comparison tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .common import (
    ComparisonConfigurationError,
    LOCAL_QUANTIZATION_CHOICES,
    LocalInferenceError,
    VIEWER_PREFIX,
    build_local_decoding_grid,
    build_prompt,
    configured_finetuned_models_root,
    defaultdict,
    discover_local_adapters,
    extract_text,
    hashlib,
    local_comparison_csv,
    os,
    parse_number_list,
    run_local_comparison,
    secret_fingerprint,
    split_human_assistant_example,
    text_columns,
)

def render_local_inference_tab(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    cached_sample: Any,
    cached_local_model_pair: Any,
) -> None:
    st.markdown(
        "Compare a local base model with its finetuned PEFT adapter on exactly "
        "the same prompt and decoding parameters. No inference API is used."
    )
    local_models_root = configured_finetuned_models_root()
    local_adapters = discover_local_adapters(local_models_root)
    local_text_columns = text_columns(inventory["schema"])
    if not local_adapters:
        st.warning(
            f"No complete PEFT adapters were found under " f"`{local_models_root}`."
        )
    elif not local_text_columns:
        st.warning(
            "This dataset has no detectable text field. Select another dataset "
            "or use custom prompt text below."
        )
    
    selected_adapter = (
        st.selectbox(
            "Local finetuned model",
            local_adapters,
            format_func=lambda item: item.label,
            disabled=not local_adapters,
            key="local-adapter",
        )
        if local_adapters
        else None
    )
    if selected_adapter is not None:
        st.caption(
            f"Base: `{selected_adapter.base_model_id}` · Adapter: "
            f"`{selected_adapter.path}`"
        )
    
    local_prompt_source = st.radio(
        "Evaluation input",
        ("Selected dataset instance", "Custom prompt"),
        horizontal=True,
        key=f"local-prompt-source:{spec.key}",
    )
    local_source_text = ""
    local_reference_text: str | None = None
    local_record_label = "custom"
    if local_prompt_source == "Selected dataset instance":
        if local_text_columns:
            local_source_controls = st.columns([2, 2, 1, 1])
            default_local_column = (
                local_text_columns.index("translation")
                if "translation" in local_text_columns
                else 0
            )
            local_source_column = local_source_controls[0].selectbox(
                "Prompt field",
                local_text_columns,
                index=default_local_column,
                key=f"local-source-column:{spec.key}",
            )
            reference_options = ["Auto / none"] + [
                column
                for column in local_text_columns
                if column != local_source_column
            ]
            local_reference_column = local_source_controls[1].selectbox(
                "Reference field (optional)",
                reference_options,
                key=f"local-reference-column:{spec.key}:{local_source_column}",
                help=(
                    "Auto extracts the ASSISTANT portion of a serialized "
                    "HUMAN/ASSISTANT example when available."
                ),
            )
            local_sample_seed = local_source_controls[2].number_input(
                "Instance seed",
                min_value=0,
                value=42,
                step=1,
                key=f"local-sample-seed:{spec.key}",
            )
            local_maximum_chars = local_source_controls[3].number_input(
                "Maximum characters",
                min_value=50,
                max_value=20_000,
                value=2_000,
                step=50,
                key=f"local-maximum-chars:{spec.key}",
            )
            local_sample_columns = [local_source_column]
            if local_reference_column != "Auto / none":
                local_sample_columns.append(local_reference_column)
            try:
                local_records = cached_sample(
                    inventory,
                    1,
                    int(local_sample_seed),
                    tuple(local_sample_columns),
                )
            except (OSError, ValueError, ImportError) as error:
                st.error(f"Could not read the evaluation instance: {error}")
                local_records = []
            if local_records:
                local_record = local_records[0]
                source_parts = extract_text(local_record.get(local_source_column))
                serialized_source = "\n\n".join(source_parts)
                local_source_text, embedded_reference = (
                    split_human_assistant_example(serialized_source)
                )
                if local_reference_column == "Auto / none":
                    local_reference_text = embedded_reference
                else:
                    reference_parts = extract_text(
                        local_record.get(local_reference_column)
                    )
                    local_reference_text = "\n\n".join(reference_parts) or None
                local_source_text = local_source_text[: int(local_maximum_chars)]
                if local_reference_text:
                    local_reference_text = local_reference_text[
                        : int(local_maximum_chars)
                    ]
                global_row = local_record.get(f"{VIEWER_PREFIX}row_index")
                local_record_label = (
                    f"{local_source_column}:{global_row}:{int(local_sample_seed)}"
                )
                st.caption(f"Selected dataset global row: {global_row}")
        else:
            st.info("Choose Custom prompt because this dataset has no text field.")
    else:
        local_source_text = "नेपालका हिमालहरूको महत्त्वबारे छोटकरीमा लेख्नुहोस्।"
    
    editable_local_source = st.text_area(
        "Model input text",
        value=local_source_text,
        height=150,
        key=(
            f"local-source-text:{spec.key}:{local_prompt_source}:"
            f"{local_record_label}"
        ),
    )
    if local_reference_text:
        with st.expander("Dataset reference answer", expanded=True):
            st.text(local_reference_text)
    
    local_prompt_columns = st.columns(2)
    local_prompt_template = local_prompt_columns[0].text_area(
        "Prompt template",
        value="{text}",
        height=100,
        key=f"local-prompt-template:{spec.key}",
        help="Use {text} where the selected dataset text should be inserted.",
    )
    local_system_prompt = local_prompt_columns[1].text_area(
        "System prompt (optional)",
        value="",
        height=100,
        key=f"local-system-prompt:{spec.key}",
    )
    try:
        local_prompt_preview = build_prompt(
            local_prompt_template, editable_local_source
        )
    except ComparisonConfigurationError as error:
        st.error(str(error))
        local_prompt_preview = ""
    if local_prompt_preview:
        with st.expander("Final local prompt preview"):
            st.text(local_prompt_preview)
    
    st.markdown("#### Local decoding grid")
    local_decoding_columns = st.columns(6)
    local_temperatures_text = local_decoding_columns[0].text_input(
        "Temperatures",
        "0, 0.7",
        key="local-temperatures",
        help="Temperature 0 uses greedy decoding.",
    )
    local_top_ps_text = local_decoding_columns[1].text_input(
        "Top-p values", "0.95", key="local-top-ps"
    )
    local_top_ks_text = local_decoding_columns[2].text_input(
        "Top-k values", "40", key="local-top-ks"
    )
    local_max_tokens = local_decoding_columns[3].number_input(
        "Maximum new tokens",
        min_value=1,
        max_value=1_024,
        value=128,
        key="local-max-new-tokens",
    )
    local_repetition_penalty = local_decoding_columns[4].number_input(
        "Repetition penalty",
        min_value=0.1,
        max_value=5.0,
        value=1.05,
        step=0.05,
        key="local-repetition-penalty",
    )
    local_generation_seed = local_decoding_columns[5].number_input(
        "Generation seed",
        min_value=0,
        value=42,
        key="local-generation-seed",
    )
    
    local_runtime_columns = st.columns(4)
    local_device = local_runtime_columns[0].selectbox(
        "Device", ("auto", "cuda", "cpu"), key="local-device"
    )
    local_dtype = local_runtime_columns[1].selectbox(
        "Weight dtype",
        ("auto", "float32", "bfloat16", "float16"),
        key="local-dtype",
    )
    local_quantization = local_runtime_columns[2].selectbox(
        "Quantization",
        LOCAL_QUANTIZATION_CHOICES,
        key="local-quantization",
        help=(
            "Auto uses CUDA 4-bit for 7B+ PEFT base models and no "
            "quantization for smaller models. GPU overflow may be offloaded "
            "to CPU or data/cache/model_offload."
        ),
    )
    local_files_only = local_runtime_columns[3].checkbox(
        "Use cached base weights only",
        value=False,
        key="local-files-only",
        help=(
            "Adapters and tokenizers are local. Base weights may be downloaded "
            "from Hugging Face once when this is off."
        ),
    )
    
    local_configs = []
    local_grid_error = None
    try:
        local_temperatures = parse_number_list(
            local_temperatures_text,
            value_type=float,
            name="temperatures",
        )
        local_top_ps = parse_number_list(
            local_top_ps_text, value_type=float, name="top-p"
        )
        local_top_ks = parse_number_list(
            local_top_ks_text, value_type=int, name="top-k"
        )
        local_configs = build_local_decoding_grid(
            local_temperatures,
            local_top_ps,
            local_top_ks,
            max_new_tokens=int(local_max_tokens),
            repetition_penalty=float(local_repetition_penalty),
            seed=int(local_generation_seed),
        )
    except (ComparisonConfigurationError, LocalInferenceError) as error:
        local_grid_error = str(error)
        st.error(local_grid_error)
    
    local_generation_count = len(local_configs) * 2
    st.info(
        f"This run will create {local_generation_count} generation(s): base and "
        "finetuned output for each decoding configuration."
    )
    if local_generation_count > 12:
        st.error("Reduce the local decoding grid to at most 12 generations.")
    
    local_result_context = hashlib.sha256(
        (
            f"{selected_adapter.key if selected_adapter else ''}\0"
            f"{local_prompt_preview}\0{local_system_prompt}\0"
            f"{local_configs}\0{local_device}\0{local_dtype}\0"
            f"{local_quantization}\0"
            f"{local_files_only}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    local_results_key = f"local-inference-results:{spec.key}:{local_result_context}"
    local_run_disabled = (
        selected_adapter is None
        or not local_prompt_preview
        or not local_configs
        or local_generation_count > 12
        or bool(local_grid_error)
    )
    if st.button(
        "Run local base vs finetuned comparison",
        type="primary",
        disabled=local_run_disabled,
    ):
        try:
            with st.spinner(
                "Loading the base model and local adapter, then generating…"
            ):
                local_bundle = cached_local_model_pair(
                    selected_adapter.key,
                    selected_adapter.label,
                    str(selected_adapter.path),
                    selected_adapter.base_model_id,
                    local_device,
                    local_dtype,
                    local_quantization,
                    True,
                    local_files_only,
                    secret_fingerprint(os.getenv("HF_TOKEN", "")),
                    os.getenv("HF_TOKEN", ""),
                )
                st.session_state[local_results_key] = run_local_comparison(
                    local_bundle,
                    local_prompt_preview,
                    local_configs,
                    system_prompt=local_system_prompt,
                )
        except (LocalInferenceError, RuntimeError, OSError) as error:
            st.error(f"Could not run local comparison: {error}")
    
    local_results = st.session_state.get(local_results_key, [])
    if local_results:
        st.markdown("#### Local comparison results")
        local_summary_rows = [
            {
                "variant": result.variant,
                "decoding": result.decoding,
                "latency_seconds": round(result.latency_seconds, 3),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "status": "error" if result.error else "ok",
                "output_preview": (result.output or result.error or "")[:240],
            }
            for result in local_results
        ]
        st.dataframe(local_summary_rows, width="stretch", hide_index=True)
        results_by_decoding: dict[str, dict[str, Any]] = defaultdict(dict)
        for result in local_results:
            results_by_decoding[result.decoding][result.variant] = result
        for decoding_name, variant_results in results_by_decoding.items():
            st.markdown(f"##### `{decoding_name}`")
            base_output_column, finetuned_output_column = st.columns(2)
            for output_column, variant in (
                (base_output_column, "Base"),
                (finetuned_output_column, "Finetuned"),
            ):
                result = variant_results.get(variant)
                output_column.markdown(f"**{variant} model**")
                if result is None:
                    output_column.warning("No result")
                elif result.error:
                    output_column.error(result.error)
                else:
                    output_column.text(result.output or "—")
                    output_column.caption(
                        f"{result.output_tokens} tokens · "
                        f"{result.latency_seconds:.2f} seconds"
                    )
        st.download_button(
            "Download local comparison as CSV",
            local_comparison_csv(local_results),
            file_name="local_base_vs_finetuned.csv",
            mime="text/csv",
        )
