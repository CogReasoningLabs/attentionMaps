"""Multi-backend comparison tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .common import (
    ARKIOS_BACKEND_NAME,
    ArkiosBackend,
    ComparisonConfigurationError,
    DEFAULT_ARKIOS_MODEL_ID,
    DEFAULT_ARKIOS_REVISION,
    DEFAULT_GEMINI_FLASH_LITE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMMA4_BASE_MODEL_ID,
    DEFAULT_GEMMA4_BASE_REVISION,
    DEFAULT_GOOGLE_GEMMA_MODEL,
    DEFAULT_HIMALAYAGPT_MODEL_ID,
    DEFAULT_HIMALAYAGPT_REVISION,
    GEMMA4_BASE_BACKEND_NAME,
    GOOGLE_GEMMA_BACKEND_NAME,
    Gemma4BaseBackend,
    HIMALAYAGPT_BACKEND_NAME,
    HimalayaGPTBackend,
    IRIISGPT2Backend,
    IRIIS_GPT2_BACKEND_OPTIONS,
    LOCAL_QUANTIZATION_CHOICES,
    LocalInferenceError,
    LocalPeftBackend,
    build_decoding_grid,
    build_prompt,
    comparison_csv,
    configured_finetuned_models_root,
    discover_local_adapters,
    extract_text,
    hashlib,
    local_model_context_limit,
    os,
    parse_model_ids,
    parse_number_list,
    run_comparison,
    secret_fingerprint,
    sentiment_column,
    sentiment_label,
    sentiment_prediction,
    text_columns,
)

def render_comparison_tab(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    cached_sample: Any,
    cached_arkios: Any,
    cached_gemma4_base: Any,
    cached_google_backend: Any,
    cached_himalayagpt: Any,
    cached_huggingface_backend: Any,
    cached_iriis_gpt2: Any,
    cached_local_model_pair: Any,
) -> None:
    st.markdown(
        "Compare the same Nepali prompt across the local finetuned GPT-2 and "
        "TinyLlama adapters, Gemini, Google-hosted Gemma, and optional Hugging "
        "Face Inference Provider models. Hosted API calls may be billed."
    )
    inference_text_columns = text_columns(inventory["schema"])
    reference_sentiment_column = sentiment_column(inventory["schema"])
    task_options = (
        "Sentiment classification",
        "Summarization",
        "Custom instruction",
    )
    task_preset = st.selectbox(
        "Task preset",
        task_options,
        index=0 if reference_sentiment_column else 1,
        help=(
            "Sentiment classification asks for exactly one of: negative, "
            "neutral, or positive. Dataset labels are shown only as references."
        ),
    )
    prompt_source = st.radio(
        "Prompt source",
        ("Selected dataset sample", "Custom source text", "Prompt only"),
        horizontal=True,
    )
    
    source_text: str | None = ""
    reference_sentiment: str | None = None
    reference_sentiment_value: Any = None
    source_widget_context = "custom"
    if prompt_source == "Selected dataset sample":
        if not inference_text_columns:
            st.warning(
                "This dataset has no detectable natural-language column. "
                "Choose Custom source text, Prompt only, or a pre-tokenization dataset."
            )
        else:
            source_controls = st.columns([2, 1, 1])
            inference_column = source_controls[0].selectbox(
                "Dataset text column",
                inference_text_columns,
                key=f"inference-column:{spec.key}",
            )
            inference_sample_seed = source_controls[1].number_input(
                "Sample seed",
                min_value=0,
                value=42,
                step=1,
            )
            maximum_prompt_chars = source_controls[2].number_input(
                "Maximum source characters",
                min_value=50,
                max_value=20_000,
                value=1_000,
                step=50,
            )
            inference_sample_columns = [inference_column]
            if (
                reference_sentiment_column
                and reference_sentiment_column != inference_column
            ):
                inference_sample_columns.append(reference_sentiment_column)
            inference_records = cached_sample(
                inventory,
                1,
                int(inference_sample_seed),
                tuple(inference_sample_columns),
            )
            if inference_records:
                inference_record = inference_records[0]
                extracted = extract_text(inference_record.get(inference_column))
                source_text = "\n\n".join(extracted)[: int(maximum_prompt_chars)]
                if reference_sentiment_column:
                    reference_sentiment_value = inference_record.get(
                        reference_sentiment_column
                    )
                    reference_sentiment = sentiment_label(reference_sentiment_value)
                source_widget_context = (
                    f"{inference_column}:{int(inference_sample_seed)}:"
                    f"{int(maximum_prompt_chars)}"
                )
    elif prompt_source == "Custom source text":
        source_text = "नेपाल एउटा विविध भाषा र संस्कृतिले भरिएको देश हो।"
    else:
        source_text = None
    
    editable_source: str | None = source_text
    if prompt_source != "Prompt only":
        editable_source = st.text_area(
            "Source text",
            value=source_text or "",
            height=180,
            key=f"inference-source:{spec.key}:{prompt_source}:{source_widget_context}",
        )
    if reference_sentiment:
        st.info(
            f"Dataset reference sentiment: **{reference_sentiment}** "
            f"(raw label: `{reference_sentiment_value}`). This label is not "
            "included in the model prompt."
        )
    
    sentiment_system_prompt = (
        "तपाईं नेपाली पाठको भावना वर्गीकरण गर्ने सहायक हुनुहुन्छ। "
        "उत्तरमा negative, neutral, वा positive मध्ये ठीक एउटा अंग्रेजी "
        "लेबल मात्र दिनुहोस्; व्याख्या नदिनुहोस्।"
    )
    system_prompt = st.text_area(
        "System prompt (optional)",
        value=(
            sentiment_system_prompt
            if task_preset == "Sentiment classification"
            else ""
        ),
        height=90,
        key=f"system-prompt:{task_preset}",
        help=(
            "Sets model behavior independently from the user prompt. "
            "It is sent through the provider's system-instruction mechanism."
        ),
    )
    if prompt_source == "Prompt only":
        default_user_prompt = (
            "यो नेपाली वाक्यको भावना वर्गीकरण गर्नुहोस्: आजको खबरले मलाई खुसी बनायो।"
            if task_preset == "Sentiment classification"
            else "नेपाली भाषामा नेपालको सांस्कृतिक विविधताबारे छोटो अनुच्छेद लेख्नुहोस्।"
        )
    elif task_preset == "Sentiment classification":
        default_user_prompt = "निम्न नेपाली पाठको भावना वर्गीकरण गर्नुहोस्:\n\n{text}"
    elif task_preset == "Summarization":
        default_user_prompt = "निम्न पाठलाई नेपालीमा संक्षेप गर्नुहोस्:\n\n{text}"
    else:
        default_user_prompt = "{text}"
    prompt_template = st.text_area(
        "User prompt" if prompt_source == "Prompt only" else "User prompt template",
        value=default_user_prompt,
        height=110,
        key=f"user-prompt:{task_preset}:{prompt_source}",
        help=(
            "Use {text} where the source document should be inserted."
            if prompt_source != "Prompt only"
            else "This prompt is sent directly without a source document."
        ),
    )
    try:
        prompt_preview = build_prompt(prompt_template, editable_source)
    except ComparisonConfigurationError as error:
        st.error(str(error))
        prompt_preview = ""
    if prompt_preview:
        with st.expander("Final prompt preview"):
            st.text(prompt_preview)
    
    comparison_local_adapters = discover_local_adapters(
        configured_finetuned_models_root()
    )
    local_backend_specs = {
        f"Local · {adapter.label}": adapter for adapter in comparison_local_adapters
    }
    backend_names = st.multiselect(
        "Inference backends",
        (
            "Gemini API",
            "Gemini Flash Lite API",
            GOOGLE_GEMMA_BACKEND_NAME,
            "Hugging Face Inference API",
            *IRIIS_GPT2_BACKEND_OPTIONS,
            GEMMA4_BASE_BACKEND_NAME,
            HIMALAYAGPT_BACKEND_NAME,
            ARKIOS_BACKEND_NAME,
            *local_backend_specs,
        ),
        default=(),
        help=(
            "Select hosted APIs and/or local finetuned adapters. Only selected "
            "models run."
        ),
    )
    if not backend_names:
        st.caption("Select one or more inference backends before running.")
    gemini_model = DEFAULT_GEMINI_MODEL
    gemini_flash_lite_model = DEFAULT_GEMINI_FLASH_LITE_MODEL
    google_gemma_model = DEFAULT_GOOGLE_GEMMA_MODEL
    hf_models_text = ""
    hf_provider = "auto"
    selected_comparison_local_specs = [
        local_backend_specs[name]
        for name in backend_names
        if name in local_backend_specs
    ]
    selected_iriis_gpt2_specs = [
        IRIIS_GPT2_BACKEND_OPTIONS[name]
        for name in backend_names
        if name in IRIIS_GPT2_BACKEND_OPTIONS
    ]
    comparison_local_device = "auto"
    comparison_local_dtype = "auto"
    comparison_local_quantization = "auto"
    comparison_local_files_only = False
    selected_arkios = ARKIOS_BACKEND_NAME in backend_names
    selected_himalayagpt = HIMALAYAGPT_BACKEND_NAME in backend_names
    selected_gemma4_base = GEMMA4_BASE_BACKEND_NAME in backend_names
    selected_full_local_models = (
        int(selected_arkios)
        + int(selected_himalayagpt)
        + int(selected_gemma4_base)
        + len(selected_iriis_gpt2_specs)
    )
    if backend_names:
        backend_columns = st.columns(len(backend_names))
        for backend_column, backend_name in zip(backend_columns, backend_names):
            with backend_column:
                if backend_name == "Gemini API":
                    gemini_model = st.text_input(
                        "Gemini model", DEFAULT_GEMINI_MODEL
                    )
                    gemini_env_available = bool(os.getenv("GEMINI_API_KEY"))
                    st.caption(
                        "GEMINI_API_KEY detected in environment"
                        if gemini_env_available
                        else "GEMINI_API_KEY is not available to this process"
                    )
                elif backend_name == "Gemini Flash Lite API":
                    gemini_flash_lite_model = st.text_input(
                        "Gemini Flash Lite model",
                        DEFAULT_GEMINI_FLASH_LITE_MODEL,
                    )
                    st.caption("Uses GEMINI_API_KEY through the Interactions API")
                elif backend_name == GOOGLE_GEMMA_BACKEND_NAME:
                    google_gemma_model = st.text_input(
                        "Google Gemma model", DEFAULT_GOOGLE_GEMMA_MODEL
                    )
                    st.caption(
                        "Uses GEMINI_API_KEY through Gemma's generate_content endpoint"
                    )
                elif backend_name == "Hugging Face Inference API":
                    hf_models_text = st.text_area(
                        "Hugging Face model IDs",
                        "",
                        height=70,
                        placeholder="account/model-name",
                        help=(
                            "Enter one Hub model ID per line. Each model must "
                            "have coverage from the selected Inference Provider."
                        ),
                    )
                    hf_provider = st.text_input("Inference Provider", "auto")
                    hf_env_available = bool(os.getenv("HF_TOKEN"))
                    st.caption(
                        "HF_TOKEN detected in environment"
                        if hf_env_available
                        else "HF_TOKEN is not available to this process"
                    )
                elif backend_name in local_backend_specs:
                    local_spec = local_backend_specs[backend_name]
                    st.caption(
                        f"Base: `{local_spec.base_model_id}`  \n"
                        f"Adapter: `{local_spec.path.name}`"
                    )
                elif backend_name in IRIIS_GPT2_BACKEND_OPTIONS:
                    iriis_spec = IRIIS_GPT2_BACKEND_OPTIONS[backend_name]
                    variant = (
                        "Instruction tuned"
                        if iriis_spec.instruction_tuned
                        else "Base"
                    )
                    st.caption(
                        f"Model: `{iriis_spec.model_id}`  \n"
                        f"{variant} · 124M parameters · 512-token context"
                    )
                elif backend_name == ARKIOS_BACKEND_NAME:
                    st.caption(
                        f"Model: `{DEFAULT_ARKIOS_MODEL_ID}`  \n"
                        "Published ChatML template · 4,096-token context"
                    )
                elif backend_name == HIMALAYAGPT_BACKEND_NAME:
                    st.caption(
                        f"Model: `{DEFAULT_HIMALAYAGPT_MODEL_ID}`  \n"
                        "Pinned custom code · 2,048-token context"
                    )
                elif backend_name == GEMMA4_BASE_BACKEND_NAME:
                    st.caption(
                        f"Model: `{DEFAULT_GEMMA4_BASE_MODEL_ID}`  \n"
                        "Pre-trained base (not instruction tuned) · text-only UI · "
                        "about 10.2 GB of BF16 weights"
                    )
        if selected_comparison_local_specs or selected_full_local_models:
            st.markdown("##### Local model runtime")
            comparison_runtime_columns = st.columns(4)
            comparison_local_device = comparison_runtime_columns[0].selectbox(
                "Local device",
                ("auto", "cuda", "cpu"),
                key="comparison-local-device",
            )
            comparison_dtype_options = (
                ("auto", "float32", "bfloat16")
                if selected_himalayagpt
                else ("auto", "float32", "bfloat16", "float16")
            )
            comparison_local_dtype = comparison_runtime_columns[1].selectbox(
                "Local weight dtype",
                comparison_dtype_options,
                key="comparison-local-dtype",
            )
            comparison_local_quantization = comparison_runtime_columns[2].selectbox(
                "PEFT quantization",
                LOCAL_QUANTIZATION_CHOICES,
                key="comparison-local-quantization",
                help=(
                    "Auto uses CUDA 4-bit for PEFT base models of 7B or "
                    "larger and permits CPU/disk offload when VRAM is full."
                ),
            )
            comparison_local_files_only = comparison_runtime_columns[3].checkbox(
                "Cached base weights only",
                value=False,
                key="comparison-local-files-only",
            )
    
    st.markdown("#### Decoding grid")
    decoding_profile = "himalayagpt" if selected_himalayagpt else "general"
    default_temperatures = "0.8" if selected_himalayagpt else "0.2, 0.7"
    default_top_ps = "1.0" if selected_himalayagpt else "0.95"
    default_top_ks = "50" if selected_himalayagpt else "40"
    default_max_tokens = 96 if selected_himalayagpt else 256
    st.caption(
        "Every temperature × top-p × top-k combination runs against every "
        "selected model. Hugging Face top-k support is provider-dependent."
    )
    if selected_himalayagpt:
        st.caption(
            "HimalayaGPT reference profile: temperature 0.8, top-k 50, "
            "maximum 96 tokens, repetition penalty 1.08, and special-token "
            "stopping. Its reference generation loop does not apply top-p."
        )
    decoding_columns = st.columns(6)
    temperatures_text = decoding_columns[0].text_input(
        "Temperatures",
        default_temperatures,
        key=f"comparison-temperatures:{decoding_profile}",
    )
    top_ps_text = decoding_columns[1].text_input(
        "Top-p values",
        default_top_ps,
        key=f"comparison-top-ps:{decoding_profile}",
    )
    top_ks_text = decoding_columns[2].text_input(
        "Top-k values",
        default_top_ks,
        key=f"comparison-top-ks:{decoding_profile}",
    )
    inference_max_tokens = decoding_columns[3].number_input(
        "Maximum new tokens",
        min_value=1,
        max_value=4_096,
        value=default_max_tokens,
        key=f"comparison-max-new-tokens:{decoding_profile}",
    )
    inference_seed = decoding_columns[4].number_input(
        "Generation seed", min_value=0, value=42
    )
    inference_thinking_level = decoding_columns[5].selectbox(
        "Thinking level",
        ("minimal", "low", "medium", "high"),
        index=0,
        help=(
            "Google thinking tokens share the maximum-output budget. Use "
            "minimal for short generation limits and comparison tasks."
        ),
    )
    
    configs = []
    grid_error = None
    try:
        temperatures = parse_number_list(
            temperatures_text, value_type=float, name="temperatures"
        )
        top_ps = parse_number_list(top_ps_text, value_type=float, name="top-p")
        top_ks = parse_number_list(top_ks_text, value_type=int, name="top-k")
        configs = build_decoding_grid(
            temperatures,
            top_ps,
            top_ks,
            max_new_tokens=int(inference_max_tokens),
            seed=int(inference_seed),
            thinking_level=inference_thinking_level,
        )
    except ComparisonConfigurationError as error:
        grid_error = str(error)
        st.error(grid_error)
    
    hf_model_ids = parse_model_ids(hf_models_text)
    hosted_model_count = (
        int("Gemini API" in backend_names)
        + int("Gemini Flash Lite API" in backend_names)
        + int(GOOGLE_GEMMA_BACKEND_NAME in backend_names)
        + (
            len(hf_model_ids)
            if "Hugging Face Inference API" in backend_names
            else 0
        )
    )
    local_model_count = (
        len(selected_comparison_local_specs) + selected_full_local_models
    )
    model_count = hosted_model_count + local_model_count
    request_count = model_count * len(configs)
    remote_request_count = hosted_model_count * len(configs)
    local_comparison_count = local_model_count * len(configs)
    st.info(
        f"This comparison will run {request_count} generation(s): "
        f"{remote_request_count} hosted request(s) and "
        f"{local_comparison_count} local generation(s)."
    )
    if request_count > 60:
        st.error("Reduce the grid to at most 60 generations per comparison.")
    
    local_context_error = ""
    for local_spec in selected_comparison_local_specs:
        context_limit = local_model_context_limit(local_spec.base_model_id)
        if int(inference_max_tokens) >= context_limit:
            local_context_error = (
                f"Maximum new tokens must be below {context_limit:,} for "
                f"{local_spec.label}."
            )
            st.error(local_context_error)
            break
    if (
        not local_context_error
        and selected_iriis_gpt2_specs
        and int(inference_max_tokens) >= 512
    ):
        local_context_error = (
            "Maximum new tokens must be below 512 for IRIIS Nepali GPT-2."
        )
        st.error(local_context_error)
    if (
        not local_context_error
        and selected_himalayagpt
        and int(inference_max_tokens) >= 2_048
    ):
        local_context_error = (
            "Maximum new tokens must be below 2,048 for HimalayaGPT 0.5B."
        )
        st.error(local_context_error)
    if (
        not local_context_error
        and selected_arkios
        and int(inference_max_tokens) >= 4_096
    ):
        local_context_error = (
            "Maximum new tokens must be below 4,096 for Arkios 1B Chat."
        )
        st.error(local_context_error)
    
    result_context = hashlib.sha256(
        (
            f"{task_preset}\0{prompt_preview}\0{system_prompt}\0"
            f"{backend_names}\0{gemini_model}\0{gemini_flash_lite_model}\0"
            f"{google_gemma_model}\0{hf_model_ids}\0{hf_provider}\0"
            f"{configs}\0{comparison_local_device}\0"
            f"{comparison_local_dtype}\0{comparison_local_quantization}\0"
            f"{comparison_local_files_only}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    results_key = f"inference-results:{spec.key}:{result_context}"
    run_disabled = (
        request_count <= 0
        or request_count > 60
        or bool(grid_error)
        or bool(local_context_error)
        or not prompt_preview
    )
    if st.button(
        "Run inference comparison",
        type="primary",
        disabled=run_disabled,
    ):
        effective_gemini_key = os.getenv("GEMINI_API_KEY", "")
        effective_hf_token = os.getenv("HF_TOKEN", "")
        try:
            if grid_error:
                raise ComparisonConfigurationError(grid_error)
            if not prompt_preview:
                raise ComparisonConfigurationError("provide a valid source prompt")
            if request_count <= 0:
                raise ComparisonConfigurationError("select at least one model")
            if request_count > 60:
                raise ComparisonConfigurationError(
                    "comparison exceeds the 60-generation safety limit"
                )
            if local_context_error:
                raise ComparisonConfigurationError(local_context_error)
    
            backends = []
            if {
                "Gemini API",
                "Gemini Flash Lite API",
                GOOGLE_GEMMA_BACKEND_NAME,
            }.intersection(backend_names):
                if not effective_gemini_key:
                    raise ComparisonConfigurationError(
                        "GEMINI_API_KEY is missing for Google inference"
                    )
            if "Gemini API" in backend_names:
                backends.append(
                    cached_google_backend(
                        gemini_model,
                        secret_fingerprint(effective_gemini_key),
                        effective_gemini_key,
                    )
                )
            if "Gemini Flash Lite API" in backend_names:
                backends.append(
                    cached_google_backend(
                        gemini_flash_lite_model,
                        secret_fingerprint(effective_gemini_key),
                        effective_gemini_key,
                    )
                )
            if GOOGLE_GEMMA_BACKEND_NAME in backend_names:
                backends.append(
                    cached_google_backend(
                        google_gemma_model,
                        secret_fingerprint(effective_gemini_key),
                        effective_gemini_key,
                    )
                )
            if "Hugging Face Inference API" in backend_names:
                if not effective_hf_token:
                    raise ComparisonConfigurationError(
                        "Hugging Face token is missing"
                    )
                for model_id in hf_model_ids:
                    backends.append(
                        cached_huggingface_backend(
                            model_id,
                            hf_provider,
                            secret_fingerprint(effective_hf_token),
                            effective_hf_token,
                        )
                    )
            for local_spec in selected_comparison_local_specs:
                local_bundle = cached_local_model_pair(
                    local_spec.key,
                    local_spec.label,
                    str(local_spec.path),
                    local_spec.base_model_id,
                    comparison_local_device,
                    comparison_local_dtype,
                    comparison_local_quantization,
                    True,
                    comparison_local_files_only,
                    secret_fingerprint(effective_hf_token),
                    effective_hf_token,
                )
                backends.append(LocalPeftBackend(local_bundle))
            for iriis_spec in selected_iriis_gpt2_specs:
                backends.append(
                    IRIISGPT2Backend(
                        cached_iriis_gpt2(
                            iriis_spec.key,
                            comparison_local_device,
                            comparison_local_dtype,
                            comparison_local_files_only,
                            secret_fingerprint(effective_hf_token),
                            effective_hf_token,
                        )
                    )
                )
            if selected_himalayagpt:
                backends.append(
                    HimalayaGPTBackend(
                        cached_himalayagpt(
                            DEFAULT_HIMALAYAGPT_MODEL_ID,
                            DEFAULT_HIMALAYAGPT_REVISION,
                            comparison_local_device,
                            comparison_local_dtype,
                            comparison_local_files_only,
                        )
                    )
                )
            if selected_arkios:
                backends.append(
                    ArkiosBackend(
                        cached_arkios(
                            DEFAULT_ARKIOS_MODEL_ID,
                            DEFAULT_ARKIOS_REVISION,
                            comparison_local_device,
                            comparison_local_dtype,
                            comparison_local_files_only,
                        )
                    )
                )
            if selected_gemma4_base:
                backends.append(
                    Gemma4BaseBackend(
                        cached_gemma4_base(
                            DEFAULT_GEMMA4_BASE_MODEL_ID,
                            DEFAULT_GEMMA4_BASE_REVISION,
                            comparison_local_device,
                            comparison_local_dtype,
                            comparison_local_files_only,
                            secret_fingerprint(effective_hf_token),
                            effective_hf_token,
                        )
                    )
                )
    
            with st.spinner(f"Running {request_count} model generations…"):
                st.session_state[results_key] = run_comparison(
                    backends,
                    [editable_source],
                    configs,
                    prompt_template=prompt_template,
                    system_prompt=system_prompt,
                )
        except (
            ComparisonConfigurationError,
            LocalInferenceError,
            ImportError,
            RuntimeError,
            OSError,
        ) as error:
            st.error(f"Could not start comparison: {error}")
    
    comparison_results = st.session_state.get(results_key, [])
    if comparison_results:
        st.markdown("#### Comparison results")
        summary_rows = [
            {
                "model": result.model,
                "decoding": result.decoding,
                "latency_seconds": result.latency_seconds,
                "status": "error" if result.error else "ok",
                "prediction": (
                    sentiment_prediction(result.output)
                    if task_preset == "Sentiment classification"
                    else None
                ),
                "reference": reference_sentiment,
                "matches_reference": (
                    sentiment_prediction(result.output) == reference_sentiment
                    if reference_sentiment and not result.error
                    else None
                ),
                "output_preview": (result.output or result.error or "")[:300],
            }
            for result in comparison_results
        ]
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        selected_result = st.selectbox(
            "Inspect complete output",
            range(len(comparison_results)),
            format_func=lambda index: (
                f"{comparison_results[index].model} · "
                f"{comparison_results[index].decoding}"
            ),
        )
        result = comparison_results[selected_result]
        if result.error:
            st.error(result.error)
        else:
            st.text(result.output)
        st.download_button(
            "Download results as CSV",
            comparison_csv(comparison_results),
            file_name="nepali_inference_comparison.csv",
            mime="text/csv",
        )
