"""FLORES and decoder benchmark tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .common import (
    ARKIOS_BACKEND_NAME,
    ArkiosBackend,
    ComparisonConfigurationError,
    DECODER_EVALUATION_TASKS,
    DEFAULT_ARKIOS_MODEL_ID,
    DEFAULT_ARKIOS_REVISION,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMMA4_BASE_MODEL_ID,
    DEFAULT_GEMMA4_BASE_REVISION,
    DEFAULT_GOOGLE_GEMMA_MODEL,
    DEFAULT_HIMALAYAGPT_MODEL_ID,
    DEFAULT_HIMALAYAGPT_REVISION,
    DecodingConfig,
    FLORES_DATASET_ID,
    FLORES_SPLIT_SIZES,
    FLORES_TRANSLATION_PROMPT,
    FloresEvaluationError,
    GEMMA4_BASE_BACKEND_NAME,
    GOOGLE_GEMMA_BACKEND_NAME,
    Gemma4BaseBackend,
    HIMALAYAGPT_BACKEND_NAME,
    HimalayaGPTBackend,
    IRIISGPT2Backend,
    IRIIS_GPT2_BACKEND_OPTIONS,
    LIMA_TEACHER_MODEL,
    LIMA_TEACHER_TEMPERATURE,
    LIMA_TEACHER_TOP_K,
    LIMA_TEACHER_TOP_P,
    LOCAL_QUANTIZATION_CHOICES,
    LocalAdapterSpec,
    LocalInferenceError,
    LocalPeftBackend,
    NLUEEvaluationError,
    NLUE_COLLECTION_URL,
    NLUE_PAPER_URL,
    configured_finetuned_models_root,
    discover_local_adapters,
    hashlib,
    json,
    os,
    published_baseline_rows,
    run_comparison,
    score_flores_results,
    score_nlue_results,
    secret_fingerprint,
)

def render_evaluation_tab(
    *,
    st: Any,
    cached_arkios: Any,
    cached_flores_examples: Any,
    cached_gemma4_base: Any,
    cached_google_backend: Any,
    cached_himalayagpt: Any,
    cached_iriis_gpt2: Any,
    cached_lima_teacher: Any,
    cached_local_model_pair: Any,
) -> None:
    st.markdown("### FLORES-200 · English → Nepali")
    st.caption(
        f"Streams the public `{FLORES_DATASET_ID}` mirror in bounded slices; "
        "the full benchmark is never materialized in memory. Performance is "
        "reported with corpus chrF++ (word order 2), the primary FLORES metric."
    )
    flores_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
    st.caption(
        "HF_TOKEN detected for higher Hugging Face rate limits."
        if flores_hf_token
        else "No HF_TOKEN detected; this public mirror supports anonymous access."
    )
    flores_controls = st.columns(3)
    flores_split = flores_controls[0].selectbox(
        "FLORES split",
        ("devtest", "dev"),
        key="flores-split",
    )
    flores_offset = flores_controls[1].number_input(
        "Starting position",
        min_value=0,
        max_value=FLORES_SPLIT_SIZES[flores_split] - 1,
        value=0,
        step=1,
        key=f"flores-offset:{flores_split}",
    )
    flores_count = flores_controls[2].number_input(
        "Examples",
        min_value=1,
        max_value=min(20, FLORES_SPLIT_SIZES[flores_split] - int(flores_offset)),
        value=min(5, FLORES_SPLIT_SIZES[flores_split] - int(flores_offset)),
        step=1,
        key=f"flores-count:{flores_split}:{int(flores_offset)}",
    )
    flores_examples_key = (
        f"flores-examples:{flores_split}:{int(flores_offset)}:"
        f"{int(flores_count)}"
    )
    if st.button("Load FLORES instances", key="load-flores-instances"):
        try:
            with st.spinner("Streaming the requested FLORES slice…"):
                st.session_state[flores_examples_key] = cached_flores_examples(
                    flores_split,
                    int(flores_offset),
                    int(flores_count),
                    secret_fingerprint(flores_hf_token),
                    flores_hf_token,
                )
        except FloresEvaluationError as error:
            st.error(str(error))
    
    flores_examples = st.session_state.get(flores_examples_key, [])
    if not flores_examples:
        st.info("Load a small FLORES slice to inspect examples and run evaluation.")
    else:
        st.markdown("#### Benchmark instances")
        st.dataframe(
            [example.as_dict() for example in flores_examples],
            width="stretch",
            hide_index=True,
        )
    
        st.markdown("#### Models")
        teacher_option = f"Teacher · {LIMA_TEACHER_MODEL}"
        google_gemma_api_option = GOOGLE_GEMMA_BACKEND_NAME
        arkios_option = "Other · Arkios 1B Chat"
        himalaya_option = "Other · HimalayaGPT 0.5B Instruct"
        gemma4_base_option = GEMMA4_BASE_BACKEND_NAME
        evaluation_adapters = discover_local_adapters(
            configured_finetuned_models_root()
        )
        local_options: dict[str, tuple[LocalAdapterSpec, bool]] = {}
        for adapter in evaluation_adapters:
            local_options[f"Base · {adapter.label}"] = (adapter, False)
            local_options[f"Finetuned · {adapter.label}"] = (adapter, True)
        evaluation_model_options = [
            teacher_option,
            google_gemma_api_option,
            *local_options,
            *IRIIS_GPT2_BACKEND_OPTIONS,
            gemma4_base_option,
            himalaya_option,
            arkios_option,
        ]
        selected_evaluation_models = st.multiselect(
            "Models to benchmark",
            evaluation_model_options,
            default=[teacher_option],
            key="flores-models",
            help=(
                "Base and finetuned choices sharing an adapter are loaded as one "
                "model pair and evaluated with the adapter disabled/enabled."
            ),
        )
        if teacher_option in selected_evaluation_models:
            st.caption(
                f"LIMA teacher detected from the translation notebook: "
                f"`{LIMA_TEACHER_MODEL}`, temperature "
                f"{LIMA_TEACHER_TEMPERATURE}, top-p {LIMA_TEACHER_TOP_P}, "
                f"top-k {LIMA_TEACHER_TOP_K}. Uses `GEMINI_API_KEY`."
            )
        evaluation_google_gemma_model = DEFAULT_GOOGLE_GEMMA_MODEL
        if google_gemma_api_option in selected_evaluation_models:
            evaluation_google_gemma_model = st.text_input(
                "Evaluation Google Gemma model",
                DEFAULT_GOOGLE_GEMMA_MODEL,
                key="flores-google-gemma-model",
                help=(
                    "Uses GEMINI_API_KEY through Google's hosted Gemma "
                    "generate_content endpoint."
                ),
            )
        if gemma4_base_option in selected_evaluation_models:
            st.caption(
                f"`{DEFAULT_GEMMA4_BASE_MODEL_ID}` is the pre-trained base "
                "checkpoint. The first local run downloads about 10.2 GB of "
                "BF16 weights and may require an accepted Hub license/HF_TOKEN."
            )
        selected_evaluation_iriis_specs = [
            IRIIS_GPT2_BACKEND_OPTIONS[option]
            for option in selected_evaluation_models
            if option in IRIIS_GPT2_BACKEND_OPTIONS
        ]
    
        has_local_evaluation = (
            any(option in local_options for option in selected_evaluation_models)
            or any(
                option in selected_evaluation_models
                for option in (gemma4_base_option, himalaya_option, arkios_option)
            )
            or bool(selected_evaluation_iriis_specs)
        )
        evaluation_device = "auto"
        evaluation_dtype = "auto"
        evaluation_quantization = "auto"
        evaluation_local_only = False
        if has_local_evaluation:
            runtime_columns = st.columns(4)
            evaluation_device = runtime_columns[0].selectbox(
                "Evaluation device",
                ("auto", "cuda", "cpu"),
                key="flores-device",
            )
            dtype_choices = (
                ("auto", "float32", "bfloat16")
                if himalaya_option in selected_evaluation_models
                else ("auto", "float32", "bfloat16", "float16")
            )
            evaluation_dtype = runtime_columns[1].selectbox(
                "Evaluation dtype",
                dtype_choices,
                key="flores-dtype",
            )
            evaluation_quantization = runtime_columns[2].selectbox(
                "PEFT quantization",
                LOCAL_QUANTIZATION_CHOICES,
                key="flores-quantization",
                help=(
                    "Auto uses CUDA 4-bit for PEFT base models of 7B or "
                    "larger and permits CPU/disk offload when VRAM is full."
                ),
            )
            evaluation_local_only = runtime_columns[3].checkbox(
                "Cached model files only",
                key="flores-local-only",
            )
    
        st.markdown("#### Translation configuration")
        translation_prompt = st.text_area(
            "Translation prompt",
            FLORES_TRANSLATION_PROMPT,
            height=110,
            key="flores-prompt",
            help="Keep the `{text}` placeholder for each English source sentence.",
        )
        generation_columns = st.columns(2)
        evaluation_max_tokens = generation_columns[0].number_input(
            "Maximum new tokens",
            min_value=16,
            max_value=1_024,
            value=256,
            step=16,
            key="flores-max-tokens",
        )
        evaluation_seed = generation_columns[1].number_input(
            "Evaluation seed",
            min_value=0,
            value=42,
            key="flores-seed",
        )
        evaluation_config = DecodingConfig(
            temperature=LIMA_TEACHER_TEMPERATURE,
            top_p=LIMA_TEACHER_TOP_P,
            top_k=LIMA_TEACHER_TOP_K,
            max_new_tokens=int(evaluation_max_tokens),
            seed=int(evaluation_seed),
            thinking_level="minimal",
        )
        evaluation_requests = len(flores_examples) * len(selected_evaluation_models)
        st.info(
            f"This run will perform {evaluation_requests} translation(s) over "
            f"{len(flores_examples)} FLORES instance(s)."
        )
        if evaluation_requests > 60:
            st.error(
                "Select fewer examples or models; at most 60 generations are allowed."
            )
        evaluation_context_error = ""
        if selected_evaluation_iriis_specs and int(evaluation_max_tokens) >= 512:
            evaluation_context_error = (
                "Maximum new tokens must be below 512 for IRIIS Nepali GPT-2."
            )
            st.error(evaluation_context_error)
    
        evaluation_context = hashlib.sha256(
            (
                f"{flores_split}\0{flores_offset}\0{flores_count}\0"
                f"{selected_evaluation_models}\0{translation_prompt}\0"
                f"{evaluation_config}\0{evaluation_device}\0{evaluation_dtype}\0"
                f"{evaluation_quantization}\0{evaluation_local_only}\0"
                f"{evaluation_google_gemma_model}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        evaluation_results_key = f"flores-results:{evaluation_context}"
        if st.button(
            "Run FLORES evaluation",
            type="primary",
            disabled=(
                not selected_evaluation_models
                or evaluation_requests > 60
                or "{text}" not in translation_prompt
                or bool(evaluation_context_error)
            ),
            key="run-flores-evaluation",
        ):
            try:
                evaluation_backends = []
                if teacher_option in selected_evaluation_models:
                    gemini_key = os.getenv("GEMINI_API_KEY", "")
                    if not gemini_key:
                        raise ComparisonConfigurationError(
                            "GEMINI_API_KEY is required to evaluate the LIMA teacher"
                        )
                    evaluation_backends.append(
                        cached_lima_teacher(
                            LIMA_TEACHER_MODEL,
                            secret_fingerprint(gemini_key),
                            gemini_key,
                        )
                    )
                if google_gemma_api_option in selected_evaluation_models:
                    gemini_key = os.getenv("GEMINI_API_KEY", "")
                    if not gemini_key:
                        raise ComparisonConfigurationError(
                            "GEMINI_API_KEY is required to evaluate Google Gemma"
                        )
                    evaluation_backends.append(
                        cached_google_backend(
                            evaluation_google_gemma_model,
                            secret_fingerprint(gemini_key),
                            gemini_key,
                        )
                    )
    
                loaded_pairs: dict[str, Any] = {}
                evaluation_adapter_required: dict[str, bool] = {}
                for selected_option in selected_evaluation_models:
                    if selected_option not in local_options:
                        continue
                    selected_adapter, use_adapter = local_options[selected_option]
                    evaluation_adapter_required[selected_adapter.key] = (
                        evaluation_adapter_required.get(selected_adapter.key, False)
                        or use_adapter
                    )
                for option in selected_evaluation_models:
                    if option not in local_options:
                        continue
                    adapter, use_adapter = local_options[option]
                    if adapter.key not in loaded_pairs:
                        loaded_pairs[adapter.key] = cached_local_model_pair(
                            adapter.key,
                            adapter.label,
                            str(adapter.path),
                            adapter.base_model_id,
                            evaluation_device,
                            evaluation_dtype,
                            evaluation_quantization,
                            evaluation_adapter_required[adapter.key],
                            evaluation_local_only,
                            secret_fingerprint(flores_hf_token),
                            flores_hf_token,
                        )
                    evaluation_backends.append(
                        LocalPeftBackend(
                            loaded_pairs[adapter.key],
                            use_adapter=use_adapter,
                        )
                    )
                for iriis_spec in selected_evaluation_iriis_specs:
                    evaluation_backends.append(
                        IRIISGPT2Backend(
                            cached_iriis_gpt2(
                                iriis_spec.key,
                                evaluation_device,
                                evaluation_dtype,
                                evaluation_local_only,
                                secret_fingerprint(flores_hf_token),
                                flores_hf_token,
                            )
                        )
                    )
                if himalaya_option in selected_evaluation_models:
                    evaluation_backends.append(
                        HimalayaGPTBackend(
                            cached_himalayagpt(
                                DEFAULT_HIMALAYAGPT_MODEL_ID,
                                DEFAULT_HIMALAYAGPT_REVISION,
                                evaluation_device,
                                evaluation_dtype,
                                evaluation_local_only,
                            )
                        )
                    )
                if arkios_option in selected_evaluation_models:
                    evaluation_backends.append(
                        ArkiosBackend(
                            cached_arkios(
                                DEFAULT_ARKIOS_MODEL_ID,
                                DEFAULT_ARKIOS_REVISION,
                                evaluation_device,
                                evaluation_dtype,
                                evaluation_local_only,
                            )
                        )
                    )
                if gemma4_base_option in selected_evaluation_models:
                    evaluation_backends.append(
                        Gemma4BaseBackend(
                            cached_gemma4_base(
                                DEFAULT_GEMMA4_BASE_MODEL_ID,
                                DEFAULT_GEMMA4_BASE_REVISION,
                                evaluation_device,
                                evaluation_dtype,
                                evaluation_local_only,
                                secret_fingerprint(flores_hf_token),
                                flores_hf_token,
                            )
                        )
                    )
    
                with st.spinner(
                    f"Running and scoring {evaluation_requests} translations…"
                ):
                    raw_results = run_comparison(
                        evaluation_backends,
                        [example.source for example in flores_examples],
                        [evaluation_config],
                        prompt_template=translation_prompt,
                    )
                    detail_rows, summary_rows = score_flores_results(
                        raw_results, flores_examples
                    )
                st.session_state[evaluation_results_key] = {
                    "details": detail_rows,
                    "summaries": summary_rows,
                }
            except (
                ComparisonConfigurationError,
                FloresEvaluationError,
                LocalInferenceError,
                ImportError,
                RuntimeError,
                OSError,
            ) as error:
                st.error(f"Could not complete FLORES evaluation: {error}")
    
        evaluation_output = st.session_state.get(evaluation_results_key)
        if evaluation_output:
            st.markdown("#### Benchmark performance")
            summary_rows = evaluation_output["summaries"]
            st.dataframe(summary_rows, width="stretch", hide_index=True)
            chart_rows = [
                row for row in summary_rows if row.get("chrF++") is not None
            ]
            if chart_rows:
                st.bar_chart(chart_rows, x="model", y="chrF++")
            st.markdown("#### Per-instance translations")
            st.dataframe(
                evaluation_output["details"],
                width="stretch",
                hide_index=True,
            )

def render_nlue_tab(
    *,
    st: Any,
    cached_arkios: Any,
    cached_gemma4_base: Any,
    cached_google_backend: Any,
    cached_himalayagpt: Any,
    cached_iriis_gpt2: Any,
    cached_local_model_pair: Any,
    cached_nlue_examples: Any,
) -> None:
    st.markdown("### Nepali decoder and generation benchmarks")
    st.markdown(
        f"Official [IRIIS dataset collection]({NLUE_COLLECTION_URL}) · "
        f"[benchmark paper and published baselines]({NLUE_PAPER_URL})"
    )
    st.caption(
        "All tasks run through text generation: 13 generatively prompted NLUE "
        "tasks plus Belebele Nepali reading comprehension, Global-MMLU Nepali "
        "knowledge/reasoning, and XL-Sum Nepali abstractive summarization. "
        "FLORES translation is available from the Translation evaluation option "
        "in this same Inference tab."
    )
    nlue_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
    nlue_task_spec = st.selectbox(
        "Decoder benchmark task",
        DECODER_EVALUATION_TASKS,
        format_func=lambda task: f"{task.category} · {task.label}",
        key="nlue-task",
    )
    task_columns = st.columns(4)
    task_columns[0].metric("Official split", nlue_task_spec.split)
    task_columns[1].metric("Split rows", f"{nlue_task_spec.split_size:,}")
    task_columns[2].metric("Evaluation type", nlue_task_spec.kind)
    task_columns[3].metric(
        "Primary metric", nlue_task_spec.primary_metric or "manual review"
    )
    st.caption(
        f"Dataset: `{nlue_task_spec.dataset_id}` · pinned revision: "
        f"`{nlue_task_spec.revision[:12]}`"
    )
    if nlue_task_spec.source_url != NLUE_COLLECTION_URL:
        st.markdown(f"[Official benchmark source]({nlue_task_spec.source_url})")
    
    baseline_rows = published_baseline_rows(nlue_task_spec.key)
    st.markdown("#### Published standard results")
    if baseline_rows:
        st.dataframe(baseline_rows, width="stretch", hide_index=True)
    else:
        st.info(
            "No directly comparable published score is attached to this task. "
            "Use the selected models below for an identical-prompt comparison."
        )
    if nlue_task_spec.kind == "manual":
        st.warning(
            "GMET has no gold-answer column. The paper used native-speaker "
            "manual judgment because more than one masked completion may be "
            "valid. Predictions are generated and exported here, but no "
            "automatic accuracy or paper delta is fabricated."
        )
    
    nlue_slice_columns = st.columns(3)
    nlue_offset = nlue_slice_columns[0].number_input(
        "Starting position",
        min_value=0,
        max_value=nlue_task_spec.split_size - 1,
        value=0,
        step=1,
        key=f"nlue-offset:{nlue_task_spec.key}",
    )
    nlue_count = nlue_slice_columns[1].number_input(
        "Examples",
        min_value=1,
        max_value=min(100, nlue_task_spec.split_size - int(nlue_offset)),
        value=min(10, nlue_task_spec.split_size - int(nlue_offset)),
        step=1,
        key=f"nlue-count:{nlue_task_spec.key}:{int(nlue_offset)}",
    )
    nlue_slice_columns[2].caption(
        "HF_TOKEN detected."
        if nlue_hf_token
        else "Public datasets; no token required."
    )
    nlue_examples_key = (
        f"nlue-examples:{nlue_task_spec.key}:{int(nlue_offset)}:"
        f"{int(nlue_count)}"
    )
    if st.button("Load benchmark instances", key="load-nlue-instances"):
        try:
            with st.spinner("Streaming the requested pinned benchmark slice…"):
                st.session_state[nlue_examples_key] = cached_nlue_examples(
                    nlue_task_spec.key,
                    int(nlue_offset),
                    int(nlue_count),
                    secret_fingerprint(nlue_hf_token),
                    nlue_hf_token,
                )
        except NLUEEvaluationError as error:
            st.error(str(error))
    nlue_examples = st.session_state.get(nlue_examples_key, [])
    if nlue_examples:
        st.markdown("#### Benchmark instances")
        st.dataframe(
            [example.as_dict() for example in nlue_examples],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Load a bounded benchmark slice before running model inference.")
    
    st.markdown("#### Models")
    nlue_adapters = discover_local_adapters(configured_finetuned_models_root())
    nlue_local_options: dict[str, tuple[LocalAdapterSpec, bool]] = {}
    for adapter in nlue_adapters:
        nlue_local_options[f"Base · {adapter.label}"] = (adapter, False)
        nlue_local_options[f"Finetuned · {adapter.label}"] = (adapter, True)
    nlue_gemini_option = "Gemini API"
    nlue_google_gemma_option = GOOGLE_GEMMA_BACKEND_NAME
    nlue_gemma_base_option = GEMMA4_BASE_BACKEND_NAME
    nlue_himalaya_option = HIMALAYAGPT_BACKEND_NAME
    nlue_arkios_option = ARKIOS_BACKEND_NAME
    selected_nlue_models = st.multiselect(
        "Decoder models to benchmark",
        [
            nlue_gemini_option,
            nlue_google_gemma_option,
            *nlue_local_options,
            *IRIIS_GPT2_BACKEND_OPTIONS,
            nlue_gemma_base_option,
            nlue_himalaya_option,
            nlue_arkios_option,
        ],
        default=[],
        key="nlue-models",
        help="Base and finetuned variants share one loaded PEFT model pair.",
    )
    nlue_api_columns = st.columns(2)
    nlue_gemini_model = DEFAULT_GEMINI_MODEL
    if nlue_gemini_option in selected_nlue_models:
        nlue_gemini_model = nlue_api_columns[0].text_input(
            "Benchmark Gemini model", DEFAULT_GEMINI_MODEL
        )
    nlue_google_gemma_model = DEFAULT_GOOGLE_GEMMA_MODEL
    if nlue_google_gemma_option in selected_nlue_models:
        nlue_google_gemma_model = nlue_api_columns[1].text_input(
            "Benchmark Google Gemma model", DEFAULT_GOOGLE_GEMMA_MODEL
        )
    
    selected_nlue_local = [
        option for option in selected_nlue_models if option in nlue_local_options
    ]
    selected_nlue_iriis_specs = [
        IRIIS_GPT2_BACKEND_OPTIONS[option]
        for option in selected_nlue_models
        if option in IRIIS_GPT2_BACKEND_OPTIONS
    ]
    nlue_has_local = (
        bool(selected_nlue_local)
        or any(
            option in selected_nlue_models
            for option in (
                nlue_gemma_base_option,
                nlue_himalaya_option,
                nlue_arkios_option,
            )
        )
        or bool(selected_nlue_iriis_specs)
    )
    nlue_device = "auto"
    nlue_dtype = "auto"
    nlue_quantization = "auto"
    nlue_local_only = False
    if nlue_has_local:
        nlue_runtime_columns = st.columns(4)
        nlue_device = nlue_runtime_columns[0].selectbox(
            "Benchmark device", ("auto", "cuda", "cpu")
        )
        nlue_dtype = nlue_runtime_columns[1].selectbox(
            "Benchmark dtype", ("auto", "float32", "bfloat16", "float16")
        )
        nlue_quantization = nlue_runtime_columns[2].selectbox(
            "Benchmark PEFT quantization",
            LOCAL_QUANTIZATION_CHOICES,
            help=(
                "Auto uses CUDA 4-bit for PEFT base models of 7B or larger "
                "and permits CPU/disk offload when VRAM is full."
            ),
        )
        nlue_local_only = nlue_runtime_columns[3].checkbox(
            "Benchmark cached weights only"
        )
    
    generation_columns = st.columns(2)
    default_nlue_max_tokens = 128 if nlue_task_spec.kind == "generation" else 16
    nlue_max_tokens = generation_columns[0].number_input(
        "Benchmark maximum new tokens",
        1,
        512,
        default_nlue_max_tokens,
        key=f"nlue-max-tokens:{nlue_task_spec.key}",
    )
    nlue_seed = generation_columns[1].number_input(
        "Benchmark evaluation seed", min_value=0, value=42
    )
    nlue_config = DecodingConfig(
        temperature=0,
        top_p=1,
        top_k=None,
        max_new_tokens=int(nlue_max_tokens),
        seed=int(nlue_seed),
        thinking_level="minimal",
    )
    nlue_request_count = len(nlue_examples) * len(selected_nlue_models)
    st.info(
        f"This run will perform {nlue_request_count} deterministic "
        f"generation(s) over {len(nlue_examples)} loaded instance(s)."
    )
    nlue_safety_error = ""
    if nlue_request_count > 200:
        nlue_safety_error = "Reduce models or examples to at most 200 generations."
        st.error(nlue_safety_error)
    if selected_nlue_iriis_specs and int(nlue_max_tokens) >= 512:
        nlue_safety_error = (
            "Benchmark maximum new tokens must be below 512 for IRIIS Nepali GPT-2."
        )
        st.error(nlue_safety_error)
    
    nlue_context = hashlib.sha256(
        (
            f"{nlue_task_spec.key}\0{nlue_offset}\0{nlue_count}\0"
            f"{selected_nlue_models}\0{nlue_gemini_model}\0"
            f"{nlue_google_gemma_model}\0{nlue_device}\0{nlue_dtype}\0"
            f"{nlue_quantization}\0{nlue_local_only}\0{nlue_config}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    nlue_results_key = f"nlue-results:{nlue_context}"
    if st.button(
        "Run decoder evaluation",
        type="primary",
        key="run-nlue-evaluation",
        disabled=(
            not nlue_examples or not selected_nlue_models or bool(nlue_safety_error)
        ),
    ):
        try:
            nlue_backends = []
            effective_gemini_key = os.getenv("GEMINI_API_KEY", "")
            if {
                nlue_gemini_option,
                nlue_google_gemma_option,
            }.intersection(selected_nlue_models) and not effective_gemini_key:
                raise ComparisonConfigurationError(
                    "GEMINI_API_KEY is required for selected Google models"
                )
            if nlue_gemini_option in selected_nlue_models:
                nlue_backends.append(
                    cached_google_backend(
                        nlue_gemini_model,
                        secret_fingerprint(effective_gemini_key),
                        effective_gemini_key,
                    )
                )
            if nlue_google_gemma_option in selected_nlue_models:
                nlue_backends.append(
                    cached_google_backend(
                        nlue_google_gemma_model,
                        secret_fingerprint(effective_gemini_key),
                        effective_gemini_key,
                    )
                )
    
            loaded_nlue_pairs: dict[str, Any] = {}
            nlue_adapter_required: dict[str, bool] = {}
            for selected_option in selected_nlue_local:
                selected_adapter, use_adapter = nlue_local_options[selected_option]
                nlue_adapter_required[selected_adapter.key] = (
                    nlue_adapter_required.get(selected_adapter.key, False)
                    or use_adapter
                )
            for option in selected_nlue_local:
                adapter, use_adapter = nlue_local_options[option]
                if adapter.key not in loaded_nlue_pairs:
                    loaded_nlue_pairs[adapter.key] = cached_local_model_pair(
                        adapter.key,
                        adapter.label,
                        str(adapter.path),
                        adapter.base_model_id,
                        nlue_device,
                        nlue_dtype,
                        nlue_quantization,
                        nlue_adapter_required[adapter.key],
                        nlue_local_only,
                        secret_fingerprint(nlue_hf_token),
                        nlue_hf_token,
                    )
                nlue_backends.append(
                    LocalPeftBackend(
                        loaded_nlue_pairs[adapter.key], use_adapter=use_adapter
                    )
                )
            for iriis_spec in selected_nlue_iriis_specs:
                nlue_backends.append(
                    IRIISGPT2Backend(
                        cached_iriis_gpt2(
                            iriis_spec.key,
                            nlue_device,
                            nlue_dtype,
                            nlue_local_only,
                            secret_fingerprint(nlue_hf_token),
                            nlue_hf_token,
                        )
                    )
                )
            if nlue_himalaya_option in selected_nlue_models:
                nlue_backends.append(
                    HimalayaGPTBackend(
                        cached_himalayagpt(
                            DEFAULT_HIMALAYAGPT_MODEL_ID,
                            DEFAULT_HIMALAYAGPT_REVISION,
                            nlue_device,
                            nlue_dtype,
                            nlue_local_only,
                        )
                    )
                )
            if nlue_arkios_option in selected_nlue_models:
                nlue_backends.append(
                    ArkiosBackend(
                        cached_arkios(
                            DEFAULT_ARKIOS_MODEL_ID,
                            DEFAULT_ARKIOS_REVISION,
                            nlue_device,
                            nlue_dtype,
                            nlue_local_only,
                        )
                    )
                )
            if nlue_gemma_base_option in selected_nlue_models:
                nlue_backends.append(
                    Gemma4BaseBackend(
                        cached_gemma4_base(
                            DEFAULT_GEMMA4_BASE_MODEL_ID,
                            DEFAULT_GEMMA4_BASE_REVISION,
                            nlue_device,
                            nlue_dtype,
                            nlue_local_only,
                            secret_fingerprint(nlue_hf_token),
                            nlue_hf_token,
                        )
                    )
                )
            with st.spinner(
                f"Running and scoring {nlue_request_count} benchmark generations…"
            ):
                nlue_raw_results = run_comparison(
                    nlue_backends,
                    [example.prompt for example in nlue_examples],
                    [nlue_config],
                    prompt_template="{text}",
                )
                nlue_details, nlue_summaries = score_nlue_results(
                    nlue_task_spec.key, nlue_raw_results, nlue_examples
                )
            st.session_state[nlue_results_key] = {
                "details": nlue_details,
                "summaries": nlue_summaries,
            }
        except (
            ComparisonConfigurationError,
            NLUEEvaluationError,
            LocalInferenceError,
            ImportError,
            RuntimeError,
            OSError,
        ) as error:
            st.error(f"Could not complete decoder evaluation: {error}")
    
    nlue_output = st.session_state.get(nlue_results_key)
    if nlue_output:
        st.markdown("#### Your model performance")
        nlue_summary_rows = nlue_output["summaries"]
        st.dataframe(nlue_summary_rows, width="stretch", hide_index=True)
        primary_metric = nlue_task_spec.primary_metric
        if primary_metric:
            chart_rows = [
                {"model": row["model"], primary_metric: row.get(primary_metric)}
                for row in nlue_summary_rows
                if row.get(primary_metric) is not None
            ]
            published = next(
                (row for row in baseline_rows if row["metric"] == primary_metric),
                None,
            )
            if published:
                chart_rows.append(
                    {
                        "model": f"Published best · {published['reference_model']}",
                        primary_metric: published["published_best"],
                    }
                )
            if chart_rows:
                st.bar_chart(chart_rows, x="model", y=primary_metric)
        st.markdown("#### Per-instance predictions")
        st.dataframe(nlue_output["details"], width="stretch", hide_index=True)
        st.download_button(
            "Download benchmark predictions as JSON",
            json.dumps(
                nlue_output,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            file_name=f"decoder_{nlue_task_spec.key}_results.json",
            mime="application/json",
        )
