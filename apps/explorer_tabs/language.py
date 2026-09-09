"""Word-cloud and tokenizer analysis tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .common import (
    DEFAULT_DATA_ROOT,
    ENGLISH_WORDCLOUD_STOPWORDS,
    Path,
    TokenizerAnalysisError,
    TokenizerSpec,
    analyses_csv,
    analyze_tokenizer,
    create_wordcloud,
    extract_text,
    find_devanagari_font,
    find_latin_font,
    font_supports_devanagari,
    hashlib,
    os,
    parse_stopwords,
    secret_fingerprint,
    text_columns,
    tokenizer_specs,
    word_frequencies,
)

def render_wordcloud_tab(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    cached_sample: Any,
    nepali_stopwords: tuple[str, ...],
    base_seed: int,
) -> None:
    candidate_columns = text_columns(inventory["schema"])
    paired_translation_clouds = {"source_text", "translation"}.issubset(
        inventory["columns"]
    )
    if not candidate_columns:
        st.info("This dataset has no detectable text-bearing columns.")
    else:
        st.markdown(
            "The cloud is computed from a deterministic random sample and reads "
            "only the selected columns. Increase the sample carefully for large corpora."
        )
        cloud_controls = st.columns([2, 1, 1, 1])
        if paired_translation_clouds:
            cloud_columns = ["source_text", "translation"]
            cloud_controls[0].markdown(
                "**Text columns**  \nOriginal (`source_text`) and Nepali (`translation`)"
            )
        else:
            cloud_columns = cloud_controls[0].multiselect(
                "Text columns",
                candidate_columns,
                default=candidate_columns[:1],
                key=f"wordcloud-columns:{spec.key}",
            )
        maximum_sample = max(1, min(5_000, int(inventory["rows"])))
        default_sample = min(500, maximum_sample)
        cloud_sample_size = cloud_controls[1].number_input(
            "Sampled rows",
            min_value=1,
            max_value=maximum_sample,
            value=default_sample,
            step=1,
        )
        maximum_words = cloud_controls[2].number_input(
            "Maximum words", min_value=10, max_value=500, value=150, step=10
        )
        minimum_frequency = cloud_controls[3].number_input(
            "Minimum frequency", min_value=1, max_value=100, value=2, step=1
        )
    
        option_columns = st.columns(3)
        minimum_characters = option_columns[0].number_input(
            "Minimum word characters", min_value=1, max_value=20, value=2
        )
        include_numbers = option_columns[1].checkbox("Include numeric tokens")
        colormap = option_columns[2].selectbox(
            "Colour map", ("viridis", "plasma", "magma", "cividis", "turbo")
        )
        if paired_translation_clouds:
            font_columns = st.columns(2)
            default_latin_font = find_latin_font()
            english_font_text = font_columns[0].text_input(
                "Original English font path",
                str(default_latin_font) if default_latin_font else "",
                help="Use a font with Latin glyph coverage, such as DejaVu Sans.",
            )
            default_devanagari_font = find_devanagari_font()
            nepali_font_text = font_columns[1].text_input(
                "Nepali translation font path",
                str(default_devanagari_font) if default_devanagari_font else "",
                help="Use a Devanagari-capable .ttf/.otf font.",
            )
            stopword_columns = st.columns(2)
            english_stopword_text = stopword_columns[0].text_area(
                "Original-text stopwords",
                ", ".join(ENGLISH_WORDCLOUD_STOPWORDS),
                height=100,
            )
            nepali_stopword_text = stopword_columns[1].text_area(
                "Nepali-translation stopwords",
                ", ".join(nepali_stopwords),
                height=100,
            )
            cloud_groups = (
                (
                    "Original (English)",
                    ("source_text",),
                    english_stopword_text,
                    english_font_text,
                    False,
                ),
                (
                    "Translation (Nepali)",
                    ("translation",),
                    nepali_stopword_text,
                    nepali_font_text,
                    True,
                ),
            )
        else:
            english_dataset = spec.key.startswith("lima:original:")
            default_font = (
                find_latin_font() if english_dataset else find_devanagari_font()
            )
            font_text = st.text_input(
                "Font path",
                str(default_font) if default_font else "",
                help=(
                    "Use a font with Latin glyph coverage."
                    if english_dataset
                    else "Use a Devanagari-capable .ttf/.otf font for Nepali text."
                ),
            )
            stopword_text = st.text_area(
                "Stopwords (comma, space, or newline separated)",
                ", ".join(
                    ENGLISH_WORDCLOUD_STOPWORDS
                    if english_dataset
                    else nepali_stopwords
                ),
                height=100,
            )
            cloud_groups = (
                (
                    "Selected text",
                    tuple(cloud_columns),
                    stopword_text,
                    font_text,
                    not english_dataset,
                ),
            )
    
        if not cloud_columns:
            st.info("Select at least one text column.")
        elif st.button("Generate word cloud", type="primary"):
            invalid_fonts = [
                Path(group_font_text).expanduser()
                for _, _, _, group_font_text, _ in cloud_groups
                if group_font_text.strip()
                and not Path(group_font_text).expanduser().is_file()
            ]
            if invalid_fonts:
                st.error(f"Font file not found: {invalid_fonts[0]}")
            else:
                incompatible_fonts = [
                    Path(group_font_text).expanduser()
                    for _, _, _, group_font_text, is_nepali in cloud_groups
                    if is_nepali
                    and group_font_text.strip()
                    and not font_supports_devanagari(
                        Path(group_font_text).expanduser()
                    )
                ]
                if incompatible_fonts:
                    st.error(
                        "Font does not contain the required Devanagari glyphs: "
                        f"{incompatible_fonts[0]}"
                    )
                    st.stop()
                try:
                    with st.spinner("Sampling text and building word frequencies…"):
                        cloud_records = cached_sample(
                            inventory,
                            int(cloud_sample_size),
                            int(base_seed),
                            tuple(cloud_columns),
                        )
                        cloud_outputs = []
                        for (
                            title,
                            columns,
                            group_stopwords,
                            group_font_text,
                            is_nepali,
                        ) in cloud_groups:
                            frequencies = word_frequencies(
                                cloud_records,
                                columns,
                                stopwords=parse_stopwords(group_stopwords),
                                min_characters=int(minimum_characters),
                                min_frequency=int(minimum_frequency),
                                include_numbers=include_numbers,
                                devanagari_only=is_nepali,
                                strip_nepali_suffixes=is_nepali,
                            )
                            cloud = create_wordcloud(
                                frequencies,
                                font_path=(
                                    Path(group_font_text).expanduser()
                                    if group_font_text.strip()
                                    else None
                                ),
                                max_words=int(maximum_words),
                                colormap=colormap,
                                seed=int(base_seed),
                            )
                            cloud_outputs.append((title, frequencies, cloud))
                except (ImportError, OSError, ValueError) as error:
                    st.error(f"Could not generate the word cloud: {error}")
                else:
                    output_columns = st.columns(len(cloud_outputs))
                    for output_column, (title, frequencies, cloud) in zip(
                        output_columns, cloud_outputs
                    ):
                        output_column.markdown(f"#### {title}")
                        output_column.image(cloud.to_array(), width="stretch")
                        output_column.caption(
                            f"{len(frequencies):,} retained word types from "
                            f"{len(cloud_records):,} sampled records."
                        )
                        top_words = [
                            {"word": word, "count": count}
                            for word, count in frequencies.most_common(50)
                        ]
                        output_column.markdown("##### Most frequent words")
                        output_column.dataframe(
                            top_words, width="stretch", hide_index=True
                        )

def render_tokenizer_tab(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    cached_sample: Any,
    cached_analysis_tokenizer: Any,
) -> None:
    st.markdown("### Nepali tokenizer coverage and efficiency")
    st.markdown(
        "Compare tokenizers without loading model weights. Vocabulary coverage "
        "shows how many non-special vocabulary entries contain Devanagari. "
        "Sample metrics show how efficiently the same Nepali text is encoded."
    )
    st.caption(
        "Hosted Gemini tokenizers are not listed because their vocabulary is "
        "not distributed as a local Hugging Face tokenizer."
    )
    
    available_tokenizer_specs = tokenizer_specs(DEFAULT_DATA_ROOT / "tokenized")
    custom_tokenizer_source = st.text_input(
        "Additional Hugging Face tokenizer/model ID (optional)",
        key="tokenizer-custom-source",
        placeholder="organization/model-name",
        help=(
            "Use this to compare a future base-model tokenizer without changing "
            "the application code. Model weights are never loaded."
        ),
    ).strip()
    if custom_tokenizer_source:
        custom_key = hashlib.sha256(
            custom_tokenizer_source.encode("utf-8")
        ).hexdigest()[:12]
        available_tokenizer_specs.append(
            TokenizerSpec(
                key=f"custom:{custom_key}",
                label=f"Custom · {custom_tokenizer_source}",
                source=custom_tokenizer_source,
            )
        )
    tokenizer_by_label = {
        tokenizer_spec.label: tokenizer_spec
        for tokenizer_spec in available_tokenizer_specs
    }
    default_tokenizers = [
        label
        for label in ("GPT-2 · base", "TinyLlama 1.1B · base")
        if label in tokenizer_by_label
    ]
    selected_tokenizer_labels = st.multiselect(
        "Tokenizers to compare",
        list(tokenizer_by_label),
        default=default_tokenizers,
        key="tokenizers-to-compare",
        help="Choose up to four tokenizers for a readable side-by-side view.",
    )
    if len(selected_tokenizer_labels) > 4:
        st.error("Select at most four tokenizers per comparison.")
    
    tokenizer_hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_token") or ""
    runtime_columns = st.columns(2)
    tokenizer_local_files_only = runtime_columns[0].checkbox(
        "Use cached tokenizer files only",
        value=False,
        key="tokenizer-local-files-only",
    )
    runtime_columns[1].caption(
        "HF_TOKEN detected for gated tokenizers."
        if tokenizer_hf_token
        else "No HF_TOKEN detected; gated tokenizers such as Llama 2 may fail."
    )
    
    tokenizer_text_mode = st.radio(
        "Comparison text",
        ("Custom Nepali text", "Selected dataset sample"),
        horizontal=True,
        key=f"tokenizer-text-mode:{spec.key}",
    )
    tokenizer_analysis_text = ""
    if tokenizer_text_mode == "Custom Nepali text":
        tokenizer_analysis_text = st.text_area(
            "Nepali text analyzed by every tokenizer",
            (
                "नेपाल प्राकृतिक सुन्दरता, सांस्कृतिक विविधता र बहुभाषिक "
                "समुदायले भरिएको देश हो। नेपाली भाषाका लागि प्रभावकारी "
                "टोकनाइजरले शब्दलाई धेरै साना टुक्रामा विभाजन गर्नु हुँदैन।"
            ),
            height=150,
            key="tokenizer-custom-text",
        )
    else:
        tokenizer_text_columns = text_columns(inventory["schema"])
        if not tokenizer_text_columns:
            st.warning(
                "The selected dataset has no detectable natural-language column."
            )
        else:
            sample_controls = st.columns([2, 1, 1, 1])
            preferred_tokenizer_column = (
                "translation"
                if "translation" in tokenizer_text_columns
                else (
                    "text"
                    if "text" in tokenizer_text_columns
                    else tokenizer_text_columns[0]
                )
            )
            tokenizer_text_column = sample_controls[0].selectbox(
                "Dataset text column",
                tokenizer_text_columns,
                index=tokenizer_text_columns.index(preferred_tokenizer_column),
                key=f"tokenizer-text-column:{spec.key}",
            )
            tokenizer_sample_rows = sample_controls[1].number_input(
                "Sample rows",
                min_value=1,
                max_value=max(1, min(200, int(inventory["rows"]))),
                value=min(20, max(1, int(inventory["rows"]))),
                step=1,
                key=f"tokenizer-sample-rows:{spec.key}",
            )
            tokenizer_sample_seed = sample_controls[2].number_input(
                "Sample seed",
                min_value=0,
                value=42,
                step=1,
                key=f"tokenizer-sample-seed:{spec.key}",
            )
            tokenizer_maximum_characters = sample_controls[3].number_input(
                "Maximum characters",
                min_value=100,
                max_value=100_000,
                value=10_000,
                step=100,
                key=f"tokenizer-max-characters:{spec.key}",
            )
            tokenizer_sample_key = (
                f"tokenizer-sample:{spec.key}:{tokenizer_text_column}:"
                f"{int(tokenizer_sample_rows)}:{int(tokenizer_sample_seed)}:"
                f"{int(tokenizer_maximum_characters)}"
            )
            if st.button(
                "Load dataset text sample",
                key=f"load-tokenizer-sample:{spec.key}",
            ):
                try:
                    sampled_tokenizer_records = cached_sample(
                        inventory,
                        int(tokenizer_sample_rows),
                        int(tokenizer_sample_seed),
                        (tokenizer_text_column,),
                    )
                    sampled_text_parts = []
                    for sampled_record in sampled_tokenizer_records:
                        sampled_text_parts.extend(
                            extract_text(sampled_record.get(tokenizer_text_column))
                        )
                    st.session_state[tokenizer_sample_key] = "\n\n".join(
                        sampled_text_parts
                    )[: int(tokenizer_maximum_characters)]
                except (ImportError, OSError, ValueError) as error:
                    st.error(f"Could not load tokenizer sample text: {error}")
            tokenizer_analysis_text = st.session_state.get(tokenizer_sample_key, "")
            if tokenizer_analysis_text:
                st.caption(
                    f"Loaded {len(tokenizer_analysis_text):,} characters from "
                    f"`{tokenizer_text_column}`."
                )
                with st.expander("Dataset text sample preview"):
                    st.text(tokenizer_analysis_text[:5_000])
            else:
                st.info(
                    "Load a bounded dataset sample before comparing tokenizers."
                )
    
    selected_tokenizer_specs = [
        tokenizer_by_label[label] for label in selected_tokenizer_labels
    ]
    tokenizer_context = hashlib.sha256(
        (
            f"{selected_tokenizer_specs}\0{tokenizer_analysis_text}\0"
            f"{tokenizer_local_files_only}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    tokenizer_results_key = f"tokenizer-results:{tokenizer_context}"
    if st.button(
        "Compare tokenizers",
        type="primary",
        key="run-tokenizer-comparison",
        disabled=(
            not selected_tokenizer_specs
            or len(selected_tokenizer_specs) > 4
            or not tokenizer_analysis_text.strip()
        ),
    ):
        tokenizer_analyses = []
        tokenizer_errors = []
        with st.spinner(
            f"Loading {len(selected_tokenizer_specs)} tokenizer(s) and analyzing…"
        ):
            for tokenizer_spec in selected_tokenizer_specs:
                try:
                    analysis_tokenizer = cached_analysis_tokenizer(
                        tokenizer_spec.key,
                        tokenizer_spec.label,
                        tokenizer_spec.source,
                        tokenizer_spec.revision,
                        tokenizer_spec.trust_remote_code,
                        tokenizer_spec.local,
                        tokenizer_local_files_only,
                        secret_fingerprint(tokenizer_hf_token),
                        tokenizer_hf_token,
                    )
                    tokenizer_analyses.append(
                        analyze_tokenizer(
                            tokenizer_spec,
                            analysis_tokenizer,
                            tokenizer_analysis_text,
                        )
                    )
                except (TokenizerAnalysisError, ImportError, OSError) as error:
                    tokenizer_errors.append(
                        {"tokenizer": tokenizer_spec.label, "error": str(error)}
                    )
        st.session_state[tokenizer_results_key] = {
            "analyses": tokenizer_analyses,
            "errors": tokenizer_errors,
        }
    
    tokenizer_output = st.session_state.get(tokenizer_results_key)
    if tokenizer_output:
        for tokenizer_error in tokenizer_output["errors"]:
            st.error(f"{tokenizer_error['tokenizer']}: {tokenizer_error['error']}")
        tokenizer_analyses = tokenizer_output["analyses"]
        if tokenizer_analyses:
            st.markdown("#### Side-by-side decision metrics")
            st.caption(
                "Prefer lower tokens per Nepali word and unknown-token percentage, "
                "and higher single-token word coverage. Vocabulary percentage is "
                "descriptive—not a model-quality score."
            )
            metric_columns = st.columns(len(tokenizer_analyses))
            for metric_column, tokenizer_analysis in zip(
                metric_columns, tokenizer_analyses
            ):
                metric_column.markdown(f"##### {tokenizer_analysis.tokenizer}")
                metric_column.caption(f"`{tokenizer_analysis.source}`")
                metric_column.metric(
                    "Devanagari vocabulary",
                    f"{tokenizer_analysis.devanagari_vocabulary_percent:.2f}%",
                    help=(
                        "Percentage of non-special vocabulary entries whose token "
                        "string contains at least one Devanagari code point."
                    ),
                )
                metric_column.metric(
                    "Tokens / Nepali word",
                    f"{tokenizer_analysis.tokens_per_nepali_word:.2f}",
                    help="Lower means less fragmentation on this exact sample.",
                )
                metric_column.metric(
                    "Single-token word coverage",
                    f"{tokenizer_analysis.single_token_nepali_word_percent:.2f}%",
                    help=(
                        "Percentage of unique Nepali words in the sample encoded as "
                        "one token."
                    ),
                )
                metric_column.metric(
                    "Unknown tokens",
                    f"{tokenizer_analysis.unknown_token_percent:.2f}%",
                )
                token_preview = " | ".join(
                    piece.raw_token.replace("\n", "\\n")
                    for piece in tokenizer_analysis.pieces[:40]
                )
                metric_column.code(token_preview or "—", language=None)
    
            summary_rows = [
                tokenizer_analysis.summary()
                for tokenizer_analysis in tokenizer_analyses
            ]
            st.dataframe(summary_rows, width="stretch", hide_index=True)
            chart_columns = st.columns(2)
            chart_columns[0].markdown("##### Devanagari vocabulary coverage")
            chart_columns[0].bar_chart(
                summary_rows,
                x="tokenizer",
                y="Devanagari vocabulary %",
            )
            chart_columns[1].markdown(
                "##### Nepali fragmentation (lower is better)"
            )
            chart_columns[1].bar_chart(
                summary_rows,
                x="tokenizer",
                y="tokens / Nepali word",
            )
            leader_columns = st.columns(3)
            vocabulary_leader = max(
                tokenizer_analyses,
                key=lambda result: result.devanagari_vocabulary_percent,
            )
            efficiency_leader = min(
                tokenizer_analyses,
                key=lambda result: result.tokens_per_nepali_word,
            )
            word_coverage_leader = max(
                tokenizer_analyses,
                key=lambda result: result.single_token_nepali_word_percent,
            )
            leader_columns[0].success(
                "Vocabulary coverage leader  \n"
                f"**{vocabulary_leader.tokenizer}**"
            )
            leader_columns[1].success(
                "Lowest fragmentation  \n" f"**{efficiency_leader.tokenizer}**"
            )
            leader_columns[2].success(
                "Single-token word leader  \n"
                f"**{word_coverage_leader.tokenizer}**"
            )
    
            detail_tokenizer = st.selectbox(
                "Inspect token pieces",
                tokenizer_analyses,
                format_func=lambda result: result.tokenizer,
                key=f"tokenizer-piece-detail:{tokenizer_context}",
            )
            st.dataframe(
                [piece.as_dict() for piece in detail_tokenizer.pieces],
                width="stretch",
                hide_index=True,
            )
            if detail_tokenizer.sample_tokens > len(detail_tokenizer.pieces):
                st.caption(
                    f"Showing the first {len(detail_tokenizer.pieces):,} of "
                    f"{detail_tokenizer.sample_tokens:,} sample tokens."
                )
            with st.expander("Devanagari vocabulary token examples"):
                vocabulary_example_columns = st.columns(len(tokenizer_analyses))
                for example_column, tokenizer_analysis in zip(
                    vocabulary_example_columns, tokenizer_analyses
                ):
                    example_column.markdown(f"**{tokenizer_analysis.tokenizer}**")
                    example_column.code(
                        "\n".join(tokenizer_analysis.devanagari_vocabulary_examples)
                        or "No Devanagari-bearing vocabulary tokens found.",
                        language=None,
                    )
            st.download_button(
                "Download tokenizer comparison as CSV",
                analyses_csv(tokenizer_analyses),
                file_name="nepali_tokenizer_comparison.csv",
                mime="text/csv",
            )
