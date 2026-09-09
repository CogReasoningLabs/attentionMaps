"""EDA and methodology tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .common import (
    DEFAULT_EDA_TERMS,
    EDAAnalysisConfig,
    EDASurveyPlan,
    EDASurveyRun,
    Path,
    analyze_eda_records,
    configured_eda_output_root,
    eda_dataset_spec,
    json,
    load_eda_cleaning_notes,
    parse_eda_terms,
    parse_stopwords,
    sample_dataset_rows,
    text_columns,
    write_survey_report,
)

def render_eda_tab(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    nepali_stopwords: tuple[str, ...],
) -> None:
    st.markdown("### Survey-ready exploratory data analysis")
    st.caption(
        "Run the shared bounded EDA pipeline on a deterministic sample of this "
        "dataset. Raw records are used only during analysis; saved artifacts "
        "contain aggregate metrics, bounded numeric samples, and top patterns."
    )
    eda_candidate_columns = text_columns(inventory["schema"])
    if not eda_candidate_columns or int(inventory["rows"]) <= 0:
        st.info("This dataset has no detectable text rows available for EDA.")
    else:
        eda_field_columns = st.columns([3, 2])
        eda_text_fields = eda_field_columns[0].multiselect(
            "Text fields to combine",
            eda_candidate_columns,
            default=eda_candidate_columns[:1],
            key=f"eda-text-fields:{spec.key}",
            help=(
                "Chat and instruction fields can be combined. Document-size "
                "metrics describe the combined text for each row."
            ),
        )
        source_candidates = [
            column
            for column in inventory["columns"]
            if column.casefold()
            in {"source", "domain", "url", "link", "dataset", "source_id"}
        ]
        eda_source_fields = eda_field_columns[1].multiselect(
            "Source / provenance fields",
            inventory["columns"],
            default=source_candidates[:1],
            key=f"eda-source-fields:{spec.key}",
        )
    
        maximum_eda_rows = max(1, min(50_000, int(inventory["rows"])))
        settings = st.columns(5)
        eda_sample_size = settings[0].number_input(
            "Sampled rows",
            min_value=1,
            max_value=maximum_eda_rows,
            value=min(5_000, maximum_eda_rows),
            step=1,
            key=f"eda-sample-size:{spec.key}",
        )
        eda_seed = settings[1].number_input(
            "EDA seed",
            min_value=0,
            value=42,
            step=1,
            key=f"eda-seed:{spec.key}",
        )
        eda_min_tokens = settings[2].number_input(
            "Minimum words",
            min_value=1,
            max_value=1_000,
            value=5,
            step=1,
            key=f"eda-min-tokens:{spec.key}",
        )
        eda_devanagari_ratio = settings[3].slider(
            "Minimum Devanagari ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key=f"eda-devanagari-ratio:{spec.key}",
        )
        eda_top_patterns = settings[4].number_input(
            "Top patterns",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key=f"eda-top-patterns:{spec.key}",
        )
    
        with st.expander("N-gram and co-occurrence settings"):
            pattern_columns = st.columns([3, 1, 1])
            eda_terms_text = pattern_columns[0].text_area(
                "Co-occurrence seed terms",
                value=", ".join(DEFAULT_EDA_TERMS),
                key=f"eda-terms:{spec.key}",
                help="Comma- or whitespace-separated token list.",
            )
            eda_window = pattern_columns[1].number_input(
                "Token window",
                min_value=1,
                max_value=50,
                value=6,
                step=1,
                key=f"eda-window:{spec.key}",
            )
            eda_pattern_cap = pattern_columns[2].number_input(
                "Tokens per document",
                min_value=100,
                max_value=50_000,
                value=5_000,
                step=100,
                key=f"eda-pattern-cap:{spec.key}",
            )
            eda_stopwords_text = st.text_area(
                "Co-occurrence stopwords",
                value=" ".join(nepali_stopwords),
                key=f"eda-stopwords:{spec.key}",
            )
    
        result_key = f"eda-result:{spec.key}"
        if st.button(
            "Run EDA for this dataset",
            type="primary",
            key=f"run-eda:{spec.key}",
            disabled=not eda_text_fields,
        ):
            progress_bar = st.progress(0, text="Preparing EDA configuration…")
            run_status = st.status("Running bounded dataset EDA…", expanded=True)
            try:
                analysis_config = EDAAnalysisConfig(
                    sample_size=int(eda_sample_size),
                    seed=int(eda_seed),
                    min_tokens=int(eda_min_tokens),
                    min_devanagari_ratio=float(eda_devanagari_ratio),
                    reservoir_size=min(10_000, max(100, int(eda_sample_size))),
                    max_vocabulary=250_000,
                    top_tokens=max(30, int(eda_top_patterns)),
                    ngram_orders=(2, 3, 4),
                    max_ngrams=150_000,
                    top_ngrams=int(eda_top_patterns),
                    max_pattern_tokens_per_document=int(eda_pattern_cap),
                    cooccurrence_terms=parse_eda_terms(eda_terms_text),
                    cooccurrence_window=int(eda_window),
                    max_cooccurrence_edges=75_000,
                    top_cooccurrence_edges=max(50, int(eda_top_patterns)),
                    cooccurrence_stopwords=tuple(
                        sorted(parse_stopwords(eda_stopwords_text))
                    ),
                )
                analysis_spec = eda_dataset_spec(
                    spec,
                    eda_text_fields,
                    eda_source_fields,
                    int(eda_sample_size),
                )
                selected_fields = tuple(
                    dict.fromkeys((*eda_text_fields, *eda_source_fields))
                )
                run_status.write(
                    f"Sampling {int(eda_sample_size):,} rows with seed "
                    f"{int(eda_seed)}…"
                )
                progress_bar.progress(10, text="Sampling dataset rows…")
                sampled_records = sample_dataset_rows(
                    inventory,
                    int(eda_sample_size),
                    int(eda_seed),
                    selected_fields,
                )
                run_status.write(
                    f"Analyzing {len(sampled_records):,} sampled rows…"
                )
    
                def update_eda_progress(
                    dataset_key: str, stage: str, completed: int
                ) -> None:
                    del dataset_key, stage
                    share = completed / max(1, len(sampled_records))
                    progress_bar.progress(
                        min(85, 25 + round(60 * share)),
                        text=f"Analyzing row {completed:,}…",
                    )
    
                profile = analyze_eda_records(
                    analysis_spec,
                    sampled_records,
                    analysis_config,
                    progress=update_eda_progress,
                )
                del sampled_records
                progress_bar.progress(88, text="Creating figures and CSV tables…")
                report_root = configured_eda_output_root()
                survey = EDASurveyRun(
                    EDASurveyPlan(
                        f"UI EDA · {spec.label}",
                        (analysis_spec,),
                        analysis_config,
                    ),
                    (profile,),
                    {},
                )
                write_survey_report(survey, report_root, plots=True)
                dataset_output = report_root / analysis_spec.key
                st.session_state[result_key] = {
                    "profile": profile,
                    "output_dir": dataset_output,
                    "text_fields": tuple(eda_text_fields),
                    "analysis": analysis_config,
                }
                progress_bar.progress(100, text="EDA complete")
                run_status.update(
                    label="EDA complete",
                    state="complete",
                    expanded=False,
                )
            except (
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                run_status.update(label="EDA failed", state="error", expanded=True)
                st.error(f"Could not complete dataset EDA: {error}")
    
        eda_result = st.session_state.get(result_key)
        if eda_result:
            profile = eda_result["profile"]
            output_dir = Path(eda_result["output_dir"])
            summary = profile.summary
            st.markdown("#### Dataset overview")
            st.caption(
                f"Latest completed run · fields: "
                f"{', '.join(eda_result['text_fields'])} · "
                f"requested rows: {summary.sample_limit:,} · seed: {summary.seed}"
            )
            overview = st.columns(6)
            overview[0].metric("Usable rows", f"{summary.usable_rows:,}")
            overview[1].metric("Median words", f"{summary.median_tokens:,.1f}")
            overview[2].metric(
                "Median sentence", f"{summary.median_sentence_tokens:,.1f} words"
            )
            overview[3].metric(
                "Median line", f"{summary.median_line_characters:,.1f} chars"
            )
            overview[4].metric(
                "Devanagari clean", f"{summary.devanagari_clean_ratio_pct:.1f}%"
            )
            overview[5].metric("Duplicates", f"{summary.duplicate_ratio_pct:.1f}%")
            for warning in summary.warnings:
                st.warning(warning)
    
            size_plot, structure_plot, ngram_plot, network_plot = st.tabs(
                [
                    "Document size",
                    "Sentence & line structure",
                    "N-grams",
                    "Co-occurrence network",
                ]
            )
            with size_plot:
                st.image(output_dir / "document_size_distribution.png")
            with structure_plot:
                st.image(output_dir / "segment_length_kde.png")
            with ngram_plot:
                st.image(output_dir / "top_ngrams.png")
            with network_plot:
                st.image(output_dir / "term_cooccurrence_network.png")
    
            st.markdown("#### Detailed metrics and evidence")
            summary_table, pattern_table, edge_table, provenance_table = st.tabs(
                [
                    "All metrics",
                    "N-gram counts",
                    "Network edges",
                    "Tokens & sources",
                ]
            )
            with summary_table:
                st.dataframe(
                    [
                        {
                            "metric": key,
                            "value": (
                                json.dumps(value, ensure_ascii=False)
                                if isinstance(value, (dict, list, tuple))
                                else str(value)
                            ),
                        }
                        for key, value in summary.as_dict().items()
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with pattern_table:
                st.dataframe(
                    [
                        {"order": order, "ngram": phrase, "count": count}
                        for order, values in sorted(profile.top_ngrams.items())
                        for phrase, count in values
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with edge_table:
                st.dataframe(
                    [
                        {
                            "seed_term": source,
                            "neighbor": target,
                            "document_count": count,
                        }
                        for source, target, count in profile.cooccurrence_edges
                    ],
                    width="stretch",
                    hide_index=True,
                )
            with provenance_table:
                token_column, source_column = st.columns(2)
                token_column.dataframe(
                    [
                        {"token": token, "count": count}
                        for token, count in profile.top_tokens
                    ],
                    width="stretch",
                    hide_index=True,
                )
                source_column.dataframe(
                    [
                        {"source": source, "count": count}
                        for source, count in profile.top_sources
                    ],
                    width="stretch",
                    hide_index=True,
                )
    
            with st.expander("Download EDA artifacts"):
                st.caption(str(output_dir))
                for artifact in sorted(output_dir.glob("*.csv")):
                    st.download_button(
                        f"Download {artifact.name}",
                        artifact.read_bytes(),
                        file_name=artifact.name,
                        mime="text/csv",
                        key=f"download-eda:{spec.key}:{artifact.name}",
                    )
                summary_path = output_dir / "eda_summary.json"
                if summary_path.is_file():
                    st.download_button(
                        "Download eda_summary.json",
                        summary_path.read_bytes(),
                        file_name=summary_path.name,
                        mime="application/json",
                        key=f"download-eda:{spec.key}:summary",
                    )
            st.info(
                "Use the adjacent Random records tab to inspect complete source "
                "rows behind the aggregate findings."
            )

def render_notes_tab(*, st: Any) -> None:
    st.markdown(load_eda_cleaning_notes())
