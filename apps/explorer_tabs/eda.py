"""EDA and methodology tab renderers for the dataset explorer."""

from __future__ import annotations

from dataclasses import replace
from statistics import median
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
    fold_seed,
    json,
    load_eda_cleaning_notes,
    parse_eda_terms,
    plan_repeated_sampling,
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
    st.info(
        "This tab profiles source samples without changing them. For the enforced "
        "Sampling → NFC → Deduplication → clean-data EDA sequence, use WORKSPACE."
    )
    st.caption(
        "Choose a population percentage and compare independently seeded EDA "
        "folds. Raw records are used only during analysis; saved artifacts contain "
        "aggregate metrics, bounded numeric samples, and top patterns."
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
    
        population_rows = int(inventory["rows"])
        scope = st.radio(
            "EDA and deduplication scope",
            ("Repeated sample folds", "Entire population"),
            horizontal=True,
            key=f"eda-scope:{spec.key}",
            help=(
                "Sample folds are diagnostic and may overlap. Entire-population "
                "analysis is enabled in this UI only when it fits the safety limit."
            ),
        )
        settings = st.columns(6)
        suggested_percentage = min(20.0, max(0.01, 500_000 / population_rows))
        eda_percentage = settings[0].number_input(
            "Population per fold (%)",
            min_value=0.01,
            max_value=100.0,
            value=float(round(suggested_percentage, 4)),
            step=0.1,
            format="%.4f",
            disabled=scope == "Entire population",
            key=f"eda-sample-percentage:{spec.key}",
        )
        eda_folds = settings[1].number_input(
            "Repeated folds",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            disabled=scope == "Entire population",
            key=f"eda-folds:{spec.key}",
        )
        eda_seed = settings[2].number_input(
            "EDA seed",
            min_value=0,
            value=42,
            step=1,
            key=f"eda-seed:{spec.key}",
        )
        eda_min_tokens = settings[3].number_input(
            "Minimum words",
            min_value=1,
            max_value=1_000,
            value=5,
            step=1,
            key=f"eda-min-tokens:{spec.key}",
        )
        eda_devanagari_ratio = settings[4].slider(
            "Minimum Devanagari ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            key=f"eda-devanagari-ratio:{spec.key}",
        )
        eda_top_patterns = settings[5].number_input(
            "Top patterns",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key=f"eda-top-patterns:{spec.key}",
        )

        estimated_population_bytes = int(
            inventory.get("memory_bytes", inventory.get("bytes", 0))
        )
        full_population_available = (
            population_rows <= 50_000
            and estimated_population_bytes <= 1024**3
        )
        selected_percentage = 100.0 if scope == "Entire population" else float(eda_percentage)
        selected_folds = 1 if scope == "Entire population" else int(eda_folds)
        sampling_plan = plan_repeated_sampling(
            population_rows,
            selected_percentage,
            folds=selected_folds,
            max_rows_per_fold=(
                population_rows if scope == "Entire population" else 50_000
            ),
        )
        sampling_metrics = st.columns(4)
        sampling_metrics[0].metric(
            "Rows / fold", f"{sampling_plan.rows_per_fold:,}"
        )
        sampling_metrics[1].metric(
            "Effective / fold",
            f"{sampling_plan.effective_percentage_per_fold:.4g}%",
        )
        sampling_metrics[2].metric(
            "Total rows read", f"{sampling_plan.total_rows_read:,}"
        )
        sampling_metrics[3].metric(
            "Expected unique coverage",
            f"{sampling_plan.expected_population_coverage_pct:.1f}%",
            help="Expectation under independent folds; actual overlap can differ.",
        )
        if sampling_plan.capped:
            st.warning(
                f"The requested {sampling_plan.requested_percentage:g}% equals "
                f"{sampling_plan.requested_rows_per_fold:,} rows per fold. The "
                "laptop-safe limit reduces this to 50,000; the effective percentage "
                "shown above is what will actually run."
            )
        if scope == "Entire population" and not full_population_available:
            st.error(
                "Entire-population analysis in Streamlit requires at most 50,000 "
                "rows and an estimated decoded size no larger than 1 GiB. Use the "
                "streaming EDA CLI or distributed DataTrove/NeMo tooling with an "
                "explicit memory budget for larger corpora."
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

        with st.expander("Multi-stage deduplication settings", expanded=True):
            st.caption(
                "Pass 1: normalized SHA-256 exact matching → Pass 2: document "
                "MinHash-LSH → Pass 3: repeated normalized paragraph screening."
            )
            dedup_columns = st.columns(6)
            dedup_normalization = dedup_columns[0].selectbox(
                "Unicode normalization",
                ("NFC", "NFKC"),
                key=f"eda-dedup-normalization:{spec.key}",
            )
            dedup_lowercase = dedup_columns[1].toggle(
                "Case-fold before hash",
                value=True,
                key=f"eda-dedup-lowercase:{spec.key}",
            )
            dedup_whitespace = dedup_columns[2].toggle(
                "Collapse whitespace",
                value=True,
                key=f"eda-dedup-whitespace:{spec.key}",
            )
            minhash_shingles = dedup_columns[3].selectbox(
                "Character shingles",
                (3, 5, 7),
                index=1,
                key=f"eda-minhash-shingles:{spec.key}",
            )
            minhash_threshold = dedup_columns[4].slider(
                "Near-duplicate threshold",
                min_value=0.60,
                max_value=1.0,
                value=0.80,
                step=0.05,
                key=f"eda-minhash-threshold:{spec.key}",
            )
            boilerplate_min_documents = dedup_columns[5].number_input(
                "Boilerplate documents",
                min_value=2,
                max_value=1_000,
                value=3,
                step=1,
                key=f"eda-boilerplate-documents:{spec.key}",
            )
            st.info(
                f"Exact hashes will use SHA-256 after {dedup_normalization} "
                f"normalization, {'case-folding' if dedup_lowercase else 'case preservation'}, "
                f"and {'collapsed' if dedup_whitespace else 'preserved'} internal whitespace."
            )
    
        result_key = f"eda-result:{spec.key}"
        if st.button(
            "Run EDA for this dataset",
            type="primary",
            key=f"run-eda:{spec.key}",
            disabled=(
                not eda_text_fields
                or scope == "Entire population" and not full_population_available
            ),
        ):
            progress_bar = st.progress(0, text="Preparing EDA configuration…")
            run_status = st.status("Running bounded dataset EDA…", expanded=True)
            try:
                analysis_config = EDAAnalysisConfig(
                    sample_size=sampling_plan.rows_per_fold,
                    seed=int(eda_seed),
                    min_tokens=int(eda_min_tokens),
                    min_devanagari_ratio=float(eda_devanagari_ratio),
                    reservoir_size=min(
                        10_000, max(100, sampling_plan.rows_per_fold)
                    ),
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
                    dedup_normalization=str(dedup_normalization),
                    dedup_lowercase=bool(dedup_lowercase),
                    dedup_collapse_whitespace=bool(dedup_whitespace),
                    minhash_shingle_size=int(minhash_shingles),
                    minhash_permutations=128,
                    minhash_bands=16,
                    near_duplicate_threshold=float(minhash_threshold),
                    boilerplate_min_documents=int(boilerplate_min_documents),
                )
                base_analysis_spec = eda_dataset_spec(
                    spec,
                    eda_text_fields,
                    eda_source_fields,
                    sampling_plan.rows_per_fold,
                    population_rows,
                )
                selected_fields = tuple(
                    dict.fromkeys((*eda_text_fields, *eda_source_fields))
                )
                run_status.write(
                    f"Running {sampling_plan.folds} fold(s) × "
                    f"{sampling_plan.rows_per_fold:,} rows…"
                )
                profiles = []
                fold_specs = []
                for fold_index in range(1, sampling_plan.folds + 1):
                    current_seed = fold_seed(int(eda_seed), fold_index)
                    progress_base = (fold_index - 1) / sampling_plan.folds
                    progress_bar.progress(
                        5 + round(75 * progress_base),
                        text=f"Sampling fold {fold_index}/{sampling_plan.folds}…",
                    )
                    sampled_records = sample_dataset_rows(
                        inventory,
                        sampling_plan.rows_per_fold,
                        current_seed,
                        selected_fields,
                    )
                    run_status.write(
                        f"Fold {fold_index}: analyzing {len(sampled_records):,} "
                        f"rows with seed {current_seed}…"
                    )
                    fold_spec = replace(
                        base_analysis_spec,
                        key=f"{base_analysis_spec.key}-fold-{fold_index:02d}",
                    )
                    fold_config = replace(analysis_config, seed=current_seed)

                    def update_eda_progress(
                        dataset_key: str,
                        stage: str,
                        completed: int,
                    ) -> None:
                        del dataset_key, stage
                        within_fold = completed / max(1, len(sampled_records))
                        total_share = (
                            fold_index - 1 + within_fold
                        ) / sampling_plan.folds
                        progress_bar.progress(
                            min(80, 5 + round(75 * total_share)),
                            text=(
                                f"Fold {fold_index}/{sampling_plan.folds}: "
                                f"row {completed:,}…"
                            ),
                        )

                    profiles.append(
                        analyze_eda_records(
                            fold_spec,
                            sampled_records,
                            fold_config,
                            progress=update_eda_progress,
                        )
                    )
                    fold_specs.append(fold_spec)
                    del sampled_records

                progress_bar.progress(82, text="Creating fold figures and CSV tables…")
                report_root = configured_eda_output_root()
                survey = EDASurveyRun(
                    EDASurveyPlan(
                        f"UI EDA · {spec.label}",
                        tuple(fold_specs),
                        analysis_config,
                    ),
                    tuple(profiles),
                    {},
                )
                write_survey_report(survey, report_root, plots=True)
                st.session_state[result_key] = {
                    "profiles": tuple(profiles),
                    "output_dirs": tuple(
                        report_root / fold_spec.key for fold_spec in fold_specs
                    ),
                    "text_fields": tuple(eda_text_fields),
                    "analysis": analysis_config,
                    "sampling": sampling_plan,
                    "scope": scope,
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
            profiles = tuple(
                eda_result.get("profiles") or (eda_result["profile"],)
            )
            output_dirs = tuple(
                eda_result.get("output_dirs") or (eda_result["output_dir"],)
            )
            selected_fold = st.selectbox(
                "Displayed EDA fold",
                range(len(profiles)),
                format_func=lambda index: (
                    f"Fold {index + 1} · seed {profiles[index].summary.seed}"
                ),
                key=f"eda-displayed-fold:{spec.key}",
            )
            profile = profiles[selected_fold]
            output_dir = Path(output_dirs[selected_fold])
            summary = profile.summary
            st.markdown("#### Dataset overview")
            st.caption(
                f"Latest completed run · {eda_result.get('scope', 'Sample')} · fields: "
                f"{', '.join(eda_result['text_fields'])} · "
                f"fold rows: {summary.sample_limit:,} · seed: {summary.seed}"
            )
            if len(profiles) > 1:
                st.markdown("##### Across-fold stability")
                st.dataframe(
                    [
                        {
                            "fold": index,
                            "seed": item.summary.seed,
                            "examined_rows": item.summary.rows_seen,
                            "usable_rows": item.summary.usable_rows,
                            "script": item.summary.script_category,
                            "quality_pass_%": item.summary.quality_pass_ratio_pct,
                            "exact_duplicate_%": round(
                                100
                                * item.summary.exact_duplicate_rows
                                / max(1, item.summary.usable_rows),
                                4,
                            ),
                            "near_duplicate_%": round(
                                100
                                * item.summary.near_duplicate_rows
                                / max(1, item.summary.usable_rows),
                                4,
                            ),
                            "all_document_duplicates_%": (
                                item.summary.duplicate_ratio_pct
                            ),
                            "boilerplate_affected_%": (
                                item.summary.boilerplate_affected_ratio_pct
                            ),
                        }
                        for index, item in enumerate(profiles, start=1)
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Median across folds · quality pass "
                    f"{median(item.summary.quality_pass_ratio_pct for item in profiles):.1f}%"
                    " · document duplicates "
                    f"{median(item.summary.duplicate_ratio_pct for item in profiles):.1f}%"
                    ". Fold samples are independently seeded and may overlap."
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
            st.markdown("##### Multi-stage duplicate screening")
            duplicate_columns = st.columns(4)
            duplicate_columns[0].metric(
                "1 · Exact SHA-256", f"{summary.exact_duplicate_rows:,}"
            )
            duplicate_columns[1].metric(
                "2 · Near MinHash-LSH", f"{summary.near_duplicate_rows:,}"
            )
            duplicate_columns[2].metric(
                "3 · Repeated paragraphs",
                f"{summary.boilerplate_unique_paragraphs:,}",
            )
            duplicate_columns[3].metric(
                "Boilerplate-affected rows",
                f"{summary.boilerplate_affected_ratio_pct:.1f}%",
            )
            st.caption(
                f"Exact comparison: {summary.dedup_hash_algorithm} after "
                f"{summary.dedup_normalization}; case-folding="
                f"{summary.dedup_lowercase}; collapse-whitespace="
                f"{summary.dedup_collapse_whitespace}. Near comparison: "
                f"{summary.minhash_shingle_size}-character shingles, "
                f"{summary.minhash_permutations} permutations, "
                f"{summary.minhash_bands} LSH bands, threshold "
                f"{summary.near_duplicate_threshold:.2f}. These EDA passes flag "
                "candidates; they do not rewrite the selected dataset."
            )
            if (
                summary.population_coverage_pct is not None
                and summary.population_coverage_pct < 100
            ):
                st.warning(
                    "Fold-level duplicate rates are sample diagnostics and usually "
                    "underestimate corpus-wide duplication because both copies must "
                    "land in the same fold. Use entire-population deduplication before "
                    "materializing training data when resources permit."
                )
            st.markdown("##### Script classification")
            script_columns = st.columns([2, 1, 1, 1, 1])
            script_columns[0].metric("Detected category", summary.script_category)
            script_columns[1].metric(
                "Devanagari letters", f"{summary.devanagari_letter_share_pct:.1f}%"
            )
            script_columns[2].metric(
                "Latin letters", f"{summary.latin_letter_share_pct:.1f}%"
            )
            script_columns[3].metric(
                "Other letters", f"{summary.other_letter_share_pct:.1f}%"
            )
            script_columns[4].metric(
                "Romanized signal",
                f"{summary.romanized_latin_token_share_pct:.1f}%",
                help=(
                    "Share of sampled Latin word tokens matching conservative "
                    "Romanized-Nepali lexical or transliteration signals."
                ),
            )
            st.caption(
                "Category is inferred from the selected text fields in this "
                "bounded sample. Romanized vs Latin/English is heuristic; review "
                "examples before using it as a filtering rule."
            )
            for warning in summary.warnings:
                st.warning(warning)
    
            size_plot, structure_plot, ngram_plot, network_plot, dedup_plot = st.tabs(
                [
                    "Document size",
                    "Sentence & line structure",
                    "N-grams",
                    "Co-occurrence network",
                    "Deduplication",
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
            with dedup_plot:
                st.image(output_dir / "deduplication_stages.png")
    
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
    st.info(
        "Survey EDA only normalizes an in-memory analysis copy; it does not "
        "remove rows or rewrite the source dataset. The notes below separately "
        "describe the materializing pretraining cleaner."
    )
    st.markdown(load_eda_cleaning_notes())
