"""Read-only presentation of reports produced by the inspection script."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.components.inspection_history import select_history_report
from attention_maps.explorer.inspection_history import decision_text
from attention_maps.explorer.language_status import LANGUAGE_COVERAGE_OPTIONS, SCRIPT_OPTIONS
from apps.components.file_formats import render_file_formats
from apps.components.sampling_vote import render_sampling_vote
from attention_maps.explorer.inspection import file_signatures
from attention_maps.explorer.inspection_runs import load_inspection_report
from attention_maps.explorer.text import format_decimal_bytes

REPORT_STATE_KEY = "dataset-inspection-report"


def _category_value(status: dict, key: str) -> str:
    allowed = LANGUAGE_COVERAGE_OPTIONS if key == "language_coverage" else SCRIPT_OPTIONS
    return status[key] if status[key] in allowed else "—"


def _display_vote_counts(counts: dict, key: str) -> dict:
    allowed = LANGUAGE_COVERAGE_OPTIONS if key == "language_coverage" else SCRIPT_OPTIONS
    shown = {label: count for label, count in counts.items() if label in allowed}
    abstentions = sum(count for label, count in counts.items() if label not in allowed)
    if abstentions:
        shown["No supported category"] = abstentions
    return shown


def _render_category_proportions(st: Any, status: dict, key: str) -> None:
    def share(count: int, total: int) -> str:
        return f"{count / total:.2%}" if total else "0.00%"

    if key == "language_coverage":
        options = LANGUAGE_COVERAGE_OPTIONS
        counts = status.get("language_category_counts")
        percentages = status.get("language_category_percentages")
        denominator = status["sampled_records"]
        no_evidence = status.get("language_no_evidence_records", 0)
        outside = status.get("language_outside_categories_records", 0)
        detail = (f"No language evidence: {no_evidence:,} ({share(no_evidence, denominator)}); "
                  f"outside listed categories: {outside:,} ({share(outside, denominator)}). "
                  "Both remain in the denominator.")
        scope = "all unique sampled records"
        detail += " Multilingual here means a single record with multiple language labels."
    else:
        options = SCRIPT_OPTIONS
        counts = status.get("script_category_counts")
        percentages = status.get("script_category_percentages")
        denominator = status.get("script_eligible_records", 0)
        no_evidence = status.get("script_no_evidence_records", 0)
        outside = status.get("script_outside_categories_records", 0)
        detail = (f"No script evidence: {no_evidence:,} ({share(no_evidence, denominator)}); "
                  f"outside listed categories: {outside:,} ({share(outside, denominator)}). "
                  "Both remain in the denominator. "
                  f"Records without Nepali evidence excluded from this conditional denominator: {status['sampled_records'] - denominator:,}.")
        scope = "Nepali-eligible unique sampled records"
    if counts is None or percentages is None:
        st.caption("Category proportions were not saved by this earlier run. Rerun the inspection script to calculate them.")
        return
    st.caption(f"Category proportions · denominator: {denominator:,} {scope}")
    st.dataframe([{"Category": name, "Records": counts[name], "Percent": percentages[name]}
                  for name in options], hide_index=True, width="stretch")
    st.caption(detail)
    if key == "language_coverage" and (status.get("language_detection") or {}).get("attempted_records"):
        st.caption("The text detector assigns one dominant language per record; bilingual and multilingual record shares require multi-language column labels or a multi-label detector.")


def render_saved_status(st: Any, status: dict) -> None:
    cards = st.columns(2)
    for card, key, label in zip(cards, ("language_coverage", "script"), ("Language coverage", "Script")):
        with card.container(border=True):
            category = _category_value(status, key)
            st.metric(label, category)
            if category != "—":
                st.write(decision_text(status, key))
            if category == "—":
                if key == "script" and status.get("nepali_covered") is False:
                    st.caption("No Nepali script category applies because Nepali is not covered.")
                else:
                    st.warning(f"This saved run has no supported {label} category. "
                               f"{status.get('language_reason' if key == 'language_coverage' else 'script_reason', '')}")
            _render_category_proportions(st, status, key)
            if key == "language_coverage" and status[key] == "Unknown":
                gate = status.get("classification_thresholds", {}).get("language_min_labelled_ratio")
                measured = status.get("labelled_record_ratio")
                if gate is not None and measured is not None and measured < gate:
                    detection = status.get("language_detection") or {}
                    message = (f"Language evidence covers {measured:.1%} of sampled records; "
                               f"the required minimum is {gate:.1%}.")
                    if detection.get("attempted_records"):
                        reasons = detection.get("reason_counts") or {}
                        message += (f" Text detector: {reasons.get('too_short', 0):,} too short, "
                                    f"{reasons.get('low_confidence', 0):,} below its "
                                    f"{detection['min_confidence']:g} score cutoff.")
                    st.warning(message)
            if key == "script" and status.get("script_policy"):
                covered = status.get("nepali_covered")
                st.caption("Nepali covered: " + ("Yes" if covered is True else "No" if covered is False else "—"))
                st.caption(status["script_reason"])
            vote = status.get("voting", {}).get(key)
            if vote:
                st.write({"Category votes": _display_vote_counts(vote["counts"], key)})
                rows = []
                for run in status["sampling"]["run_results"]:
                    row = {"Run": run["run"], "Records": run["sampled_records"],
                           "Vote": _category_value(run, key) if _category_value(run, key) != "—" else
                           "—" if key == "script" and run.get("nepali_covered") is False else "Abstained"}
                    if key == "language_coverage":
                        row["Evidence"] = run["language_basis"]
                        row["Reason"] = run.get("language_reason", "")
                        if run.get("language_detection"):
                            detected = run["language_detection"]
                            row["Predictions accepted"] = detected["accepted_records"]
                            row["Predictions rejected"] = detected["rejected_records"]
                    else:
                        percentages = run.get("script_analysis_percentages", run["script_percentages"])
                        row["Devanagari %"] = percentages.get("Devanagari", 0)
                        row["Latin %"] = percentages.get("Latin", 0)
                        row["Reason"] = run.get("script_reason", "")
                    rows.append(row)
                st.dataframe(rows, hide_index=True, width="stretch")
            elif key == "language_coverage":
                st.caption(status["language_basis"])
    if status.get("language_detection"):
        detection = status["language_detection"]
        st.caption(f"Text detection: {detection['accepted_records']:,} accepted / "
                   f"{detection['attempted_records']:,} attempted predictions on sampled records "
                   f"(minimum score {detection['min_confidence']:g}). "
                   "Metadata-labelled records bypass the detector.")
        with st.expander("Text language detector and rejected predictions"):
            st.json(detection)
    if status.get("language_evidence_policy"):
        with st.expander("Language evidence origins"):
            st.caption(status["language_evidence_rule"])
            sources = status.get("language_evidence_sources", [])
            if sources:
                st.json(sources)
            else:
                st.info("Neither a dataset language column nor an explicit Hugging Face language partition supplied evidence.")
        if status.get("language_evidence_conflict"):
            st.warning("Dataset-column languages differ from the selected Hugging Face language partition. The language result uses sampled column labels; both origins are saved above.")
    else:
        st.info("Historical report: this run predates explicit language evidence tracking. Rerun inspect_dataset.py to apply the current rule.")
    if status.get("classification_thresholds"):
        with st.expander("Classification thresholds and measured ratios"):
            st.json(status["classification_thresholds"])
            st.caption("Language ratios: fraction of records with metadata labels or accepted predictions mentioning each language (one count per language per record). Pairs can sum above 100%.")
            st.json({"labelled_fraction_of_sample": status.get("labelled_record_ratio"),
                     "language_record_ratios": status.get("language_record_ratios", {}),
                     "script_decision_ratios": status.get("script_decision_ratios", {})})
            st.caption("Script dominance uses Devanagari + Latin characters. Other-script tolerance uses all eligible letters/marks. These pooled diagnostics may differ from individual runs.")
            runs = status.get("sampling", {}).get("run_results", [])
            if runs:
                st.json({"Per-run decisions": [
                    {"run": run["run"], "language_ratios": run.get("language_record_ratios", {}),
                     "labelled_fraction": run.get("labelled_record_ratio"),
                     "script_ratios": run.get("script_decision_ratios", {}),
                     "language_reason": run.get("language_reason"), "script_reason": run.get("script_reason")}
                    for run in runs
                ]}, expanded=False)
    unit = "unique sampled records" if status.get("sampling") else "sampled records"
    st.caption(f"Language evidence: {status['language_basis']} · {status['sampled_records']:,} {unit}")
    render_sampling_vote(st, status)
    st.caption(status["note"])
    if status.get("script_policy"):
        with st.expander("Nepali script analysis scope"):
            st.caption(status["script_scope"])
            st.json(status["script_analysis_percentages"])
    if status.get("script_policy") != "nepali_required_v3":
        st.info("Historical script result: rerun inspect_dataset.py to apply the current language/script ratio thresholds.")
    if status.get("script_percentages"):
        st.caption("All sampled text: raw Unicode script percentages (diagnostics, independent of Nepali coverage).")
        st.json(status["script_percentages"], expanded=False)


def render_inspection_report(st: Any, report: dict) -> None:
    """Display stored values verbatim; never inspect, filter, or resample a source."""
    inventory = report["inventory"]
    settings = report["settings"]
    st.subheader("Dataset inspection script results")
    st.caption(f"Run completed: {report.get('created_at', 'Unknown')} · report version {report['report_version']}")
    provider = inventory.get("provider") or ("huggingface" if inventory.get("format") == "huggingface" else "local")
    st.caption(f"Source provider: {provider}")
    st.code(inventory.get("source_uri") or inventory.get("dataset_id") or settings.get("local") or "Local dataset", language=None)
    if provider == "kaggle":
        st.write({"Dataset": inventory["dataset_id"], "Version": inventory["dataset_version"],
                  "Selected files": inventory.get("selected_files", [])})
    elif inventory.get("dataset_id"):
        st.write({"Configuration": inventory.get("dataset_config"),
                  "Split": inventory.get("dataset_split"), "Revision": inventory.get("dataset_revision")})
    cards = st.columns(3)
    rows, size = inventory.get("rows"), inventory.get("bytes")
    cards[0].metric("Selected source rows", f"{rows:,}" if rows is not None else "Unknown")
    cards[1].metric("Selected source files", str(inventory.get("files", "Unknown")))
    cards[2].metric("Selected file size", format_decimal_bytes(size) if size is not None else "Unknown")
    selection = inventory.get("row_selection")
    if selection:
        with st.container(border=True):
            st.markdown("#### Language selection before sampling")
            st.json(selection["filters"])
            st.write(f"{selection['rows_matched']:,} matching rows out of {selection['rows_scanned']:,}; "
                     f"{selection['rows_excluded']:,} excluded before sampling.")
            st.caption(selection["rule"])
            st.caption("Language coverage and script votes below describe the selected subset. "
                       "File size still describes the selected source files, not the filtered rows.")
    render_file_formats(st, inventory.get("file_formats"))
    if settings.get("formats_only"):
        st.caption("Formats-only run: records were not parsed or sampled.")
    st.caption("Processing order: choose configuration/split/shards and matching rows first; "
               "then sample or apply optional Devanagari filtering within that portion. "
               "Language/script thresholds classify those samples; excluded corpus records do not contribute.")
    st.markdown("#### Selected portion language coverage and script")
    st.caption(f"Sample scope: {report.get('sample_scope', 'selected_source_sample')}")
    render_saved_status(st, report["language_status"])
    filtered = report.get("filter_result")
    if filtered is not None:
        st.markdown("#### Devanagari filtering result")
        st.caption(f"Minimum Devanagari ratio: {filtered['min_devanagari_ratio']:g} · Whole records retained")
        if not filtered["selection_exhausted"]:
            st.info("This run scanned a limited prefix. Filtering counts describe that prefix only.")
        cards = st.columns(3)
        for card, label, key in zip(cards, ("Records scanned", "Records kept", "Records dropped"),
                                    ("rows_scanned", "rows_kept", "rows_dropped")):
            card.metric(label, f"{filtered[key]:,}")
        st.code(filtered["output_path"], language=None)
        st.caption(f"Output file format: {filtered.get('output_format', 'Unknown').upper()}")
        st.caption(f"Filtered output: {format_decimal_bytes(filtered['output_bytes'])} · {filtered['scope']}")
        st.markdown("#### Filtered output language coverage and script")
        render_saved_status(st, filtered["language_status"])
    with st.expander("Exact script settings and selected files"):
        st.json(settings)
        st.write(inventory.get("dataset_shards") or inventory.get("selected_files") or inventory.get("source_files") or [])
    with st.expander("Full saved report"):
        st.json(report)


def render_script_results(st: Any, *, default_path: str = "") -> None:
    st.sidebar.caption("Run scripts/inspect_dataset.py, then browse the single CSV containing every inspection run.")
    mode = st.sidebar.radio("Result source", ("History sheet", "Report file"), index=1 if default_path and Path(default_path).is_file() else 0)
    st.sidebar.button("Reload report", help="Read saved files again. This never reruns the script.")
    st.info("This view displays saved script output. Run the script manually for each dataset to record its results and votes.")
    try:
        if mode == "History sheet":
            report = select_history_report(st)
            if report is None:
                st.session_state.pop(REPORT_STATE_KEY, None)
                return
        else:
            path_text = st.sidebar.text_input(
                "Inspection report path", value=default_path or "artifacts/dataset_inspection/history.csv",
                key="inspection-report-path",
            ).strip()
            if not path_text:
                st.session_state.pop(REPORT_STATE_KEY, None)
                return
            report = load_inspection_report(Path(path_text).expanduser())
    except (OSError, ValueError) as error:
        st.session_state.pop(REPORT_STATE_KEY, None)
        st.warning(f"Could not load inspection report: {error}")
        return
    st.session_state[REPORT_STATE_KEY] = report
    render_inspection_report(st, report)


def report_matches_source(report: dict, inventory: dict, spec: Any) -> bool:
    saved = report["inventory"]
    if saved.get("row_filters", {}) != inventory.get("row_filters", {}):
        return False
    aliases = {"kaggle": "xlsx", "kaggle_text": "text"}
    if aliases.get(saved.get("format"), saved.get("format")) != aliases.get(inventory.get("format"), inventory.get("format")):
        return False
    if inventory.get("format") == "huggingface":
        fields = ("dataset_id", "dataset_config", "dataset_split", "dataset_revision",
                  "filter_column", "filter_value")
        return all(saved.get(key) == inventory.get(key) for key in fields) and (
            sorted(saved.get("dataset_shards") or []) == sorted(inventory.get("dataset_shards") or [])
        )
    try:
        files = spec.files or ((Path(inventory["path"]),) if inventory.get("path") else ())
        return saved.get("source_signatures") == [list(item) for item in file_signatures(files)]
    except OSError:
        return False


def render_language_status(st: Any, *, inventory: dict, spec: Any) -> None:
    st.markdown("#### Language coverage and script")
    report = st.session_state.get(REPORT_STATE_KEY)
    if report and report_matches_source(report, inventory, spec):
        st.caption(f"From the inspection script run at {report.get('created_at', 'Unknown')}")
        st.caption(f"Sample scope: {report.get('sample_scope', 'selected_source_sample')}")
        render_saved_status(st, report["language_status"])
    else:
        st.info("Run scripts/inspect_dataset.py and open its report in the Script results view to display language/script evidence for this exact source selection.")
