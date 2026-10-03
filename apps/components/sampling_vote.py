"""Read-only display of repeated-sampling coverage and stored vote results."""

from typing import Any

from attention_maps.explorer.language_status import SCRIPT_OPTIONS, saved_language_options
from attention_maps.explorer.category_proportions import category_breakdown


def _category_value(status: dict, key: str) -> str:
    allowed = saved_language_options(status) if key == "language_coverage" else SCRIPT_OPTIONS
    return status[key] if status[key] in allowed else "—"


def render_sampling_vote(st: Any, status: dict) -> None:
    sampling = status.get("sampling")
    if sampling is None:
        return
    votes = status["voting"]
    st.caption(
        f"{sampling['runs']} independent {sampling['fraction_requested']:.0%} random runs · "
        f"{sampling['rows_per_run']:,} records per run · all selected text analyzed"
    )
    cards = st.columns(3)
    cards[0].metric("Unique sample coverage", f"{sampling['unique_coverage']:.2%}")
    cards[1].metric("Unique records analyzed", f"{sampling['unique_sampled_records']:,}")
    cards[2].metric("Population scanned", f"{sampling['population_rows']:,}")
    for key, label in (("language_coverage", "Language"), ("script", "Script")):
        vote = votes[key]
        if vote["decision"] == "majority":
            st.caption(f"{label} vote: {vote['winning_votes']}/{vote['runs']} runs agree ({vote['agreement']:.1%}).")
        else:
            st.info(f"{label} vote is inconclusive: no supported category meets the required vote threshold.")
    st.caption(votes["note"])
    with st.expander("Sampling runs and voting details"):
        st.caption(
            f"Population basis: {sampling['population_basis']}. "
            f"Expected unique coverage: {sampling['expected_unique_coverage']:.2%}. "
            "Runs may overlap; coverage counts each source record position once."
        )
        st.dataframe([
            {"Run": run["run"], "Seed": run["seed"], "Records": run["sampled_records"],
             "Characters": run["sampled_characters"], "Language coverage": _category_value(run, "language_coverage"),
             "Language evidence": run["language_basis"], "Script": _category_value(run, "script"),
             "Nepali covered": "Yes" if run.get("nepali_covered") is True else "No" if run.get("nepali_covered") is False else "—",
             "Devanagari %": run.get("script_analysis_percentages", run["script_percentages"]).get("Devanagari", 0),
             "Latin %": run.get("script_analysis_percentages", run["script_percentages"]).get("Latin", 0)}
            for run in sampling["run_results"]
        ], hide_index=True, width="stretch")
        if all("language_category_percentages" in run and "script_category_percentages" in run
               for run in sampling["run_results"]):
            st.caption("Language category % by run · denominator: sampled records in that run")
            language_rows = []
            for run in sampling["run_results"]:
                language_rows.append({"Run": run["run"], **category_breakdown(run, "language")[1],
                                      "No language evidence (records)": run["language_no_evidence_records"],
                                      "Outside listed categories (records)": run["language_outside_categories_records"]})
            st.dataframe(language_rows, hide_index=True, width="stretch")
            st.caption("Nepali script category % by run · denominator: Nepali-eligible records in that run")
            st.dataframe([{"Run": run["run"], **category_breakdown(run, "script")[1],
                           "No script evidence (records)": run["script_no_evidence_records"],
                           "Outside listed categories (records)": run["script_outside_categories_records"]}
                          for run in sampling["run_results"]], hide_index=True, width="stretch")
        st.caption("For current reports, per-run script percentages describe eligible Nepali/English records; raw source percentages remain separate diagnostics.")
        st.json({key: votes[key] for key in ("language_coverage", "script")})
        if status.get("language_evidence_policy"):
            st.json({"Language evidence by run": [
                {"run": run["run"], "sources": run.get("language_evidence_sources", []),
                 "sources_disagree": run.get("language_evidence_conflict", False)}
                for run in sampling["run_results"]
            ]}, expanded=False)
        pooled_language = _category_value({**status, "language_coverage": status["pooled_language_coverage"]}, "language_coverage")
        pooled_script = _category_value({"script": status["pooled_script"]}, "script")
        st.caption(
            f"Pooled unique-sample evidence: {pooled_language} · {pooled_script}. "
            "Script percentages below describe this pool; the final cards show majority votes."
        )
