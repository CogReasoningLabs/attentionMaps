"""Browse saved dataset inspections and remove individual history entries."""

import json
import os
from pathlib import Path

from attention_maps.explorer.inspection_history import (
    DEFAULT_HISTORY_SHEET, delete_history_run, load_history_report, read_history,
)
from attention_maps.explorer.language_status import LANGUAGE_COVERAGE_OPTIONS, SCRIPT_OPTIONS
from attention_maps.explorer.category_proportions import category_breakdown


def _language_distribution(row):
    """Return language category shares and the no-evidence share for a saved run."""
    try:
        status = json.loads(row["Report JSON"])["language_status"]
        total = status["sampled_records"]
        if not isinstance(total, int) or total < 1:
            return {}, None
        percentages = status["language_category_percentages"]
        no_evidence = status["language_no_evidence_records"]
        if not isinstance(percentages, dict) or not isinstance(no_evidence, int) or no_evidence < 0:
            return {}, None
        percentages = category_breakdown(status, "language")[1]
        return percentages, percentages.get("No language evidence")
    except (KeyError, TypeError, ValueError):
        return {}, None


def select_history_report(st):
    sheet = Path(st.sidebar.text_input("Inspection history sheet", value=os.getenv(
        "DATASET_INSPECTION_HISTORY", str(DEFAULT_HISTORY_SHEET)))).expanduser()
    st.subheader("Dataset inspection history")
    st.caption("One row per completed script run. Repeated datasets keep their earlier results and votes.")
    notice_key = f"inspection-history-notice:{sheet.resolve()}"
    if notice := st.session_state.pop(notice_key, None):
        st.success(notice)
    rows = read_history(sheet, allow_additive_fields=True)
    if not rows:
        st.info("No recorded inspections yet. Run scripts/inspect_dataset.py with a dataset; history is saved automatically.")
        return None
    distribution_options = (*LANGUAGE_COVERAGE_OPTIONS, "Other", "Unknown", "Outside listed categories")
    language_columns = tuple(f"{category} %" for category in distribution_options)
    columns = ("Completed at", "Processing seconds", "Provider", "Dataset", "Configuration", "Split", "Selected files", "Row filters", "Rows before language selection", "Rows after language selection", "Eligible instances", "Skipped instances", "Sample scope",
               "Language Coverage", "Script", *language_columns, "No language evidence %", "Nepali script category %",
               "Devanagari %", "Sampling runs", "Unique coverage %")
    display_rows = []
    for row in reversed(rows):
        display = {key: row.get(key, "") for key in columns}
        percentages, no_evidence = _language_distribution(row)
        # Missing historical categories are null, not string values in numeric columns.
        display.update({f"{category} %": percentages.get(category) for category in distribution_options})
        display["No language evidence %"] = no_evidence
        if display["Language Coverage"] not in LANGUAGE_COVERAGE_OPTIONS:
            display["Language Coverage"] = "—"
        if display["Script"] not in SCRIPT_OPTIONS:
            display["Script"] = "—"
        display_rows.append(display)
    st.dataframe(display_rows, hide_index=True, width="stretch")
    st.caption("Language shares include every sampled record: Other means evidence outside the listed categories; Unknown means no accepted evidence. Older runs show these shares under their original Outside listed categories and No language evidence labels. Shares total 100% before rounding. Devanagari % measures letters/marks instead of records.")
    st.download_button("Download inspection history CSV", sheet.read_bytes(), file_name=sheet.name, mime="text/csv")
    choices = {row["Run ID"]: row for row in reversed(rows)}

    def label(run_id):
        row = choices[run_id]
        selection = " / ".join(value for value in (row["Dataset"], row["Configuration"], row["Split"]) if value)
        return f"{row['Completed at']} · {row['Provider']} · {selection} · {run_id[:8]}"

    selected = st.selectbox("Inspection run", list(choices), format_func=label)
    if st.button("Delete selected run", key=f"inspection-history-delete:{sheet.resolve()}:{selected}",
                 help="Remove this inspection entry from the history CSV. Other runs and source dataset files are kept."):
        try:
            deleted = delete_history_run(sheet, selected)
        except (OSError, ValueError) as error:
            st.error(f"Could not delete the selected run: {error}")
        else:
            st.session_state[notice_key] = (f"Deleted inspection run {selected[:8]} from {sheet.name}."
                                          if deleted else "That inspection run has already been removed.")
            st.rerun()
    return load_history_report(sheet, choices[selected])
