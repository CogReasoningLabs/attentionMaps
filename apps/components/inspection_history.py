"""Read-only history sheet selector for script-owned dataset inspections."""

import os
from pathlib import Path

from attention_maps.explorer.inspection_history import DEFAULT_HISTORY_SHEET, load_history_report, read_history
from attention_maps.explorer.language_status import LANGUAGE_COVERAGE_OPTIONS, SCRIPT_OPTIONS


def select_history_report(st):
    sheet = Path(st.sidebar.text_input("Inspection history sheet", value=os.getenv(
        "DATASET_INSPECTION_HISTORY", str(DEFAULT_HISTORY_SHEET)))).expanduser()
    st.subheader("Dataset inspection history")
    st.caption("One row per completed script run. Repeated datasets keep their earlier results and votes.")
    rows = read_history(sheet)
    if not rows:
        st.info("No recorded inspections yet. Run scripts/inspect_dataset.py with a dataset; history is saved automatically.")
        return None
    columns = ("Completed at", "Provider", "Dataset", "Configuration", "Split", "Selected files", "Row filters", "Rows before language selection", "Rows after language selection", "Sample scope",
               "Language Coverage", "Script", "Language category %", "Nepali script category %",
               "Devanagari %", "Sampling runs", "Unique coverage %")
    display_rows = []
    for row in reversed(rows):
        display = {key: row.get(key, "") for key in columns}
        if display["Language Coverage"] not in LANGUAGE_COVERAGE_OPTIONS:
            display["Language Coverage"] = "—"
        if display["Script"] not in SCRIPT_OPTIONS:
            display["Script"] = "—"
        display_rows.append(display)
    st.dataframe(display_rows, hide_index=True, width="stretch")
    st.caption("Devanagari % describes the pooled unique sample's letters/marks, not the percentage of dataset records. Metadata-only runs have no sampling votes.")
    st.download_button("Download inspection history CSV", sheet.read_bytes(), file_name=sheet.name, mime="text/csv")
    choices = {row["Run ID"]: row for row in reversed(rows)}

    def label(run_id):
        row = choices[run_id]
        selection = " / ".join(value for value in (row["Dataset"], row["Configuration"], row["Split"]) if value)
        return f"{row['Completed at']} · {row['Provider']} · {selection} · {run_id[:8]}"

    selected = st.selectbox("Inspection run", list(choices), format_func=label)
    return load_history_report(sheet, choices[selected])
