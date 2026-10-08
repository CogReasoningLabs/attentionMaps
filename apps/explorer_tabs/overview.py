"""Overview tab renderers for the dataset explorer."""

from __future__ import annotations

from typing import Any

from .selection import render_row_filter_controls

from .common import VIEWER_PREFIX, default_columns, preview_records, render_full_record

def render_details_tab(*, st: Any, inventory: dict[str, Any], spec: Any) -> None:
    st.dataframe(inventory["schema"], width="stretch", hide_index=True)
    with st.expander("Dataset files"):
        displayed_files = "\n".join(str(path) for path in (spec.dataset_shards or spec.files))
        st.code(displayed_files or str(spec.location), language=None)

def render_sample_tab(*, st: Any, inventory: dict[str, Any], spec: Any,
                      cached_sample: Any, cached_filtered_sample: Any, cached_filter_values: Any) -> int:
    controls = st.columns([1, 1, 3])
    sample_size = controls[0].number_input(
        "Records", min_value=1, max_value=20, value=5, step=1
    )
    base_seed = controls[1].number_input(
        "Random seed", min_value=0, value=42, step=1
    )
    selected_columns = controls[2].multiselect(
        "Columns to read",
        inventory["columns"],
        default=default_columns(inventory["columns"]),
        key=f"sample-columns:{spec.key}",
        help="Choose the fields to display. Row filters can use other columns.",
    )
    with st.expander("Row filters"):
        row_filters, missing_values, max_scan_rows = render_row_filter_controls(
            st, inventory, spec.key, cached_filter_values,
        )
        st.caption("To preprocess this selection, open WORKSPACE and choose Use Source sample filters and columns.")
    st.session_state[f"source-sample-selection:{spec.key}"] = {
        "columns": list(selected_columns), "row_filters": row_filters,
        "missing_values": missing_values, "max_scan_rows": max_scan_rows,
    }
    preview_limit = st.slider(
        "Table preview characters per field", 100, 2_000, 400, 100
    )
    
    refresh_key = f"refresh:{spec.key}"
    if refresh_key not in st.session_state:
        st.session_state[refresh_key] = 0
    if st.button("New random sample", type="primary"):
        st.session_state[refresh_key] += 1
    effective_seed = int(base_seed) + st.session_state[refresh_key]
    
    if missing_values:
        st.info(f"Select or enter at least one allowed value for: {', '.join(missing_values)}.")
    elif not selected_columns:
        st.info("Select at least one column.")
    else:
        try:
            with st.spinner("Reading sampled records…"):
                if row_filters:
                    result = cached_filtered_sample(
                        inventory, int(sample_size), effective_seed, tuple(selected_columns),
                        row_filters, int(max_scan_rows),
                    )
                    records = result["records"]
                    st.caption(f"{result['rows_scanned']:,} source rows scanned · "
                               f"{result['rows_matched']:,} matching rows.")
                    if not records:
                        st.info("No matching rows found in the scanned portion. "
                                "Check the exact values or increase Maximum rows to scan.")
                    if result["scan_limit_reached"]:
                        st.caption("Scan limit reached; rows beyond this limit were not searched.")
                else:
                    records = cached_sample(
                        inventory, int(sample_size), effective_seed, tuple(selected_columns),
                    )
        except (OSError, ValueError, ImportError) as error:
            st.error(f"Could not read the sample: {error}")
            records = []
    
        if records:
            sampling_description = (
                "Parquet predicate-filtered streaming sample"
                if inventory.get("filter_column")
                else (
                    "Bounded streaming shuffle"
                    if inventory["format"] == "huggingface"
                    else (
                        "Memory-bounded uniform reservoir sample"
                        if inventory["format"] == "kaggle"
                        else (
                            "Random-offset streaming line sample"
                            if inventory["format"] == "kaggle_text"
                            else "Uniform random sample"
                        )
                    )
                )
            )
            if row_filters:
                sampling_description = "Uniform sample of matching rows in the scanned portion"
            st.caption(
                f"{sampling_description} using effective seed {effective_seed}. "
                "The table is shortened only for display; the full record below is not."
            )
            st.dataframe(
                preview_records(records, preview_limit),
                width="stretch",
                hide_index=True,
            )
            record_index = st.selectbox(
                "Full record",
                range(len(records)),
                format_func=lambda index: (
                    f"Sample {index + 1} · global row "
                    f"{records[index].get(f'{VIEWER_PREFIX}row_index')}"
                ),
            )
            render_full_record(st, records[record_index])
    return int(base_seed)
