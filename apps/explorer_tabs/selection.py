"""Shared exact-value filter controls for previews and preprocessing."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from attention_maps.explorer.inspection import project_inventory
from attention_maps.explorer.sample_filters import discover_filter_values
from attention_maps.explorer.workspace_selection import materialize_workspace_selection


def render_row_filter_controls(st, inventory, spec_key, cached_filter_values, *, prefix="sample", label_prefix=""):
    row_filters = {}
    missing_values = []
    filter_columns = st.multiselect(
        f"{label_prefix}Filter rows by", inventory["columns"], key=f"{prefix}-filter-columns:{spec_key}",
        disabled=inventory["format"] in {"text", "kaggle_text"},
        help="Choose metadata columns, such as language. Plain text files have no metadata columns.",
    )
    st.caption(
        "Select the values stored in your dataset; matching includes case. "
        "A row must match every selected column and any allowed value within each column. "
        "Filters run before column selection."
    )
    max_scan_rows = st.number_input(
        f"{label_prefix}Maximum rows to scan", min_value=1, max_value=10_000_000,
        value=None if f"{prefix}-filter-limit:{spec_key}" in st.session_state else 100_000, step=1_000,
        key=f"{prefix}-filter-limit:{spec_key}",
        disabled=not filter_columns and not (prefix == "workspace" and (
            inventory.get("rows") is None or inventory.get("row_filters") or inventory.get("filter_column")
        )),
        help="Available values and filtered samples use this prefix of the selected source. "
             "Increase the limit to find values farther into the dataset.",
    )
    discovered = {"counts": {}}
    if filter_columns:
        try:
            with st.spinner("Finding available filter values…"):
                discovered = cached_filter_values(inventory, tuple(filter_columns), int(max_scan_rows))
            st.caption(f"Available values observed in {discovered['rows_scanned']:,} source rows. "
                       "Counts are per column, before applying these row filters.")
            if discovered["scan_limit_reached"]:
                st.caption("More values may exist beyond the scan limit. Increase Maximum rows to scan to look farther.")
            if discovered["truncated_columns"]:
                st.caption("Showing the first 200 distinct values for: "
                           + ", ".join(discovered["truncated_columns"]) + ". Other values can be entered below.")
        except (OSError, ValueError, ImportError) as error:
            st.warning(f"Could not list available values: {error}. You can enter exact values below.")
    for column in filter_columns:
        counts = discovered["counts"].get(column, {})
        value_key = f"{prefix}-filter-options:{spec_key}:{column}"
        # Keep selections active if a smaller scan no longer observes them.
        options = list(dict.fromkeys([*counts, *st.session_state.get(value_key, [])]))
        selected_values = st.multiselect(
            f"{label_prefix}Available values for {column}", options, key=value_key,
            format_func=lambda value, counts=counts: (
                f"{value} · {counts[value]:,} rows" if value in counts else f"{value} · not observed in this scan"
            ),
            help="Choose one or more values. Counts refer to scanned rows, not the full dataset.",
        )
        if not counts:
            st.caption(f"No selectable text values found for {column} in this scan.")
        raw_values = st.text_area(
            f"{label_prefix}Allowed values for {column}", key=f"{prefix}-filter-values:{spec_key}:{column}",
            placeholder="Optional: values not listed above, one per line", height=68,
            help="Additional exact values to allow alongside your dropdown selections. Leave blank to use only selected values.",
        )
        values = list(dict.fromkeys([*selected_values, *(value for value in raw_values.splitlines() if value.strip())]))
        if values:
            row_filters[column] = values
        else:
            missing_values.append(column)
    if max_scan_rows is None:
        st.info("Enter a maximum row count to scan.")
        missing_values.append("Maximum rows to scan")
    return row_filters, missing_values, int(max_scan_rows or 100_000)


def render_workspace_selection(st, inventory, spec, output_root, cached_filter_values=None):
    """Return a selected inventory only when its current predicates were applied."""
    cached_filter_values = cached_filter_values or st.cache_data(show_spinner=False)(discover_filter_values)
    with st.expander("Workspace row and column filters", expanded=True):
        st.caption(
            "The sidebar configuration, split and shards define the source. "
            "These filters constrain every workspace step, starting with sampling. "
            "For a language-specific split, choose that split and leave row filters empty."
        )
        preview = st.session_state.get(f"source-sample-selection:{spec.key}")
        if st.button("Use Source sample filters and columns", key=f"workspace-import-selection:{spec.key}",
                     disabled=not preview or bool(preview.get("missing_values")) or not preview.get("columns")):
            st.session_state[f"workspace-columns:{spec.key}"] = preview["columns"]
            st.session_state[f"workspace-filter-columns:{spec.key}"] = list(preview["row_filters"])
            st.session_state[f"workspace-filter-limit:{spec.key}"] = preview["max_scan_rows"]
            st.session_state[f"workspace-scan-all:{spec.key}"] = False
            for column, values in preview["row_filters"].items():
                st.session_state[f"workspace-filter-options:{spec.key}:{column}"] = values
                st.session_state[f"workspace-filter-values:{spec.key}:{column}"] = ""
        columns = st.multiselect(
            "Workspace columns", inventory["columns"],
            default=None if f"workspace-columns:{spec.key}" in st.session_state else inventory["columns"],
            key=f"workspace-columns:{spec.key}",
            help="Keep the text and metadata fields you need. A row filter can use a column excluded here.",
        )
        filters, missing, limit = render_row_filter_controls(
            st, inventory, spec.key, cached_filter_values, prefix="workspace", label_prefix="Workspace · ",
        )
        needs_scan = bool(filters or inventory.get("row_filters") or inventory.get("filter_column")
                          or inventory.get("rows") is None)
        scan_all = st.checkbox(
            "Scan entire selected source", value=False, key=f"workspace-scan-all:{spec.key}",
            disabled=not needs_scan,
            help="Read the entire selected split/shards to count and save all matches. Otherwise use the scan limit above.",
        )
        settings = {"columns": columns, "row_filters": filters,
                    "max_scan_rows": None if scan_all or not needs_scan else limit}
        fingerprint = hashlib.sha256(json.dumps(
            {"inventory": inventory, "selection": settings, "missing": missing},
            sort_keys=True, default=str,
        ).encode()).hexdigest()
        if missing or not columns:
            st.info("Select workspace columns and at least one value for every row-filter column.")
            return None, fingerprint
        if not needs_scan:
            selected = project_inventory(inventory, columns)
            selected["workspace_selection"] = {
                **{key: inventory[key] for key in (
                    "dataset_id", "dataset_config", "dataset_split", "dataset_revision", "dataset_shards",
                ) if key in inventory},
                **settings, "source_population_rows": inventory["rows"], "matching_rows": inventory["rows"],
                "source_format": inventory["format"], "population_scope": "selected source configuration/split/shards",
                "rows_scanned": None, "scan_limit_reached": False,
            }
            st.caption(f"All {inventory['rows']:,} rows in the selected source · {len(columns)} columns available to the workspace.")
            return selected, fingerprint
        selection_key = f"workspace-applied-selection:{spec.key}"
        applied = st.session_state.get(selection_key)
        if st.button("Apply workspace filters", key=f"workspace-apply-selection:{spec.key}", type="primary"):
            try:
                with st.status("Scanning and saving matching rows…", expanded=True) as status:
                    destination = (Path(output_root) / "workspace_selections" / fingerprint[:16]
                                   / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
                    selected = materialize_workspace_selection(
                        inventory, columns, filters, destination, max_scan_rows=settings["max_scan_rows"],
                        progress=lambda scanned, matched: status.update(
                            label=f"Scanned {scanned:,} rows · {matched:,} matches"),
                    )
                    applied = {"fingerprint": fingerprint, "inventory": selected}
                    st.session_state[selection_key] = applied
                    status.update(label="Workspace selection ready", state="complete", expanded=False)
            except (OSError, ValueError, RuntimeError, TypeError, ImportError) as error:
                st.session_state.pop(selection_key, None)
                applied = None
                st.error(f"Could not apply workspace filters: {error}")
        if not applied or applied["fingerprint"] != fingerprint or not Path(applied["inventory"]["path"]).is_file():
            st.info("Apply workspace filters to count matching rows before Step 1. Population percentages will use those matches.")
            return None, fingerprint
        selected = applied["inventory"]
        report = selected["workspace_selection"]
        st.caption(f"{report['rows_scanned']:,} source rows scanned · {report['matching_rows']:,} matching rows · {len(columns)} columns.")
        if report["scan_limit_reached"]:
            st.warning("Population and sampling percentages cover matching rows in the scanned prefix only. "
                       "Select Scan entire selected source and apply again for full-source filtering.")
        if not selected["rows"]:
            st.info("No rows match this selection. Change the values or scan more rows, then apply again.")
            return None, fingerprint
        # A fresh application is a new immutable snapshot even with identical predicates.
        return selected, fingerprint + str(selected["path"])
