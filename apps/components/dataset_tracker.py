"""Human-reviewed tracker annotations for the currently explored dataset."""

import csv
from pathlib import Path

from attention_maps.datasets.schemas import DATASET_ROLES, STANDARD_DATASET_SCHEMAS, parse_dataset_roles
from attention_maps.explorer.dataset_tracker import (
    DEFAULT_TRACKER, ROLE_COLUMN, TASK_COLUMN, PRIMARY_TASKS, ACQUISITION_METHODS,
    format_role_cell, matching_tracker_rows, read_tracker, update_tracker_cells,
)


def _choice(st, label, suggestions, current, key):
    options = list(dict.fromkeys([*suggestions, *([current] if current else [])]))
    return st.selectbox(label, options, index=options.index(current) if current else None,
                        key=key, accept_new_options=True,
                        placeholder="Choose a value or type your own") or ""


def _render_role_guide(st):
    schemas = {schema.key: schema for schema in STANDARD_DATASET_SCHEMAS}
    with st.expander("Role and instance-schema guide"):
        st.dataframe([
            {"Dataset role": role.label, "Target instance fields": ", ".join(schemas[role.training_schema].required_fields),
             "Use when": role.description}
            for role in DATASET_ROLES
        ], hide_index=True, width="stretch")
        st.caption(
            "Roles describe intended use and may overlap. For domain-specific conversations, select "
            "both Instruction SFT and Domain-specific fine-tuning; unlabelled domain documents can be Pretraining. "
            "Traditional NLP may contain token labels, spans or sentence pairs that need task-specific mapping. "
            "Saving these annotations updates the tracker; it does not convert source records into a training schema."
        )


def render_dataset_tracker(st, spec, *, tracker_path=DEFAULT_TRACKER):
    """Return the saved role context; unsaved form edits never affect processing."""
    st.markdown("### Dataset tracker annotations")
    st.caption("Inspect Source sample, choose the tracker row below, then save the columns you have reviewed.")
    path = Path(tracker_path).expanduser().resolve()
    st.caption(f"Tracker CSV: {path.name}")
    snapshot_key = f"tracker-snapshot:{path}"
    version_key = f"tracker-form-version:{path}"
    if st.button("Reload tracker CSV", key=f"tracker-reload:{spec.key}"):
        st.session_state.pop(snapshot_key, None)
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
    try:
        if snapshot_key not in st.session_state:
            st.session_state[snapshot_key] = read_tracker(path)
        sheet = st.session_state[snapshot_key]
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        st.info(f"Tracker unavailable: {error}")
        return None
    matches = matching_tracker_rows(sheet, spec)
    options = list(dict.fromkeys([*matches, *sheet.dataset_rows]))
    index = options.index(matches[0]) if len(matches) == 1 else None
    row_index = st.selectbox(
        "Tracker dataset row", options, index=index,
        key=f"tracker-row:{spec.key}:{st.session_state.get(version_key, 0)}",
        placeholder="Choose the dataset row to update",
        format_func=lambda row: (
            f"{sheet.cell(row, 'Dataset Name')} · sheet row {row + 1} · {sheet.cell(row, 'URL(s)')}"
        ),
        help="Exact source URLs suggest a row. Names alone are not used to match datasets.",
    )
    if len(matches) > 1:
        st.info("This source URL appears in multiple tracker rows. Choose the intended dataset entry explicitly.")
    elif not matches:
        st.caption("No exact source URL match. Select the corresponding tracker row manually if this is a local copy or an alternate URL.")
    _render_role_guide(st)
    if row_index is None:
        return None
    st.caption(
        f"Editing {sheet.cell(row_index, 'Dataset Name')} (sheet row {row_index + 1}). "
        "This annotation belongs to the tracker entry, not just the current sample or language filter."
    )
    if row_index not in matches:
        st.caption("The selected tracker URL differs from the loaded source. Check that this is the intended entry before saving.")
    columns = [ROLE_COLUMN, TASK_COLUMN]
    if sheet.acquisition_column:
        columns.append(sheet.acquisition_column)
    else:
        st.info("No acquisition/generation-method column found. Add Data Acquisition Method to your sheet and reload to edit it here.")
    labels = {ROLE_COLUMN: "Dataset role", TASK_COLUMN: "Primary task",
              sheet.acquisition_column: "Data Acquisition Method"}
    if sheet.acquisition_column:
        st.caption(f"Data Acquisition Method saves to the existing “{sheet.acquisition_column}” column.")
    context = f"{spec.key}:{row_index}:{st.session_state.get(version_key, 0)}"
    selected = st.multiselect("Tracker columns to update", columns, default=columns,
                              format_func=lambda column: labels[column], key=f"tracker-columns:{context}")
    roles = {role.key: role for role in DATASET_ROLES}
    with st.form(f"tracker-edit:{context}"):
        updates = {}
        if ROLE_COLUMN in selected:
            current_roles = parse_dataset_roles(sheet.cell(row_index, ROLE_COLUMN))
            selected_roles = st.multiselect(
                "Dataset role(s)", list(dict.fromkeys([*roles, *current_roles])), default=list(current_roles),
                format_func=lambda key: roles[key].label if key in roles else f"Unrecognized: {key}",
                key=f"tracker-roles:{context}",
                help="Choose one or more intended uses. Leave empty to mark this entry as unclassified.",
            )
        if TASK_COLUMN in selected:
            updates[TASK_COLUMN] = _choice(st, "Primary task", PRIMARY_TASKS, sheet.cell(row_index, TASK_COLUMN), f"tracker-task:{context}")
        if sheet.acquisition_column in selected:
            updates[sheet.acquisition_column] = _choice(
                st, "Data Acquisition Method", ACQUISITION_METHODS, sheet.cell(row_index, sheet.acquisition_column),
                f"tracker-acquisition:{context}",
            )
        submitted = st.form_submit_button("Save selected columns to tracker CSV", type="primary", disabled=not selected)
    if submitted:
        try:
            if ROLE_COLUMN in selected:
                updates[ROLE_COLUMN] = format_role_cell(selected_roles)
            sheet = update_tracker_cells(sheet, row_index, updates)
            st.session_state[snapshot_key] = sheet
            st.success(f"Saved {', '.join(labels[column] for column in updates)} for {sheet.cell(row_index, 'Dataset Name')}.")
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            st.error(f"Tracker was not updated: {error}")
    st.dataframe([{"CSV column": column, "Saved value": sheet.cell(row_index, column)} for column in columns],
                 hide_index=True, width="stretch")
    st.download_button("Download tracker CSV", sheet.raw, file_name=path.name, mime="text/csv",
                       key=f"tracker-download:{spec.key}")
    return {
        "identity": (str(path), row_index, sheet.cell(row_index, "Dataset Name"), sheet.cell(row_index, "URL(s)")),
        "roles": tuple(key for key in parse_dataset_roles(sheet.cell(row_index, ROLE_COLUMN)) if key in roles),
        "role_value": sheet.cell(row_index, ROLE_COLUMN),
    }


def sync_saved_tracker_role(st, spec, context):
    """Seed the processing role when the selected row or its saved roles change."""
    if context is None:
        return
    marker = f"tracker-workspace-role:{spec.key}"
    signature = (*context["identity"], context["role_value"])
    if st.session_state.get(marker) != signature:
        st.session_state[f"workspace-schema:{spec.key}"] = context["roles"][0] if len(context["roles"]) == 1 else None
        st.session_state[marker] = signature
