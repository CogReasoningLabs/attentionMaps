"""Stepwise sampling, normalization, deduplication, and clean-data EDA UI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from attention_maps.common.google_drive import (
    DriveUploadProgress,
    GoogleDriveUploadError,
    upload_path_to_google_drive,
)
from attention_maps.common.pipeline_logging import pipeline_logger
from attention_maps.eda.contracts import AnalysisConfig, SurveyPlan, SurveyRun
from attention_maps.eda.deduplication import DeduplicationConfig
from attention_maps.eda.pipeline import analyze_records
from attention_maps.eda.reporting import write_survey_report
from attention_maps.eda.sampling import fold_seed, plan_repeated_sampling
from attention_maps.eda.workspace import (
    deduplicate_workspace_documents,
    normalize_workspace_sample,
    persist_workspace_artifacts,
    workspace_audit_csv,
    workspace_documents_jsonl,
)
from attention_maps.explorer import dataset_size_bucket, format_bytes

from .common import (
    configured_eda_output_root,
    eda_dataset_spec,
    sample_dataset_rows,
    text_columns,
)

LOGGER = pipeline_logger("workspace")


def render_workspace_tab(*, st: Any, inventory: dict[str, Any], spec: Any) -> None:
    st.markdown("### Dataset cleaning WORKSPACE")
    st.caption(
        "Run each gate in order. The selected source dataset stays immutable; "
        "the workspace creates a sampled, NFC-normalized, deduplicated copy for EDA."
    )
    candidates = text_columns(inventory["schema"])
    if not candidates or int(inventory["rows"]) <= 0:
        st.info("This dataset has no detectable text fields for the workspace.")
        return

    field_columns = st.columns([3, 2])
    selected_text = field_columns[0].multiselect(
        "Workspace text fields",
        candidates,
        default=candidates[:1],
        key=f"workspace-text-v2:{spec.key}",
    )
    source_candidates = [
        value
        for value in inventory["columns"]
        if value.casefold() in {"source", "domain", "url", "dataset", "source_id"}
    ]
    selected_source = field_columns[1].multiselect(
        "Workspace provenance fields",
        inventory["columns"],
        default=source_candidates[:1],
        key=f"workspace-source:{spec.key}",
    )
    population = int(inventory["rows"])
    controls = st.columns(4)
    suggested = min(20.0, max(0.01, 500_000 / population))
    percentage = controls[0].number_input(
        "Workspace population per fold (%)",
        min_value=0.01,
        max_value=100.0,
        value=float(round(suggested, 4)),
        step=0.1,
        format="%.4f",
        key=f"workspace-percentage:{spec.key}",
    )
    folds = controls[1].number_input(
        "Workspace sample folds",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key=f"workspace-folds:{spec.key}",
    )
    seed = controls[2].number_input(
        "Workspace seed",
        min_value=0,
        value=42,
        step=1,
        key=f"workspace-seed:{spec.key}",
    )
    disjoint_folds = controls[3].checkbox(
        "Non-overlapping folds",
        value=True,
        key=f"workspace-disjoint-folds:{spec.key}",
        help=(
            "Recommended for WORKSPACE coverage. Samples rows once, then partitions "
            "them into disjoint batches. Disable for independent repeated samples."
        ),
    )
    sampling = plan_repeated_sampling(
        population,
        float(percentage),
        folds=int(folds),
        max_rows_per_fold=50_000,
        disjoint_folds=bool(disjoint_folds),
    )
    size_bucket = dataset_size_bucket(inventory)
    plan_columns = st.columns(5)
    plan_columns[0].metric("Population", f"{population:,}")
    plan_columns[1].metric("Rows / fold", f"{sampling.rows_per_fold:,}")
    plan_columns[2].metric("Total reads", f"{sampling.total_rows_read:,}")
    plan_columns[3].metric(
        "Unique coverage" if sampling.disjoint_folds else "Expected unique coverage",
        f"{sampling.expected_population_coverage_pct:.1f}%",
    )
    plan_columns[4].metric(
        "Dataset size bucket",
        size_bucket.label.split(" (")[0],
        delta=(
            format_bytes(size_bucket.size_bytes)
            if size_bucket.size_bytes is not None
            else "size unavailable"
        ),
        help=f"Bucketed using {size_bucket.size_basis}: {size_bucket.label}.",
        delta_color="off",
    )
    if sampling.capped:
        st.warning(
            "The requested percentage exceeds the 50,000-row-per-fold UI safety "
            f"limit. The effective fold percentage is "
            f"{sampling.effective_percentage_per_fold:.4g}%."
        )

    state_key = f"cleaning-workspace:{spec.key}"
    state = st.session_state.setdefault(state_key, {})
    if st.button("Reset this dataset workspace", key=f"workspace-reset:{spec.key}"):
        LOGGER.info("WORKSPACE RESET dataset=%s", spec.key)
        st.session_state[state_key] = {}
        state = st.session_state[state_key]

    shared_key = eda_dataset_spec(
        spec, (candidates[0],), (), 1, population
    ).key
    _render_shared_runs(
        st,
        configured_eda_output_root() / "workspace" / shared_key,
    )

    _render_sampling_step(
        st,
        state,
        inventory,
        spec,
        sampling,
        int(seed),
        tuple(selected_text),
        tuple(selected_source),
    )
    _render_normalization_step(st, state, spec)
    _render_deduplication_step(st, state, spec)
    _render_clean_eda_step(st, state, spec)
    _render_drive_upload(st, state, spec)


def _render_shared_runs(st: Any, dataset_workspace: Path) -> None:
    manifests = sorted(
        dataset_workspace.glob("*/workspace_manifest.json"), reverse=True
    )[:20]
    with st.expander(f"Shared team runs ({len(manifests)})"):
        if not manifests:
            st.caption("No completed Step 3 runs for this dataset yet.")
            return
        rows = []
        for path in manifests:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            result = manifest.get("deduplication_result", {})
            rows.append(
                {
                    "created_at_utc": manifest.get("created_at_utc"),
                    "clean_documents": result.get("retained_documents"),
                    "retention_%": result.get("retention_pct"),
                    "path": str(path.parent),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_sampling_step(
    st: Any,
    state: dict[str, Any],
    inventory: dict[str, Any],
    spec: Any,
    sampling: Any,
    seed: int,
    text_fields: tuple[str, ...],
    source_fields: tuple[str, ...],
) -> None:
    st.markdown("#### Step 1 · Sampling")
    completed = bool(state.get("sampling_complete"))
    st.write("✅ Complete" if completed else "⏳ Waiting to run")
    if st.button(
        "Start Step 1 · Sample dataset",
        type="primary",
        disabled=not text_fields,
        key=f"workspace-step1:{spec.key}",
    ):
        progress = st.progress(0, text="Preparing deterministic fold samples…")
        status = st.status("Sampling selected dataset…", expanded=True)
        try:
            LOGGER.info(
                "STEP 1 START dataset=%s population=%d requested_pct=%.4f "
                "effective_pct=%.4f folds=%d rows_per_fold=%d seed=%d disjoint=%s",
                spec.key,
                int(inventory["rows"]),
                sampling.requested_percentage,
                sampling.effective_percentage_per_fold,
                sampling.folds,
                sampling.rows_per_fold,
                seed,
                sampling.disjoint_folds,
            )
            selected_fields = tuple(dict.fromkeys((*text_fields, *source_fields)))
            unique: dict[tuple[str, str], dict[str, Any]] = {}
            actual_reads = 0
            disjoint_rows = (
                sample_dataset_rows(
                    inventory,
                    sampling.total_rows_read,
                    seed,
                    selected_fields,
                )
                if sampling.disjoint_folds
                else None
            )
            for fold_index in range(1, sampling.folds + 1):
                current_seed = fold_seed(seed, fold_index)
                status.write(
                    f"Fold {fold_index}/{sampling.folds} · "
                    + (
                        "disjoint partition"
                        if sampling.disjoint_folds
                        else f"seed {current_seed}"
                    )
                )
                if sampling.disjoint_folds:
                    LOGGER.info(
                        "STEP 1 FOLD START dataset=%s fold=%d/%d mode=disjoint_partition",
                        spec.key,
                        fold_index,
                        sampling.folds,
                    )
                else:
                    LOGGER.info(
                        "STEP 1 FOLD START dataset=%s fold=%d/%d seed=%d",
                        spec.key,
                        fold_index,
                        sampling.folds,
                        current_seed,
                    )
                if disjoint_rows is not None:
                    start = (fold_index - 1) * sampling.rows_per_fold
                    rows = list(disjoint_rows[start : start + sampling.rows_per_fold])
                else:
                    rows = sample_dataset_rows(
                        inventory,
                        sampling.rows_per_fold,
                        current_seed,
                        selected_fields,
                    )
                actual_reads += len(rows)
                for fallback, row in enumerate(rows):
                    stable_identity = row.get(
                        "__viewer_identity_stable",
                        inventory.get("format") != "huggingface",
                    )
                    key = (
                        str(row.get("__viewer_file", "")),
                        (
                            str(row.get("__viewer_row_index", fallback))
                            if stable_identity
                            else f"fold-{fold_index}:position-{fallback}"
                        ),
                    )
                    row["__viewer_workspace_identity"] = key[1]
                    unique.setdefault(key, row)
                progress.progress(
                    round(100 * fold_index / sampling.folds),
                    text=f"Completed sample fold {fold_index}/{sampling.folds}",
                )
                LOGGER.info(
                    "STEP 1 FOLD COMPLETE dataset=%s fold=%d/%d rows_read=%d unique_so_far=%d",
                    spec.key,
                    fold_index,
                    sampling.folds,
                    len(rows),
                    len(unique),
                )
            state.clear()
            state.update(
                {
                    "sample_records": tuple(unique.values()),
                    "sampling_complete": True,
                    "sample_unique_rows": len(unique),
                    "sample_reads": actual_reads,
                    "sample_plan": sampling,
                    "text_fields": text_fields,
                    "source_fields": source_fields,
                    "seed": seed,
                }
            )
            LOGGER.info(
                "STEP 1 COMPLETE dataset=%s reads=%d unique_rows=%d",
                spec.key,
                actual_reads,
                len(unique),
            )
            status.update(label="Step 1 complete", state="complete", expanded=False)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            LOGGER.exception("STEP 1 FAILED dataset=%s", spec.key)
            status.update(label="Step 1 failed", state="error", expanded=True)
            st.error(f"Sampling failed: {error}")
    if state.get("sampling_complete"):
        st.success(
            f"Sample ready: {state['sample_unique_rows']:,} unique rows from "
            f"{state['sample_reads']:,} fold reads."
        )


def _render_normalization_step(st: Any, state: dict[str, Any], spec: Any) -> None:
    st.markdown("#### Step 2 · NFC normalization")
    available = "sample_records" in state
    completed = "normalization" in state
    stage_status = (
        "✅ Complete"
        if completed
        else "🔒 Complete Step 1 first" if not available else "⏳ Ready"
    )
    st.write(stage_status)
    if st.button(
        "Start Step 2 · Normalize NFC",
        type="primary",
        disabled=not available,
        key=f"workspace-step2:{spec.key}",
    ):
        progress = st.progress(0, text="Extracting selected text fields…")
        status = st.status("Applying NFC normalization…", expanded=True)
        try:
            LOGGER.info(
                "STEP 2 START dataset=%s rows=%d normalization=NFC",
                spec.key,
                len(state["sample_records"]),
            )
            last_decile = -1

            def update(stage: str, completed_rows: int, total: int) -> None:
                nonlocal last_decile
                progress.progress(
                    round(100 * completed_rows / max(1, total)),
                    text=f"NFC-normalized {completed_rows:,}/{total:,} rows",
                )
                decile = min(10, (10 * completed_rows) // max(1, total))
                if decile > last_decile:
                    LOGGER.info(
                        "STEP 2 PROGRESS dataset=%s stage=%s rows=%d/%d percent=%d",
                        spec.key,
                        stage,
                        completed_rows,
                        total,
                        decile * 10,
                    )
                    last_decile = decile

            result = normalize_workspace_sample(
                state["sample_records"],
                state["text_fields"],
                state["source_fields"],
                progress=update,
            )
            for downstream_key in (
                "deduplication",
                "dedup_config",
                "workspace_artifacts",
                "workspace_dir",
                "eda_profile",
                "eda_output",
                "drive_upload",
            ):
                state.pop(downstream_key, None)
            state["normalization"] = result
            state["lexical_text_rows"] = sum(
                any(character.isalpha() for character in document.text)
                for document in result.documents
            )
            state.pop("sample_records", None)
            LOGGER.info(
                "STEP 2 COMPLETE dataset=%s sampled_rows=%d documents=%d missing_text=%d",
                spec.key,
                result.sampled_rows,
                result.normalized_rows,
                result.missing_text_rows,
            )
            status.update(label="Step 2 complete", state="complete", expanded=False)
        except (RuntimeError, TypeError, ValueError) as error:
            LOGGER.exception("STEP 2 FAILED dataset=%s", spec.key)
            status.update(label="Step 2 failed", state="error", expanded=True)
            st.error(f"Normalization failed: {error}")
    if "normalization" in state:
        result = state["normalization"]
        columns = st.columns(3)
        columns[0].metric("Sampled rows", f"{result.sampled_rows:,}")
        columns[1].metric("NFC documents", f"{result.normalized_rows:,}")
        columns[2].metric("Missing text", f"{result.missing_text_rows:,}")
        if state.get("lexical_text_rows", 0) == 0:
            st.error(
                "The selected text field contains no lexical words and appears to "
                "be an identifier column. Reset this workspace, select the actual "
                "text/tweet/word field, and rerun Steps 1–2."
            )


def _render_deduplication_step(st: Any, state: dict[str, Any], spec: Any) -> None:
    st.markdown("#### Step 3 · Deduplication layers")
    normalization = state.get("normalization")
    lexical_rows = state.get("lexical_text_rows")
    if lexical_rows is None and normalization is not None:
        lexical_rows = sum(
            any(character.isalpha() for character in document.text)
            for document in normalization.documents
        )
    available = normalization is not None and bool(lexical_rows)
    completed = "deduplication" in state
    stage_status = (
        "✅ Complete"
        if completed
        else (
            "⚠️ Select an actual text field and rerun Steps 1–2"
            if normalization is not None and not lexical_rows
            else "🔒 Complete Step 2 first" if not available else "⏳ Ready"
        )
    )
    st.write(stage_status)
    settings = st.columns(3)
    threshold = settings[0].slider(
        "Workspace MinHash threshold",
        0.60,
        1.0,
        0.80,
        0.05,
        key=f"workspace-threshold:{spec.key}",
    )
    boilerplate_documents = settings[1].number_input(
        "Repeated paragraph documents",
        min_value=2,
        max_value=1_000,
        value=3,
        key=f"workspace-boilerplate:{spec.key}",
    )
    min_paragraph_characters = settings[2].number_input(
        "Minimum paragraph characters",
        min_value=10,
        max_value=1_000,
        value=40,
        key=f"workspace-paragraph-length:{spec.key}",
    )
    st.warning(
        "Step 3 materializes removals only in the workspace copy. MinHash and "
        "boilerplate thresholds are consequential—review the audit and clean sample "
        "before promoting it to training data."
    )
    if st.button(
        "Start Step 3 · Run deduplication",
        type="primary",
        disabled=not available,
        key=f"workspace-step3:{spec.key}",
    ):
        progress = st.progress(0, text="Preparing SHA-256 exact-match pass…")
        status = st.status("Running deduplication layers…", expanded=True)
        try:
            config = DeduplicationConfig(
                normalization="NFC",
                lowercase=True,
                collapse_whitespace=True,
                shingle_size=5,
                minhash_permutations=128,
                minhash_bands=16,
                near_duplicate_threshold=float(threshold),
                boilerplate_min_documents=int(boilerplate_documents),
                boilerplate_min_characters=int(min_paragraph_characters),
                seed=int(state["seed"]),
            )
            LOGGER.info(
                "STEP 3 START dataset=%s documents=%d normalization=NFC threshold=%.2f "
                "boilerplate_min_documents=%d boilerplate_min_characters=%d",
                spec.key,
                len(state["normalization"].documents),
                config.near_duplicate_threshold,
                config.boilerplate_min_documents,
                config.boilerplate_min_characters,
            )
            logged_progress: dict[str, int] = {}

            def update(stage: str, completed_rows: int, total: int) -> None:
                if stage == "document_deduplication":
                    share = 0.8 * completed_rows / max(1, total)
                    label = "Exact SHA-256 + MinHash-LSH"
                else:
                    share = 0.8 + 0.2 * completed_rows / max(1, total)
                    label = "Repeated paragraph removal"
                progress.progress(
                    round(100 * share),
                    text=f"{label}: {completed_rows:,}/{total:,}",
                )
                decile = min(10, (10 * completed_rows) // max(1, total))
                if decile > logged_progress.get(stage, -1):
                    LOGGER.info(
                        "STEP 3 PROGRESS dataset=%s stage=%s rows=%d/%d percent=%d",
                        spec.key,
                        stage,
                        completed_rows,
                        total,
                        decile * 10,
                    )
                    logged_progress[stage] = decile

            result = deduplicate_workspace_documents(
                state["normalization"].documents,
                config,
                progress=update,
            )
            workspace_key = eda_dataset_spec(
                spec,
                ("text",),
                ("source",),
                max(1, len(result.documents)),
                max(1, len(result.documents)),
            ).key
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            workspace_dir = (
                configured_eda_output_root()
                / "workspace"
                / workspace_key
                / run_id
            )
            workspace_artifacts = persist_workspace_artifacts(
                result,
                workspace_dir,
                metadata={
                    "dataset_key": spec.key,
                    "dataset_label": spec.label,
                    "dataset_location": str(spec.location),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "sampling": state["sample_plan"].as_dict(),
                    "normalization": "NFC",
                    "deduplication_config": asdict(config),
                },
            )
            state["deduplication"] = result
            state["dedup_config"] = config
            state["workspace_artifacts"] = workspace_artifacts
            state["workspace_dir"] = workspace_dir
            state.pop("eda_profile", None)
            state.pop("eda_output", None)
            state.pop("drive_upload", None)
            LOGGER.info(
                "STEP 3 COMPLETE dataset=%s input=%d exact_removed=%d near_removed=%d "
                "paragraphs_removed=%d retained=%d output=%s",
                spec.key,
                result.input_documents,
                result.exact_documents_removed,
                result.near_documents_removed,
                result.paragraphs_removed,
                result.retained_documents,
                workspace_dir,
            )
            status.update(label="Step 3 complete", state="complete", expanded=False)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            LOGGER.exception("STEP 3 FAILED dataset=%s", spec.key)
            status.update(label="Step 3 failed", state="error", expanded=True)
            st.error(f"Deduplication failed: {error}")
    if "deduplication" in state:
        result = state["deduplication"]
        metrics = st.columns(6)
        metrics[0].metric("Input", f"{result.input_documents:,}")
        metrics[1].metric("Exact removed", f"{result.exact_documents_removed:,}")
        metrics[2].metric("Near removed", f"{result.near_documents_removed:,}")
        metrics[3].metric("Paragraphs removed", f"{result.paragraphs_removed:,}")
        metrics[4].metric("Clean documents", f"{result.retained_documents:,}")
        metrics[5].metric("Retention", f"{result.retention_pct:.1f}%")
        if state.get("workspace_dir"):
            st.success(f"Shared workspace artifacts: {state['workspace_dir']}")
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download clean workspace JSONL",
            workspace_documents_jsonl(result.documents),
            file_name=f"{spec.key}-clean-workspace.jsonl",
            mime="application/x-ndjson",
            key=f"workspace-clean-download:{spec.key}",
        )
        download_columns[1].download_button(
            "Download deduplication audit CSV",
            workspace_audit_csv(result),
            file_name=f"{spec.key}-deduplication-audit.csv",
            mime="text/csv",
            key=f"workspace-audit-download:{spec.key}",
        )


def _render_drive_upload(st: Any, state: dict[str, Any], spec: Any) -> None:
    with st.expander("Upload completed workspace to Google Drive"):
        st.caption(
            "Uploads the complete Step 3 run folder (clean JSONL, audit CSV, and "
            "manifest) with resumable transfers. Authenticate with ADC/service "
            "account credentials, or let each teammate sign in through a browser."
        )
        completed = bool(state.get("workspace_dir"))
        auth_mode = st.radio(
            "Google Drive authentication",
            ("ADC / service account", "Browser OAuth (local Streamlit only)"),
            horizontal=True,
            key=f"workspace-drive-auth:{spec.key}",
        )
        oauth_client = None
        oauth_token = os.getenv(
            "ATTENTION_MAPS_DRIVE_OAUTH_TOKEN", ".google-drive-token.json"
        )
        if auth_mode.startswith("Browser OAuth"):
            oauth_client = st.text_input(
                "Desktop OAuth client JSON path",
                value=os.getenv("ATTENTION_MAPS_DRIVE_OAUTH_CLIENT", ""),
                key=f"workspace-drive-oauth-client:{spec.key}",
                help=(
                    "The first upload opens Google sign-in on the machine running "
                    "Streamlit. The refresh token remains local and is not uploaded."
                ),
            ).strip()
            oauth_token = st.text_input(
                "Local OAuth token cache path",
                value=oauth_token,
                key=f"workspace-drive-oauth-token:{spec.key}",
            ).strip()
            st.warning(
                "Use browser OAuth only when Streamlit is running on your local "
                "computer. For a remote team server, configure ADC or a service account."
            )
        parent_folder_id = st.text_input(
            "Destination Google Drive folder ID",
            value=os.getenv("ATTENTION_MAPS_DRIVE_PARENT_ID", ""),
            key=f"workspace-drive-parent:{spec.key}",
            help="Share this folder/Shared Drive with the authenticated account.",
        ).strip()
        full_access = st.checkbox(
            "Use full Drive scope",
            value=False,
            key=f"workspace-drive-full-scope:{spec.key}",
            help="Leave off unless drive.file cannot access the team destination.",
        )
        if not completed:
            st.info("Complete Step 3 before uploading a workspace run.")
        if st.button(
            "Upload Step 3 run folder",
            type="primary",
            disabled=(
                not completed
                or not parent_folder_id
                or (auth_mode.startswith("Browser OAuth") and not oauth_client)
            ),
            key=f"workspace-drive-upload:{spec.key}",
        ):
            source = Path(state["workspace_dir"])
            upload_progress = st.progress(0, text="Starting Drive upload…")
            status = st.status("Uploading workspace to Google Drive…", expanded=True)
            last_percent = -1
            try:
                LOGGER.info(
                    "DRIVE UPLOAD REQUEST dataset=%s source=%s full_scope=%s",
                    spec.key,
                    source,
                    full_access,
                )

                def report(update: DriveUploadProgress) -> None:
                    nonlocal last_percent
                    if update.total_bytes:
                        percent = round(
                            100 * update.uploaded_bytes / update.total_bytes
                        )
                    else:
                        percent = round(
                            100 * update.completed_files / max(1, update.total_files)
                        )
                    upload_progress.progress(
                        min(100, percent),
                        text=(
                            f"{update.local_path.name} · files "
                            f"{update.completed_files}/{update.total_files} · "
                            f"{format_bytes(update.uploaded_bytes)}/"
                            f"{format_bytes(update.total_bytes)}"
                        ),
                    )
                    if percent >= last_percent + 10 or percent == 100:
                        LOGGER.info(
                            "DRIVE UPLOAD PROGRESS dataset=%s percent=%d files=%d/%d",
                            spec.key,
                            percent,
                            update.completed_files,
                            update.total_files,
                        )
                        last_percent = percent

                summary = upload_path_to_google_drive(
                    source,
                    parent_folder_id,
                    oauth_client_secrets_file=oauth_client,
                    oauth_token_file=oauth_token,
                    full_drive_access=full_access,
                    progress=report,
                )
                state["drive_upload"] = summary
                upload_progress.progress(100, text="Drive upload complete")
                status.update(
                    label="Google Drive upload complete",
                    state="complete",
                    expanded=False,
                )
            except (GoogleDriveUploadError, OSError, ValueError) as error:
                LOGGER.exception("DRIVE UPLOAD FAILED dataset=%s", spec.key)
                status.update(
                    label="Google Drive upload failed", state="error", expanded=True
                )
                st.error(str(error))
        if "drive_upload" in state:
            summary = state["drive_upload"]
            st.success(
                f"Uploaded {summary.files_uploaded:,} files "
                f"({format_bytes(summary.bytes_uploaded)}) and created "
                f"{summary.folders_created:,} folders."
            )
            st.link_button(
                "Open uploaded workspace in Google Drive",
                f"https://drive.google.com/drive/folders/{summary.root_drive_id}",
            )


def _render_clean_eda_step(st: Any, state: dict[str, Any], spec: Any) -> None:
    st.markdown("#### Step 4 · EDA on preprocessed dataset")
    available = "deduplication" in state and bool(state["deduplication"].documents)
    completed = "eda_profile" in state
    st.write(
        "✅ Complete"
        if completed
        else "🔒 Complete Step 3 first" if not available else "⏳ Ready"
    )
    st.caption(
        "Fixed outputs: corpus profile, document size, text structure, recurring "
        "phrases, residual duplicate audit, and WordCloud. EDA never runs on "
        "the raw source."
    )
    if st.button(
        "Start Step 4 · Run preprocessed EDA",
        type="primary",
        disabled=not available,
        key=f"workspace-eda:{spec.key}",
    ):
        progress = st.progress(5, text="Preparing clean workspace EDA…")
        status = st.status("Analyzing preprocessed data…", expanded=True)
        try:
            documents = state["deduplication"].documents
            base_spec = eda_dataset_spec(
                spec,
                ("text",),
                ("source",),
                len(documents),
                len(documents),
            )
            analysis_spec = replace(base_spec, key=f"{base_spec.key}-clean-workspace")
            config = AnalysisConfig(
                sample_size=len(documents),
                seed=int(state["seed"]),
                dedup_normalization="NFC",
                top_tokens=150,
            )
            LOGGER.info(
                "CLEAN EDA START dataset=%s documents=%d output=%s",
                spec.key,
                len(documents),
                state["workspace_dir"],
            )
            last_decile = -1

            def update(dataset: str, stage: str, completed_rows: int) -> None:
                nonlocal last_decile
                progress.progress(
                    min(80, 5 + round(75 * completed_rows / len(documents))),
                    text=f"Analyzing clean row {completed_rows:,}/{len(documents):,}",
                )
                decile = min(10, (10 * completed_rows) // max(1, len(documents)))
                if decile > last_decile:
                    LOGGER.info(
                        "CLEAN EDA PROGRESS dataset=%s stage=%s rows=%d/%d percent=%d",
                        dataset,
                        stage,
                        completed_rows,
                        len(documents),
                        decile * 10,
                    )
                    last_decile = decile

            profile = analyze_records(
                analysis_spec,
                (document.as_record() for document in documents),
                config,
                progress=update,
            )
            progress.progress(85, text="Writing clean-data figures and tables…")
            output_root = Path(state["workspace_dir"]) / "eda"
            run = SurveyRun(
                SurveyPlan("Clean workspace EDA", (analysis_spec,), config),
                (profile,),
                {},
            )
            write_survey_report(
                run,
                output_root,
                plots=True,
                plot_names=(
                    "profile",
                    "document_size",
                    "text_structure",
                    "ngrams",
                    "deduplication",
                    "wordcloud",
                ),
            )
            state["eda_profile"] = profile
            state["eda_output"] = output_root / analysis_spec.key
            progress.progress(100, text="Preprocessed EDA complete")
            status.update(
                label="Preprocessed EDA complete", state="complete", expanded=False
            )
            LOGGER.info(
                "CLEAN EDA COMPLETE dataset=%s output=%s",
                spec.key,
                state["eda_output"],
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            LOGGER.exception("CLEAN EDA FAILED dataset=%s", spec.key)
            status.update(
                label="Preprocessed EDA failed", state="error", expanded=True
            )
            st.error(f"Preprocessed EDA failed: {error}")
    if "eda_profile" in state:
        summary = state["eda_profile"].summary
        output = Path(state["eda_output"])
        metrics = st.columns(5)
        metrics[0].metric("Clean rows", f"{summary.usable_rows:,}")
        metrics[1].metric("Median words", f"{summary.median_tokens:.1f}")
        metrics[2].metric("Quality pass", f"{summary.quality_pass_ratio_pct:.1f}%")
        metrics[3].metric("Script", summary.script_category)
        metrics[4].metric("Residual duplicates", f"{summary.duplicate_ratio_pct:.1f}%")
        st.markdown("##### Defined EDA visualizations")
        profile_column, size_column = st.columns(2)
        profile_column.image(
            output / "dataset_profile.png",
            caption="Corpus profile: length, script composition, tokens, provenance",
        )
        size_column.image(
            output / "document_size_distribution.png",
            caption="Document character and word-count distributions",
        )
        structure_column, phrase_column = st.columns(2)
        structure_column.image(
            output / "segment_length_kde.png",
            caption="Sentence and line-length structure",
        )
        phrase_column.image(
            output / "top_ngrams.png",
            caption="Most frequent bigrams, trigrams, and four-grams",
        )
        dedup_column, wordcloud_column = st.columns(2)
        dedup_column.image(
            output / "deduplication_stages.png",
            caption="Residual duplicate and boilerplate audit after preprocessing",
        )
        wordcloud_path = output / "wordcloud.png"
        if wordcloud_path.is_file():
            wordcloud_column.image(
                wordcloud_path,
                caption="Frequent meaningful words after fixed stopword filtering",
            )
        else:
            wordcloud_column.info(
                "Rerun Step 4 once to generate the new WordCloud for this older run."
            )
