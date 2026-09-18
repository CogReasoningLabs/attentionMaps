"""Interactive D2 pruning for task-specific clean workspace datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from attention_maps.datasets import (
    STANDARD_TRAINING_SCHEMAS,
    TASK_SPECIFIC_SUPERVISED_SCHEMA,
    infer_training_schema,
)
from attention_maps.pruning import D2PruningConfig
from attention_maps.pruning.workspace import (
    difficulty_from_confidence_artifact,
    load_aligned_npz,
    run_workspace_d2,
)


USE_CASE_LABELS = {
    "traditional_nlp": "Traditional NLP task",
    "domain_specific_finetuning": "Domain-specific fine-tuning",
}
DIFFICULTY_CONFIDENCE = "Correct-label confidence history (.npz)"
DIFFICULTY_SCORES = "Precomputed difficulty scores (.npz)"
DIFFICULTY_UNIFORM = "Uniform diagnostic ablation"
EMBEDDINGS_MODEL = "Generate XLM-R embeddings"
EMBEDDINGS_UPLOAD = "Upload precomputed embeddings (.npz)"


def render_d2_pruning_step(*, st: Any, state: dict[str, Any], spec: Any) -> None:
    st.markdown("#### Step 4 · D2 supervised data pruning")
    schema_keys = [None, *(schema.key for schema in STANDARD_TRAINING_SCHEMAS)]
    labels = {schema.key: schema.label for schema in STANDARD_TRAINING_SCHEMAS}
    inferred = infer_training_schema(spec.primary_purpose)
    selected_schema = st.selectbox(
        "Canonical training-data schema",
        schema_keys,
        index=schema_keys.index(inferred) if inferred in schema_keys else 0,
        format_func=lambda key: labels.get(key, "Not classified"),
        key=f"d2-schema:{spec.key}",
        help=(
            "The project defines four standard schemas. D2 is currently enabled "
            "only for task-specific supervised data."
        ),
    )
    st.caption(
        "Schema contract: docs/standard-training-data-schemas.md. The source and "
        "full deduplicated workspace remain unchanged."
    )
    if selected_schema != TASK_SPECIFIC_SUPERVISED_SCHEMA:
        st.info(
            "D2 is intentionally unavailable for this schema. Select "
            "Task-specific supervised only when the records are labelled "
            "traditional-NLP or domain-specific training examples."
        )
        return
    if _is_protected_split(spec):
        st.error("D2 cannot run on validation or test datasets.")
        return
    confirmed_train = st.checkbox(
        "I confirm this workspace contains training-split records only",
        value=_is_known_train_split(spec),
        key=f"d2-train-confirmation:{spec.key}",
        help="D2 must never use validation or test examples.",
    )

    use_case = st.radio(
        "Approved D2 use case",
        tuple(USE_CASE_LABELS),
        format_func=USE_CASE_LABELS.get,
        horizontal=True,
        key=f"d2-use-case:{spec.key}",
    )
    deduplication = state.get("deduplication")
    documents = tuple(deduplication.documents) if deduplication else ()
    available = len(documents) >= 2
    st.write(
        "⏳ Ready"
        if available
        else "🔒 Complete Step 3 with at least two clean documents first"
    )

    settings = st.columns(4)
    retention = settings[0].slider(
        "D2 retention (%)",
        5,
        100,
        50,
        5,
        key=f"d2-retention:{spec.key}",
    )
    max_neighbors = max(1, min(50, len(documents) - 1))
    neighbors = settings[1].number_input(
        "Nearest neighbors (k)",
        min_value=1,
        max_value=max_neighbors,
        value=min(10, max_neighbors),
        key=f"d2-neighbors:{spec.key}",
    )
    backend = settings[2].selectbox(
        "Graph backend",
        ("auto", "exact", "faiss"),
        key=f"d2-backend:{spec.key}",
    )
    device = settings[3].selectbox(
        "Embedding device",
        ("auto", "cuda", "cpu"),
        key=f"d2-device:{spec.key}",
    )

    labels_complete = bool(documents) and all(document.label for document in documents)
    label_balanced = st.checkbox(
        "Preserve proportional class budgets",
        value=labels_complete,
        disabled=not labels_complete,
        key=f"d2-label-balanced:{spec.key}",
        help=(
            "Select a Workspace label field before Step 1. Every retained row "
            "must have a label."
        ),
    )
    if documents and not labels_complete:
        st.warning(
            "No complete label column is present. D2 can run unbalanced, but the "
            "recommended supervised workflow preserves class proportions."
        )

    difficulty_mode = st.radio(
        "Difficulty input",
        (DIFFICULTY_CONFIDENCE, DIFFICULTY_SCORES, DIFFICULTY_UNIFORM),
        key=f"d2-difficulty:{spec.key}",
        help=(
            "Confidence history is an epochs × documents matrix containing the "
            "model probability of each record's correct label."
        ),
    )
    difficulty_upload = None
    if difficulty_mode != DIFFICULTY_UNIFORM:
        expected = (
            "confidences + doc_ids"
            if difficulty_mode == DIFFICULTY_CONFIDENCE
            else "scores + doc_ids"
        )
        difficulty_upload = st.file_uploader(
            f"Difficulty artifact ({expected})",
            type=("npz",),
            key=f"d2-difficulty-upload:{spec.key}:{difficulty_mode}",
        )

    embedding_mode = st.radio(
        "Embedding input",
        (EMBEDDINGS_MODEL, EMBEDDINGS_UPLOAD),
        horizontal=True,
        key=f"d2-embedding-mode:{spec.key}",
    )
    embedding_upload = None
    model_id = "FacebookAI/xlm-roberta-base"
    revision = "main"
    pooling = "cls"
    batch_size = 32
    max_length = 256
    if embedding_mode == EMBEDDINGS_UPLOAD:
        embedding_upload = st.file_uploader(
            "Embedding artifact (embeddings + doc_ids)",
            type=("npz",),
            key=f"d2-embedding-upload:{spec.key}",
        )
    else:
        with st.expander("Embedding model settings"):
            model_fields = st.columns(3)
            model_id = model_fields[0].text_input(
                "Embedding model",
                value=model_id,
                key=f"d2-model:{spec.key}",
            )
            revision = model_fields[1].text_input(
                "Model revision",
                value=revision,
                key=f"d2-revision:{spec.key}",
                help="Pin a model commit for recorded experiments.",
            )
            pooling = model_fields[2].selectbox(
                "Pooling", ("cls", "mean"), key=f"d2-pooling:{spec.key}"
            )
            limits = st.columns(2)
            batch_size = limits[0].number_input(
                "Embedding batch size",
                min_value=1,
                max_value=512,
                value=32,
                key=f"d2-batch:{spec.key}",
            )
            max_length = limits[1].number_input(
                "Maximum tokens",
                min_value=16,
                max_value=4096,
                value=256,
                step=16,
                key=f"d2-max-length:{spec.key}",
            )

    with st.expander("D2 message-passing settings"):
        gamma_fields = st.columns(2)
        gamma_forward = gamma_fields[0].number_input(
            "Forward gamma",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key=f"d2-gamma-forward:{spec.key}",
        )
        gamma_reverse = gamma_fields[1].number_input(
            "Reverse gamma",
            min_value=0.0,
            value=0.8,
            step=0.1,
            key=f"d2-gamma-reverse:{spec.key}",
        )

    missing_input = (
        not confirmed_train
        or (difficulty_mode != DIFFICULTY_UNIFORM and difficulty_upload is None)
        or (embedding_mode == EMBEDDINGS_UPLOAD and embedding_upload is None)
    )
    if st.button(
        "Start Step 4 · Run D2 pruning",
        type="primary",
        disabled=not available or missing_input,
        key=f"workspace-d2:{spec.key}",
    ):
        _run_d2(
            st=st,
            state=state,
            documents=documents,
            use_case=use_case,
            retention=retention,
            neighbors=int(neighbors),
            backend=backend,
            device=device,
            label_balanced=bool(label_balanced),
            difficulty_mode=difficulty_mode,
            difficulty_upload=difficulty_upload,
            embedding_mode=embedding_mode,
            embedding_upload=embedding_upload,
            model_id=model_id,
            revision=revision,
            pooling=pooling,
            batch_size=int(batch_size),
            max_length=int(max_length),
            gamma_forward=float(gamma_forward),
            gamma_reverse=float(gamma_reverse),
        )
    if "d2_result" in state:
        _render_result(st, state["d2_result"])


def _run_d2(**values: Any) -> None:
    st = values["st"]
    state = values["state"]
    documents = values["documents"]
    progress_bar = st.progress(0, text="Preparing D2 inputs…")
    status = st.status("Running D2 pruning…", expanded=True)
    try:
        document_ids = tuple(document.row_id for document in documents)
        difficulty_mode = values["difficulty_mode"]
        difficulty_payload = None
        if difficulty_mode == DIFFICULTY_CONFIDENCE:
            difficulty_payload = values["difficulty_upload"].getvalue()
            difficulty = difficulty_from_confidence_artifact(
                difficulty_payload, document_ids
            )
        elif difficulty_mode == DIFFICULTY_SCORES:
            difficulty_payload = values["difficulty_upload"].getvalue()
            difficulty = load_aligned_npz(
                difficulty_payload, "scores", document_ids
            )
        else:
            difficulty = np.ones(len(documents), dtype=np.float64)
        embeddings = None
        embedding_payload = None
        if values["embedding_mode"] == EMBEDDINGS_UPLOAD:
            embedding_payload = values["embedding_upload"].getvalue()
            embeddings = load_aligned_npz(
                embedding_payload, "embeddings", document_ids
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output_dir = Path(state["workspace_dir"]) / "d2" / run_id
        config = D2PruningConfig(
            enabled=True,
            use_case=values["use_case"],
            retention_fraction=values["retention"] / 100,
            n_neighbors=values["neighbors"],
            gamma_forward=values["gamma_forward"],
            gamma_reverse=values["gamma_reverse"],
            graph_backend=values["backend"],
            label_balanced=values["label_balanced"],
            embedding_model_id=values["model_id"],
            embedding_revision=values["revision"],
            embedding_pooling=values["pooling"],
            embedding_batch_size=values["batch_size"],
            embedding_max_length=values["max_length"],
            embedding_device=values["device"],
            embeddings_path=(
                output_dir / "embeddings_input.npz"
                if embedding_payload is not None
                else None
            ),
            difficulty_mode=(
                "uniform" if difficulty_mode == DIFFICULTY_UNIFORM else "external"
            ),
            difficulty_scores_path=(
                output_dir / "difficulty_input.npz"
                if difficulty_mode != DIFFICULTY_UNIFORM
                else None
            ),
        )
        stage_labels = {
            "embeddings": "Generating transformer embeddings",
            "graph": "Building nearest-neighbor graph",
            "forward": "Forward message passing",
            "selection": "Reverse-message selection",
            "label_groups": "Selecting within label groups",
            "artifacts": "Writing coreset and audit artifacts",
        }
        stage_ranges = {
            "embeddings": (0.0, 0.40),
            "graph": (0.40, 0.68),
            "forward": (0.68, 0.78),
            "selection": (0.78, 0.95),
            "label_groups": (0.40, 0.95),
            "artifacts": (0.95, 1.0),
        }
        active_stage = None

        def update(stage: str, completed: int, total: int) -> None:
            nonlocal active_stage
            start, end = stage_ranges.get(stage, (0.0, 1.0))
            share = completed / max(1, total)
            percent = round(100 * (start + (end - start) * share))
            label = stage_labels.get(stage, stage.replace("_", " ").title())
            progress_bar.progress(
                min(100, percent),
                text=f"{label}: {completed:,}/{total:,}",
            )
            if stage != active_stage:
                status.write(label)
                active_stage = stage

        result = run_workspace_d2(
            documents,
            config,
            output_dir,
            difficulty,
            embeddings=embeddings,
            difficulty_artifact=difficulty_payload,
            embeddings_artifact=embedding_payload,
            progress=update,
        )
        state["d2_result"] = result
        state["d2_output"] = output_dir
        progress_bar.progress(100, text="D2 pruning complete")
        status.update(label="D2 pruning complete", state="complete", expanded=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        status.update(label="D2 pruning failed", state="error", expanded=True)
        st.error(f"D2 pruning failed: {error}")


def _render_result(st: Any, result: Any) -> None:
    metrics = st.columns(4)
    metrics[0].metric("D2 input", f"{result.source_documents:,}")
    metrics[1].metric("Selected", f"{len(result.selected_documents):,}")
    metrics[2].metric("Retention", f"{result.retention_pct:.1f}%")
    metrics[3].metric("Graph backend", result.selection.backend)
    st.success(f"D2 artifacts: {result.output_dir}")

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    pruned = [point for point in result.projection if not point.selected]
    selected = [point for point in result.projection if point.selected]
    if pruned:
        axis.scatter(
            [point.x for point in pruned],
            [point.y for point in pruned],
            color="#aeb6bf",
            alpha=0.45,
            s=18,
            label="Pruned",
        )
    axis.scatter(
        [point.x for point in selected],
        [point.y for point in selected],
        color="#e67e22",
        alpha=0.9,
        s=28,
        label="Selected",
    )
    axis.set_title("D2 retained versus pruned embedding projection")
    axis.set_xlabel("Randomized PCA component 1")
    axis.set_ylabel("Randomized PCA component 2")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.caption(
        "The projection is diagnostic only; D2 operates on the complete embedding "
        "vectors, not these two displayed dimensions."
    )

    if all(document.label for document in result.selected_documents):
        input_counts: dict[str, int] = {}
        selected_counts: dict[str, int] = {}
        for point in result.projection:
            input_counts[point.label] = input_counts.get(point.label, 0) + 1
            if point.selected:
                selected_counts[point.label] = selected_counts.get(point.label, 0) + 1
        st.dataframe(
            [
                {
                    "label": label,
                    "input": count,
                    "selected": selected_counts.get(label, 0),
                    "retention_%": round(
                        100 * selected_counts.get(label, 0) / count, 2
                    ),
                }
                for label, count in sorted(input_counts.items())
            ],
            width="stretch",
            hide_index=True,
        )
    downloads = st.columns(3)
    downloads[0].download_button(
        "Download D2 coreset JSONL",
        result.coreset_path.read_bytes(),
        file_name="d2_coreset.jsonl",
        mime="application/x-ndjson",
    )
    downloads[1].download_button(
        "Download D2 scores CSV",
        result.scores_path.read_bytes(),
        file_name="d2_scores.csv",
        mime="text/csv",
    )
    downloads[2].download_button(
        "Download D2 manifest",
        result.manifest_path.read_bytes(),
        file_name="d2_manifest.json",
        mime="application/json",
    )


def _is_protected_split(spec: Any) -> bool:
    split = str(spec.dataset_split or "").casefold()
    if not split:
        tail = str(spec.key).rsplit(":", 1)[-1].casefold()
        split = tail if tail in {"train", "validation", "test", "dev"} else ""
    return split in {"validation", "test", "dev", "devtest"}


def _is_known_train_split(spec: Any) -> bool:
    split = str(spec.dataset_split or "").casefold()
    if not split:
        tail = str(spec.key).rsplit(":", 1)[-1].casefold()
        split = tail if tail in {"train", "validation", "test", "dev"} else ""
    return split == "train"
