"""Report-only within-dataset deduplication and cross-dataset overlap UI."""

from __future__ import annotations

import csv
import io
from typing import Any, Sequence

from attention_maps.eda.deduplication import DeduplicationConfig
from attention_maps.eda.overlap import DatasetOverlapAnalysis, analyze_dataset_overlap

from .common import extract_text, text_columns


_MAX_TOTAL_SAMPLED_ROWS = 50_000


def render_overlap_tab(
    *,
    st: Any,
    specs: Sequence[Any],
    current_spec: Any,
    inventory_for_spec: Any,
    cached_sample: Any,
) -> None:
    st.markdown("### Dataset duplication and directional containment")
    st.info(
        "Report-only analysis: source datasets are sampled and compared but are "
        "never rewritten or removed. Select two datasets for a pair comparison, "
        "or more datasets to calculate every selected combination."
    )
    unique_specs = {spec.key: spec for spec in specs}
    options = list(unique_specs)
    if len(options) < 2:
        st.warning("At least two catalog datasets are required for overlap analysis.")
        return
    second_default = next(key for key in options if key != current_spec.key)
    select_all = st.checkbox(
        "Select every available catalog dataset",
        help="This includes remote sources and may require their credentials.",
    )
    if select_all:
        selected_keys = options
        st.caption(f"Selected all {len(selected_keys):,} available datasets.")
    else:
        selected_keys = st.multiselect(
            "Datasets to compare",
            options,
            default=[current_spec.key, second_default],
            format_func=lambda key: (
                f"{unique_specs[key].label} · {unique_specs[key].provider}"
            ),
            help=(
                "Two datasets produce one pair view. Three or more produce the full "
                "ordered containment matrix for all selected pairs."
            ),
        )
    controls = st.columns(5)
    sample_size = controls[0].number_input(
        "Rows per dataset",
        min_value=10,
        max_value=10_000,
        value=1_000,
        step=100,
        help="Containment is sample-based unless this includes the full population.",
    )
    seed = controls[1].number_input(
        "Overlap seed", min_value=0, value=42, step=1
    )
    shingle_size = controls[2].number_input(
        "Token shingle size", min_value=1, max_value=10, value=5, step=1
    )
    jaccard_threshold = controls[3].slider(
        "Jaccard threshold", 0.50, 1.0, 0.80, 0.05
    )
    edit_threshold = controls[4].slider(
        "Edit threshold", 0.50, 1.0, 0.80, 0.05
    )
    st.caption(
        "Near matching uses 128 MinHash permutations and 16 LSH bands for "
        "candidate retrieval, followed by exact Jaccard and token edit verification."
    )
    unordered_pair_count = len(selected_keys) * (len(selected_keys) - 1) // 2
    requested_rows = len(selected_keys) * int(sample_size)
    st.caption(
        f"Requested upper bound: {requested_rows:,} sampled rows across "
        f"{unordered_pair_count:,} dataset pairs."
    )
    workload_too_large = requested_rows > _MAX_TOTAL_SAMPLED_ROWS
    if workload_too_large:
        st.warning(
            f"Reduce rows per dataset or selected sources to stay at or below "
            f"{_MAX_TOTAL_SAMPLED_ROWS:,} sampled rows per interactive run."
        )
    if any(
        unique_specs[key].format in {"huggingface", "kaggle"}
        for key in selected_keys
    ):
        st.warning(
            "Selected remote datasets may require downloads or authentication. "
            "Start with a small sample and two datasets."
        )

    state_key = "dataset-overlap:last-result"
    if st.button(
        "Calculate selected dataset overlaps",
        type="primary",
        disabled=len(selected_keys) < 2 or workload_too_large,
    ):
        status = st.status("Loading bounded dataset samples…", expanded=True)
        progress = st.progress(0, text="Preparing overlap analysis")
        sampled_text: dict[str, list[str]] = {}
        labels: dict[str, str] = {}
        fields: dict[str, str] = {}
        errors: list[str] = []
        for position, key in enumerate(selected_keys, start=1):
            spec = unique_specs[key]
            status.write(f"Loading {spec.label}")
            try:
                inventory = inventory_for_spec(spec)
                candidates = text_columns(inventory["schema"])
                if not candidates:
                    raise ValueError("no detectable text field")
                field = candidates[0]
                rows = cached_sample(
                    inventory,
                    min(int(sample_size), int(inventory["rows"])),
                    int(seed),
                    (field,),
                )
                texts = [
                    text
                    for row in rows
                    if (
                        text := "\n".join(extract_text(row.get(field))).strip()
                    )
                ]
                if not texts:
                    raise ValueError(f"field {field!r} yielded no text")
                sampled_text[key] = texts
                labels[key] = spec.label
                fields[key] = field
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
                errors.append(f"{spec.label}: {error}")
            progress.progress(
                round(70 * position / len(selected_keys)),
                text=f"Loaded {position}/{len(selected_keys)} dataset samples",
            )
        if len(sampled_text) >= 2:
            status.write(
                "Running internal deduplication and global cross-source matching"
            )
            config = DeduplicationConfig(
                shingle_size=int(shingle_size),
                minhash_permutations=128,
                minhash_bands=16,
                near_duplicate_threshold=float(jaccard_threshold),
                edit_similarity_threshold=float(edit_threshold),
                seed=int(seed),
            )
            try:
                analysis = analyze_dataset_overlap(sampled_text, config)
            except (RuntimeError, ValueError) as error:
                st.session_state.pop(state_key, None)
                progress.empty()
                status.update(label="Overlap analysis failed", state="error")
                status.write(str(error))
            else:
                st.session_state[state_key] = {
                    "analysis": analysis,
                    "labels": labels,
                    "fields": fields,
                    "errors": tuple(errors),
                    "sample_size": int(sample_size),
                }
                progress.progress(100, text="Overlap analysis complete")
                status.update(label="Overlap analysis complete", state="complete")
        else:
            st.session_state.pop(state_key, None)
            progress.empty()
            status.update(
                label="Fewer than two datasets could be loaded", state="error"
            )
            for error in errors:
                status.write(error)

    state = st.session_state.get(state_key)
    if state:
        _render_result(st, **state)


def _render_result(
    st: Any,
    analysis: DatasetOverlapAnalysis,
    labels: dict[str, str],
    fields: dict[str, str],
    errors: tuple[str, ...],
    sample_size: int,
) -> None:
    st.markdown("#### Latest report-only result")
    st.caption(
        f"Requested at most {sample_size:,} rows per dataset. Automatic text "
        "fields: "
        + ", ".join(
            f"{labels[key]} → {field}" for key, field in fields.items()
        )
    )
    for error in errors:
        st.warning(error)

    internal_rows = [
        {
            "dataset": labels.get(row.dataset_id, row.dataset_id),
            "sampled": row.sampled_documents,
            "usable": row.usable_documents,
            "skipped": row.skipped_documents,
            "exact_duplicates": row.exact_duplicates,
            "near_duplicates": row.near_duplicates,
            "unique_documents": row.unique_documents,
            "internal_duplicate_%": round(100 * row.internal_duplicate_ratio, 2),
        }
        for row in analysis.datasets
    ]
    st.markdown("##### Within-dataset duplication")
    st.dataframe(internal_rows, width="stretch", hide_index=True)

    directional_rows = [
        {
            "source_dataset": labels.get(row.source_dataset_id, row.source_dataset_id),
            "covered_by": labels.get(row.covering_dataset_id, row.covering_dataset_id),
            "source_unique_documents": row.source_unique_documents,
            "exact_matches": row.exact_matched_documents,
            "near_matches": row.near_matched_documents,
            "matched_documents": row.matched_documents,
            "containment_%": round(100 * row.containment_ratio, 2),
            "matched_document_token_%": round(
                100 * row.matched_document_token_ratio, 2
            ),
        }
        for row in analysis.directional_containment
    ]
    st.markdown("##### Directional cross-dataset containment")
    if len(analysis.datasets) == 2:
        metrics = st.columns(2)
        for column, row in zip(metrics, analysis.directional_containment):
            column.metric(
                f"{labels.get(row.source_dataset_id, row.source_dataset_id)} "
                "covered by "
                f"{labels.get(row.covering_dataset_id, row.covering_dataset_id)}",
                f"{100 * row.containment_ratio:.2f}%",
                help=(
                    f"{row.matched_documents:,} of "
                    f"{row.source_unique_documents:,} internally unique sampled "
                    "documents"
                ),
            )
    st.dataframe(directional_rows, width="stretch", hide_index=True)
    _render_heatmap(st, analysis, labels)
    st.download_button(
        "Download within-dataset summary CSV",
        _csv_bytes(internal_rows),
        file_name="within_dataset_dedup.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download directional containment CSV",
        _csv_bytes(directional_rows),
        file_name="directional_containment.csv",
        mime="text/csv",
    )


def _render_heatmap(
    st: Any, analysis: DatasetOverlapAnalysis, labels: dict[str, str]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    dataset_ids = [row.dataset_id for row in analysis.datasets]
    positions = {dataset_id: index for index, dataset_id in enumerate(dataset_ids)}
    matrix = np.full((len(dataset_ids), len(dataset_ids)), np.nan)
    for row in analysis.directional_containment:
        matrix[positions[row.source_dataset_id], positions[row.covering_dataset_id]] = (
            100 * row.containment_ratio
        )
    size = min(18.0, max(7.0, 1.0 + len(dataset_ids) * 0.55))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100)
    display_labels = [labels.get(dataset_id, dataset_id) for dataset_id in dataset_ids]
    axis.set_xticks(range(len(dataset_ids)), display_labels, rotation=60, ha="right")
    axis.set_yticks(range(len(dataset_ids)), display_labels)
    axis.set_xlabel("Covering dataset")
    axis.set_ylabel("Dataset being covered")
    axis.set_title("Directional document containment (%)")
    if len(dataset_ids) <= 12:
        for row_index in range(len(dataset_ids)):
            for column_index in range(len(dataset_ids)):
                if row_index != column_index:
                    value = matrix[row_index, column_index]
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
    figure.colorbar(image, ax=axis, label="Containment (%)")
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")
