"""Read saved semantic runs; never invoke model loading, embedding, or clustering."""

import json
from pathlib import Path

import numpy as np

from attention_maps.explorer.semantic_artifacts import load_run, open_records, pair_similarity
from apps.components.semantic_instance import render_instance
from apps.components.semantic_wordclouds import render_cluster_wordcloud
from attention_maps.explorer.semantic_variants import discover_clusterings, load_clustering
from apps.components.semantic_sources import select_source_runs


def discover_runs(directory):
    root = Path(directory).expanduser()
    paths = [root] if (root / "report.json").is_file() else sorted(path.parent for path in root.glob("*/report.json"))
    found = []
    for path in paths:
        try:
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            if report.get("report_type") == "dataset_semantic_analysis":
                found.append((path, report))
        except (ValueError, OSError):
            continue
    return found


def dataset_label(report):
    source = report["inventory"]
    name = source.get("dataset_id") or source.get("path") or source.get("source_files", ["Dataset"])[0]
    parts = [name, source.get("dataset_config"), source.get("dataset_split")]
    return " / ".join(str(value) for value in parts if value) + " · " + report["corpus_fingerprint"][:10]


def plot_records(root, report, *, limit=10000, clustering_root=None):
    n = report["embedded_records"]
    ids = np.sort(np.random.default_rng(report["seed"]).choice(n, min(limit, n), replace=False))
    clustering_root = clustering_root or root
    coordinates = np.load(clustering_root / "coordinates.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(clustering_root / "labels.npy", mmap_mode="r", allow_pickle=False)
    points = np.asarray(coordinates[ids])
    return {"record_id": ids, "cluster": np.asarray(labels[ids]).astype(str),
            "x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}


def render_semantic_results(st, default_root="artifacts/dataset_embeddings"):
    st.header("Dataset embeddings and clusters")
    directory = st.text_input("Embedding runs directory", value=default_root)
    st.button("Refresh saved runs")
    found = discover_runs(directory)
    choices = select_source_runs(st, found, directory, dataset_label)
    if not choices:
        return
    model = st.selectbox("Embedding model", sorted({report["model"]["id"] for _, report in choices}))
    choices = [(path, report) for path, report in choices if report["model"]["id"] == model]
    path = st.selectbox("Saved embedding run", [path for path, _ in choices], format_func=lambda path: path.name)
    try:
        root, report = load_run(path)
        st.caption(f"{report['scope']} · {report['embedded_records']:,} embedded records · {report['empty_records_skipped']:,} empty records skipped")
        st.caption(f"Embedding task: {report['model']['task']} · prompt: {report['model']['prompt'] or 'none'}")
        st.caption(report["teacher_hypothesis"])
        saved_clusterings = []
        for candidate in discover_clusterings(root):
            try:
                saved_clusterings.append((candidate, load_clustering(root, report, candidate)))
            except (OSError, ValueError, KeyError) as error:
                st.warning(f"Skipping clustering result {candidate.name}: {error}")
        cluster_reports = dict(saved_clusterings)
        cluster_root = st.selectbox("Saved clustering result", list(cluster_reports),
            format_func=lambda value: f"{cluster_reports[value]['clustering']['requested_clusters']} clusters · {'Original' if value == root else value.name}",
            key=f"clustering:{root}")
        selected_clustering = cluster_reports[cluster_root]
        clustering = selected_clustering["clustering"]
        st.caption("Saved results only. Generate another cluster count with cluster_dataset.py recluster; this view does not run clustering.")
        cards = st.columns(3)
        cards[0].metric("Embedded records", report["embedded_records"])
        cards[1].metric("Occupied clusters", clustering["occupied_clusters"])
        cards[2].metric("Requested clusters", clustering["requested_clusters"])
        if clustering["occupied_clusters"] < clustering["requested_clusters"]:
            st.info("Some requested clusters are empty. Repeated or very similar embeddings can produce fewer occupied clusters.")
        st.subheader("Compare any two dataset records")
        columns = st.columns(2)
        maximum = report["embedded_records"] - 1
        first = int(columns[0].number_input("Record A ID", min_value=0, max_value=maximum, value=0, step=1))
        second = int(columns[1].number_input("Record B ID", min_value=0, max_value=maximum, value=min(1, maximum), step=1))
        result = pair_similarity(root, first, second)
        definition = result["instance_definition"]
        if definition:
            st.caption(f"Schema: {definition['label']} · One instance = {definition['unit']}")
            st.caption(f"Similarity input: {definition['embedding_policy']}")
            if definition["text_record_unit"] == "line":
                st.info("This TXT run treats each non-empty line as one instance. It does not reconstruct documents spanning multiple lines.")
        else:
            st.info("This older run used source rows and selected text fields. Full saved records are shown below; schema-aware similarity requires a new embedding run.")
            if report["inventory"]["format"] == "text":
                st.caption("This saved TXT run used one non-empty line per record; additional document context was not included in its embeddings.")
        for column, key in zip(columns, ("first", "second")):
            with column:
                render_instance(st, result[key], definition, key=f"{root}:{key}:{result[key]['embedding_id']}")
        st.metric("Cosine similarity", f"{result['cosine_similarity']:.6f}")
        st.caption("Computed from original document vectors. Cosine is in [-1, 1]; it is not a probability or a calibrated quality score.")
        comparisons = []
        for other, saved in found:
            if saved["corpus_fingerprint"] == report["corpus_fingerprint"]:
                try:
                    score = pair_similarity(other, first, second)["cosine_similarity"]
                    comparisons.append({"Run": other.name, "Model": saved["model"]["id"], "Task": saved["model"]["task"], "Cosine": score})
                except (OSError, ValueError):
                    continue
        if len(comparisons) > 1:
            st.dataframe(comparisons, hide_index=True, width="stretch")
            st.caption("Only identical saved corpora are compared. Higher raw cosine across different models does not establish better semantic quality.")
        st.subheader("Cluster visualization")
        view = st.radio("Visualization dimensions", ("3D", "2D"), horizontal=True)
        dimensions = 3 if view == "3D" else 2
        limit = int(st.number_input("Maximum plotted records", min_value=1, max_value=100000, value=10000, step=1000))
        points = plot_records(root, report, limit=limit, clustering_root=cluster_root)
        import plotly.express as px
        plot = px.scatter_3d if dimensions == 3 else px.scatter
        figure = plot(points, x="x", y="y", **({"z": "z"} if dimensions == 3 else {}),
                      color="cluster", hover_data=["record_id"],
                      labels={"x": "PC1", "y": "PC2", "z": "PC3", "cluster": "Cluster"})
        figure.update_traces(marker_size=3 if dimensions == 3 else 5)
        if dimensions == 2:
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
        if st.checkbox("Show all occupied cluster centroids", value=True):
            import plotly.graph_objects as go
            centers = np.load(cluster_root / "centroids.npy", allow_pickle=False)
            with np.load(cluster_root / "projection.npz", allow_pickle=False) as projection:
                projected = (centers - projection["mean"]) @ projection["components"].T
            positions = np.zeros((len(centers), 3))
            positions[:, :projected.shape[1]] = projected
            occupied = np.flatnonzero(clustering["cluster_sizes"])
            centroid_trace = go.Scatter3d if dimensions == 3 else go.Scatter
            figure.add_trace(centroid_trace(
                x=positions[occupied, 0], y=positions[occupied, 1],
                **({"z": positions[occupied, 2]} if dimensions == 3 else {}),
                mode="markers", marker={"size": 7, "color": "black", "symbol": "diamond"},
                customdata=occupied, name="Cluster centroids", hovertemplate="Cluster %{customdata} centroid<extra></extra>"))
        figure.update_layout(height=650)
        st.plotly_chart(figure, width="stretch")
        st.caption(f"Showing {len(points['record_id']):,} / {report['embedded_records']:,} records. All embedded records were clustered. {view} distances are a lossy visualization; similarity and clustering use original vectors.")
        st.caption(f"Variance retained in {view}: {sum(clustering['projection']['explained_variance_ratio'][:dimensions]):.1%}")
        st.caption("2D uses saved PC1 and PC2; 3D adds saved PC3. Switching views does not recompute the projection or clusters.")
        cluster = st.selectbox("Browse cluster", list(range(clustering["requested_clusters"])),
                               format_func=lambda index: f"Cluster {index} ({clustering['cluster_sizes'][index]:,} records)",
                               key=f"cluster:{cluster_root}")
        render_cluster_wordcloud(st, cluster_root, cluster, embedding_root=root)
        labels = np.load(cluster_root / "labels.npy", mmap_mode="r", allow_pickle=False)
        ids = np.flatnonzero(labels == cluster)
        page = int(st.number_input("Cluster page (25 records)", min_value=1, max_value=max(1, (len(ids) + 24) // 25), value=1, step=1, key=f"page:{cluster_root}:{cluster}"))
        with open_records(root) as connection:
            rows = [connection.execute("SELECT embedding_id, source_row, substr(text, 1, 400) FROM records WHERE embedding_id=?", (int(index),)).fetchone()
                    for index in ids[(page - 1) * 25:page * 25]]
        st.dataframe([{"Record ID": row[0], "Source row": row[1], "Text preview": row[2]} for row in rows], hide_index=True, width="stretch")
        with st.expander("Exact model, source, and clustering settings"):
            st.json(report)
            if cluster_root != root:
                st.json(selected_clustering)
        _render_sampling(st, report)
    except (OSError, ValueError, KeyError, ImportError) as error:
        st.error(f"Cannot display this run: {error}")


def _render_sampling(st, embedding_report):
    st.subheader("B1–B3 sampling results")
    path = st.text_input("Sampling report path (optional)", value="")
    if not path.strip():
        st.caption("Run cluster_dataset.py sample with random, density, or semdedup, then select its report.json here.")
        return
    result = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if result.get("report_type") != "dataset_semantic_sampling" or result.get("corpus_fingerprint") != embedding_report["corpus_fingerprint"] or result.get("model") != embedding_report["model"]:
        raise ValueError("Sampling report must match this embedding model/run and corpus")
    status = result["language_status"]
    st.caption(f"Method: {result['method']} · eligible records: {result['eligible_records']:,}")
    st.caption(result["note"])
    st.metric("Sampling unique coverage", f"{status['sampling']['unique_coverage']:.2%}")
    st.write({"Language coverage": status["language_coverage"], "Script": status["script"]})
    st.caption(status["voting"]["note"])
    st.dataframe([{"Run": row["run"], "Records": row["sampled_records"], "Language": row["language_coverage"], "Script": row["script"]}
                  for row in status["sampling"]["run_results"]], hide_index=True, width="stretch")
    with st.expander("Sampling parameters and votes"):
        st.json(result)
