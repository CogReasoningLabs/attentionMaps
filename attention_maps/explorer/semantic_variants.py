"""Saved clustering variants that share one immutable embedding run."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from .semantic_artifacts import load_run, open_embeddings, read_json
from .semantic_clustering import cluster_embeddings
from .semantic_wordclouds import find_font, save_wordclouds


def embedding_identity(report):
    return {key: report[key] for key in ("corpus_fingerprint", "model", "created_at", "embedded_records")}


def discover_clusterings(root):
    return [root] + sorted(path.parent for path in (root / "clusterings").glob("*/report.json") if not path.parent.name.startswith("."))


def load_clustering(root, embedding_report, directory):
    """Validate variant ownership and arrays before joining saved record IDs."""
    directory = Path(directory)
    if directory == root:
        return embedding_report
    report = read_json(directory / "report.json")
    if report.get("report_type") != "dataset_clustering" or report.get("report_version") != 1:
        raise ValueError("Choose a completed clustering result")
    if report.get("embedding_identity") != embedding_identity(embedding_report):
        raise ValueError("Clustering result does not match this embedding run")
    n, dimension = embedding_report["embedded_records"], embedding_report["model"]["dimension"]
    k = report["clustering"]["requested_clusters"]
    if type(k) is not int or not 1 <= k <= n:
        raise ValueError("Invalid saved cluster count")
    for filename, shape in (("labels.npy", (n,)), ("centroid_distance.npy", (n,)),
                            ("centroids.npy", (k, dimension)), ("coordinates.npy", (n, 3))):
        array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"Invalid clustering artifact: {filename}")
    labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= k:
        raise ValueError("Invalid cluster labels")
    if np.bincount(labels, minlength=k).tolist() != report["clustering"]["cluster_sizes"]:
        raise ValueError("Cluster sizes do not match saved labels")
    return report


def recluster_run(args):
    root, parent = load_run(args.run)
    if args.seed < 0 or args.cluster_epochs < 1 or args.cluster_batch_size < 1:
        raise ValueError("seed must be non-negative; clustering epochs and batch size must be positive")
    if not 1 <= args.clusters <= parent["embedded_records"]:
        raise ValueError(f"clusters must be between 1 and {parent['embedded_records']}")
    find_font(args.wordcloud_font)
    name = args.name or f"k{args.clusters}-seed{args.seed}"
    if name.startswith(".") or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Clustering name must be a single non-hidden directory name")
    destination = root / "clusterings" / name
    if destination.exists():
        raise ValueError("Clustering result already exists; use --name for a different run")
    destination.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=destination.parent))
    try:
        vectors = open_embeddings(root, parent)
        print(f"Clustering {parent['embedded_records']:,} saved embeddings into {args.clusters} clusters…", file=sys.stderr)
        clustering = cluster_embeddings(vectors, temporary, clusters=args.clusters, seed=args.seed,
                                        batch_size=args.cluster_batch_size, epochs=args.cluster_epochs,
                                        projection_source=root, projection_metadata=parent["clustering"]["projection"])
        print("Saving word clouds from every cluster record…", file=sys.stderr)
        wordclouds = save_wordclouds(root, temporary, np.load(temporary / "labels.npy", mmap_mode="r"),
                                    definition=parent.get("instance_definition"), seed=args.seed,
                                    font_path=args.wordcloud_font)
        report = {"report_type": "dataset_clustering", "report_version": 1,
                  "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                  "embedding_identity": embedding_identity(parent), "clustering": clustering,
                  "wordclouds": wordclouds, "projection_reused": True,
                  "note": "Uses the parent run's saved embeddings and 3D coordinates. No source loading or model inference."}
        (temporary / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
        return destination, report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def backfill_wordclouds(args):
    """Add missing clouds for existing assignments; never fit clusters or projections."""
    root, parent = load_run(args.run)
    directory = root
    if args.clustering:
        name = args.clustering
        if name.startswith(".") or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Clustering name must be a single non-hidden directory name")
        directory = root / "clusterings" / name
    report = load_clustering(root, parent, directory)
    destination = directory / "wordclouds"
    if destination.exists() and not args.refresh:
        raise ValueError("Word clouds already exist; use --refresh to apply the current stopwords")
    find_font(args.wordcloud_font)
    labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
    k = report["clustering"]["requested_clusters"]
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() >= k:
        raise ValueError("Invalid saved cluster labels")
    if np.bincount(labels, minlength=k).tolist() != report["clustering"]["cluster_sizes"]:
        raise ValueError("Cluster sizes do not match saved labels")
    temporary = Path(tempfile.mkdtemp(prefix=".wordclouds-", dir=directory))
    backup = temporary / "previous-wordclouds"
    published = False
    try:
        print("Generating word clouds using current stopwords and existing cluster assignments…", file=sys.stderr)
        summary = save_wordclouds(root, temporary, labels, definition=parent.get("instance_definition"),
                                 seed=report["seed"], font_path=args.wordcloud_font)
        if destination.exists():
            destination.rename(backup)
        try:
            (temporary / "wordclouds").rename(destination)
            published = True
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
        return destination, summary
    finally:
        if backup.exists() and not published:
            print(f"Previous word clouds retained for recovery at {backup}", file=sys.stderr)
        else:
            shutil.rmtree(temporary)
