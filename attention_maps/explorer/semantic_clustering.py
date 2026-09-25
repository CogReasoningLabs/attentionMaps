"""Batched clustering in original embedding space and a separate 3D projection."""

import shutil

import numpy as np
from numpy.lib.format import open_memmap


def batches(rows: int, size: int, minimum: int = 1):
    start = 0
    while start < rows:
        end = min(rows, start + size)
        if 0 < rows - end < minimum:
            end = rows
        yield start, end
        start = end


def cluster_embeddings(embeddings, directory, *, clusters=10, batch_size=1024, epochs=3, seed=42, projection_source=None, projection_metadata=None):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import IncrementalPCA
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits

    n, dimension = embeddings.shape
    if not 1 <= clusters <= n:
        raise ValueError(f"clusters must be between 1 and {n} non-empty records")
    rng = np.random.default_rng(seed)
    size = max(batch_size, clusters, 3)
    model = MiniBatchKMeans(n_clusters=clusters, batch_size=size, random_state=seed,
                           n_init=3, reassignment_ratio=0)
    # Seed from across the corpus, never just the first source shard.
    initial = rng.choice(n, size=min(n, max(size, 3 * clusters)), replace=False)
    ranges = list(batches(n, size))
    with threadpool_limits(limits=2):
        model.partial_fit(np.asarray(embeddings[initial]))
        for _ in range(epochs):
            for block in rng.permutation(len(ranges)):
                start, end = ranges[block]
                values = np.array(embeddings[start:end])
                rng.shuffle(values)
                model.partial_fit(values)
        labels = open_memmap(directory / "labels.npy", mode="w+", dtype="int32", shape=(n,))
        distance = open_memmap(directory / "centroid_distance.npy", mode="w+", dtype="float32", shape=(n,))
        inertia = 0.0
        for start, end in ranges:
            values = np.asarray(embeddings[start:end])
            assigned = model.predict(values)
            labels[start:end] = assigned
            delta = values - model.cluster_centers_[assigned]
            distance[start:end] = np.linalg.norm(delta, axis=1)
            inertia += float(np.square(delta).sum(dtype=np.float64))
        labels.flush()
        distance.flush()
        np.save(directory / "centroids.npy", model.cluster_centers_, allow_pickle=False)
        if projection_source is not None:
            # Fixed coordinates let cluster-count comparisons share the same 3D space.
            for filename in ("coordinates.npy", "projection.npz"):
                shutil.copyfile(projection_source / filename, directory / filename)
            variance = projection_metadata["explained_variance_ratio"]
        else:
            components = min(3, n, dimension)
            projection = IncrementalPCA(n_components=components, batch_size=size)
            if n > 1:
                for start, end in batches(n, size, components):
                    projection.partial_fit(np.asarray(embeddings[start:end]))
            coordinates = open_memmap(directory / "coordinates.npy", mode="w+", dtype="float32", shape=(n, 3))
            coordinates[:] = 0
            if n > 1:
                for start, end in ranges:
                    coordinates[start:end, :components] = projection.transform(np.asarray(embeddings[start:end]))
                variance = np.nan_to_num(projection.explained_variance_ratio_).tolist()
                np.savez(directory / "projection.npz", components=projection.components_, mean=projection.mean_)
            else:
                variance = [0.0]
                np.savez(directory / "projection.npz", components=np.zeros((components, dimension)), mean=embeddings[0])
            coordinates.flush()
        scores = None
        metric_ids = rng.choice(n, size=min(n, 2000), replace=False)
        metric_labels = np.asarray(labels[metric_ids])
        if 1 < len(np.unique(metric_labels)) < len(metric_ids):
            scores = float(silhouette_score(np.asarray(embeddings[metric_ids]), metric_labels, metric="cosine"))
    counts = np.bincount(labels, minlength=clusters)
    return {"algorithm": "MiniBatchKMeans", "space": "original L2-normalized document embeddings",
            "requested_clusters": clusters, "occupied_clusters": int(np.count_nonzero(counts)),
            "cluster_sizes": counts.tolist(), "epochs": epochs, "batch_size": size, "inertia": inertia,
            "silhouette_cosine": scores, "silhouette_sample_rows": len(metric_ids),
            "projection": {"method": "IncrementalPCA", "dimensions": 3, "fitted_rows": n,
                           "explained_variance_ratio": variance,
                           "note": "3D distances are a lossy visualization; similarity and clustering use original vectors."}}
