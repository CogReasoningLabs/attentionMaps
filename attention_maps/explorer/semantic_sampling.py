"""B1 random, B2 p-stable density/IPS, and B3 within-cluster SemDeDup."""

import hashlib
from decimal import Decimal, ROUND_CEILING
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .language_status import LanguageEvidence, language_context, record_language_evidence, status_from_evidence
from .sampling_vote import majority_vote
from .semantic_artifacts import load_run, open_embeddings, open_records
from .semantic_clustering import batches


def density_scores(vectors, directory, *, rows=64, bins=2048, width=0.5, seed=42, batch_size=1024):
    """Two-pass collision sketch, Gaussian (2-stable) projections, mean bucket count."""
    if rows < 1 or bins < 2 or not math.isfinite(width) or width <= 0:
        raise ValueError("Density requires positive sketch rows/width and at least two bins")
    rng = np.random.default_rng(seed)
    projections = rng.normal(size=(vectors.shape[1], rows)).astype("float32")
    shifts = rng.uniform(0, width, size=rows).astype("float32")
    sketch = np.zeros((rows, bins), dtype=np.int64)

    def hashes(values):
        return np.floor((values @ projections + shifts) / width).astype(np.int64) % bins

    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        for start, end in batches(len(vectors), batch_size):
            positions = hashes(vectors[start:end])
            np.add.at(sketch, (np.arange(rows)[None, :], positions), 1)
        scores = open_memmap(directory / "density.npy", mode="w+", dtype="float64", shape=(len(vectors),))
        for start, end in batches(len(vectors), batch_size):
            scores[start:end] = sketch[np.arange(rows)[None, :], hashes(vectors[start:end])].mean(axis=1)
        scores.flush()
    np.savez(directory / "density_sketch.npz", counts=sketch, projections=projections, shifts=shifts, width=width)
    return scores


def semdedup_survivors(vectors, labels, distance, directory, *, threshold=0.95, block_size=512):
    """Prefer farther-from-centroid examples; compare to ALL preceding cluster members.

    Blockwise implementation of ordered max-cosine SemDeDup, not a connected
    component or a comparison limited to previously retained representatives.
    """
    if not math.isfinite(threshold) or not -1 <= threshold <= 1 or block_size < 1:
        raise ValueError("SemDeDup requires a cosine threshold in [-1, 1] and positive block size")
    n = len(vectors)
    maxima = open_memmap(directory / "semdedup_max_similarity.npy", mode="w+", dtype="float32", shape=(n,))
    matches = open_memmap(directory / "semdedup_neighbor.npy", mode="w+", dtype="int64", shape=(n,))
    maxima[:] = -1
    matches[:] = -1
    ordered = np.argsort(labels, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels[ordered])) + 1, n]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            ids = ordered[a:b]
            ids = ids[np.argsort(-np.asarray(distance[ids]), kind="stable")]
            for start, end in batches(len(ids), block_size):
                query = np.asarray(vectors[ids[start:end]])
                best = np.full(end - start, -np.inf)
                neighbor = np.full(end - start, -1, dtype=np.int64)
                for left, right in batches(end, block_size):
                    matrix = query @ np.asarray(vectors[ids[left:right]]).T
                    matrix[np.arange(left, right)[None, :] >= np.arange(start, end)[:, None]] = -np.inf
                    index = matrix.argmax(axis=1)
                    value = matrix[np.arange(len(query)), index]
                    update = value > best
                    best[update] = value[update]
                    neighbor[update] = ids[left:right][index[update]]
                maxima[ids[start:end]] = np.clip(best, -1, 1)
                matches[ids[start:end]] = neighbor
            print(f"SemDeDup: cluster {int(labels[ids[0]])}, {len(ids):,} records", file=sys.stderr)
    maxima.flush()
    matches.flush()
    return np.flatnonzero((matches < 0) | (maxima <= threshold))


def select_indices(n, count, seed, *, scores=None, eligible=None):
    if not 0 <= count <= n:
        raise ValueError("Invalid requested sample size")
    rng = np.random.default_rng(seed)
    pool = np.arange(n) if eligible is None else np.asarray(eligible)
    count = min(count, len(pool))
    if scores is not None:
        if np.any(scores <= 0) or not np.isfinite(scores).all():
            raise ValueError("Density scores must be finite and positive")
        weights = 1 / np.asarray(scores[pool], dtype=np.float64)
        weights /= weights.sum()
    else:
        weights = None
    return np.sort(rng.choice(pool, size=count, replace=False, p=weights))


def curate_run(args):
    root, report = load_run(args.run)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("Sampling output directory already exists; choose a new run name")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        vectors = open_embeddings(root, report)
        n = len(vectors)
        count = int((Decimal(str(args.sample_fraction)) * n).to_integral_value(rounding=ROUND_CEILING))
        scores = eligible = None
        if args.method == "density":
            scores = density_scores(vectors, temporary, rows=args.density_rows, bins=args.density_bins,
                                    width=args.density_width, seed=args.seed)
        elif args.method == "semdedup":
            labels = np.load(root / "labels.npy", mmap_mode="r", allow_pickle=False)
            distances = np.load(root / "centroid_distance.npy", mmap_mode="r", allow_pickle=False)
            eligible = semdedup_survivors(vectors, labels, distances, temporary,
                                         threshold=args.similarity_threshold, block_size=args.block_size)
            np.save(temporary / "semdedup_survivors.npy", eligible, allow_pickle=False)
        context = language_context(report["inventory"])
        union = np.zeros(n, dtype=bool)
        combined = LanguageEvidence()
        results = []
        with open_records(root) as connection:
            for run_index in range(args.sampling_runs):
                seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{run_index + 1}".encode()).digest()[:8], "big")
                selected = select_indices(n, count, seed, scores=scores, eligible=eligible)
                np.save(temporary / f"sample-{run_index + 1}.npy", selected, allow_pickle=False)
                evidence = LanguageEvidence()
                with (temporary / f"sample-{run_index + 1}.jsonl").open("w", encoding="utf-8") as stream:
                    for index in selected:
                        has_instance = report.get("instance_definition") is not None
                        columns = "record_json" + (", instance_json" if has_instance else "")
                        row = connection.execute(f"SELECT {columns} FROM records WHERE embedding_id=?", (int(index),)).fetchone()
                        if row is None:
                            raise ValueError("Sample record missing from saved corpus")
                        stream.write(row[0] + "\n")
                        record = json.loads(row[0])
                        fields = report["text_columns"]
                        if has_instance:
                            record.update(json.loads(row[1]))
                            schema = report["instance_definition"]["schema"]
                            fields = {"pretraining": ["text"], "task_specific_supervised": ["text"],
                                      "instruction_finetuning": ["messages"],
                                      "preference_tuning": ["prompt", "chosen", "rejected"],
                                      "evaluation": ["input", "reference"]}[schema]
                        item = record_language_evidence(record, fields)
                        evidence.add(item)
                        if not union[index]:
                            combined.add(item)
                            union[index] = True
                results.append({"run": run_index + 1, "seed": str(seed), **status_from_evidence(evidence, **context)})
        status = status_from_evidence(combined, **context)
        status.update(pooled_language_coverage=status["language_coverage"], pooled_script=status["script"])
        votes = {key: majority_vote([run[key] for run in results]) for key in ("language_coverage", "script")}
        status.update({key: value["label"] for key, value in votes.items()})
        status["voting"] = {"rule": "strict_majority", **votes,
                            "note": "Agreement measures run consistency, not correctness. Density and SemDeDup samples are intentionally biased toward coverage; language declarations remain shared evidence."}
        k = results[0]["sampled_records"]
        expected = 1 - (1 - k / n) ** args.sampling_runs if args.method == "random" else None
        status["sampling"] = {"method": args.method, "fraction_requested": args.sample_fraction,
                              "runs": args.sampling_runs, "population_rows": n, "population_basis": "saved non-empty embedded records",
                              "rows_scanned": n, "rows_per_run": k, "requested_rows_per_run": count,
                              "unique_sampled_records": combined.records, "unique_coverage": combined.records / n,
                              "sampled_record_occurrences": k * args.sampling_runs, "expected_unique_coverage": expected,
                              "run_results": results}
        result = {"report_type": "dataset_semantic_sampling", "report_version": 1, "embedding_run": str(root),
                  "corpus_fingerprint": report["corpus_fingerprint"], "model": report["model"],
                  "method": args.method, "seed": args.seed, "language_status": status,
                  "parameters": {"density_rows": args.density_rows, "density_bins": args.density_bins,
                                 "density_width": args.density_width, "similarity_threshold": args.similarity_threshold,
                                 "block_size": args.block_size},
                  "eligible_records": n if eligible is None else len(eligible),
                  "note": "B2 uses inverse collision-density weights without replacement. B3 removes cosine > threshold within each cluster, prefers farthest-from-centroid records, and never pads a shortfall with duplicates. Exported samples preserve whole original records."}
        (temporary / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
        return output, result
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
