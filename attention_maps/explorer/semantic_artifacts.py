"""Portable, disk-backed corpus embeddings and read-only similarity queries."""

import hashlib
from contextlib import contextmanager
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .semantic_instances import resolve_instance_definition, make_instance, iter_instance_records
from .semantic_clustering import cluster_embeddings
from .semantic_source import verify_source


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(path):
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    report = read_json(root / "report.json")
    if report.get("report_type") != "dataset_semantic_analysis" or report.get("report_version") != 1:
        raise ValueError("Choose a completed cluster_dataset.py run directory")
    n, d = report.get("embedded_records"), report.get("model", {}).get("dimension")
    if type(n) is not int or n < 1 or type(d) is not int or d < 1:
        raise ValueError("Invalid embedding dimensions in report")
    if (root / "embeddings.f32").stat().st_size != n * d * 4:
        raise ValueError("Embedding file is incomplete or does not match the report")
    for filename, shape in (("labels.npy", (n,)), ("coordinates.npy", (n, 3)), ("centroid_distance.npy", (n,))):
        array = np.load(root / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"Invalid {filename}")
    return root, report


def open_embeddings(root, report):
    return np.memmap(Path(root) / "embeddings.f32", mode="r", dtype="float32",
                     shape=(report["embedded_records"], report["model"]["dimension"]))


@contextmanager
def open_records(root):
    connection = sqlite3.connect((Path(root) / "records.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        yield connection
    finally:
        connection.close()


def get_record(root, index):
    with open_records(root) as connection:
        has_instance = "instance_json" in {item[1] for item in connection.execute("PRAGMA table_info(records)")}
        columns = "source_row, text, record_json, chunks" + (", instance_json" if has_instance else "")
        row = connection.execute(f"SELECT {columns} FROM records WHERE embedding_id=?", (int(index),)).fetchone()
    if row is None:
        raise ValueError(f"Embedded record ID {index} does not exist")
    return {"embedding_id": int(index), "source_row": row[0], "text": row[1], "record": json.loads(row[2]), "chunks": row[3],
            "instance": json.loads(row[4]) if has_instance else None}


def pair_similarity(path, first, second):
    root, report = load_run(path)
    if any(type(index) is not int or not 0 <= index < report["embedded_records"] for index in (first, second)):
        raise ValueError("Pair IDs must be valid zero-based embedded record IDs")
    vectors = open_embeddings(root, report)
    a, b = np.asarray(vectors[first], dtype=np.float64), np.asarray(vectors[second], dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid pair embeddings")
    score = float(np.clip(a @ b / denominator, -1, 1))
    return {"cosine_similarity": score, "model": report["model"],
            "instance_definition": report.get("instance_definition"),
            "first": get_record(root, first), "second": get_record(root, second)}


def build_run(inventory, fields, encoder, args, token=None):
    definition = resolve_instance_definition(inventory, fields, args)
    identity = {key: inventory.get(key) for key in ("dataset_id", "dataset_revision", "dataset_version", "dataset_config", "dataset_split", "dataset_shards", "source_files", "path", "row_filters")}
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; choose a new run name")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    connection = None
    try:
        connection = sqlite3.connect(temporary / "records.sqlite")
        connection.execute("CREATE TABLE records (embedding_id INTEGER PRIMARY KEY, source_row INTEGER UNIQUE NOT NULL, text TEXT NOT NULL, record_json TEXT NOT NULL, chunks INTEGER NOT NULL, instance_id TEXT UNIQUE NOT NULL, instance_json TEXT NOT NULL)")
        corpus_hash = hashlib.sha256(json.dumps(definition, sort_keys=True, ensure_ascii=False).encode())
        scanned = embedded = empty = multi_chunk = total_chunks = 0
        generated_ids = unknown_splits = 0
        exhausted = True
        pending = []
        with (temporary / "embeddings.f32").open("wb") as stream:
            def flush():
                nonlocal embedded, multi_chunk, total_chunks
                vectors, chunks = encoder.encode([item[1] for item in pending])
                if vectors.shape != (len(pending), encoder.dimension) or not np.isfinite(vectors).all():
                    raise ValueError("Encoder returned inconsistent embeddings")
                from .semantic_models import normalize_rows
                normalize_rows(vectors).astype("float32").tofile(stream)
                for (source_row, text, record, instance), count in zip(pending, chunks):
                    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
                    corpus_hash.update(json.dumps([source_row, text, payload], ensure_ascii=False).encode("utf-8"))
                    try:
                        connection.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (embedded, source_row, text, payload, int(count), str(instance["id"]), json.dumps(instance, ensure_ascii=False, default=str)))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate instance id at source row {source_row}: {instance['id']!r}") from error
                    embedded += 1
                    multi_chunk += int(count > 1)
                    total_chunks += int(count)
                connection.commit()
                pending.clear()
                print(f"Embedded {embedded:,} records ({total_chunks:,} chunks)", file=sys.stderr)

            records = iter_instance_records(inventory, definition, token=token)
            try:
                for source_row, record in enumerate(records):
                    if args.max_records is not None and scanned >= args.max_records:
                        exhausted = False
                        break
                    scanned += 1
                    try:
                        instance, text, generated = make_instance(record, definition, source_row=source_row,
                            source_identity=identity, source_split=inventory.get("dataset_split"))
                    except ValueError as error:
                        raise ValueError(f"Invalid {definition['schema']} instance at source row {source_row}: {error}") from error
                    if not text.strip():
                        empty += 1
                        continue
                    generated_ids += int(generated)
                    unknown_splits += int(instance["split"] is None)
                    pending.append((source_row, text, record, instance))
                    if len(pending) == args.batch_size:
                        flush()
                if pending:
                    flush()
            finally:
                records.close()
            stream.flush()
            os.fsync(stream.fileno())
        connection.close()
        connection = None
        if not embedded:
            raise ValueError("No non-empty text records to embed")
        if definition["text_record_unit"] != "blank_line" and exhausted and inventory.get("rows") is not None and inventory["rows"] != scanned:
            raise ValueError("Source row count changed; semantic run was not published")
        verify_source(inventory)
        vectors = np.memmap(temporary / "embeddings.f32", mode="r", dtype="float32", shape=(embedded, encoder.dimension))
        print(f"Clustering all {embedded:,} embedded records into {args.clusters} clusters…", file=sys.stderr)
        clustering = cluster_embeddings(vectors, temporary, clusters=args.clusters, batch_size=args.cluster_batch_size,
                                        epochs=args.cluster_epochs, seed=args.seed)
        from .semantic_wordclouds import save_wordclouds
        print("Saving cluster word clouds from all embedded records…", file=sys.stderr)
        wordclouds = save_wordclouds(temporary, temporary, np.load(temporary / "labels.npy", mmap_mode="r"),
                                    definition=definition, seed=args.seed, font_path=getattr(args, "wordcloud_font", None))
        report = {"wordclouds": wordclouds, "report_type": "dataset_semantic_analysis", "report_version": 1,
                  "created_at": datetime.now(timezone.utc).isoformat(), "inventory": inventory,
                  "text_columns": definition["text_columns"], "instance_definition": definition,
                  "generated_instance_ids": generated_ids, "instances_without_split": unknown_splits, "model": encoder.metadata, "seed": args.seed,
                  "settings": getattr(args, "effective_settings", {}),
                  "source_records_scanned": scanned, "embedded_records": embedded, "empty_records_skipped": empty,
                  "scope": "complete_selection" if exhausted else "limited_prefix", "max_records": args.max_records,
                  "records_with_multiple_chunks": multi_chunk, "total_chunks": total_chunks,
                  "corpus_fingerprint": corpus_hash.hexdigest(), "clustering": clustering,
                  "teacher_hypothesis": "EmbeddingGemma may capture Nepali semantics better; no superiority is established by this run."}
        (temporary / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        # A completed directory is the publication boundary; failed runs leave no report.
        temporary.rename(destination)
        return destination, report
    finally:
        if connection is not None:
            connection.close()
        if temporary.exists():
            shutil.rmtree(temporary)
