# Batched Drive preprocessing pipeline

## Purpose

This CLI materializes an auditable clean EDA dataset without changing the source:

```text
Google Drive or local path
  → deterministic sampling
  → process-pooled normalization and quality gates
  → exact or MinHash-LSH document deduplication
  → repeated-paragraph removal
  → clean Parquet shards
  → 1/2/3/4-gram tables, WordCloud, figures, and PDF
  → checksums and Zip64 package
  → optional Google Drive upload
```

Supported input files are Parquet, JSONL/NDJSON, JSON arrays, CSV, UTF-8 text,
and ZIP archives containing those formats. A `.txt` file is one document. JSONL
or Parquet is strongly preferred for large inputs because a JSON array must be
decoded in memory. The generated `eda_report.pdf` is a report; source-PDF text
extraction is not part of this CLI.

## Configuration

Copy `configs/eda/drive_preprocessing.example.yaml` and set a unique `run_name`.
Select exactly one input:

```yaml
input:
  local_path: /data/source
  # google_drive: https://drive.google.com/drive/folders/...
  text_columns: [text]
  source_columns: [source]
```

Sampling uses SHA-256 over `(seed, source-file, row-number)`, so a fraction is
repeatable. `max_records` is applied after the fraction and therefore depends on
stable source traversal order.

Deduplication modes are:

- `none`: retain every record that passes cleaning;
- `exact`: normalized SHA-256 matching;
- `multistage`: exact SHA-256; token-shingle MinHash-LSH candidate generation;
  exact Jaccard and token edit-similarity verification; then repeated normalized
  paragraph removal.

`near_duplicate_threshold` is the exact token-shingle Jaccard threshold, while
`edit_similarity_threshold` independently controls normalized token edit
similarity. MinHash only generates candidates and never removes a document by
itself. Documents shorter than `shingle_size` tokens receive exact-document
deduplication only.

The coordinator makes deduplication decisions in source order even when worker
batches complete at different times. This keeps one-worker and multi-worker runs
equivalent.

## Run locally

```bash
.venv/bin/python scripts/run_drive_preprocessing_pipeline.py \
  --config configs/eda/drive_preprocessing.yaml \
  --no-upload
```

For Google Drive on a server, use Application Default Credentials or a service
account that can read the source and write to a Shared Drive:

```bash
.venv/bin/python scripts/run_drive_preprocessing_pipeline.py \
  --config configs/eda/drive_preprocessing.yaml \
  --credentials-file /secure/service-account.json
```

For browser OAuth on a local workstation:

```bash
.venv/bin/python scripts/run_drive_preprocessing_pipeline.py \
  --config configs/eda/drive_preprocessing.yaml \
  --oauth-client-secrets /secure/desktop-client.json
```

The default full Drive scope is required to read an arbitrary shared input.
Credential and OAuth token files must remain outside version control.
For a personal My Drive destination, use browser OAuth: service accounts have
no personal Drive storage quota even when the destination folder is shared
with them.

## Concurrency and memory

`execution.workers` controls CPU processes. A batch is flushed when either
`batch_size` rows or `batch_max_mib` of encoded input is reached.
`max_pending_batches` bounds queued work and should normally be `2 × workers`.
The main process reduces batches in input order and writes Parquet incrementally.
Four threads are used for output checksums, while Drive transfers use
resumable/retryable chunks.

Recommended starting settings for `r8i.4xlarge` are eight workers, 512 rows per
batch, sixteen pending batches, and 50,000 rows per Parquet shard. Reduce batch
size for unusually large documents. The current MinHash-LSH index is retained in
RAM, so multi-million-document runs must be measured before increasing workers.

## Run structure and restart behavior

Each `run_name` owns one directory under `output_root`. A successful directory is
immutable: rerunning it is rejected. If a run has a `running` manifest, rerunning
keeps downloaded input and `.part` files but rebuilds generated preprocessing,
EDA, and package stages.

The final package contains clean Parquet shards, the preprocessing audit, EDA
JSON/CSV/PNG products, a PDF report, `run_manifest.json`, and
`checksums.sha256`. The uploader sends the ZIP, ZIP checksum, and manifest to a
new folder under the configured destination.

## Standalone Drive transfer check

Use this command to test only download and upload connectivity and throughput.
It does not run sampling, preprocessing, deduplication, EDA, PDF generation, or
packaging.

```bash
.venv/bin/python scripts/check_google_drive_transfer.py \
  "https://drive.google.com/file/d/SOURCE_ID/view" \
  --destination-folder-id "DESTINATION_FOLDER_ID" \
  --credentials-file /secure/service-account.json
```

For a local browser login, replace `--credentials-file` with
`--oauth-client-secrets`. Downloaded data and `transfer_check.json` are retained
under `artifacts/drive-transfer-checks/<UTC timestamp>/`. The report records
download and upload seconds, MiB/s, total time, byte and file counts, chunk size,
and resulting Drive IDs.

Google-native files are exported during download: Docs to PDF, Sheets to XLSX,
and Slides or Drawings to PDF. The uploaded check artifact is the exported file,
not a newly created Google-native document. Google Drive limits each native-file
export response to 10 MB; export larger items manually before running the check.
