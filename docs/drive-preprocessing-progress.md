# Drive preprocessing and deduplication progress

Branch: `feature/drive-batched-preprocessing-pipeline`

Last updated: 2026-09-14

This is the durable work log for the batched corpus preprocessing branch. Raw
datasets, credentials, OAuth tokens, generated reports, transfer-check runs,
and other runtime logs remain local and are not committed.

## Objective

Materialize a reproducible clean Nepali corpus from a local path or Google
Drive while bounding preprocessing work, preserving provenance, auditing every
rejection, producing EDA products, and packaging immutable outputs.

## Implemented

- [x] Create a dedicated feature branch.
- [x] Read Parquet, JSONL/NDJSON, JSON arrays, CSV, UTF-8 text, and supported
  files inside ZIP archives.
- [x] Recursively download Drive files or folders into a run-local input area.
- [x] Export Google Docs as PDF, Sheets as XLSX, and Slides/Drawings as PDF.
- [x] Add deterministic hash-based sampling with an optional record cap.
- [x] Bound batches by both row count and encoded MiB.
- [x] Run normalization and quality checks in a bounded process pool while the
  coordinator preserves source order.
- [x] Materialize clean Zstandard-compressed Parquet shards and an audit JSONL.
- [x] Implement normalized SHA-256 exact-document deduplication.
- [x] Upgrade near deduplication using the methodology of Lee et al. (2022):
  token 5-gram MinHash-LSH candidate generation followed by exact shingle
  Jaccard and banded token edit-similarity verification.
- [x] Store compact 64-bit token/shingle fingerprints rather than duplicate
  full document text in the near-duplicate verifier.
- [x] Detect and remove repeated normalized paragraph patterns in the
  materialized batch pipeline.
- [x] Generate top 1/2/3/4-gram tables, WordCloud and diagnostic figures, a PDF
  report, run manifest, checksums, and a Zip64 package.
- [x] Add recursive resumable Drive upload with size/checksum validation.
- [x] Add a standalone Drive download/upload timing checker and JSON report.
- [x] Add a report-only multi-dataset overlap service and a dedicated explorer
  tab for within-dataset duplicate ratios and all selected directional pairs.
- [x] Add exact/verified-near match counts, directional document containment,
  matched-document token mass, CSV downloads, and a containment heatmap.
- [x] Document local, service-account, browser-OAuth, concurrency, memory, and
  restart behavior.

## Verification record

| Date | Check | Result |
|---|---|---|
| 2026-09-14 | Focused batch and EDA tests | 25 passed |
| 2026-09-14 | Full `unittest` suite after verified-near-dedup upgrade | 210 passed, 4 skipped |
| 2026-09-14 | Full suite with multi-dataset overlap and explorer integration | 213 passed, 4 skipped |
| 2026-09-14 | Native Google document transfer | Download/export succeeded; PDF size was measured from exported bytes |
| 2026-09-14 | Service-account upload to personal My Drive | Correctly rejected by Google because service accounts have no personal storage quota |

Use this command to reproduce the current automated check:

```bash
MPLCONFIGDIR=/tmp/attentionmaps-matplotlib \
.venv/bin/python -m unittest discover -s tests
```

## Current deduplication behavior

```text
NFC comparison normalization
  → normalized SHA-256 exact-document check
  → Nepali-aware token 5-gram fingerprints
  → MinHash-LSH candidate lookup
  → exact token-shingle Jaccard threshold
  → normalized token edit-similarity threshold
  → deterministic source-order keeper
  → repeated normalized paragraph removal
```

The default Jaccard and edit-similarity thresholds are both `0.80`; they are
operational starting values, not yet validated optimal values for Nepali.
MinHash never removes a document by itself. Documents shorter than the shingle
size participate only in exact-document deduplication.

## Multi-dataset overlap calculation

The explorer's **Dataset overlap** tab accepts two selected datasets for a pair
view or any larger selection for every combination. For each dataset `A`, the
service first performs an independent internal deduplication pass. It reports:

```text
internal_duplicate_ratio(A)
  = (exact_duplicates(A) + near_duplicates(A)) / usable_sampled_documents(A)

directional_containment(A covered by B)
  = internally_unique_documents_from_A_matched_in_B
    / internally_unique_documents_in_A
```

The reverse direction has its own denominator and is therefore generally
different. Token-weighted containment is the fraction of A's retained token
mass belonging to matched documents. Exact normalized hashes are exhaustive;
near matches are MinHash-LSH candidates accepted only after exact Jaccard and
token edit-similarity verification. One shared cross-source index avoids a
quadratic document-by-document scan. Inputs remain unchanged, and the UI only
stores the latest in-session report.

Interactive runs are capped at 50,000 requested sampled rows. With 50 sources,
the default 1,000 rows per source evaluates 1,225 unordered pairs and emits
2,450 directional results. This is an exploratory estimate, not a population
claim, unless the sample includes every usable document.

## Known boundaries

- The batch materialization CLI still treats all input files as one corpus.
  The explorer can compare catalog datasets, but a formal manifest that groups
  several physical files into one logical dataset is not implemented yet.
- Near-duplicate keepers are deterministic but source-order based. A future
  source-aware policy must prefer license, provenance, quality, and intended
  use rather than whichever row arrives first.
- MinHash buckets and compact document fingerprints still grow with document
  count and token volume; multi-million-document runs require measurement.
- Exact repeated-substring removal using a disk-backed suffix array is not
  implemented. It is a separate global stage with much larger RAM/disk needs.
- Repeated-paragraph removal is exact after normalization; it is not fuzzy
  paragraph matching and may require a keep-one versus remove-all policy.
- Source-PDF text extraction is outside this CLI. A Google Doc exported as PDF
  verifies transfer behavior but is not automatically converted into input
  text for preprocessing.
- Google-native exports through Drive are limited to 10 MB per export response.
- Service accounts can download shared personal-Drive data but cannot consume
  personal My Drive storage. Personal destinations require browser OAuth;
  service-account uploads require a Shared Drive or delegated Workspace user.

## Remaining multi-dataset extension

The next data contract must distinguish logical datasets from their physical
files. Every document should retain `dataset_id`, `source_file`, `source_row`,
`document_id`, `duplicate_cluster_id`, and `canonical_document_id`.

Planned stages:

1. Load a manifest mapping each logical dataset to one or more heterogeneous
   files and its schema, license, purpose, and priority.
2. Report duplicates within each file and across files of the same dataset.
3. Persist verified duplicate clusters across all datasets without performing
   quadratic document comparisons; the current UI result is session-only.
4. Promote sampled directional containment and token-weighted overlap into
   versioned pipeline artifacts.
5. Select cluster representatives using source quality and license priority.
6. Produce dataset-selection evidence based on unique contribution, quality,
   language composition, provenance, and intended training purpose.

Expected artifacts include `dataset_inventory.csv`,
`within_dataset_dedup.csv`, `cross_dataset_overlap.csv`,
`document_duplicate_clusters.parquet`, an overlap heatmap, and a dataset
selection report. Cross-dataset removal should initially run in `report_only`
mode until reviewed.

## Open validation work

- Label a stratified Nepali duplicate-pair sample.
- Compare token shingle sizes 3/5/7 and Jaccard/edit thresholds 0.80/0.85/0.90.
- Benchmark MinHash configurations for candidate recall, runtime, and peak RSS.
- Review repeated short-pattern behavior and paragraph removal semantics.
- Measure full-corpus throughput on the selected AWS instance class.
- Define the dataset manifest, source priorities, license policy, and protected
  evaluation datasets before enabling inter-dataset removal.

## References

- [Batched pipeline runbook](scripts/drive-preprocessing-pipeline.md)
- [EDA, cleaning, and deduplication notes](eda-cleaning-notes.md)
- [Deduplicating Training Data Makes Language Models Better](https://aclanthology.org/2022.acl-long.577/)
