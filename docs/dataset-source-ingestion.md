# Identifier-driven dataset ingestion

The Streamlit preprocessing workspace accepts five source types without first
adding every dataset to the repository's curated catalog:

| Source | Identifier entered in the UI | Loading policy |
|---|---|---|
| Hugging Face | `owner/dataset` | Streams a bounded sample from the requested configuration, split, and revision. |
| Kaggle | `owner/dataset` or `owner/dataset/versions/N` | KaggleHub caches the requested file or folder. A blank internal path downloads the dataset before file selection. |
| Google Drive | A file/folder ID or standard sharing URL | Recursively stages the source in a resumable local cache. |
| S3-compatible storage | Exact `s3://bucket/key` object URI | Checks object size and atomically stages one object locally. |
| Local | File or Parquet directory path | Reads immutable source metadata directly. |

Some very large Hub datasets, including CulturaX, omit split counts from the
streaming `DatasetInfo`. In that case the loader uses Hugging Face Dataset
Viewer size metadata for the requested configuration and split; it does not
scan the corpus to count rows.

After registration, every source enters the same workflow:

```text
source inventory
  -> bounded deterministic sampling
  -> NFC normalization
  -> exact/near deduplication
  -> optional supervised D2 pruning
  -> EDA and persisted audit artifacts
```

Remote files are cached below `data/cache/source-imports/`; Hugging Face
streaming does not materialize the complete split. KaggleHub uses its managed
cache. Cached source material is excluded from Git.

## Streamlit usage

1. Run `python scripts/run_dataset_explorer.py`.
2. Choose **Data source** in the sidebar.
3. Select one of the four standard training-data schemas. This preserves the
   correct atomic unit and restricts D2 to task-specific supervised datasets.
4. Enter the provider's standard identifier and select **Load ... source**.
5. For Hugging Face, choose **Dataset configuration**, **Dataset split**, and
   **All shards** or **Choose shards**. Sizes and samples follow this selection.
   For a staged folder or multi-file Kaggle dataset, select the dataset file.
6. Use **WORKSPACE** to sample, normalize, deduplicate, optionally prune, and
   generate EDA artifacts.

Pin Hugging Face revisions to a commit for research runs. A branch such as
`main` is convenient during exploration but can change over time.

## Credentials

Copy `.env.example` to `.env` and set only the providers in use. Never commit
`.env`.

```dotenv
HF_TOKEN=
KAGGLE_API_TOKEN=

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_SESSION_TOKEN=
AWS_DEFAULT_REGION=
S3_ENDPOINT_URL=

GOOGLE_APPLICATION_CREDENTIALS=
ATTENTION_MAPS_DRIVE_OAUTH_CLIENT=
ATTENTION_MAPS_DRIVE_OAUTH_TOKEN=.google-drive-token.json
```

`S3_ENDPOINT_URL` is optional and supports MinIO or another S3-compatible
service. Google Drive requires OAuth, application-default, or service-account
credentials; a simple API key cannot read private files.

Secrets are loaded into the process environment. They are not accepted by a
Streamlit widget, stored in session state, or written to workspace manifests.

## Formats and resource policy

Local and staged remote sources support Parquet, JSON arrays, JSONL/NDJSON,
CSV, UTF-8 line corpora, and XLSX. Parquet uses indexed row sampling. CSV,
JSONL, TXT, and XLSX use deterministic reservoir sampling with bounded memory.
Hugging Face uses bounded streaming shuffle and does not hold the split in RAM.

For Kaggle, provide the internal file path whenever it is known to prevent an
unnecessary full-dataset download. S3 currently requires an exact object URI
for the same reason. Drive folders are staged recursively because the Drive API
does not expose them as a single tabular stream.

## Operating an 80-dataset study

Datasets do not need to be downloaded or hard-coded in advance. Register each
provider identifier when it is scheduled for analysis, pin its immutable
version where supported, and preserve the resulting workspace manifest.
Provider caches make repeated inspection resumable.

The source loader and statistical sampling policy are separate concerns. This
ingestion layer preserves the workspace's deterministic sampling behavior.
Schema-aware stratified allocation can be added without changing provider
adapters.

## Current boundaries

- Only one source is active in a Streamlit workspace at a time.
- S3 prefix enumeration is intentionally disabled; select an exact object.
- A blank Kaggle internal path can download every file in that dataset.
- Hugging Face streaming shuffle is approximate when its buffer is smaller than
  the dataset.
- Full-corpus batch preprocessing from Hugging Face, S3, and Kaggle remains a
  separate extension. This UI processes the selected bounded workspace sample
  while leaving the source immutable.

## Language, script, and shard inspection

The standalone `scripts/inspect_dataset.py` command supports both
`--provider huggingface` and `--provider kaggle`. It discovers Hub configurations,
splits/shards or versioned Kaggle files and produces the same language/script report.
Kaggle `--list` and `--formats-only` use remote metadata; normal inspection caches
only the selected `--dataset-file`. Its YAML/JSON
settings file is the main input for repeatable runs, including optional whole-record
Devanagari filtering. Default language/script analysis uses five independent
20% random samples and strict-majority voting, with per-run evidence and measured
unique coverage. Full streaming passes can still be required to draw those samples.
Reports track each selected file's extension, compression,
record format, and local batch reader requirements. `--formats-only` catalogs
files without parsing records, including mixed and unsupported local formats. Streamlit's **Script results** view reads the saved report
without executing processing or maintaining separate filter rules. The interactive
explorer keeps configuration/split/shard selectors and displays a loaded report's
language/script evidence only for a matching source selection.
See [the inspector reference](scripts/inspect-dataset.md) for commands, settings,
metadata scope, filter behavior, and sample limits.

## Embedding-based corpus analysis

`cluster_dataset.py run` reuses the same Hugging Face config/split/shards and
Kaggle file selection, embeds every non-empty selected record, and clusters in
the original vector space. `pair` reads any two saved vectors; `sample` exports
B1 random, B2 density-weighted, or B3 SemDeDup selections with shared language
evidence and voting. Streamlit’s **Embedding results** view only reads saved
artifacts. See [the embedding guide](scripts/cluster-dataset.md) for model
research, full-dataset commands, 3D visualization, and resource limits.
