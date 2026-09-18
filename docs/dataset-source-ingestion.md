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
5. For a staged folder or multi-file Kaggle dataset, select the dataset file.
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
