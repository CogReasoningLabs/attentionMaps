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
scan the corpus to count rows. File-size lookup is independent: when row counts
exist but download bytes are missing (for example `Sidharth1743/indicphi`), the
loader still requests the selected split's Viewer size. The UI labels original
files versus converted Parquet and identifies split versus configuration scope.
An optional file-size lookup failure leaves inspection usable with an unknown
file size; decoded Arrow bytes are never labelled as Hub storage. Optional
Viewer file-size fallback is skipped for pinned source revisions because the
Viewer describes the default revision.

`Nandan007/NepFakeV2` has a failed Viewer size job: automatic JSON discovery
includes both its records and `data/stats.json`. Its adapter explicitly selects
only `data/nepfakev2.csv`, so neither summary statistics nor duplicate JSON
exports enter the corpus. Use configuration empty or `default`, split `train`.
The CSV is downloaded into the Hugging Face cache and parsed in batches of
1,000 rows into disk-backed Arrow. Inspection gets exact row counts, typed
columns, original file bytes, and decoded bytes from that cache. Sampling and
the EDA CLI use the same adapter and resolved source revision; inspection does
not depend on the Viewer or published statistics.

Legacy Hub repositories such as `MBZUAI/Bactrian-X` still contain Python
loading scripts that modern `datasets` versions reject. The configuration/shard
discovery flow reads Bactrian-X's `data/<config>.json.gz` directly at the selected
commit. For Nepali, choose configuration `ne`, split `train`. Source sampling,
inspection, and embedding runs retain this JSON loader and the selected files.
The older inventory API and EDA CLI also support a converted-Parquet fallback
for legacy script repositories without executing repository code.

The converted-Parquet fallback cannot guarantee an arbitrary source commit or
tag; such requests fail explicitly instead of silently using another revision.
Use a versioned script-free export for those runs. Missing or partial conversions
also fail with an actionable error. See the official
[Parquet conversion API](https://huggingface.co/docs/dataset-viewer/parquet).

`csebuetnlp/CrossSum` has a legacy loading script and no working Viewer
conversion. Its adapter reads the original language-pair archive directly,
without executing repository Python. A configuration is required:
`nepali-nepali` selects Nepali articles/summaries, `english-nepali` selects
English articles with Nepali summaries, and `nepali-english` reverses that
pair. Splits are `train`, `validation`, and `test` (`validation` maps to `_val`).

The selected archive is downloaded once at a resolved Hub commit. Only the
exact selected JSONL member is read, without extracting archive paths onto the
filesystem, and records are written to disk-backed Arrow in batches of 1,000.
The four source columns (`source_url`, `target_url`, `summary`, `text`) remain
intact. Empty splits stay empty; no placeholder records are generated. Row and
decoded-byte counts describe the selected split. **Hub file size covers the
compressed language-pair archive, including all three splits**, which the UI
labels explicitly. Inspection, sampling, and the EDA CLI use this same reader.
See the [CrossSum dataset card](https://huggingface.co/datasets/csebuetnlp/CrossSum).

`facebook/flores` requires approved Hugging Face access and a read token from
that account (`HF_TOKEN`, including gated-repository read permission). Its public
splits are `dev` and `devtest`, not `train`; use `npi_Deva` for Nepali or
`eng_Latn-npi_Deva` for aligned English/Nepali. The loader validates these inputs,
checks access, resolves the requested revision to a commit, and uses the official
Parquet-based configuration with bounded streaming. File bytes come from the
selected split's Hub files, not the sum of all configurations or splits. Older
script-only revisions fail explicitly; the loader does not substitute another
repository. The workspace labels this source **Evaluation / benchmark**. See the
[FLORES access page and dataset card](https://huggingface.co/datasets/facebook/flores).

`google/IndicGenBench_flores_in` needs a separate nested-JSON adapter: its
records live under `examples`, and its Dataset Viewer is disabled. Use
configuration `ne` (the workspace's language selector) and split `validation`
or `test`; there is no `train` split. This selects both English→Nepali and
Nepali→English files and preserves `source`, `target`, `lang`, and
`translation_direction`. The top-level canary stays outside the data rows.

The selected language/split is cached as memory-mapped Arrow on disk, giving
exact row counts and a schema without the disabled Viewer API. The cache reads
one source JSON file at a time; it does download the selected files before
sampling. The resolved Hub commit is reused for subsequent sampling. This
source is labelled **Evaluation / benchmark**, including when the generic
schema selector remains on its default. Keep these benchmark examples out of
pretraining data. See the
[dataset card](https://huggingface.co/datasets/google/IndicGenBench_flores_in).

CrossSum, Flores-IN, and NepFakeV2 also participate in the configuration/split
selectors, `inspect_dataset.py`, and embedding source inspection. Discovery and
`--formats-only` read repository metadata without downloading records. Normal
inspection builds their disk-backed row index. These adapters require the
complete selected split's source files: Flores-IN keeps both translation
directions together, and CrossSum reads the chosen member of its pair archive.
Partial file selections are rejected instead of silently expanded. Ordinary
Hugging Face sources, including the FLORES Parquet export, retain individual
shard selection.

After registration, every source enters the same workflow:

```text
source inventory
  -> source sample and metadata exploration
  -> dataset role selection in WORKSPACE
  -> bounded deterministic sampling
  -> NFC normalization
  -> exact/near deduplication
  -> optional supervised D2 pruning
  -> EDA and persisted audit artifacts
```

Remote files are cached below `data/cache/source-imports/`. Ordinary Hugging Face
streaming does not materialize the complete split; the Flores-IN, NepFakeV2,
and CrossSum adapters instead use Hugging Face's disk caches for the selected
files or language-pair archive and the selected split's Arrow index.
KaggleHub uses its managed cache. Cached source material is excluded from Git.

## Streamlit usage

1. Run `python scripts/run_dataset_explorer.py`.
2. Choose **Data source** in the sidebar.
3. Enter the provider's standard identifier and select **Load ... source**.
   No dataset role or training schema is required to load it. New sources remain
   unclassified; known FLORES benchmarks retain their evaluation purpose.
4. For Hugging Face, choose **Dataset configuration**, **Dataset split**, and
   **All shards** or **Choose shards**. Sizes and samples follow this selection.
   For a staged folder or multi-file Kaggle dataset, select the dataset file.
5. Explore **Source sample** and **Metadata** to understand the actual fields,
   complete records, labels, and intended use.
6. In **WORKSPACE**, choose **Dataset role** after inspection. Its choices are
   the four training schemas plus evaluation; D2 is available only for
   task-specific supervised data. Changing the role preserves the loaded source
   and selected files, clears earlier in-memory processing results, and records
   the chosen schema in new workspace manifests.
7. Use **WORKSPACE** to sample, normalize, deduplicate, optionally prune, and
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
