# `download_nepali_pretrain_corpus.py`

## Purpose

Downloads Parquet shards and metadata from a Hugging Face dataset repository
into `data/raw/`. It records the resolved commit, selected files, sizes, and
download status in `download_manifest.json` so the source can be reproduced.

This command does not clean text, combine datasets, make splits, or tokenize.

## Basic usage

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  himalaya-ai/nepali-news-corpus
```

It also accepts a Hugging Face tree URL:

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  https://huggingface.co/datasets/himalaya-ai/nepali_pdf_corpus/tree/main/data
```

## Parameters

| Parameter | Default | Purpose and importance |
|---|---|---|
| `dataset` | `himalaya-ai/nepali-pretrain-corpus` | Positional dataset ID or Hugging Face dataset URL. This is the preferred way to select a repository. |
| `--dataset-id` | unset | Compatibility form of `dataset`. Do not pass both forms; doing so is an error. |
| `--output-dir` | `data/raw/<normalized-dataset-name>` | Destination for immutable source files and the manifest. Hyphens and other separators in the repository name are normalized to underscores. |
| `--revision` | URL revision or `main` | Branch, tag, or commit to resolve. A commit hash gives the strongest reproducibility. |
| `--path-prefix` | prefix from URL, otherwise none | Restricts downloads to one repository folder, such as `data`. Avoids unrelated large files. |
| `--max-workers` | `4` | Number of concurrent Hugging Face download workers. Must be positive. More workers may improve network throughput but increase simultaneous I/O. |
| `--dry-run` | false | Resolves metadata and prints selected files, total known size, and available disk without downloading. Use before large downloads. |
| `--all-files` | false | Downloads all files instead of the default Parquet/README/JSON selection. This can greatly increase disk usage. |

### `dataset`

Accepted forms:

```text
organization/dataset-name
https://huggingface.co/datasets/organization/dataset-name
https://huggingface.co/datasets/organization/dataset-name/tree/revision/folder
```

The URL form can carry both a revision and folder prefix. Explicit
`--revision` and `--path-prefix` values override those parsed from the URL.

### `--output-dir`

The output directory contains downloaded files plus
`download_manifest.json`. The script refuses to mix data into a non-empty
directory without a compatible manifest. If a manifest belongs to another
dataset or path prefix, the run fails instead of corrupting provenance.

Use an explicit versioned directory when comparing revisions:

```bash
--output-dir data/raw/nepali_news_corpus_v2
```

### `--revision`

Examples:

```bash
--revision main
--revision v1.0
--revision 64226874e938d803827729d57ccf7f6befc1b27e
```

The script resolves a branch or tag to a commit and stores that resolved commit
in the manifest. Passing a commit directly avoids changes if `main` moves.

### `--path-prefix`

Use this when Parquet files live under a specific repository folder:

```bash
--path-prefix data
```

Without `--all-files`, the default selection includes Parquet shards, JSON
metadata, and the repository README. With `--all-files`, every file below the
prefix is selected, plus the root README.

### `--max-workers`

This affects download speed only, not dataset contents. Reduce it on slow disks
or unstable networks:

```bash
--max-workers 1
```

### `--dry-run`

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  himalaya-ai/nepali-news-corpus \
  --dry-run
```

The command still queries Hugging Face metadata, so network access is required.
It does not download repository payloads. Before a real download, the script
requires at least 10% more free space than the known selected-file size.

### `--all-files`

Use only when non-Parquet repository files are genuinely required:

```bash
--all-files
```

The downloader still requires the selected repository content to contain at
least one Parquet file. Large audio/model archives can make this option
expensive.

## Output

```text
data/raw/<dataset-name>/
├── download_manifest.json
├── README.md
└── data/
    └── *.parquet
```

Manifest status is written as `downloading` before transfer and `complete` only
after every selected file is verified locally.

## Recommended commands

PDF corpus:

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  https://huggingface.co/datasets/himalaya-ai/nepali_pdf_corpus/tree/main/data \
  --output-dir data/raw/nepali_pdf_corpus
```

News corpus:

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  himalaya-ai/nepali-news-corpus \
  --output-dir data/raw/nepali_news_corpus
```
