# `build_pretraining_dataset.py`

## Purpose

Consumes the three canonical datasets from `data/cleaned/` and creates the
final text dataset used before tokenization:

```text
data/cleaned/ → sampling → exact deduplication → deterministic splits
             → data/processed/<run>/train.parquet
             → data/processed/<run>/validation.parquet
             → data/processed/<run>/test.parquet
```

It does not clean raw text and does not tokenize. Run
`clean_pretraining_sources.py` first.

## Default command

```bash
.venv/bin/python scripts/build_pretraining_dataset.py
```

The default includes 100% of all three cleaned sources and creates 98% train,
1% validation, and 1% test splits under
`data/processed/nepali_pretraining/`.

## Parameters

| Parameter | Default | Purpose and importance |
|---|---:|---|
| `--pdf-dir` | `data/cleaned/nepali_pdf_corpus/data` | Cleaned canonical PDF input directory. |
| `--news-dir` | `data/cleaned/nepali_news_corpus/data` | Cleaned canonical news input directory. |
| `--lyrics-dir` | `data/cleaned/nepali_music_lyrics/data` | Cleaned canonical song-document input directory. |
| `--output-dir` | `data/processed/nepali_pretraining` | Destination for split Parquet files and `build_manifest.json`. Must be empty. |
| `--sample-fraction` | `1.0` | Global deterministic inclusion fraction multiplied into every source fraction. |
| `--pdf-fraction` | `1.0` | Fraction of cleaned PDF documents considered for output. |
| `--news-fraction` | `1.0` | Fraction of cleaned news documents considered for output. Usually the primary size control because news is largest. |
| `--lyrics-fraction` | `1.0` | Fraction of cleaned song documents considered for output. Values cannot oversample this small source. |
| `--train-ratio` | `0.98` | Share of selected unique documents assigned to train. |
| `--validation-ratio` | `0.01` | Share assigned to validation. |
| `--test-ratio` | `0.01` | Share assigned to test. |
| `--seed` | `42` | Controls deterministic sampling and split assignment. |
| `--deduplicate` / `--no-deduplicate` | deduplicate | Enables or disables exact cleaned-text deduplication across all sources. |
| `--batch-size` | `512` | Rows scanned from cleaned Parquet per Arrow batch. Controls memory/throughput, not selection. |
| `--write-buffer-size` | `2,048` | Output records buffered per split before a Parquet write. |
| `--log-every` | `100,000` | Progress interval in scanned rows. `0` disables progress messages. |
| `--compression` | `zstd` | Output Parquet codec; alternatives: `zstd`, `snappy`. |

## Input parameters

### `--pdf-dir`, `--news-dir`, `--lyrics-dir`

Each directory must contain Parquet files emitted by the cleaning stage. The
builder requires canonical columns including `source`, `source_id`, `text`,
`language`, `url`, `text_sha256`, and `metadata_json`. It verifies that the
source label inside every row matches the expected input.

When cleaning into a versioned root, pass all three paths:

```bash
--pdf-dir data/cleaned/run_v2/nepali_pdf_corpus/data \
--news-dir data/cleaned/run_v2/nepali_news_corpus/data \
--lyrics-dir data/cleaned/run_v2/nepali_music_lyrics/data
```

Even a source with fraction `0` currently needs a valid cleaned input directory.

### `--output-dir`

The script refuses a non-empty directory. This prevents two runs with different
fractions, seeds, or ratios from being silently mixed. Use a descriptive,
versioned name:

```bash
--output-dir data/processed/nepali_pretraining_news_1pct_seed42
```

## Source sampling parameters

The effective inclusion probability for each source is:

```text
effective PDF fraction    = sample_fraction × pdf_fraction
effective news fraction   = sample_fraction × news_fraction
effective lyrics fraction = sample_fraction × lyrics_fraction
```

All fraction values must be in `[0, 1]`. `0.01` means approximately 1%; use
`1.0`, not `100`, for 100%.

### `--sample-fraction`

A global scale control. For example:

```bash
--sample-fraction 0.10 \
--pdf-fraction 1.0 \
--news-fraction 0.20 \
--lyrics-fraction 1.0
```

Effective fractions become 10% PDF, 2% news, and 10% lyrics. Because lyrics is
small, a global fraction can remove too many songs; source-specific fractions
are generally easier to reason about.

### `--pdf-fraction`, `--news-fraction`, `--lyrics-fraction`

These are inclusion rates, not final mixture percentages. The sources have very
different sizes, so setting every value to `1.0` does not make a 33/33/33
mixture. News remains dominant.

Sampling is stable hash-based filtering over `(seed, source, source_id)`:

- It does not depend on Parquet row order.
- Repeating the same command selects the same documents.
- Counts are approximate, not exact quotas.
- Adding new source rows does not reshuffle existing selections.
- Fractions below `1` downsample only; they cannot oversample lyrics.

For more lyrics influence, use training-time source weights or a weighted
sampler later rather than duplicating rows before splitting.

## Split parameters

### `--train-ratio`, `--validation-ratio`, `--test-ratio`

The three values must be non-negative and sum to exactly `1.0` within numerical
tolerance. Training must be positive.

```bash
--train-ratio 0.98 \
--validation-ratio 0.01 \
--test-ratio 0.01
```

Splits are assigned by hashing the stable `doc_id` with the seed. Therefore:

- Every document belongs to exactly one split.
- Row order does not affect the split.
- Repeating the run with the same seed preserves membership.
- Ratios produce approximate counts rather than exact quotas.

Sampling happens before deduplication, and deduplication happens before split
assignment. This prevents an exact duplicate from appearing in two splits.

### `--seed`

The seed controls both source sampling and split assignment. Changing it creates
a different sample and different held-out membership. Record it for every
experiment; it is saved in `build_manifest.json`.

## Deduplication

### `--deduplicate`, `--no-deduplicate`

Exact deduplication is enabled by default. It hashes cleaned UTF-8 text and
keeps the first occurrence encountered in source order:

```text
nepali_pdf → nepali_news → nepali_lyrics
```

This removes repeated PDF pages/articles and prevents exact leakage. It is not
semantic or near-duplicate detection: paraphrases and slightly different text
remain.

Disable only for a controlled ablation:

```bash
--no-deduplicate
```

The builder also rejects duplicate `(source, source_id)` identifiers that would
otherwise produce the same `doc_id`.

## Performance and disk parameters

### `--batch-size`

Controls cleaned input rows decoded per Arrow scan batch. Lower values reduce
peak memory; higher values can reduce scanning overhead. It does not affect
which documents are selected.

### `--write-buffer-size`

Controls how many output records are accumulated separately for each split
before writing. Lower values reduce memory but increase write frequency.

### `--log-every`

Progress reporting only:

```bash
--log-every 10000
--log-every 0
```

### `--compression`

- `zstd`: smaller output, usually more CPU work; recommended for limited disk.
- `snappy`: faster compression with larger output.

Compression is lossless and does not alter sampling or text.

## Disk-size interpretation

Approximate disk use depends on:

1. Effective source fractions.
2. Text lengths in selected documents.
3. Exact duplicates removed.
4. Compression codec and content compressibility.
5. Split ratios.

Split disk sizes are approximately proportional to split ratios only when text
length distributions are similar. Hash-based assignment makes this generally
true for large corpora, but very small validation/test sets can vary.

## Output schema

Every split has these columns:

| Column | Meaning |
|---|---|
| `doc_id` | Stable SHA-256 identifier derived from source and source ID |
| `text` | Cleaned training text |
| `source` | `nepali_pdf`, `nepali_news`, or `nepali_lyrics` |
| `source_id` | Original document/song identifier |
| `language` | Language metadata, normally `ne` |
| `url` | Source provenance; not part of model text |
| `text_sha256` | Exact cleaned-text hash used for deduplication checks |
| `metadata_json` | Source and cleaning metadata encoded as JSON text |

`build_manifest.json` records inputs, embedded cleaning manifests, fractions,
effective fractions, split ratios/counts, deduplication statistics, seed, and
canonical schema.

## Recommended run sizes

Small research run—retain all PDF/lyrics and about 1% of news:

```bash
.venv/bin/python scripts/build_pretraining_dataset.py \
  --pdf-fraction 1.0 \
  --news-fraction 0.01 \
  --lyrics-fraction 1.0 \
  --train-ratio 0.98 \
  --validation-ratio 0.01 \
  --test-ratio 0.01 \
  --seed 42 \
  --output-dir data/processed/nepali_pretraining_small
```

Medium run:

```bash
--pdf-fraction 1.0 \
--news-fraction 0.10 \
--lyrics-fraction 1.0
```

Full run:

```bash
--pdf-fraction 1.0 \
--news-fraction 1.0 \
--lyrics-fraction 1.0
```
