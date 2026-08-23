# Cleaning CLI parameter reference

This document describes the command-line parameters in
`scripts/clean_pretraining_sources.py`.

The script implements only the **raw → cleaned** stage:

```text
data/raw/<source> → data/cleaned/<source>/data/part-*.parquet
```

It cleans PDF, news, and song-level lyrics independently and converts them to a
common Parquet schema. It does **not** combine sources, sample training data,
deduplicate across sources, create train/validation/test splits, train a
tokenizer, or tokenize text. Those operations happen later in
`scripts/build_pretraining_dataset.py` and the tokenization pipeline.

## Default command

```bash
.venv/bin/python scripts/clean_pretraining_sources.py
```

With no arguments, this reads all three default raw sources, uses strict
Nepali-only cleaning, and writes separate cleaned datasets under
`data/cleaned/`.

## Parameter summary

| Parameter | Default | Purpose | Why it matters |
|---|---:|---|---|
| `--pdf-dir` | `data/raw/nepali_pdf_corpus/data` | Raw PDF Parquet directory | Selects the physical PDF input without changing source code. |
| `--news-dir` | `data/raw/nepali_news_corpus/data` | Raw news Parquet directory | Allows alternate downloads, versions, or mounted datasets. |
| `--lyrics-dir` | `data/raw/nepali_music_lyrics/archive/Final Dataset` | Extracted lyrics/audio metadata root | Must contain artist-level `metadata.csv` files used to construct songs. |
| `--output-root` | `data/cleaned` | Parent directory for all cleaned sources | Keeps cleaned data separate from immutable raw data. |
| `--sources` | all three sources | Chooses which sources to clean | Useful for rerunning or debugging one corpus without scanning the others. |
| `--mode` | `strict` | Controls whether Latin/unsupported characters survive | Defines whether output is Nepali-only or Nepali-dominant mixed-script text. |
| `--normalization` | `NFC` | Unicode normalization form | Prevents visually identical text from having different byte sequences. |
| `--min-devanagari-ratio` | `0.80` | PDF/news Nepali-dominance threshold | Rejects English-heavy or corrupted documents before character stripping. |
| `--lyrics-min-devanagari-ratio` | `0.10` | Song-level Nepali threshold | Preserves Nepali portions of mixed songs while rejecting non-Nepali songs. |
| `--min-devanagari-letters` | `10` | Minimum Devanagari letters in any document | Prevents punctuation, dates, or one Nepali word from passing the language gate. |
| `--min-characters` | `20` | Minimum visible cleaned PDF/news characters | Removes empty fragments and very short OCR debris. |
| `--lyrics-min-characters` | `10` | Minimum visible cleaned song characters | Uses a smaller limit appropriate for short lyric documents. |
| `--batch-size` | `512` | Rows read per Arrow scan batch | Trades scanning overhead against memory usage. |
| `--write-buffer-size` | `2,048` | Cleaned rows buffered before writing | Trades write frequency against memory usage. |
| `--rows-per-shard` | `100,000` | Approximate maximum rows per output file | Controls Parquet file size and downstream parallelism. |
| `--compression` | `zstd` | Parquet compression codec | Controls disk size and read/write CPU cost. |
| `--log-every` | `100,000` | Progress-report interval in scanned rows | Makes long news-corpus runs observable. Set to `0` to disable. |

## Input and output parameters

### `--pdf-dir`

Path to a directory containing the raw PDF corpus Parquet files. Each input row
must provide `id`, `text`, `url`, and `language` columns.

```bash
--pdf-dir data/raw/nepali_pdf_corpus/data
```

This parameter is important for reproducibility: different dataset revisions
can be stored at different paths and cleaned without modifying the script.

### `--news-dir`

Path to the raw news Parquet shards. It requires the same columns as the PDF
source.

```bash
--news-dir data/raw/nepali_news_corpus/data
```

News is the largest source, so accidentally pointing at duplicate or unrelated
shards can materially change training scale and source balance.

### `--lyrics-dir`

Path to the extracted music dataset. The cleaner reads one authoritative
`metadata.csv` from each artist directory and groups naturally ordered lyric
segments into one document per song. It deliberately does not ingest the
per-song CSV/XLSX copies because those duplicate the artist metadata rows.

```bash
--lyrics-dir "data/raw/nepali_music_lyrics/archive/Final Dataset"
```

### `--output-root`

Parent directory where the three cleaned source directories are created:

```text
<output-root>/
├── nepali_pdf_corpus/
│   ├── cleaning_manifest.json
│   └── data/part-*.parquet
├── nepali_news_corpus/
│   ├── cleaning_manifest.json
│   └── data/part-*.parquet
└── nepali_music_lyrics/
    ├── cleaning_manifest.json
    └── data/part-*.parquet
```

The script refuses to write into a non-empty selected output directory. This
protects an existing reproducible run from being silently mixed or overwritten.
Use a versioned root for another experiment:

```bash
--output-root data/cleaned/run_v2
```

### `--sources`

Accepts one or more of:

- `nepali_pdf`
- `nepali_news`
- `nepali_lyrics`

Clean only PDF data:

```bash
.venv/bin/python scripts/clean_pretraining_sources.py \
  --sources nepali_pdf \
  --output-root data/cleaned/pdf_experiment
```

Clean PDF and news but not lyrics:

```bash
--sources nepali_pdf nepali_news
```

## Text-policy parameters

### `--mode {strict,preserve}`

Both modes perform the common cleanup steps:

1. Unicode normalization.
2. HTML entity decoding.
3. HTML tag, comment, script, and style removal.
4. URL and email removal.
5. Unsafe Unicode control-character removal.
6. Nepali-content threshold checks.
7. Whitespace normalization.

The difference is character filtering after the document passes the Nepali
language gate:

- `strict` keeps Devanagari, whitespace, ASCII digits, Nepali-compatible
  punctuation, ZWNJ, and ZWJ. Latin/English and unsupported characters are
  removed. This is the current default research policy.
- `preserve` keeps mixed-script terms such as English names, acronyms, and
  technical expressions inside an already Nepali-dominant document.

```bash
--mode strict
```

The language ratio is checked **before strict character stripping**. This is
important: an English document containing only a few Nepali words must be
rejected, not converted into a tiny Nepali fragment that appears 100%
Devanagari afterward.

### `--normalization {NFC,NFKC}`

- `NFC` canonically composes Unicode sequences without aggressively changing
  compatibility characters. It is the safest default for corpus preservation.
- `NFKC` additionally folds compatibility forms. It can improve consistency in
  noisy OCR, but may change typography or symbols more aggressively.

Normalization makes hashing and exact deduplication reliable. Without it, two
visually identical strings can have different underlying code points.

### `--min-devanagari-ratio`

Minimum Devanagari share for PDF and news documents. The value is a fraction in
`[0, 1]`, so `0.80` means 80%, not 0.8% or 80.

The denominator contains Unicode **letters only**. Digits, spaces, and
punctuation do not reduce the ratio.

```text
Devanagari ratio = Devanagari letters / all Unicode letters
```

Examples:

- `1.00`: every letter is Devanagari.
- `0.80`: four out of every five letters are Devanagari.
- `0.20`: probably English-heavy, Romanized, or noisy text.

Increasing the threshold improves Nepali purity but rejects more code-mixed
content. Lowering it retains more documents but risks admitting English-heavy
or corrupted text.

### `--lyrics-min-devanagari-ratio`

Separate song-level threshold for lyrics. The default is `0.10` because a song
may contain English or Romanized segments alongside useful Devanagari verses.
After passing this gate, strict mode removes the Latin portions.

This parameter is separate because applying the PDF/news value of `0.80` to
lyrics would reject useful mixed songs. Increase it when the experiment demands
high-purity Devanagari songs:

```bash
--lyrics-min-devanagari-ratio 0.80
```

### `--min-devanagari-letters`

Minimum absolute number of Devanagari letters required in every accepted
document. The ratio alone is insufficient: a row containing one Nepali letter
and punctuation could otherwise have a ratio of 100%.

The default `10` is a basic content floor. Raising it removes more short
fragments but can discard valid headlines or short lyric lines.

### `--min-characters`

Minimum number of visible, non-whitespace characters remaining in cleaned PDF
and news documents. It is evaluated after cleanup and strict filtering.

The default `20` removes tiny OCR fragments and navigation debris. This is a
document-quality limit, not a tokenizer sequence-length setting.

### `--lyrics-min-characters`

Lyrics-specific visible-character minimum. The default `10` is lower because
song documents may be shorter than articles or PDF extractions.

## Performance and storage parameters

These parameters change resource usage and physical file layout. They should
not change which documents pass cleaning.

### `--batch-size`

Number of raw Parquet rows Arrow scans at a time.

- Lower values reduce peak input memory but cause more scan overhead.
- Higher values can improve throughput but retain more decoded text in memory.

`512` is a conservative default for corpora containing occasional very large
PDF documents.

### `--write-buffer-size`

Number of accepted rows accumulated before a Parquet write.

- Lower values use less memory but create more write calls and row groups.
- Higher values improve write efficiency but use more memory.

This buffer contains cleaned rows, so unusually long documents matter more
than the row count alone.

### `--rows-per-shard`

Approximate row count at which the active `part-*.parquet` file is closed and a
new shard begins. Because rows are flushed in buffers, a shard can exceed this
target by a small amount.

Larger shards mean fewer files; smaller shards can improve parallel loading and
make partial inspection easier. This controls physical storage layout, not
train/validation/test ratios.

### `--compression {zstd,snappy}`

- `zstd`: usually smaller files with higher compression/decompression CPU cost.
  Recommended when disk capacity is important.
- `snappy`: usually faster but produces larger files.

Compression is lossless and does not alter text.

### `--log-every`

Prints progress after every N scanned source rows. It does not affect the data.

```bash
--log-every 10000  # more frequent updates
--log-every 0      # disable progress output
```

## Recommended commands

### Current strict Nepali-only policy

```bash
.venv/bin/python scripts/clean_pretraining_sources.py \
  --mode strict \
  --normalization NFC \
  --min-devanagari-ratio 0.80 \
  --lyrics-min-devanagari-ratio 0.10 \
  --min-devanagari-letters 10 \
  --min-characters 20 \
  --lyrics-min-characters 10
```

These values are already the defaults; spelling them out makes an experiment
command self-documenting.

### Preserve English names and technical terms

```bash
.venv/bin/python scripts/clean_pretraining_sources.py \
  --mode preserve \
  --output-root data/cleaned/preserve_mixed_terms
```

Use this as a separate experiment. Do not mix its output into a strict run.

### Lower-memory run

```bash
.venv/bin/python scripts/clean_pretraining_sources.py \
  --batch-size 128 \
  --write-buffer-size 512 \
  --output-root data/cleaned/low_memory_run
```

## Running the next stage

For the default cleaned root:

```bash
.venv/bin/python scripts/build_pretraining_dataset.py \
  --pdf-fraction 1.0 \
  --news-fraction 0.01 \
  --lyrics-fraction 1.0 \
  --train-ratio 0.98 \
  --validation-ratio 0.01 \
  --test-ratio 0.01 \
  --output-dir data/processed/nepali_pretraining_small
```

For a versioned cleaning root such as `data/cleaned/run_v2`, pass its three
source directories explicitly:

```bash
.venv/bin/python scripts/build_pretraining_dataset.py \
  --pdf-dir data/cleaned/run_v2/nepali_pdf_corpus/data \
  --news-dir data/cleaned/run_v2/nepali_news_corpus/data \
  --lyrics-dir data/cleaned/run_v2/nepali_music_lyrics/data \
  --output-dir data/processed/nepali_pretraining_v2
```

The source fractions and split ratios belong to this second stage. They should
not be confused with cleaning thresholds such as `--min-devanagari-ratio`.
