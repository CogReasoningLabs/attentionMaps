# Data layout

The data pipeline has four explicit stages. Code may read from a later stage,
but it must never write back into an earlier one.

```text
data/
├── raw/                         # immutable downloads and extracted source files
│   ├── nepali_pdf_corpus/
│   ├── nepali_news_corpus/
│   └── nepali_music_lyrics/
├── cleaned/                     # one canonical, cleaned dataset per source
│   ├── nepali_pdf_corpus/
│   ├── nepali_news_corpus/
│   └── nepali_music_lyrics/
├── processed/                   # combined, deduplicated train/validation/test data
│   └── nepali_pretraining/
├── tokenized/                   # frozen tokenizer plus document token IDs
│   └── nepali_bpe/
└── cache/                       # disposable model, tokenizer, and framework caches
```

## Stage 1: raw source data

Raw data preserves the downloaded files, schemas, manifests, archive, and audio.
It is the reproducible input and is not cleaned in place.

## Stage 2: cleaned source data

Run:

```bash
python scripts/clean_pretraining_sources.py
```

Each source is converted independently to canonical sharded Parquet. The
default `strict` policy performs Unicode NFC normalization, decodes HTML
entities, removes HTML tags and script/style payloads, removes URLs/emails and
unsafe controls, filters non-Nepali-dominant documents, removes Latin and other
unsupported characters, and normalizes whitespace. Lyrics use a lower
Devanagari-ratio gate so mixed songs can retain their Nepali portions; fully
non-Nepali songs still fail the minimum Devanagari-content gate.

Every source directory contains `data/part-*.parquet` and a
`cleaning_manifest.json`. No train/validation/test decision is made here.
See [`docs/cleaning-cli-reference.md`](../docs/cleaning-cli-reference.md) for
the purpose, behavior, and tuning implications of every cleaning parameter.

## Stage 3: processed pretraining data

Run:

```bash
python scripts/build_pretraining_dataset.py
```

This stage reads only `data/cleaned/`, applies deterministic source sampling,
deduplicates exact cleaned text across all sources, assigns stable document
IDs, and creates `train.parquet`, `validation.parquet`, `test.parquet`, and
`build_manifest.json` under `data/processed/nepali_pretraining/`.

## Stage 4: tokenized data

Tokenizer training and encoding come after the processed text splits are
reviewed. The materialized pipeline writes to `data/tokenized/<run>/`, never in
`raw/`, `cleaned/`, or the processed text directory:

```text
data/tokenized/<run>/
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── train/part-*.parquet
├── validation/part-*.parquet
├── test/part-*.parquet
└── tokenization_manifest.json
```

Only `train.parquet` learns vocabulary and BPE merges. Validation and test are
encoded afterward with the frozen tokenizer. Profile-driven training may still
use `data/cache/` for its disposable derived cache.
