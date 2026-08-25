# Nepali pretraining workflow

This is the canonical end-to-end path for preparing and training on the three
current Nepali sources:

1. `himalaya-ai/nepali_pdf_corpus`
2. `himalaya-ai/nepali-news-corpus`
3. local music lyrics under
   `data/raw/nepali_music_lyrics/archive/Final Dataset`

## Data contract

```text
data/raw/          immutable source files
    ↓ clean_pretraining_sources.py
data/cleaned/      one canonical Parquet dataset per source
    ↓ build_pretraining_dataset.py
data/processed/    combined train/validation/test text splits
    ↓ attention_maps.tokenization
data/tokenized/    frozen tokenizer and document-level token IDs
    ↓ attention_maps.training
runs/              checkpoints, metrics, generations, and attention maps
```

The directories contain large or reproducible artifacts and are ignored by
Git. Only [data/README.md](../data/README.md) documents their contract.

## 1. Download Hugging Face sources

Use the downloader for each remote dataset. Local music files do not require a
download step.

```bash
.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  https://huggingface.co/datasets/himalaya-ai/nepali_pdf_corpus/tree/main/data \
  --output-dir data/raw/nepali_pdf_corpus

.venv/bin/python scripts/download_nepali_pretrain_corpus.py \
  himalaya-ai/nepali-news-corpus \
  --output-dir data/raw/nepali_news_corpus
```

Use a Hub revision or commit with `--revision` when reproducibility matters.

## 2. Clean and standardize each source

```bash
.venv/bin/python scripts/clean_pretraining_sources.py
```

The default strict policy performs Unicode NFC normalization, removes markup,
script/style bodies, URLs, emails, unsafe controls, Latin/English debris, and
invalid punctuation remnants, then applies minimum Devanagari and length
thresholds. Lyrics use deliberately lower ratio/length thresholds because
song text is shorter and more likely to contain mixed metadata.

Output records share these important fields:

```text
doc_id, text, source, source_id, language,
url, text_sha256, metadata_json
```

Each cleaned source also has `cleaning_manifest.json`. See the
[cleaning parameter reference](cleaning-cli-reference.md) before changing
language thresholds.

## 3. Sample, deduplicate, and split

The source fractions control composition; split ratios control where the
selected documents go. They are separate decisions.

```bash
.venv/bin/python scripts/build_pretraining_dataset.py \
  --pdf-fraction 1.0 \
  --news-fraction 0.02 \
  --lyrics-fraction 1.0 \
  --train-ratio 0.98 \
  --validation-ratio 0.01 \
  --test-ratio 0.01 \
  --output-dir data/processed/nepali_pretraining_small
```

`--sample-fraction` scales all source fractions together. With the command
above, every accepted PDF/lyrics document is eligible while approximately 2%
of news is eligible. Sampling and split assignment are seeded and stable.
Exact cleaned text is deduplicated across all sources before the document-level
split is written.

Inspect `build_manifest.json` for exact input, accepted, duplicate, sampled,
and per-split counts. Full parameter details are in the
[dataset builder reference](scripts/build-pretraining-dataset.md).

## 4. Inspect before tokenization

The notebooks are ordered by pipeline stage:

- `01_nepali_pdf_corpus_exploration.ipynb`
- `02_nepali_news_corpus_exploration.ipynb`
- `03_nepali_music_lyrics_exploration.ipynb`
- `04_three_corpus_comparative_eda.ipynb`
- `05_final_pretraining_splits_inspection.ipynb`

The final notebook verifies schemas, source proportions, split sizes, text
lengths, duplicate hashes, representative full records, and train/test
separation.

## 5. Try the available Hugging Face BPE tokenizer

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml
```

This experiment downloads an existing third-party Nepali BPE tokenizer from a
configured Hugging Face repository. The `account/repository` format is a Hub
location; its account namespace is not the tokenizer architecture or the name
of a model in this project. The recorded commit makes the experiment
reproducible, while the downloaded vocabulary and IDs remain unchanged. The
pipeline adds configured BOS/EOS IDs already present in that vocabulary and
writes document-level `input_ids` for all three splits. It does not train BPE.

Use `configs/tokenizer/nepali_bpe.yaml` only for a deliberate new-tokenizer
experiment. That path learns merges exclusively from the training split and
then freezes the result before encoding validation/test.

## 6. Verify and run decoder training

Run the five-step smoke configuration first:

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_smoke.yaml
```

Then start the small configured experiment:

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_small.yaml \
  --device cuda \
  --run-name nepali_decoder_v1
```

The small YAML currently selects a 4-layer, 256-dimensional causal decoder
with 8 attention heads and a 4-expert top-2 MoE feed-forward path. Setting
`use_moe: false` or passing `--no-moe` switches to the dense feed-forward
network; it does not select a different YAML automatically.

Materialized token IDs are packed into fixed blocks. Each target is the input
shifted by one token, causal masking prevents future access, learned token and
absolute-position embeddings are summed, and cross-entropy trains next-token
prediction. See [tokenization and decoder training](tokenization-and-decoder-training.md)
for the detailed tensor flow.

## Reproducibility checklist

- pin Hub dataset and tokenizer revisions;
- preserve raw downloads and generated manifests outside Git;
- record source fractions, split ratios, and random seed;
- never train tokenizer merges on validation or test;
- keep tokenization and training YAML files with the run;
- evaluate with `best.pt`; use `latest.pt` only when resuming optimizer state.
