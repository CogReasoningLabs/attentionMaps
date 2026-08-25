# Attention Lab

A small, modular PyTorch project for training a tiny causal language model on
profile-defined UTF-8 corpora and visualizing attention maps. The pipeline is
language-neutral; the current research profile targets Nepali PDF text.

## What this project is for

This repo is designed to help you **learn and compare attention mechanisms** on a model small enough to train locally.

It includes:

- local Parquet and Hugging Face dataset profiles
- SentencePiece subword tokenization with a legacy word-tokenizer fallback
- interchangeable attention variants
- a tiny transformer language model
- training and validation loops
- attention map visualization for a chosen input sentence

## Attention variants included

- `softmax` — standard scaled dot-product causal self-attention
- `cosine` — cosine-similarity attention with causal masking
- `linear` — a simple feature-map-based linear attention approximation for causal decoding

These are intentionally compact educational implementations.

## Project structure

```text
attention_maps/
├── common/          # shared checkpoint and artifact utilities
├── training/        # decoder model, data loading, and training CLI
├── tokenization/    # tokenizer implementations and BPE pipeline
├── inference/       # generation and NepBERTa diagnostics
├── visualization/   # reusable attention plotting and checkpoint CLI
└── config.py        # shared profile/configuration schema

scripts/             # raw → cleaned → processed data commands
configs/             # tokenizer configuration
profiles/            # language/dataset experiment profiles
notebooks/           # corpus and split inspection
```

See [`docs/code-structure.md`](docs/code-structure.md) for module boundaries,
canonical commands, and compatibility entry points.

Project-level documentation:

- [`docs/project-progress.md`](docs/project-progress.md) — completed work,
  design decisions, and next research steps
- [`docs/nepali-pretraining-workflow.md`](docs/nepali-pretraining-workflow.md) —
  raw data through tokenization and decoder training
- [`docs/model-inference-and-attention.md`](docs/model-inference-and-attention.md) —
  decoder, NepBERTa, Gemma 4, and attention-map workflows

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Language profiles

The data pipeline is being moved to versioned, language-neutral experiment
profiles. Phase 1 provides validated profiles for:

- `profiles/en_wikitext103.json` — the existing English corpus
- `profiles/ne_wikipedia.json` — Nepali Wikipedia with deterministic held-out
  split settings
- `profiles/ne_pdf_local.json` — the prepared local Nepali PDF corpus
- `profiles/local_text.example.json` — a template for any UTF-8 local corpus

Each profile describes the corpus source, text column, language metadata,
split strategy, and tokenizer training settings. Language is metadata rather
than branching logic: the same schema can describe any UTF-8 corpus.

`python -m attention_maps.training --profile ...` resolves the configured
dataset, trains or restores the configured tokenizer, encodes all splits, and
stores the profile and tokenizer inside the checkpoint. Encoded tensors are
cached under `data/cache/`.

Profiles are UTF-8 JSON and can be validated without downloading their data:

```bash
python -c "from attention_maps.config import load_experiment_profile; print(load_experiment_profile('profiles/ne_wikipedia.json'))"
```

Run the configuration tests with:

```bash
python -m unittest discover -s tests -v
```

## Train

Complete command and parameter documentation is indexed in
[`docs/scripts/README.md`](docs/scripts/README.md).

### Clean and standardize the three-source corpus

The pipeline is deliberately staged as `raw → cleaned → processed splits`.
Raw downloads are never edited in place. First clean each source independently:

```bash
python scripts/clean_pretraining_sources.py
```

The default strict policy removes English/Latin debris, HTML tags and entities,
script/style payloads, URLs, emails, unsafe control characters, and malformed
spacing while retaining Devanagari, digits, and Nepali-compatible punctuation.
It writes canonical source datasets and cleaning manifests under
`data/cleaned/`.

Then combine the cleaned sources, deduplicate, sample, and split:

```bash
python scripts/build_pretraining_dataset.py \
  --output-dir data/processed/nepali_pretraining \
  --sample-fraction 1.0 \
  --train-ratio 0.98 \
  --validation-ratio 0.01 \
  --test-ratio 0.01
```

`--sample-fraction` creates deterministic dataset sizes. For example, `0.10`
selects approximately 10% of each source. Source-specific fractions are
multiplied by the global fraction, so the following keeps all PDF and lyrics
documents while using approximately 2% of news:

```bash
python scripts/build_pretraining_dataset.py \
  --output-dir data/processed/nepali_balanced_small \
  --sample-fraction 1.0 \
  --pdf-fraction 1.0 \
  --news-fraction 0.02 \
  --lyrics-fraction 1.0
```

Every processed record has the canonical columns `doc_id`, `text`, `source`,
`source_id`, `language`, `url`, `text_sha256`, and `metadata_json`. Cleaning is
recorded in each source's `cleaning_manifest.json`; sampling, exact cross-source
deduplication, and document-level split assignment are recorded in
`build_manifest.json`. See `data/README.md` for the complete directory contract.

Encode all three splits with the exact pinned `Aananda-giri/NepaliBPE`
tokenizer:

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/aananda_nepali_bpe.yaml
```

The pipeline preserves the published vocabulary and IDs. Use
`configs/tokenizer/nepali_bpe.yaml` only when intentionally training a new,
independent BPE vocabulary. See the
[`tokenization reference`](docs/scripts/tokenize-pretraining-data.md) for every
parameter and output field.

After tokenization, verify the packed-token training pipeline:

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_smoke.yaml
```

Then start the configured decoder run with
`configs/training/nepali_decoder_small.yaml`. The complete tokenization,
packing, embedding, and positional-encoding flow is documented in
[`docs/tokenization-and-decoder-training.md`](docs/tokenization-and-decoder-training.md).

EDA notebooks:

- `notebooks/01_nepali_pdf_corpus_exploration.ipynb` — PDF corpus
- `notebooks/02_nepali_news_corpus_exploration.ipynb` — news corpus
- `notebooks/03_nepali_music_lyrics_exploration.ipynb` — music/lyrics at segment and song level
- `notebooks/04_three_corpus_comparative_eda.ipynb` — matched comparison across all three sources
- `notebooks/05_final_pretraining_splits_inspection.ipynb` — final train/validation/test audit before tokenization

### Nepali PDF corpus

Download the pinned dataset files. To run only the PDF part of the shared
cleaning stage, select that source explicitly:

```bash
python scripts/download_nepali_pretrain_corpus.py \
  https://huggingface.co/datasets/himalaya-ai/nepali_pdf_corpus/tree/main/data

python scripts/clean_pretraining_sources.py --sources nepali_pdf
```

The checked-in `profiles/ne_pdf_local.json` trains an 8,000-piece BPE
SentencePiece tokenizer from 50,000 bounded text segments, then encodes every
accepted document. Start causal next-token pretraining with:

```bash
python -m attention_maps.training \
  --profile profiles/ne_pdf_local.json \
  --attention softmax \
  --epochs 3 \
  --batch-size 8 \
  --seq-len 128 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 2 \
  --d-ff 256 \
  --no-moe \
  --run-name ne_pdf_run
```

Use `--device cuda` on a CUDA machine. The first run builds the tokenizer and
encoded cache; later runs with the same profile and source files reuse it. Pass
`--no-data-cache` only when intentionally rebuilding. Checkpoints and metrics
are written beneath `runs/<run-name>/`.

The objective is standard autoregressive language modeling. For each token
sequence `(x1, ..., xT)`, training minimizes the mean cross-entropy of predicting
`x(t+1)` from the preceding Nepali/subword context `(x1, ..., xt)`.

### Legacy WikiText path

Standard softmax attention:

```bash
python -m attention_maps.training --attention softmax --epochs 3
```

Cosine attention:

```bash
python -m attention_maps.training --attention cosine --epochs 3
```

Linear attention:

```bash
python -m attention_maps.training --attention linear --epochs 3
```

## NepBERTa fill-mask inference

`NepBERTa/NepBERTa` is a bidirectional masked-language model, not a causal text
generator. Its native inference task predicts a token that replaces `[MASK]`.
Install its optional TensorFlow dependencies separately, then run:

```bash
.venv/bin/pip install -r requirements-nepberta.txt

.venv/bin/python -m attention_maps.inference.nepberta \
  "नेपाल एक [MASK] देश हो।" \
  --top-k 5
```

The portable `{mask}` placeholder is also accepted, and `--json` returns
machine-readable output. The first run downloads roughly 534 MB of model files
from Hugging Face.

## Visualize attention

After training, checkpoints are saved under `runs/<run-name>/checkpoints/`.

Example:

```bash
python -m attention_maps.visualization \
  --checkpoint runs/nepali_decoder_v1/checkpoints/best.pt \
  --text "नेपाल एउटा सुन्दर देश हो।" \
  --layer 0 \
  --head 0 \
  --run-name sentence_01
```

This saves a heatmap under `artifacts/`. See
[`docs/attention-visualization.md`](docs/attention-visualization.md) for the
reusable plotting API and complete examples.

Generate and inspect decoder output with:

```bash
.venv/bin/python -m attention_maps.inference.generate \
  --checkpoint runs/nepali_decoder_v1/checkpoints/best.pt \
  --prompt "नेपालको राजधानी" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --do-sample \
  --seed 42 \
  --device cuda \
  --gen-name sample_01
```

See [`docs/generation-inference.md`](docs/generation-inference.md) for greedy,
sampled, and generation-attention inspection.

## Notes

- `linear` attention in this repo is an educational approximation. It is useful for experiments and intuition, but it does **not** produce the exact same attention matrix as softmax attention.
- For `softmax` and `cosine`, full attention maps are directly available.
- For `linear`, the script produces an approximate token-token influence map by replaying prefix computations.

## Good first experiments

1. Train all three attention variants for 1–3 epochs.
2. Compare validation loss.
3. Visualize the same sentence with each checkpoint.
4. Increase sequence length and see which variant degrades less in runtime.
5. Reduce model size and check whether the qualitative patterns remain stable.

## Suggested laptop-safe starting point

- embedding dim: 128
- heads: 4
- layers: 2
- sequence length: 64
- batch size: 32 on GPU, 8–16 on CPU
