# Attention Lab

A modular language-model research codebase covering causal pretraining,
continual pretraining, supervised finetuning, and—later—preference/reward
modeling and RLHF. Nepali is the current research language, while dataset,
tokenizer, and training choices must remain configurable for English and other
UTF-8 languages.

## What this project is for

The currently implemented path is strongest from corpus preparation through
decoder pretraining, generation, and attention inspection. Finetuning and
alignment are the next planned stages; they are not yet claimed as completed.

Attention maps are an optional diagnostic for pretraining and supervised
finetuning when a model exposes attention weights. They are not a required
dependency for reward modeling or RLHF.

It includes:

- local Parquet and Hugging Face dataset profiles
- SentencePiece subword tokenization with a legacy word-tokenizer fallback
- interchangeable attention variants
- a tiny transformer language model
- training and validation loops
- attention map visualization for a chosen input sentence

The staged roadmap and the decision between continual pretraining and
supervised finetuning are documented in
[`docs/project-progress.md`](docs/project-progress.md).

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
apps/                # interactive, read-only dataset inspection
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

For the current experiment, try the available Nepali BPE tokenizer hosted on
Hugging Face:

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml
```

The configuration records its Hub repository and commit for reproducibility
and preserves the downloaded vocabulary and IDs. The account namespace in the
repository ID is only a hosting location; it is not the tokenizer architecture
or a project model name. Use
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

Notebooks:

- `notebooks/01_nepali_pdf_corpus_exploration.ipynb` — PDF corpus
- `notebooks/02_nepali_news_corpus_exploration.ipynb` — news corpus
- `notebooks/03_nepali_music_lyrics_exploration.ipynb` — music/lyrics at segment and song level
- `notebooks/04_three_corpus_comparative_eda.ipynb` — matched comparison across all three sources
- `notebooks/05_final_pretraining_splits_inspection.ipynb` — final train/validation/test audit before tokenization
- `notebooks/Llama2_7B_Nepali_MultiDataset_QLoRA.ipynb` — selectable
  `saillab/alpaca-nepali-cleaned` or pipeline-cleaned translated-LIMA QLoRA
  instruction tuning for `meta-llama/Llama-2-7b-chat-hf`

### Interactive dataset explorer

Inspect raw, cleaned, processed, or tokenized Parquet records without loading a
whole dataset into memory:

```bash
streamlit run apps/dataset_explorer.py
```

Install `requirements.txt` in that same Python environment first.
HimalayaGPT's serialized tokenizer requires `tiktoken`. The app disables
Streamlit's source watcher because its inspection of Transformers can otherwise
import optional `torchvision` modules in this text-only application.

The app discovers the present Nepali datasets beneath `data/`, displays schema
and manifest information, draws uniform random records with a full-text view,
and creates configurable Unicode-aware word clouds. Pipeline selection is
limited to raw, cleaned, preprocessed, and tokenized data. Original-English
and Gemini/Gemma-translated Nepali LIMA views appear directly in the dataset
dropdown. Use the custom-path option for a Parquet file or directory elsewhere.
The dataset dropdown also includes the remote
`himalaya-ai/nepali-sft-dataset`. Its 3.8 GB training split is never downloaded
as one in-memory object: schema metrics come from streaming metadata, while
records, word clouds, and inference prompts use a bounded shuffle buffer.
Aya's 4,002-row Nepali training view is also available with the filter fixed to
`language_code=npi`. Hugging Face's Parquet predicate filtering is applied in
streaming mode, so multilingual rows are not loaded into the Streamlit process.
The train and test splits of `IRIIS-RESEARCH/Nepali-Text-Corpus` are available
through the same bounded Hugging Face streaming path. Its `Article` field is
used as natural-language text and `Source` remains available as provenance.
It also includes the Kaggle Nepali movie-review sentiment dataset and separate
lexicon/tweet views of the Kaggle Nepali hate-speech collection. Only the
selected Excel file is downloaded and cached; workbook inspection is read-only,
and previews use memory-bounded reservoir sampling.
The Kaggle OSCAR Nepali corpus is exposed through its smaller deduplicated
`ne_dedup.txt` file. Because that file is approximately 1.2 GB, the app asks for
explicit confirmation before downloading it. It then counts lines as a stream
and samples from random byte offsets without loading the corpus into memory.

The **Local base vs finetuned** tab discovers the final TinyLlama and GPT-2
PEFT adapters under `finetuned_models/`. It samples an instance from the
currently selected evaluation dataset (or accepts a custom prompt), applies a
temperature/top-p/top-k decoding grid, and displays each base-model output
beside its finetuned output. LIMA-style `HUMAN`/`ASSISTANT` records are split
into a model prompt and reference answer automatically. The adapter weights
and tokenizers are local; the corresponding base checkpoint is downloaded
from Hugging Face on first use unless cached-only mode is selected.

The **Tokenizer analysis** tab compares up to four base or repository
tokenizers without loading model weights. It reports the percentage of
non-special vocabulary entries containing Devanagari, token fragmentation on
the same Nepali sample, single-token Nepali word coverage, unknown-token rate,
and per-token pieces. Use the sample-efficiency metrics alongside vocabulary
coverage when choosing a base model; vocabulary percentage alone is not a
model-quality score.

The **Model comparison** tab can turn the selected dataset record into a
prompt and compare Gemini Flash, Gemini Flash-Lite, Google-hosted Gemma, the
two local finetuned adapters, the Hugging Face `google/gemma-4-E2B` base model,
HimalayaGPT 0.5B Instruct, Arkios 1B Chat, and
Hugging Face Inference Provider models over a shared temperature/top-p/top-k
grid. Local choices do not make an inference API request. Arkios and
HimalayaGPT and Gemma 4 E2B each have a dedicated implementation under
`attention_maps/inference/`.
Source text may come from the dataset, be entered manually, or be omitted in
prompt-only mode. System and user prompts are configured separately, and no
model is selected automatically, preventing accidental API calls.
Credentials are read only from the `GEMINI_API_KEY` and `HF_TOKEN` environment
variables; the UI never accepts or displays their values. Remote requests may
require provider access or incur cost.

The **Evaluation** tab streams bounded slices from a pinned public FLORES-200
mirror, using the `eng_Latn` and `npi_Deva` columns from its dev or devtest
split. `HF_TOKEN` is optional and only improves Hub rate limits. The tab compares
the LIMA teacher (`gemini-3.5-flash-lite`), local base and finetuned adapter
variants, HimalayaGPT, and Arkios on the same English sentences. It reports
per-instance and corpus chrF++ scores, coverage, latency, and a model comparison
chart without loading the complete benchmark into memory.

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
