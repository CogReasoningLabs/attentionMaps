# Attention Lab

A modular language-model research codebase covering causal pretraining,
continual pretraining, supervised finetuning, and—later—preference/reward
modeling and RLHF. Nepali is the current research language, while dataset,
tokenizer, and training choices must remain configurable for English and other
UTF-8 languages.

## What this project is for

The current priority is dataset research: understanding candidate corpora,
establishing cleaning and deduplication evidence, and building reproducible
synthetic datasets. Earlier decoder-pretraining and attention-visualization
experiments remain in the repository as useful foundations, but they are not
treated as a completed model or attention research program.

The intended sequence is **dataset research → frozen/versioned training data →
model building → attention analysis and visualization**. Finetuning, alignment,
and comprehensive attention research are not yet claimed as complete.

It includes:

- local Parquet and Hugging Face dataset profiles
- SentencePiece subword tokenization with a legacy word-tokenizer fallback
- interchangeable attention variants
- a tiny transformer language model
- training and validation loops
- a prototype attention-map CLI from earlier model experiments

The staged roadmap and the decision between continual pretraining and
supervised finetuning are documented in
[`docs/project-progress.md`](docs/project-progress.md).

## Three project paths

### 1. Dataset exploration application — active

Use this first to select local or remote data, verify a bounded source sample,
run the gated cleaning pipeline, and perform EDA only on its preprocessed
output. Synthetic pipeline artifacts remain separate from the corpus catalog.

```bash
.venv/bin/python scripts/run_dataset_explorer.py
```

This supervised launcher shuts down Streamlit and any active model, EDA, or
upload workers together. Press `Ctrl+C` once for a graceful stop; press it a
second time to force an immediate stop. Extra Streamlit flags can be appended,
for example `--server.port=8502`.

Primary outputs are clean workspace datasets, deduplication audits, manifests,
and a fixed set of reproducible post-preprocessing EDA reports. See
[Path 1 detailed guide](#path-1-detailed-guide--dataset-exploration-application).

### 2. Synthetic dataset generation application — active foundation

Use this after selecting and understanding source data. It creates bounded,
structured records with hosted or local models, displays live progress, and
persists successful runs for JSONL/CSV export.

```bash
cp .env.example .env
.venv/bin/python -m streamlit run apps/dataset_generator.py
```

The **Synthetic data generation** workspace is registry-driven: LIMA
translation is the first dataset family, with translated-Nepali and
original-English variants. Future families and variants can be registered
without adding another hard-coded application path. See
[Path 2 detailed guide](#path-2-detailed-guide--synthetic-dataset-generation-application).

### 3. Model building and attention visualization — planned next research path

This path begins after the dataset survey, cleaning policy, deduplication rules,
and training-data versions are finalized. The repository contains earlier
pretraining code, checkpoints, three experimental attention implementations,
and a command-line heatmap prototype. Those pieces are foundations—not evidence
that the model-building or attention study is finished.

The next phase must define controlled model experiments, train comparable
checkpoints on frozen datasets, validate attention extraction across layers,
heads, tokens, and generation steps, and then design an interactive
visualization experience. See
[Path 3 roadmap](#path-3-roadmap--model-building-and-attention-visualization).

## Existing experimental attention foundations

- `softmax` — standard scaled dot-product causal self-attention
- `cosine` — cosine-similarity attention with causal masking
- `linear` — a simple feature-map-based linear attention approximation for causal decoding

These are intentionally compact educational implementations from the earlier
pretraining phase. They require further research and validation before being
used for conclusions about model behavior.

## Project structure

```text
attention_maps/
├── common/          # shared checkpoint and artifact utilities
├── explorer/        # dataset catalog, bounded inspection, and text services
├── generation/      # synthetic-family registry, records, exports, persistence
├── training/        # decoder model, data loading, and training CLI
├── tokenization/    # tokenizer implementations and BPE pipeline
├── inference/       # hosted/local model loading, generation, and comparison
├── evaluation/      # FLORES, NLUE, and decoder benchmark services
├── visualization/   # prototype attention plotting and checkpoint CLI
└── config.py        # shared profile/configuration schema

scripts/             # raw → cleaned → processed data commands
configs/             # EDA, tokenizer, and training configuration
profiles/            # language/dataset experiment profiles
notebooks/           # corpus and split inspection
apps/
├── components/      # UI shared by explorer and generator
├── explorer_tabs/   # feature-oriented explorer renderers
├── dataset_explorer.py
└── dataset_generator.py
```

See [`docs/code-structure.md`](docs/code-structure.md) for module boundaries,
canonical commands, and compatibility entry points.
The refactor rules and enforced module-size budget are documented in
[`docs/modularization.md`](docs/modularization.md).

Project-level documentation:

- [`docs/project-progress.md`](docs/project-progress.md) — completed work,
  design decisions, and next research steps
- [`docs/eda-survey-progress.md`](docs/eda-survey-progress.md) — active dataset
  survey work log
- [`docs/eda-cleaning-notes.md`](docs/eda-cleaning-notes.md) — EDA metrics and
  the planned auditable cleaning/deduplication logic shown in the UI
- [`docs/google-drive-workspace.md`](docs/google-drive-workspace.md) — team
  workspace uploads, credentials, resumable transfers, and size buckets
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

## Dataset preparation and historical training reference

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

## Path 1 detailed guide — Dataset exploration application

Inspect raw, cleaned, processed, or tokenized Parquet records without loading a
whole dataset into memory:

```bash
.venv/bin/python scripts/run_dataset_explorer.py
```

Install `requirements.txt` in that same Python environment first.
HimalayaGPT's serialized tokenizer requires `tiktoken`. The app disables
Streamlit's source watcher because its inspection of Transformers can otherwise
import optional `torchvision` modules in this text-only application.

The app is intentionally limited to four tabs: **WORKSPACE**, **Source sample**,
**Inference**, and **Metadata**. WORKSPACE enforces Sampling → NFC normalization → multi-stage
deduplication → EDA. EDA is unavailable until preprocessing succeeds and shows
only the fixed corpus-profile, document-size, text-structure, n-gram, and
residual-duplicate visualizations plus a fixed-default WordCloud. Source sample is a bounded verification view;
Inference contains one selector for model comparison, local base-vs-finetuned,
translation evaluation, or decoder benchmarks. Metadata contains schema and
manifest evidence. Use the custom-path option for a Parquet file or directory
elsewhere.

The dataset dropdown also includes the remote
`himalaya-ai/nepali-sft-dataset`. Its 3.8 GB training split is never downloaded
as one in-memory object: schema metrics come from streaming metadata and
workspace samples use a bounded shuffle buffer.
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

## Path 2 detailed guide — Synthetic dataset generation application

Generate bounded, structured training-data batches from the same local,
Hugging Face, and Kaggle sources exposed by the dataset explorer:

```bash
cp .env.example .env
.venv/bin/python -m streamlit run apps/dataset_generator.py
```

Choose **Dataset catalog** for ordinary source corpora or **Synthetic data
generation** for registry-managed pipeline families and variants. Both this app
and the explorer use the same synthetic-family registry, so a newly registered
family appears consistently in both places.

Add only the provider credentials you intend to use to `.env`. The app does
not make a request until **Generate dataset** is pressed, shows the request
count first, and applies configurable local per-minute and per-day limits.
Its backend selector distinguishes **Gemma 4 · Google API** from **Gemma 4
E2B Base · Hugging Face local**; the latter runs locally after its weights are
available.
Local PEFT adapters, including `ckpt-504-llama7b`, appear in this selector when
their adapter directory contains `adapter_config.json` and adapter weights.
Set `ATTENTION_MAPS_FINETUNED_MODELS_ROOT` when the weights live outside the
active repository checkout. PEFT quantization defaults to `auto`: 7B and larger
base models use CUDA 4-bit NF4 loading, while smaller adapters remain
unquantized. Install the project requirements in a GPU-enabled environment;
large adapters stop with a clear error instead of falling back entirely to CPU
float32. When the quantized model exceeds VRAM, Accelerate may keep overflow
modules in full precision on CPU or spill them under
`data/cache/model_offload/`; this is slower but avoids the dispatch failure.
During a run, the UI shows the active model and record, live completion counts,
the latest response, and a rolling table of recently generated records.
Successful records can be downloaded as JSONL or CSV using the configured
output schema. Failed provider calls remain visible as diagnostics but are not
included in training-data downloads. Run metadata and records are persisted in
the ignored `.dataset_generator.sqlite3` database so successful records from a
previous run can be downloaded after restarting the app.

## Historical pretraining reference

The following commands document earlier decoder-pretraining experiments. Keep
them for reproducibility, but do not treat them as the current project phase or
as a completed attention study.

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

## Multi-dataset corpus EDA

The survey EDA pipeline streams heterogeneous Hugging Face corpora through a
shared schema and metric contract. It keeps bounded numeric samples and compact
duplicate/vocabulary state, but does not persist sampled source text.
Per-dataset outputs include log-scaled document-size histograms, sentence/line
KDE plots, top 2/3/4-gram charts, seed-term co-occurrence networks, and the CSV
tables needed to reproduce each figure.

Run one configured dataset:

```bash
.venv/bin/python -m attention_maps.eda \
  --config configs/eda/nepali_corpus_survey.json \
  --datasets nepali-corpus-compile \
  --output-dir artifacts/eda/smoke
```

Remove `--datasets` to run all 17 configured corpora. See the
[methodology](docs/eda-survey-methodology.md) and active
[work log](docs/eda-survey-progress.md) before interpreting cross-dataset
results.

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

## Path 3 roadmap — Model building and attention visualization

**Status: planned after dataset research.** There is not yet a complete
attention-visualization Streamlit application or a finished comparative
attention study. The existing CLI below is a prototype retained from earlier
pretraining work and can be used to validate old or future compatible
checkpoints.

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

This prototype saves a heatmap under `artifacts/`. See
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

## Future attention research plan

1. Complete the dataset survey and freeze versioned train/validation/test data.
2. Define attention research questions, controls, metrics, and comparison sets.
3. Train comparable model checkpoints with fixed data and tokenizer versions.
4. Validate attention capture across layers, heads, prompt tokens, and generated
   tokens.
5. Compare attention variants using validation quality, runtime, memory, and
   carefully scoped qualitative analysis.
6. Build an interactive visualization application for inspecting trained models.
7. Record limitations: attention weights or influence approximations are not,
   by themselves, causal explanations of model behavior.

### Historical laptop-safe starting point

- embedding dim: 128
- heads: 4
- layers: 2
- sequence length: 64
- batch size: 32 on GPU, 8–16 on CPU
