# Code structure

Reusable application code lives in the `attention_maps` package. Data
preparation commands remain in `scripts/` because they are pipeline entry
points rather than model internals.

The standalone `apps/dataset_explorer.py` entry point provides interactive
dataset inspection, bounded survey EDA, word-cloud analysis, model comparison,
methodology notes, and benchmark evaluation. It remains outside the training package because
exploration is not a training dependency.

The entrypoint now only composes cached services and feature renderers. See
[`modularization.md`](modularization.md) for the module-size guardrail,
dependency rules, and research sources behind the split.

The current project sequence is dataset research, reproducible synthetic-data
generation, then model building and attention research. The inference and
visualization packages below are retained experimental foundations; their
presence does not mean the third path is complete.

```text
apps/
├── dataset_explorer.py           explorer composition root and cache wiring
├── dataset_generator.py          generator composition root and cache wiring
├── components/
│   └── synthetic_data.py         shared synthetic-family/variant selector
└── explorer_tabs/                feature-oriented explorer renderers
```

```text
attention_maps/
├── config.py                     shared experiment/profile configuration
├── common/
│   └── artifacts.py              checkpoints, JSON, and run directories
├── explorer/
│   ├── catalog.py                dataset taxonomy, lineage, and discovery
│   ├── inspection.py             bounded local/remote inspection and sampling
│   ├── text.py                   Unicode-aware explorer analysis helpers
│   └── presentation.py           record and manifest presentation helpers
├── generation/
│   ├── catalog.py                synthetic pipeline-family/variant registry
│   ├── records.py                generated-record schemas and exports
│   └── persistence.py            local rate ledger and run archive
├── training/
│   ├── sft_data.py               schema-flexible SFT normalization and cleaning
│   ├── attention.py              attention implementations
│   ├── model.py                  decoder model and MoE blocks
│   ├── data.py                   corpus loading, encoding, and batch sampling
│   ├── tokenized_data.py         saved token IDs and packed DataLoaders
│   ├── configuration.py          YAML decoder-training configuration
│   ├── metrics.py                training-loss plots
│   ├── cli.py                    training command
│   └── __main__.py               python -m attention_maps.training
├── tokenization/
│   ├── analysis.py               tokenizer-only Nepali coverage and efficiency metrics
│   ├── models.py                 portable runtime tokenizer adapters
│   ├── pipeline.py               Hub/local/custom tokenizers and Parquet encoding
│   ├── cli.py                    materialized-tokenization command
│   └── __main__.py               python -m attention_maps.tokenization
├── inference/
│   ├── generate.py               causal decoder generation
│   ├── comparison.py             shared hosted/local decoding comparison engine
│   ├── local_comparison.py       local PEFT base-versus-adapter comparison
│   ├── gemini_translation.py     LIMA Gemini teacher translation backend
│   ├── arkios.py                 Arkios 1B Chat loading and generation
│   ├── himalayagpt.py            HimalayaGPT 0.5B IT loading and generation
│   ├── gemma4_base.py             Google Gemma 4 E2B base loading and generation
│   ├── iriis_gpt2.py               IRIIS Nepali GPT-2 base/instruct local generation
│   ├── nepberta.py               optional masked-LM diagnostics
│   └── himalaya_gemma.py         Gemma 4 base + PEFT adapter generation
├── eda/
│   ├── contracts.py              survey configuration and result contracts
│   ├── text.py                   schema extraction and Unicode text metrics
│   ├── pipeline.py               bounded streaming analysis orchestration
│   ├── reporting.py              derived JSON/CSV reports and plots
│   └── cli.py                    multi-dataset EDA command line
├── evaluation/
│   └── flores.py                 streamed FLORES loading and chrF++ scoring
└── visualization/
    ├── attention.py              reusable attention-map rendering
    ├── cli.py                    checkpoint visualization command
    └── __main__.py               python -m attention_maps.visualization
```

Streamlit feature renderers live under `apps/explorer_tabs/`; they depend on
the package APIs above and never own reusable data or model behavior.
Shared UI cards live under `apps/components/`. The synthetic-data component
reads a registry of pipeline families and materialized variants, keeping them
separate from the ordinary catalog while serving both applications. LIMA
translation is the first registered family rather than a hard-coded UI path.

## Canonical commands

```bash
# Train the decoder
.venv/bin/python -m attention_maps.training --profile profiles/ne_pdf_local.json

# Train directly from the frozen materialized token IDs
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_small.yaml

# Try the configured Hugging Face BPE tokenizer on the processed splits
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml

# Generate from a decoder checkpoint
.venv/bin/python -m attention_maps.inference.generate \
  --checkpoint runs/<run>/checkpoints/best.pt \
  --prompt "नेपाल"

# Visualize decoder attention
.venv/bin/python -m attention_maps.visualization \
  --checkpoint runs/<run>/checkpoints/best.pt \
  --text "नेपाल एउटा सुन्दर देश हो।"

# Optional NepBERTa fill-mask diagnostic
.venv/bin/python -m attention_maps.inference.nepberta \
  "नेपाल एक [MASK] देश हो।"
```

The former root commands (`train.py`, `generate.py`,
`visualize_attention.py`) and the two former script commands remain as small
compatibility launchers. They contain no training, tokenization, or inference
logic. New code and tests should import from `attention_maps.*`.

## Dependency direction

```text
config ─────────────┐
tokenization.models ├──> training.data/model/cli
common.artifacts ───┘

training + tokenization + common ──> inference
training + tokenization ───────────> visualization
visualization API ─────────────────> optional generation attention export

apps entrypoints ──> apps.components/apps.explorer_tabs ──> package APIs
apps.components.synthetic_data ──> generation.catalog ──> explorer.catalog
```

Training does not import inference or visualization code. The tokenization
pipeline does not import the model. Visualization owns plotting and Unicode
font handling; inference calls its public API only when attention export is
requested. NepBERTa remains isolated from decoder training.

`attention_maps.generation.catalog` is the single registry for generated
dataset families and materialized variants. Both Streamlit entrypoints consume
it through the shared component; new synthetic dataset varieties should be
registered there instead of adding family-specific branches to either app.

## Planned stage boundaries

Finetuning and alignment are the next package boundaries, but are not yet
implemented. They should be added without moving language rules into trainers:

```text
attention_maps/
├── finetuning/       # SFT schemas, chat formatting, loss masking, trainer CLI
├── reward_modeling/  # preference schemas, pairwise scoring, evaluation
└── alignment/        # offline preference optimization and optional RLHF loops
```

Shared checkpoint, tokenizer, model-loading, provenance, and evaluation
interfaces should remain outside those stage-specific packages. The expected
dependency direction is:

```text
canonical data + tokenizer/model adapters
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
 pretraining     SFT    reward/alignment
        │         │
        └────┬────┘
             ▼
 optional attention inspection
```

Attention visualization may inspect compatible pretraining and SFT models, but
reward-model and RLHF execution must not depend on visualization. Stage
configuration should include language metadata and template/policy selection;
the implementation itself should work for Nepali, English, or another UTF-8
language through configuration.
