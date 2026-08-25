# Code structure

Reusable application code lives in the `attention_maps` package. Data
preparation commands remain in `scripts/` because they are pipeline entry
points rather than model internals.

```text
attention_maps/
├── config.py                     shared experiment/profile configuration
├── common/
│   └── artifacts.py              checkpoints, JSON, and run directories
├── training/
│   ├── attention.py              attention implementations
│   ├── model.py                  decoder model and MoE blocks
│   ├── data.py                   corpus loading, encoding, and batch sampling
│   ├── tokenized_data.py         saved token IDs and packed DataLoaders
│   ├── configuration.py          YAML decoder-training configuration
│   ├── metrics.py                training-loss plots
│   ├── cli.py                    training command
│   └── __main__.py               python -m attention_maps.training
├── tokenization/
│   ├── models.py                 portable runtime tokenizer adapters
│   ├── pipeline.py               Hub/local/custom tokenizers and Parquet encoding
│   ├── cli.py                    materialized-tokenization command
│   └── __main__.py               python -m attention_maps.tokenization
├── inference/
│   ├── generate.py               causal decoder generation
│   ├── nepberta.py               optional masked-LM diagnostics
│   └── himalaya_gemma.py         Gemma 4 base + PEFT adapter generation
└── visualization/
    ├── attention.py              reusable attention-map rendering
    ├── cli.py                    checkpoint visualization command
    └── __main__.py               python -m attention_maps.visualization
```

## Canonical commands

```bash
# Train the decoder
.venv/bin/python -m attention_maps.training --profile profiles/ne_pdf_local.json

# Train directly from the frozen materialized token IDs
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_small.yaml

# Apply the exact pinned NepaliBPE and encode final processed splits
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/aananda_nepali_bpe.yaml

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
```

Training does not import inference or visualization code. The tokenization
pipeline does not import the model. Visualization owns plotting and Unicode
font handling; inference calls its public API only when attention export is
requested. NepBERTa remains isolated from decoder training.
