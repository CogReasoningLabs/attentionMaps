# `infer_nepberta.py`

## Purpose

Runs fill-mask inference using `NepBERTa/NepBERTa`, a bidirectional masked
language model. It is an optional diagnostic tool and is **not** part of causal
decoder pretraining, dataset cleaning, splitting, tokenization, or generation.

Install its separate TensorFlow dependencies first:

```bash
.venv/bin/pip install -r requirements-nepberta.txt
```

## Basic usage

```bash
.venv/bin/python -m attention_maps.inference.nepberta \
  "नेपाल एक {mask} देश हो।" \
  --top-k 5
```

Exactly one `[MASK]` token or portable `{mask}` placeholder is required.

## Parameters

| Parameter | Default | Purpose and importance |
|---|---|---|
| `text` | `नेपाल एक {mask} देश हो।` | Positional Nepali sentence containing exactly one mask. |
| `--model-id` | `NepBERTa/NepBERTa` | Hugging Face model repository to load. A replacement must support masked-language modeling. |
| `--revision` | `main` | Model branch, tag, or commit. Pin a commit for reproducibility. |
| `--top-k` | `5` | Maximum number of predictions to return. Must be positive. |
| `--target` | unset; repeatable | Restricts scoring to supplied candidate tokens. Each candidate must tokenize to exactly one WordPiece token. |
| `--device` | `cpu` | TensorFlow device policy: `cpu`, `gpu`, or `auto`. |
| `--cache-dir` | `data/cache/huggingface` | Local Hugging Face model/tokenizer cache. |
| `--local-files-only` | false | Forbids network downloads and uses only cached files. |
| `--json` | false | Emits machine-readable JSON instead of a formatted table. |

### `text`

Valid:

```text
नेपाल एक [MASK] देश हो।
नेपाल एक {mask} देश हो।
```

`{mask}` is preferred when experimenting with another tokenizer because the
script replaces it with that tokenizer's actual mask token.

The command fails if text is empty, contains zero/multiple masks, or exceeds the
model's maximum position count.

### `--model-id`

```bash
--model-id NepBERTa/NepBERTa
```

The script loads `AutoTokenizer` and `TFAutoModelForMaskedLM`. A causal decoder
checkpoint is not compatible with this command.

### `--revision`

```bash
--revision main
--revision <commit-hash>
```

`main` may change over time. Pin a commit for comparable inference results.

### `--top-k`

Without targets, returns the highest-probability vocabulary tokens. With
targets, ranks only the supplied candidates. If `top-k` exceeds the vocabulary
or candidate count, the available count is used.

### `--target`

Repeat once per candidate:

```bash
.venv/bin/python -m attention_maps.inference.nepberta \
  "नेपाल एक {mask} देश हो।" \
  --target सुन्दर \
  --target सानो \
  --target स्वतन्त्र \
  --top-k 3
```

Each candidate must map to one tokenizer token. Multi-piece words are rejected
because this command scores only one masked position.

### `--device {cpu,gpu,auto}`

- `cpu`: disables CUDA visibility and TensorFlow GPUs. This is the stable
  default and avoids GPU/native-library problems.
- `gpu`: requires TensorFlow to detect a GPU; otherwise the command fails.
- `auto`: leaves TensorFlow device selection enabled.

This controls inference only and does not affect model files.

### `--cache-dir`

The first online run downloads the tokenizer/config/checkpoint. Keeping the
cache under `data/cache/` separates disposable model artifacts from datasets.

```bash
--cache-dir data/cache/huggingface
```

### `--local-files-only`

Use after the model is cached:

```bash
--local-files-only
```

The command fails instead of accessing the network if required files are
missing locally.

### `--json`

```bash
--json
```

Outputs model, revision, task, normalized input, mask token, and ranked
predictions with token ID, token string, probability, and completed sequence.
Use it for evaluation scripts or saved experiment results.

## Environment behavior

The script deliberately enables TensorFlow and disables PyTorch/Flax before
importing Transformers. It also defaults to legacy Keras compatibility,
disables oneDNN optimizations, and disables the optional Xet downloader to avoid
the native-library crashes previously observed in mixed environments.
