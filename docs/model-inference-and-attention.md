# Model inference and attention

> **Project status:** this is a reference for experimental model tooling that
> already exists. The active phase is dataset exploration, cleaning design, and
> synthetic-data generation. A controlled model-building and comparative
> attention study is planned after the datasets are reviewed and versioned.

The repository supports three distinct model workflows. They should not share
loading logic because their objectives and runtime dependencies differ.

## Locally pretrained causal decoder

Generate a continuation from the best validation checkpoint:

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

Without `--do-sample`, generation is greedy. The command saves prompt IDs,
completion IDs, full output, decoded text, and generation parameters under the
checkpoint run. Add `--save-attention` to export every layer/head for the final
pre-sampling context.

## Standalone attention maps

```bash
.venv/bin/python -m attention_maps.visualization \
  --checkpoint runs/nepali_decoder_v1/checkpoints/best.pt \
  --text "नेपाल एउटा सुन्दर देश हो।" \
  --run-name sentence_01 \
  --device cuda
```

Rows are query tokens and columns are key tokens. The masked upper-right region
is expected for a causal model. Attention weight is useful for comparing heads
and layers, but it is not a causal explanation by itself. Unicode-aware labels,
shared color scaling, validation, and plotting live in the independent
`attention_maps.visualization` package.

## NepBERTa masked-language diagnostics

NepBERTa is a bidirectional encoder and predicts a replacement for one mask;
it is not used for decoder generation.

```bash
.venv-nepberta/bin/pip install -r requirements-nepberta.txt

.venv-nepberta/bin/python -m attention_maps.inference.nepberta \
  "नेपाल एक [MASK] देश हो।" \
  --top-k 5
```

This TensorFlow path is isolated from PyTorch before Transformers import to
avoid native Triton/PyTorch initialization problems. It uses the repository's
native TensorFlow weights and supports `{mask}` as a portable placeholder.

## Himalaya Gemma 4 PEFT generation

`Govsovereign/himalaya-gemma-4-e2b-it-merged-s1-qlora` is not a standalone
merged checkpoint despite its name. It publishes a roughly 203 MB PEFT LoRA
adapter whose metadata points to the approximately 5B-parameter
`himalaya-ai/himalaya-gemma-4-e2b-it` base model. The loader therefore:

1. reads the adapter configuration;
2. loads the adapter tokenizer and chat template;
3. loads/quantizes the full base model;
4. attaches and redispatches the PEFT adapter;
5. formats chat messages and decodes only newly generated tokens.

Gemma 4 needs Transformers 5.x, so use its separate environment:

```bash
python -m venv .venv-gemma
.venv-gemma/bin/pip install -r requirements.txt
.venv-gemma/bin/pip install -r requirements-himalaya-gemma.txt
```

### Preferred 4-bit single-GPU inference

```bash
.venv-gemma/bin/python -m attention_maps.inference.himalaya_gemma \
  "नेपाली भाषाको महत्त्व के हो?" \
  --device cuda \
  --device-map single \
  --quantization 4bit \
  --max-new-tokens 128
```

This avoids the conservative swap reserve of automatic placement. If it raises
CUDA OOM, first stop other GPU processes and retry.

### Supported 8-bit CPU/disk offload

```bash
.venv-gemma/bin/python -m attention_maps.inference.himalaya_gemma \
  "नेपाली भाषाको महत्त्व के हो?" \
  --device cuda \
  --quantization 8bit \
  --cpu-offload \
  --gpu-max-memory 6GiB \
  --cpu-max-memory 24GiB \
  --offload-dir data/cache/huggingface/offload/himalaya_gemma \
  --max-new-tokens 128 \
  --debug
```

Do not combine 4-bit bitsandbytes loading with `--cpu-offload`; that combination
produced meta-tensor dispatch failures in the current stack. CPU-offloaded
quantized modules are held in full precision, so offload may use substantial
RAM and is much slower.

The offload directory and memory ceilings are passed both to Transformers and
to PEFT's second dispatch. For offline mode, the adapter tokenizer receives the
base Gemma configuration explicitly because an adapter repository contains
`adapter_config.json`, not necessarily a standalone `config.json`.

After both base and adapter files have been cached, add
`--local-files-only`. Use `--debug` when reporting a loader failure so the full
traceback identifies whether it occurred during base loading, PEFT dispatch,
tokenization, or generation.

## Dependency boundary

Use independent environments for NepBERTa and Gemma:

| Workflow | Framework | Transformers line |
|---|---|---|
| Local decoder | PyTorch | project default |
| NepBERTa | TensorFlow/Keras | 4.x compatibility requirements |
| Gemma 4 + PEFT | PyTorch/Accelerate/bitsandbytes | 5.x requirements |

Detailed parameters are documented in
[decoder generation](generation-inference.md),
[attention visualization](attention-visualization.md),
[NepBERTa inference](scripts/infer-nepberta.md), and
[Himalaya Gemma inference](scripts/infer-himalaya-gemma.md).
