# Himalaya Gemma 4 inference

`attention_maps.inference.himalaya_gemma` performs text generation with:

- adapter: `Govsovereign/himalaya-gemma-4-e2b-it-merged-s1-qlora`
- base model: read from the adapter metadata, currently
  `himalaya-ai/himalaya-gemma-4-e2b-it`

Despite `merged` in the repository name, its published files are a PEFT LoRA
adapter. The command therefore downloads the full 5B base checkpoint and then
attaches the adapter. The adapter tokenizer and chat template are used exactly
as published.

## Installation

Gemma 4 needs Transformers 5.x, while the older TensorFlow NepBERTa command in
this repository is pinned to Transformers 4.x. A separate virtual environment
is the safest setup:

```bash
python -m venv .venv-gemma
.venv-gemma/bin/pip install --upgrade pip
.venv-gemma/bin/pip install -r requirements.txt
.venv-gemma/bin/pip install -r requirements-himalaya-gemma.txt
```

If Hugging Face requests authentication, log in once with
`.venv-gemma/bin/hf auth login`. Files are cached under
`data/cache/huggingface/` by default.

## Recommended GPU command

Four-bit NF4 loading substantially reduces VRAM use:

```bash
.venv-gemma/bin/python -m attention_maps.inference.himalaya_gemma \
  "नेपालमा जलवायु परिवर्तनको प्रभावबारे छोटकरीमा लेख्नुहोस्।" \
  --device cuda \
  --quantization 4bit \
  --do-sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 256
```

The equivalent compatibility launcher is:

```bash
.venv-gemma/bin/python scripts/infer_himalaya_gemma.py \
  "नेपालको इतिहासबारे तीन वाक्य लेख्नुहोस्।" \
  --device cuda --quantization 4bit
```

For deterministic greedy decoding, omit `--do-sample`. To save all inference
metadata and the completion:

```bash
.venv-gemma/bin/python -m attention_maps.inference.himalaya_gemma \
  "नेपाली भाषाको महत्त्व के हो?" \
  --device cuda --quantization 4bit \
  --output runs/himalaya_gemma/example.json
```

## Important options

| Option | Purpose |
|---|---|
| `--adapter-id` | Hub adapter repository or local adapter directory |
| `--base-model-id` | Optional override; normally inferred from adapter metadata |
| `--device` | `auto`, `cuda`, or `cpu` |
| `--quantization` | `none`, `4bit`, or `8bit`; quantization requires CUDA |
| `--dtype` | Unquantized weight dtype or quantized compute dtype |
| `--system-prompt` | Optional system instruction before the user message |
| `--enable-thinking` | Enables the chat template's optional thinking mode |
| `--local-files-only` | Prevents network access after files are cached |
| `--adapter-revision`, `--base-revision` | Pin a tag or commit for reproducibility |
| `--trust-remote-code` | Explicitly allow Hub Python code; normally unnecessary |

CPU inference is supported with `--device cpu --quantization none`, but a 5B
checkpoint requires substantial RAM and will be much slower than a GPU.

## Insufficient VRAM and offloading

If `device_map="auto"` reports that modules were dispatched to CPU or disk,
first check `nvidia-smi` and stop other GPU processes if possible. This is the
fastest option because the complete quantized model remains on the GPU.

When the GPU is intrinsically too small, explicitly enable offloading:

```bash
.venv-gemma/bin/python -m attention_maps.inference.himalaya_gemma \
  "नेपाली भाषाको महत्त्व के हो?" \
  --device cuda \
  --quantization 8bit \
  --cpu-offload \
  --gpu-max-memory 5GiB \
  --cpu-max-memory 24GiB \
  --offload-dir data/cache/huggingface/offload/himalaya_gemma \
  --max-new-tokens 128
```

CPU/disk offload is considerably slower. Modules placed on the CPU remain in
full precision, so this mode can require much more system RAM than the 4-bit
GPU footprint suggests. Bitsandbytes CPU offload is supported for 8-bit
loading; do not combine `--quantization 4bit` with `--cpu-offload`. Once the
base checkpoint and adapter are fully cached, add
`--local-files-only` to avoid another network download.

The same offload directory and memory limits are forwarded to both loading
stages: first the base Transformers model, then PEFT's second dispatch after
the adapter has been attached.

For offline loading, the tokenizer is given the base model's cached Gemma 4
configuration explicitly. This matters because the adapter repository has an
`adapter_config.json`, not a standalone `config.json`.

The default CUDA placement is `--device-map single`, which puts the complete
quantized checkpoint on CUDA:0. This avoids the conservative swap-space reserve
used by automatic mapping. `--cpu-offload` automatically selects
`--device-map auto`; use `--debug` if an upstream loading error needs a full
traceback.
