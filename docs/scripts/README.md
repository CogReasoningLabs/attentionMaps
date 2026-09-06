# Script reference

This directory documents every maintained file under `scripts/` that has a
user-facing command or reusable API.

## Required data-pipeline order

```text
1. download_nepali_pretrain_corpus.py   Hugging Face → data/raw/
2. clean_pretraining_sources.py         data/raw/ → data/cleaned/
3. build_pretraining_dataset.py         data/cleaned/ → data/processed/ splits
4. python -m attention_maps.tokenization
                                        data/processed/ → data/tokenized/
5. decoder pretraining                  consumes frozen token IDs
```

The downloader is required only for Hugging Face sources that are not already
present locally. The music archive is already local, so it does not need that
step.

| File | Type | Purpose | Documentation |
|---|---|---|---|
| `download_nepali_pretrain_corpus.py` | CLI | Download versioned Hugging Face Parquet sources into immutable raw storage | [Download reference](download-nepali-pretrain-corpus.md) |
| `clean_pretraining_sources.py` | CLI | Clean and canonicalize PDF, news, and song documents independently | [Cleaning reference](../cleaning-cli-reference.md) |
| `build_pretraining_dataset.py` | CLI | Sample sources, deduplicate, and create deterministic train/validation/test Parquet files | [Build/split reference](build-pretraining-dataset.md) |
| `tokenize_pretraining_data.py` | Compatibility CLI | Forwards to `attention_maps.tokenization`; new workflows should use the package command | [Tokenization reference](tokenize-pretraining-data.md) |
| `infer_nepberta.py` | Compatibility CLI | Forwards to `attention_maps.inference.nepberta` | [NepBERTa reference](infer-nepberta.md) |
| `infer_himalaya_gemma.py` | Compatibility CLI | Loads the Himalaya Gemma 4 base model plus its PEFT adapter for generation | [Himalaya Gemma reference](infer-himalaya-gemma.md) |
| `nepali_inference_compare.py` | CLI | Compares hosted models and local GPT-2/TinyLlama finetuned adapters over a decoding grid | [Inference comparison](nepali-inference-compare.md) |
| `utils/nepali_text.py` | Internal API | Shared Unicode, markup, language-ratio, and strict-character cleaning functions | [Utility API reference](nepali-text-utilities.md) |
| `utils/tokenization.py` | Compatibility import | Re-exports `attention_maps.tokenization.pipeline` for older imports | [Tokenization reference](tokenize-pretraining-data.md#modular-code-boundary) |
| `__init__.py`, `utils/__init__.py` | Package markers | Make imports reliable; they are not commands | No parameters |

## Typical workflow

If raw data is already available:

```bash
.venv/bin/python scripts/clean_pretraining_sources.py

.venv/bin/python scripts/build_pretraining_dataset.py \
  --pdf-fraction 1.0 \
  --news-fraction 0.01 \
  --lyrics-fraction 1.0 \
  --train-ratio 0.98 \
  --validation-ratio 0.01 \
  --test-ratio 0.01 \
  --output-dir data/processed/nepali_pretraining_small
```

Only the second command controls source inclusion and split sizes. Cleaning
thresholds decide whether individual documents are valid; they are not dataset
sampling percentages.

Then try the configured third-party Hugging Face BPE tokenizer:

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml
```
