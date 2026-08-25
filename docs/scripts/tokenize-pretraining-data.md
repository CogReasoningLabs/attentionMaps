# Tokenize pretraining data

`python -m attention_maps.tokenization` is stage 4 of the corpus pipeline. It
can try an existing Hugging Face tokenizer, load local artifacts, or train a
custom BPE:

```text
data/processed/<run>/{train,validation,test}.parquet
    → resolve one frozen tokenizer
    → data/tokenized/<run>/tokenizer/
    → data/tokenized/<run>/{train,validation,test}/part-*.parquet
```

The default is `configs/tokenizer/huggingface_nepali_bpe.yaml`. It tries an
available third-party Nepali BPE tokenizer hosted on Hugging Face and records a
specific commit for reproducibility. The account portion of the Hub repository
ID is only a storage namespace; it is not the tokenizer architecture or a model
name used by this project. This path does not train new merge rules, and the
downloaded 50,006-entry vocabulary and token IDs remain unchanged.

## Run

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml
```

Override the dataset run or destination without editing the YAML:

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/huggingface_nepali_bpe.yaml \
  --input-dir data/processed/nepali_pretraining_v2 \
  --output-dir data/tokenized/huggingface_nepali_bpe_v2
```

To train a new custom BPE instead, select its separate configuration:

```bash
.venv/bin/python -m attention_maps.tokenization \
  --config configs/tokenizer/nepali_bpe.yaml
```

For a local tokenizer, set `tokenizer.source.type: local` and
`tokenizer.source.path` in YAML. `--tokenizer-dir PATH` is a convenient local
override. The path must contain `tokenizer.json`; the YAML must map the correct
existing special tokens. The pipeline rejects missing tokens and never adds
tokens automatically.

The command refuses to write into a non-empty output directory. Use a new
output path for every tokenizer experiment so the tokenizer, encoded data, and
manifest cannot silently disagree.

## CLI parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `--config` | `configs/tokenizer/huggingface_nepali_bpe.yaml` | Versioned YAML containing source, boundary, and output settings. |
| `--input-dir` | YAML value | Optional override for the processed split directory. |
| `--output-dir` | YAML value | Optional override for the tokenized artifact directory. |
| `--tokenizer-dir` | unset | Override the configured source with a local tokenizer directory. |
| `--no-progress` | off | Disable the BPE trainer progress display, useful in logs and CI. |

## YAML: `input`

| Field | Purpose |
|---|---|
| `dataset_dir` | Directory containing final split Parquet files. Relative paths resolve from the YAML file. |
| `splits` | Splits that will be encoded. |
| `training_split` | The only split used to learn vocabulary and BPE merges; normally `train`. |
| `text_column` | Clean document text to encode. |
| `id_column` | Stable document identifier copied to tokenized output. |
| `source_column` | Corpus source copied for composition diagnostics. |
| `hash_column` | Clean-text hash copied for provenance and joining. |

## YAML: `tokenizer`

### Source

| Field | Importance |
|---|---|
| `source.type` | `huggingface` loads a Hub snapshot, `local` loads local artifacts, and `train` learns a new BPE. |
| `source.repository` | Hub repository ID in `account/repository` form. The account is only a hosting namespace. |
| `source.revision` | Branch, tag, or preferably immutable commit. Pinning a commit prevents upstream changes from changing IDs. |
| `source.path` | Directory containing local `tokenizer.json`; used only by `local`. Relative paths resolve from the YAML. |
| `source.cache_dir` | Hugging Face snapshot cache. It may be reused offline after the pinned snapshot is present. |
| `source.local_files_only` | If true, fail rather than access the network when the snapshot is absent. |

### Token roles and custom-BPE training

The BPE learning fields below apply only to `source.type: train`; Hub and local
sources execute the normalization, pre-tokenization, and model already stored
inside their `tokenizer.json`.

| Field | Importance |
|---|---|
| `vocab_size` | For `source.type: train` only: maximum vocabulary size including special tokens. |
| `min_frequency` | Minimum pair frequency for a BPE merge. Raising it avoids entries learned from rare OCR artifacts. |
| `normalization` | `NFC`, `NFKC`, or `none`. `NFC` matches the default cleaning policy without aggressive compatibility folding. |
| `pre_tokenizer` | `whitespace` separates punctuation as well as whitespace; `whitespace_split` splits only at whitespace. |
| `continuing_subword_prefix` | Optional marker on noninitial subwords. Empty by default because `</w>` marks word endings. |
| `end_of_word_suffix` | Marker on the final subword of each word. `</w>` follows the reference tokenizer. |
| `max_token_length` | Prevents abnormally long repeated strings from becoming vocabulary tokens. It does not truncate documents. |
| `max_training_segments` | Optional trainer-input cap. `null` uses the complete training split and avoids source-order bias. |
| `max_chars_per_segment` | Splits huge PDF documents into bounded trainer inputs; it does not remove text from encoded output. |
| `add_bos`, `add_eos` | Add document boundaries during encoding, keeping adjacent documents distinguishable in a causal token stream. |
| `special_tokens` | Maps semantic roles to token strings. For Hub/local sources, each non-null token must already exist. `pad` may be null when packing unpadded causal streams. |

The downloaded tokenizer metadata does not assign semantic roles, so our
config maps its already-existing tokens explicitly: UNK=1, BOS=50000, and
EOS=50001. BOS/EOS are inserted around each document by the corpus pipeline;
the tokenizer vocabulary itself is not modified.

## YAML: `output`

| Field | Purpose |
|---|---|
| `output_dir` | Root for tokenizer files, tokenized shards, and the manifest. |
| `rows_per_shard` | Maximum documents per output Parquet shard. |
| `batch_size` | Documents read and encoded together; increase only within available RAM. |
| `compression` | `zstd` for smaller files or `snappy` for somewhat faster I/O. |
| `model_max_length` | Transformers metadata. It does not truncate documents here; training chunks IDs to its sequence length later. |

## Output record

Each row preserves one input document:

```text
doc_id:       string
source:       string
text_sha256:  string
input_ids:    list<int32>
num_tokens:   int32
```

The tokenized copy omits full text to avoid duplicating a large corpus.
`doc_id` and `text_sha256` link it to the processed source row.

`tokenization_manifest.json` records input hashes, effective configuration,
special-token IDs, requested and resolved Hub revisions, tokenizer SHA-256,
vocabulary size, output shards, token counts, source counts, document-length
statistics, and unknown-token rates for every split. For external tokenizers,
upstream metadata is copied unchanged; `attention_maps_tokenizer_config.json`
stores our token-role and document-boundary policy separately.

## Modular code boundary

- `attention_maps/tokenization/cli.py`: argument parsing and status output.
- `attention_maps/tokenization/pipeline.py`: source resolution, optional custom BPE
  training, Parquet streaming, export, encoding, statistics, and manifests.

Notebooks and future training loaders should import the utility module rather
than importing the CLI script.
