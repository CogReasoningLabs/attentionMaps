# Hugging Face dataset-size export

Use `scripts/get_huggingface_dataset_sizes.py` to turn a newline-delimited list
of Hugging Face dataset identifiers into a spreadsheet-ready CSV. The command
queries Dataset Viewer metadata; it does not download dataset shards.

## Input

Create a text file containing one target per line. Blank lines and lines that
start with `#` are ignored.

```text
himalaya-ai/nepali-news-corpus
himalaya-ai/nepali_pdf_corpus
uonlp/CulturaX|ne|train
https://huggingface.co/datasets/uonlp/CulturaX/viewer/ne/train
```

The accepted forms are:

- `owner/dataset` for the entire dataset across all configurations and splits;
- `owner/dataset|config` for one configuration;
- `owner/dataset|config|split` for one split; or
- a Hugging Face dataset or Dataset Viewer URL.

For example, `uonlp/CulturaX` reports the entire multilingual dataset, whereas
`uonlp/CulturaX|ne|train` reports only its Nepali training split.

A ready-to-edit input template is available at
`configs/datasets/huggingface_dataset_ids.example.txt`.

## Run

From the repository root:

```bash
cp configs/datasets/huggingface_dataset_ids.example.txt \
  configs/datasets/huggingface_dataset_ids.txt

# Edit configs/datasets/huggingface_dataset_ids.txt, then run:
.venv/bin/python scripts/get_huggingface_dataset_sizes.py \
  configs/datasets/huggingface_dataset_ids.txt \
  --output huggingface_dataset_sizes.csv
```

Public datasets usually need no credentials. For gated or private datasets,
put the token in the project `.env` file:

```dotenv
HF_TOKEN=hf_your_token_here
```

The output includes the identifier, optional configuration and split, row
count, automatic B/KB/MB/GB/TB file size, raw byte counts, decoded size,
status, and any per-dataset error. A failed identifier does not stop the rest
of the list.

`file_size` prefers the size of the original source files reported by Hugging
Face. When that value is unavailable, it falls back to the generated Parquet
files and records the choice in `file_size_basis`. `decoded_size` estimates the
in-memory size and is not the download size.
