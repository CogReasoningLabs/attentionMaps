# Attention Lab

A small, modular PyTorch project for training a tiny causal language model on WikiText and visualizing attention maps.

## What this project is for

This repo is designed to help you **learn and compare attention mechanisms** on a model small enough to train locally.

It includes:
- WikiText-2 data loading
- a lightweight word-level tokenizer
- interchangeable attention variants
- a tiny transformer language model
- training and validation loops
- attention map visualization for a chosen input sentence

## Attention variants included

- `softmax` — standard scaled dot-product causal self-attention
- `cosine` — cosine-similarity attention with causal masking
- `linear` — a simple feature-map-based linear attention approximation for causal decoding

These are intentionally compact educational implementations.

## Project structure

```text
attention_lab/
├── README.md
├── requirements.txt
├── config.py
├── data.py
├── tokenizer.py
├── attention_variants.py
├── model.py
├── train.py
├── visualize_attention.py
└── utils.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Train

Standard softmax attention:

```bash
python train.py --attention softmax --epochs 3
```

Cosine attention:

```bash
python train.py --attention cosine --epochs 3
```

Linear attention:

```bash
python train.py --attention linear --epochs 3
```

## Visualize attention

After training, a checkpoint is saved under `checkpoints/`.

Example:

```bash
python visualize_attention.py \
  --checkpoint checkpoints/tiny_lm_softmax.pt \
  --text "the history of science is the history of attention" \
  --layer 0 \
  --head 0
```

This saves a heatmap under `artifacts/`.

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

