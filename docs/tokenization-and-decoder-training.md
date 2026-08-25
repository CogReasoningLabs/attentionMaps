# From Nepali text to decoder training

## How tokenization happened

The current run uses the exact published tokenizer rather than learning a new
vocabulary. The sequence was:

```text
Aananda-giri/NepaliBPE at pinned commit fa173a1...
→ load its serialized tokenizer.json unchanged
→ verify its 50,006-entry vocabulary and configured token roles
→ encode train, validation, and test with the frozen tokenizer
→ add BOS and EOS around every document
→ save document-level input_ids to Parquet
```

The published BPE splits text using its own serialized normalizer,
pre-tokenizer, merge model, decoder, and `</w>` end-of-word convention. No
locally configured normalization or merge rule replaces those components.

For the saved tokenizer:

```text
text:   राम ले भात खायो ।
pieces: राम</w>, ले</w>, भात</w>, खायो</w>, ।</w>
raw IDs: 1621, 285, 14413, 27675, 251
saved IDs: 50000, 1621, 285, 14413, 27675, 251, 50001
```

The configured roles point to tokens already present upstream:

```text
PAD=not configured  UNK=1  BOS=50000  EOS=50001
```

The model uses packed, fixed-length streams, so it does not need a padding
token. The pipeline does not call `add_tokens` or `add_special_tokens`; doing so
would create a different vocabulary.

For a custom tokenizer experiment,
`configs/tokenizer/nepali_bpe.yaml` trains BPE merges from the training split
only. Validation and test remain excluded from tokenizer training. Such output
has an independent token-ID space and must not be combined with this one.

The saved `input_ids` are integer vocabulary references—not embeddings. They
are stable and reusable. Embeddings are floating-point model parameters that
change on every optimizer step, so they belong in model checkpoints rather
than the dataset.

## How token IDs become training examples

`attention_maps.training.tokenized_data` validates the tokenization manifest,
loads each split independently, and concatenates its document token lists.
Every document already ends with EOS and the next starts with BOS, so document
boundaries remain explicit after packing.

For `sequence_length=4`, a stream like this:

```text
10 11 12 13 14 15 ...
```

becomes:

```text
x = 10 11 12 13
y = 11 12 13 14
```

The next block starts at token `14`. Blocks are non-overlapping as model inputs,
but the final target of one block is the first input of the next. Training
blocks are shuffled deterministically; validation and test blocks retain their
order. At most `sequence_length` residual tokens are unused per split, and no
validation/test tokens enter the training DataLoader.

## Embedding and position processing

Inside `TinyTransformerLM.forward`:

```python
token_vectors = self.token_emb(input_ids)
positions = torch.arange(sequence_length, device=input_ids.device)
position_vectors = self.pos_emb(positions)
hidden_states = self.drop(token_vectors + position_vectors)
```

For the small config, an input batch shaped `(batch, 512)` becomes hidden
states shaped `(batch, 512, 256)`. Causal attention prevents a position from
seeing future positions. The output head is weight-tied to the token embedding,
and cross-entropy trains it to predict `y` at every position.

This implementation currently uses learned absolute positional embeddings.
RoPE or another positional method can be introduced later as a model experiment
without changing the tokenized Parquet files.

## Run a five-step smoke test

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_smoke.yaml
```

This verifies loading, packing, embedding lookup, positional embeddings,
forward/backward passes, evaluation, and checkpoint serialization. It is not a
meaningful pretrained model.

## Start the small training run

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_small.yaml
```

Command-line values override YAML defaults. For example:

```bash
.venv/bin/python -m attention_maps.training \
  --training-config configs/training/nepali_decoder_small.yaml \
  --device cuda \
  --batch-size 2 \
  --grad-accum 8 \
  --max-steps 1000 \
  --run-name nepali_decoder_trial_001
```

The run stores the model configuration, original training YAML, tokenization
manifest, portable tokenizer state, metrics, and checkpoints under `runs/`.
