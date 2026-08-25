# Attention visualization

Attention visualization is an independent package rather than part of training
or inference:

```text
attention_maps/visualization/
├── attention.py    reusable validation, Unicode labels, scaling, and plotting
├── cli.py          checkpoint loading and visualization command
└── __main__.py     python -m attention_maps.visualization
```

## Command

Save every head from every layer:

```bash
.venv/bin/python -m attention_maps.visualization \
  --checkpoint runs/nepali_decoder_v1/checkpoints/best.pt \
  --text "नेपाल एउटा सुन्दर देश हो।" \
  --run-name sentence_01 \
  --device cuda
```

Save one selected map:

```bash
.venv/bin/python -m attention_maps.visualization \
  --checkpoint runs/nepali_decoder_v1/checkpoints/best.pt \
  --text "नेपाल एउटा सुन्दर देश हो।" \
  --layer 0 \
  --head 1 \
  --run-name sentence_01 \
  --device cuda
```

The module checks layer/head bounds and verifies that token labels match the
square attention matrix. It selects an installed Unicode font capable of
rendering each Devanagari label, converts upstream BOS/EOS tokens into readable
labels, and removes the tokenizer's display-only `</w>` suffix. Multi-head
exports use a shared 95th-percentile color scale, making layers and heads within
one inspection visually comparable.

By default, output is associated with the trained checkpoint:

```text
runs/nepali_decoder_v1/attention_maps/sentence_01/
├── attention_metadata.json
├── layer_00/head_00.png ... head_07.png
...
└── layer_03/head_00.png ... head_07.png
```

For a selected head, the metadata JSON is stored beside its PNG. Inputs longer
than 64 labels are rejected by default because the resulting heatmap is not
readable; `--max-display-tokens` can explicitly change that guard.

Rows are query positions and columns are key positions. The black upper-right
triangle is expected: causal masking prevents each query from attending to
future keys. A bright cell indicates a larger attention weight for that head;
it is not, by itself, a causal explanation of the model's prediction.

## Reusable API

Import plotting functions without invoking the CLI:

```python
from attention_maps.visualization.attention import (
    compute_global_attention_scale,
    plot_attention,
    save_all_attention_maps,
)
```

`plot_attention` handles a single `(tokens, tokens)` tensor.
`save_all_attention_maps` handles an iterable of
`(batch, heads, tokens, tokens)` layer tensors and returns the saved paths.

Decoder generation may optionally call this API with `--save-attention`, but
the visualization package does not depend on the generation CLI. It receives
the exact model context corresponding to the exported attention matrix.
See [generation inference](generation-inference.md) for that workflow.
