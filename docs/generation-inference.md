# Decoder generation inference

> This page documents model-output inference from a decoder checkpoint. It is
> distinct from the Streamlit synthetic dataset generation pipeline. The
> decoder workflow is retained as an experimental foundation for the planned
> model-building phase.

Use the checkpoint with the lowest validation loss for inference:

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

`best.pt` is intended for inference. `latest.pt` additionally contains the
optimizer and scheduler states needed to resume training.

Without `--do-sample`, decoding is greedy and reproducible: the highest-scoring
next token is selected at every step. With sampling enabled, `temperature`
controls distribution sharpness, `top-k` limits candidates by rank, `top-p`
limits them by cumulative probability, and `seed` makes the same command
repeatable.

Each inspection is written under:

```text
runs/nepali_decoder_v1/generations/<gen-name>/
├── generation.json
└── generation.txt
```

The JSON distinguishes `prompt_ids`, `completion_ids`, complete output IDs,
raw tokenizer pieces, readable labels, completion-only text, and full text.

## Inspect generation attention

Add `--save-attention` to the generation command. It exports all four layers
and eight heads for the final pre-sampling model context:

```text
generations/<gen-name>/attention_maps/
├── layer_00/head_00.png ... head_07.png
...
└── layer_03/head_00.png ... head_07.png
```

Only the final step's maps are retained, preventing memory from growing with
every generated token. The newly sampled token is correctly excluded from the
labels because it did not participate in the attention pass that produced it.

The current model was trained for one epoch on a corpus dominated by formal
PDF/news language. Fluent but repetitive administrative phrasing is therefore
plausible; generation quality should be judged across several fixed prompts and
seeds rather than from one sample.
