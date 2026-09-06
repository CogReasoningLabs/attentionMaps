# Nepali inference comparison

The comparison engine is available through the **Model comparison** tab in
`apps/dataset_explorer.py` and through a batch CLI. Both entry points share
`attention_maps.inference.comparison`, so parameter validation and provider
behavior remain consistent.

## Interactive use

```bash
python -m streamlit run apps/dataset_explorer.py
```

Select a dataset and open **Model comparison**. There are three prompt
modes:

- **Selected dataset sample** inserts a deterministic Parquet record into the
  user prompt template.
- **Custom source text** inserts manually entered text.
- **Prompt only** sends a direct user prompt without requiring source text.

Templates used with source text must contain `{text}`. The optional system
prompt is separate from the user prompt and is sent through each provider's
system-instruction mechanism.

When `Raw · NepCOV19Tweets · sentiment` is selected, the sentiment preset is
chosen automatically. It samples from `Sentences`, asks for exactly one of
`negative`, `neutral`, or `positive`, and displays the dataset's `Sentiment`
label separately. The reference label is never included in the model prompt.
The result table extracts the predicted label and reports whether it matches
the sampled reference.

No model is selected automatically. Select one or more of Gemini 3.6 Flash,
Gemini 3.5 Flash-Lite, Google-hosted Gemma, configured Hugging Face models,
local GPT-2 LoRA, local TinyLlama QLoRA, HimalayaGPT 0.5B Instruct, or Arkios
1B Chat before running. Only selected models run. Local adapter choices use
weights beneath `finetuned_models/`; Arkios and HimalayaGPT use Hugging Face
snapshots. Weights come from the cache or are downloaded once. The Google models use
`GEMINI_API_KEY`. Gemini uses the
Interactions API, while Gemma uses its documented `generate_content` endpoint.
The default minimal thinking level keeps short output budgets available for
visible answers. Hugging Face Inference Providers remain
optional for models that have provider coverage; multiple Hub model IDs may be
entered one per line. The UI evaluates the Cartesian product of temperature,
top-p, and top-k values and limits one run to 60 total generations. Results include
output, latency, individual provider errors, and CSV download.

Credentials are read only from environment variables; the UI does not accept
or display token values:

```bash
export GEMINI_API_KEY="..."
export HF_TOKEN="..."
```

Keys are used only to construct cached API clients and are not written into
comparison results. The UI displays only whether each variable was detected.
Calls may be billed, rate-limited, gated by a model license, or unavailable on
a chosen provider.

`google/gemma-2-2b-it` can return `model_not_supported` when no enabled Hugging
Face Inference Provider serves it. This is a provider-availability error, not a
problem with the prompt or token. The default Gemma comparison therefore uses
Google's `gemma-4-26b-a4b-it` endpoint. A Hub model can still be selected through
the optional Hugging Face backend when an enabled provider serves it.

## CLI use

Use a UTF-8 file with one prompt source per line:

```bash
python scripts/nepali_inference_compare.py \
  --models gemini gemini-flash-lite google-gemma \
  --input-file samples.txt \
  --temperatures 0.2 0.7 \
  --top-p 0.95 \
  --top-k 40 \
  --max-new-tokens 128 \
  --thinking-level minimal \
  --system-prompt "स्पष्ट र स्वाभाविक नेपालीमा उत्तर दिनुहोस्।" \
  --output artifacts/nepali_inference_results.csv
```

Run both repository finetuned models on the same inputs with:

```bash
python scripts/nepali_inference_compare.py \
  --models local-gpt2 local-tinyllama \
  --input-file samples.txt \
  --temperatures 0.2 0.7 \
  --top-p 0.95 \
  --top-k 40 \
  --max-new-tokens 128 \
  --local-device auto \
  --local-dtype auto \
  --output artifacts/local_finetuned_comparison.csv
```

Use `--local-files-only` to prohibit base-weight downloads. The local choices
are `local-gpt2` and `local-tinyllama`; `--local-models-root` overrides the
default `finetuned_models/` directory.

Run the two standalone instruction/chat models with:

```bash
python scripts/nepali_inference_compare.py \
  --models himalayagpt arkios \
  --input-file samples.txt \
  --max-new-tokens 128 \
  --local-device auto \
  --local-dtype auto \
  --output artifacts/nepali_chat_models.csv
```

HimalayaGPT defaults to the pinned revision supplied by its reference runner
and executes that revision's custom Transformers code. Arkios uses its
published ChatML template. Each implementation is isolated in
`attention_maps/inference/himalayagpt.py` and `attention_maps/inference/arkios.py`.
The HimalayaGPT UI profile defaults to its reference settings: temperature
0.8, top-k 50, 96 new tokens, repetition penalty 1.08, and special-token
stopping. Its reference-compatible manual loop does not apply top-p.

Without `--input-file`, the CLI streams the configured Hugging Face dataset.
Use `--google-gemma-model` to change the Google-hosted Gemma model.
`--gemma-model` remains a compatibility alias for `--hf-model`, and the legacy
`gemma` backend name remains an alias for the generic `huggingface` backend.
`--system-prompt` is optional and remains separate from `--prompt-template`.

## Decoding notes

- `temperature` controls randomness and must be between 0 and 2.
- `top_p` retains the smallest token set reaching the selected probability
  mass and must be in `(0, 1]`.
- `top_k` restricts generation to the most likely K tokens. Google-hosted Gemini
  and Gemma support it; Hugging Face chat completion forwards it as a
  provider-specific option, so a provider may reject it.
- `max_new_tokens` limits completion length and defaults to 256. Local GPT-2
  requires it below 1,024, TinyLlama and HimalayaGPT below 2,048, and Arkios
  below 4,096; prompt tokens also occupy that context window.
- `seed` requests repeatability, although hosted APIs do not guarantee bitwise
  identical output.
- `thinking_level` controls hidden reasoning by Google models. Thinking and
  visible output share the output-token budget, so `minimal` is the default for
  these short summarization comparisons. Increase the token limit when using
  `medium` or `high`.

The backend checks the Gemini interaction status. An `incomplete` response is
reported as an error with thought/output token counts instead of being labeled
`ok` merely because it contains a partial fragment.

The script uses the current `google-genai` SDK rather than the deprecated
`google-generativeai` package. It also bootstraps the repository import path, so
direct execution from the project root works.

Gemini uses the Interactions API with `store=False`; dataset prompts are not
intentionally retained as server-side conversation state. Gemma uses the
stateless `generate_content` endpoint. The defaults are `gemini-3.6-flash`,
`gemini-3.5-flash-lite`, and `gemma-4-26b-a4b-it`.
