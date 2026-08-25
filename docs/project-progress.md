# Project progress

This document records the research and engineering work completed on the
`feature/language-agnostic-phase-1` branch. Generated datasets, caches,
checkpoints, and run artifacts are intentionally local and are not versioned.

## Research direction

The original educational decoder was primarily exercised on English text. The
pipeline is now language-agnostic at its core: language, source, schema, split,
and tokenizer choices are configuration rather than hard-coded model behavior.
The current research profile targets Nepali.

The decoder pretraining objective remains causal next-token prediction. For a
token sequence `(x1, ..., xT)`, the model minimizes cross-entropy for predicting
`x(t+1)` from `(x1, ..., xt)`. Nepali changes the corpus and tokenizer, not the
mathematical objective.

## Completed work

| Area | Current state |
|---|---|
| Source data | PDF, news, and music-lyrics sources are represented under a consistent raw-data layout |
| Cleaning | Shared Unicode, HTML, URL/email, control-character, whitespace, and Devanagari filtering utilities |
| Canonical schema | Every cleaned source becomes sharded Parquet with stable document and provenance fields |
| Dataset building | Deterministic source sampling, exact-text deduplication, and document-level train/validation/test splitting |
| EDA | Individual PDF, news, and lyrics notebooks; comparative three-source notebook; final-split inspection notebook |
| Tokenization | Exact pinned `Aananda-giri/NepaliBPE` support plus an optional train-your-own BPE path |
| Materialized training data | Document token IDs stored in Parquet and packed into fixed causal training blocks |
| Decoder training | YAML-driven training, standard/cosine/linear attention, optional sparse MoE, checkpointing, and resume support |
| Generation | Greedy or sampled continuation with completion-only output and reproducible seeds |
| Attention analysis | Modular per-layer/per-head visualization for prompts and final generation context |
| External diagnostics | NepBERTa fill-mask inference and Himalaya Gemma 4 PEFT generation are isolated from decoder training |
| Packaging | Core logic moved into `attention_maps`; root scripts remain compatibility entry points |
| Tests | Data stages, tokenization, packed training data, configuration, inference, and visualization have focused unit coverage |

## Important design decisions

- Raw files are immutable. Each stage writes into a new directory:
  `raw → cleaned → processed → tokenized → runs`.
- Cleaning does not decide train/validation/test membership.
- Split assignment is document-level and deterministic to prevent leakage.
- A published tokenizer is loaded from its exact pinned revision without
  changing its vocabulary or token IDs.
- Embeddings and positional encodings are model parameters; they are not stored
  in the tokenized dataset.
- NepBERTa is an encoder-style masked-language model and is not used by the
  causal decoder.
- Himalaya Gemma inference is a separate PEFT/base-model workflow and uses a
  separate Transformers 5.x environment.

## Repository boundaries

Version-controlled files should include source code, tests, YAML/JSON configs,
documentation, and notebooks. These local products are excluded by
`.gitignore`:

- virtual environments;
- raw, cleaned, processed, tokenized, and cached data;
- Hugging Face model weights and offload files;
- decoder checkpoints, run directories, generated attention maps, and logs;
- Python, pytest, editor, and notebook caches;
- secrets such as `.env` and access tokens.

## Current checkpoint and next direction

The local decoder pipeline has been exercised through training, generation,
and attention export. Early output is grammatical in places but repetitive and
biased toward administrative notices, reflecting corpus composition and a
short training run.

The Himalaya Gemma loader now supports:

- explicit PEFT adapter plus base-model loading;
- 4-bit single-GPU loading;
- supported 8-bit CPU/disk offload;
- shared offload settings across Transformers and PEFT dispatch;
- fully cached/offline configuration and tokenizer loading;
- completion-only decoding and optional JSON output.

End-to-end Gemma generation still needs confirmation on the target GPU. After
that, the next research work is to establish fixed Nepali evaluation prompts,
measure source balance and held-out perplexity, tune sampling/training, and
compare standard attention with MoE and other attention variants.

## Related documentation

- [Nepali pretraining workflow](nepali-pretraining-workflow.md)
- [Model inference and attention](model-inference-and-attention.md)
- [Code structure](code-structure.md)
- [Detailed script reference](scripts/README.md)

