# Project progress

> Active multi-dataset EDA work on `feature/multi-dataset-eda` is tracked in
> [eda-survey-progress.md](eda-survey-progress.md).

This document separates current research from earlier experimental foundations.
Generated datasets, caches, checkpoints, and run artifacts are intentionally
local and are not versioned.

| Project path | Status | Current boundary |
|---|---|---|
| Dataset exploration | Active | catalog taxonomy, bounded inspection, survey EDA, and cleaning/deduplication design |
| Synthetic dataset generation | Active foundation | shared family registry, live bounded generation, persistence, and exports; LIMA translation is the first family |
| Model building and attention visualization | Planned next | earlier training, inference, and CLI visualization code is retained for reuse but is not a completed research program |

## Research direction

The long-term goal is one modular research codebase for:

1. causal pretraining and continual pretraining;
2. supervised finetuning (SFT);
3. preference-data and reward-model training;
4. alignment methods, including RLHF where it is justified.

The original educational decoder was primarily exercised on English text. The
current research focus is Nepali, but the core pipeline must remain
language-neutral: language, source, schema, cleaning policy, split, tokenizer,
and model choices belong in configuration rather than hard-coded Nepali
branches. An English experiment should use the same stage interfaces with a
different profile and cleaning policy.

The decoder pretraining objective remains causal next-token prediction. For a
token sequence `(x1, ..., xT)`, the model minimizes cross-entropy for predicting
`x(t+1)` from `(x1, ..., xt)`. Nepali changes the corpus and tokenizer, not the
mathematical objective.

## Choosing continual pretraining or SFT

These stages solve different problems and are often used in sequence:

- **Continual pretraining (CPT)** continues causal next-token training on raw,
  unlabeled text. Use it when a candidate base model has weak Nepali language
  modeling, weak coverage of the target domain, or unsuitable tokenization.
- **Supervised finetuning (SFT)** trains on prompt/response or chat examples.
  Use it when the base model already represents Nepali adequately but needs to
  follow instructions, adopt a task format, or produce desired response styles.

The decision should be evaluation-gated rather than assumed:

```text
candidate base checkpoint
        │
        ▼
Nepali base-model evaluation
   ├── weak language/domain modeling ──> CPT ──> reevaluate
   └── adequate language modeling ─────────────> SFT
                                                   │
                                                   ▼
                                      preference/alignment data
                                                   │
                                      reward model, DPO, or RLHF
```

For the small decoder trained from scratch in this repository, additional
next-token training is simply further pretraining. For an external multilingual
checkpoint, the same objective on the Nepali corpus is language- or
domain-adaptive continual pretraining. SFT should begin only after the selected
base checkpoint passes a fixed Nepali evaluation set.

## Implemented foundations

| Area | Current state |
|---|---|
| Source data | PDF, news, and music-lyrics sources are represented under a consistent raw-data layout |
| Cleaning | Shared Unicode, HTML, URL/email, control-character, whitespace, and Devanagari filtering utilities |
| Canonical schema | Every cleaned source becomes sharded Parquet with stable document and provenance fields |
| Dataset building | Deterministic source sampling, exact-text deduplication, and document-level train/validation/test splitting |
| Dataset explorer | Provider/lineage and purpose taxonomy, dependent filters, local/remote inspection, complete-record views, bounded EDA, tokenizer/model comparison, and evaluation tabs |
| Survey EDA | Incremental size/script/quality/provenance metrics, exact and approximate duplicate screening, n-grams, co-occurrence networks, evidence exports, and in-app methodology notes |
| Synthetic generation | Registry-driven family/variant selection shared by both apps, live bounded generation, rate accounting, run persistence, and JSONL/CSV exports |
| Tokenization | Tried an available third-party Hugging Face BPE tokenizer; also retained a separate custom-BPE experiment path |
| Materialized training data | Document token IDs stored in Parquet and packed into fixed causal training blocks |
| Historical decoder foundation | YAML-driven training, standard/cosine/linear attention, optional sparse MoE, checkpointing, and resume support |
| Historical inference foundation | Greedy or sampled continuation with completion-only output and reproducible seeds |
| Attention prototype | Command-line per-layer/per-head visualization for prompts and final generation context; comparative research and a Streamlit attention UI remain planned |
| External diagnostics | NepBERTa fill-mask inference and Himalaya Gemma 4 PEFT generation are isolated from decoder training |
| Packaging | Core logic lives in cohesive `attention_maps` packages; Streamlit feature renderers and shared components are separated from domain services; a 1,000-line module cap is tested |
| Tests | Data stages, tokenization, packed training data, configuration, inference, and visualization have focused unit coverage |

## Important design decisions

- Raw files are immutable. Each stage writes into a new directory:
  `raw → cleaned → processed → tokenized → runs`.
- Cleaning does not decide train/validation/test membership.
- Split assignment is document-level and deterministic to prevent leakage.
- The external tokenizer experiment records the Hub repository and commit and
  does not change the downloaded vocabulary or token IDs. A repository account
  namespace is treated only as a location, not as an architecture/model name.
- Embeddings and positional encodings are model parameters; they are not stored
  in the tokenized dataset.
- NepBERTa is an encoder-style masked-language model and is not used by the
  causal decoder.
- Himalaya Gemma inference is a separate PEFT/base-model workflow and uses a
  separate Transformers 5.x environment.

## Stage data contracts

Each stage should consume a canonical schema while preserving language and
source provenance:

| Stage | Minimum logical fields |
|---|---|
| Pretraining/CPT | `id`, `text`, `language`, `source`, `metadata` |
| SFT | `id`, `messages` (system/user/assistant), `language`, `source`, `metadata` |
| Preference/reward modeling | `id`, `prompt` or `messages`, `chosen`, `rejected`, `language`, `source`, `metadata` |

Source-specific readers may map different raw schemas into these contracts.
Core trainers must not contain checks such as “if Nepali”; language-specific
cleaning and quality thresholds belong in configured preprocessing components.
For example, the strict Devanagari filter is suitable for the present Nepali
corpus, while English data would use a policy without that script gate.

## Attention-map scope

Attention inspection remains a cross-cutting, optional diagnostic:

- use it on selected evaluation prompts during pretraining/CPT and SFT;
- support both local decoder checkpoints and compatible Hugging Face models
  when they expose attention tensors;
- keep it outside the core optimization loop so training does not depend on
  plotting libraries or eager attention execution;
- do not require it for reward modeling or RLHF.

Some optimized attention backends do not return full attention matrices. An
inspection run may therefore need an eager attention implementation even when
training uses a faster backend. Attention maps show model behavior, but should
not be treated as causal explanations by themselves.

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

## Current phase and roadmap

The current phase is dataset research. Survey results are diagnostic and
bounded unless the UI reports full population coverage. Cleaning and
deduplication are documented but do not yet rewrite source datasets. Synthetic
generation has a reusable application foundation, with LIMA translation as the
first registered family.

The local decoder pipeline was previously exercised through training,
generation, and attention export. Early output was grammatical in places but
repetitive and biased toward administrative notices, reflecting corpus
composition and a short training run. This is historical evidence, not the
baseline for the next controlled model study.

The Himalaya Gemma loader now supports:

- explicit PEFT adapter plus base-model loading;
- 4-bit single-GPU loading;
- supported 8-bit CPU/disk offload;
- shared offload settings across Transformers and PEFT dispatch;
- fully cached/offline configuration and tokenizer loading;
- completion-only decoding and optional JSON output.

End-to-end Gemma generation still needs confirmation on the target GPU.

The next implementation sequence is:

1. validate every catalog schema/revision and complete the comparative dataset
   survey;
2. approve quality, provenance, license, and exact/near-duplicate policies,
   then materialize auditable versioned datasets without changing raw inputs;
3. expand the synthetic-family registry only for reviewed pipeline contracts
   and freeze generated dataset versions with complete lineage;
4. establish fixed Nepali evaluation sets and use them to choose a base model
   and decide whether continual pretraining is needed before SFT;
5. train comparable checkpoints against frozen data/tokenizer versions;
6. validate attention extraction across layers, heads, prompts, and generation
   steps before building the interactive attention visualization application;
7. add preference schemas and alignment only after SFT contracts, evaluation,
   and checkpoint lineage are stable.

Reward modeling, RLHF, and a complete attention research application remain
planned work, not current capabilities.

## Related documentation

- [Nepali pretraining workflow](nepali-pretraining-workflow.md)
- [Model inference and attention](model-inference-and-attention.md)
- [Code structure](code-structure.md)
- [Modularization guardrails](modularization.md)
- [Detailed script reference](scripts/README.md)
- [Multi-dataset EDA survey progress](eda-survey-progress.md)
- [Corpus survey EDA methodology](eda-survey-methodology.md)
