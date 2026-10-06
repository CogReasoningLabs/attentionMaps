Preprocessing across training stages — research and code review

Reviewed 2026-09-24 against repository revision `be6ce9d` and the working tree.
This note records findings and recommendations; it does not change pipeline
behavior. Review scope included the project documentation, package/application
structure, preprocessing implementations, tokenization and training data paths,
SFT notebooks, and relevant tests. It was not a full-corpus quality audit or a
new model-training experiment.

The recommendation is one shared preprocessing framework with policies selected
by dataset schema, task, language, and model. Shared infrastructure should not
mean identical text transformations or thresholds.

The question listed three training types. The repository's
[four canonical schemas](standard-training-data-schemas.md) add preference
tuning, so it is included here. Continued/domain-adaptive pretraining belongs
with pretraining for data representation, while requiring the existing model's
tokenizer. Instruction tuning and task-specific tuning overlap: a sentiment
task expressed as a chat instruction needs both conversation and task checks.

The common unit is an intact training example:

| Dataset | Unit that must remain intact | Main emphasis |
| --- | --- | --- |
| Pretraining/CPT | Document, with parent-document identity retained for chunks | Extraction quality, coverage, duplication, token mixture |
| Instruction SFT | Conversation or prompt plus response | Roles, response quality, formatting, target loss mask |
| Task-specific supervised | Input plus labels, spans, pairs, or target text | Annotation correctness and input-target alignment |
| Preference tuning | Shared prompt plus chosen/rejected responses | Pair integrity and reliability of the preference |

All four should share source revision tracking, immutable originals, stable IDs,
schema validation, duplicate/contamination analysis, split protection, audit
artifacts, and tokenization checks. The implementation of each check must know
which fields represent inputs, targets, and metadata.

Recommended shared sequence:

```text
immutable source + dataset/version/parent IDs
  -> schema adapter preserving the complete payload and existing split
  -> conservative field-aware normalization + separate comparison views
  -> structural validation and quality flags
  -> duplicate/conflict clusters + protected-benchmark overlap checks
  -> preserve existing splits or assign new splits by related-record group
  -> training-only selection, balancing, augmentation, and learned preprocessing
  -> objective/model-specific formatting, tokenization, labels, and masks
  -> post-transform validation + versioned dataset and audit manifest
```

For an already published benchmark, keep the evaluation data fixed and remove
or quarantine overlapping training candidates. For a new unsplit dataset,
identify related documents/conversations and duplicate clusters before assigning
splits. Split before chunking or generating translated/augmented variants, or
ensure every derived example inherits its parent's split. Seeded random row
splitting alone does not protect against related examples crossing boundaries.

Normalization should default to NFC for ordinary prose, with explicit exceptions
for code, exact strings, and span-annotated data. Keep original content and
offset mappings when a transformation changes annotated text. NFKC can erase
meaningful compatibility distinctions and should require a domain-specific
reason. This follows the distinction in the
[Unicode normalization specification](https://www.unicode.org/reports/tr15/).

Create comparison text separately from training content. Case folding and
whitespace collapse can help retrieve duplicate candidates in prose but can
change code, identifiers, or labels. Preserve meaningful indentation, line
breaks, numbers, emoji, punctuation, URLs, and Devanagari joiners. Remove source
HTML wrappers only when they are extraction artifacts; an HTML answer in a
coding task is content.

Use language/script identification as evidence, with confidence and review of
uncertain records. A Devanagari ratio does not distinguish Nepali from Hindi or
other languages using that script. Romanized Nepali and Nepali-English mixtures
need explicit coverage policies. FineWeb2 supports adapting language
identification and filtering to language-specific evidence; it does not validate
this repository's Nepali thresholds or establish universal SFT filters.
[FineWeb2 paper](https://arxiv.org/html/2506.20920v1).

Deduplication and evaluation decontamination are shared requirements, but their
keys differ. Deduplicate documents for pretraining, complete examples for SFT,
task input/target structures for supervision, and complete preference tuples
for preference tuning. Also group identical/related prompts across responses to
protect splits. Same input with different labels or answers is a conflict to
inspect, not a license to silently keep the first. Multiple valid responses can
be intentional. Near similarity alone does not establish redundant supervision.
The evidence for reducing memorization and evaluation contamination through
deduplication comes from
[Lee et al. (2022)](https://aclanthology.org/2022.acl-long.577/).

Source permission/license metadata and sensitive-content handling should also
be recorded consistently. Removal/redaction policies depend on the task and
must preserve annotation alignment. In particular, a generic toxicity filter
would remove useful positive examples from a hate-speech detection dataset.

The preprocessing policies should differ as follows:

| Operation | Pretraining/CPT | Instruction SFT | Task-specific supervised | Preference |
| --- | --- | --- | --- | --- |
| Boilerplate removal | Source-aware extraction/repetition policy | Off by default; repeated system prompts and instructions can be valid | Off unless validated for the task | Off by default |
| Near-duplicate removal | Calibrated document matching | Compare prompts and responses separately; preserve useful answer diversity | Inspect labels, context, and rare classes | Compare complete pairs; preserve preference distinctions |
| Minimum length | Source/document-specific | Separate prompt and response rules; short answers can be valid | Task-specific; one-token labels can be valid | Both responses must retain a meaningful comparison |
| Sampling/balancing | Control source/domain contributions by tokens as well as rows | Control task, language, response-length, and conversation coverage | Training-only class/task/group policy | Prompt coverage, label reliability, source/judge coverage |
| Tokenizer | Train on train split only for a new model; reuse existing tokenizer for CPT | Reuse selected checkpoint's tokenizer/template | Reuse model tokenizer plus task label alignment | Reuse model tokenizer/template consistently for both candidates |
| Targets | Causal next-token labels in this repository | Assistant/completion tokens under an explicit policy | Class IDs, token labels, spans, or generated targets | Pairwise preference objective |
| D2 in this repo | Disabled | Disabled | Optional, supported supervised use cases only | Disabled |

For pretraining, emphasize HTML/PDF/OCR extraction, repeated headers and footers,
corrupt text, spam/repetition, cross-source overlap, and useful domain coverage.
Removing all Latin text is not a general Nepali quality policy: names, units,
acronyms, technical terms, and code may be useful. Select strict monolingual
filtering only for an explicit experiment and measure what it removes.

Keep document boundaries during packing and record whether attention may cross
them. EOS separates documents symbolically; it does not itself create an
attention barrier. The current packed decoder concatenates documents within
each split and uses ordinary causal attention. That is a viable pretraining
choice, but should not silently become the SFT packing policy. Recheck hashes
after content-changing operations such as boilerplate removal, since distinct
documents can become identical afterward.

For instruction SFT, emphasize valid role order, complete targets, faithful
translation, instruction-answer agreement, and preservation of code/JSON/tool
fields. Explicitly choose last-assistant-turn versus all-assistant-turn
supervision. TRL supports completion-only loss for prompt/completion data and
assistant-only loss when the chat template exposes assistant masks. Inspect
actual collated labels instead of assuming a configuration flag produced the
intended behavior. Padding and excluded prompt tokens should be ignored by
the loss, while intended target and end-of-turn tokens remain supervised.
[TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer).

Use the chosen model's chat format and correct end-of-turn tokens. When rendering
a template to a string and tokenizing afterward, avoid adding special tokens
twice. Fully specified training conversations normally use
`add_generation_prompt=False`; inference prompts have different requirements.
[Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating).

Measure prompt and response token lengths separately. Reject or deliberately
restructure examples whose targets disappear under truncation; do not accept an
all-masked example. Check that truncation preserves valid JSON, complete tool
calls, and the context needed for the answer. For translated/generated data,
retain original-example ID, teacher revision, generation settings, and review
status. Keep original/translation/paraphrase families in the same split.

Task-specific fine-tuning needs further adapters:

| Task | Separate preprocessing to emphasize |
| --- | --- |
| Sentiment/hate speech | Documented label vocabulary, conflicting-label audit, negation/emoji/slur preservation, class-aware reporting, author/thread/source grouping when relevant |
| NER/POS | Word-to-subword label alignment, valid tag sequences, ignored special/padding positions, retained sentence boundaries |
| Extractive QA | Question/context separation, correct character spans, offset remapping after normalization, answer-aware context windows |
| Translation | Aligned source-target pairs, direction/language checks per field, omissions and entity/number fidelity |
| Summarization | Intact document-summary pairing, grounding checks, shared source-document split |
| Proofreading | Preserve mistakes in the input and corrections in the target; never autocorrect the input during cleaning |
| Code/structured output | Preserve whitespace/case/syntax and validate outputs with task-appropriate parsers or checks |

Token classification requires aligning word labels to model subwords, while
extractive QA requires converting answer offsets to token positions. These are
different transformations and cannot be replaced by a single text-plus-scalar-
label adapter. See the official
[token classification](https://huggingface.co/docs/transformers/tasks/token_classification)
and [question answering](https://huggingface.co/docs/transformers/tasks/question_answering)
recipes. Generative versions of these tasks additionally use the SFT formatting
and masking rules above.

For preference data, preserve both completions under the same prompt and keep
all related comparisons together during splitting. Check chosen/rejected
inequality after normalization and after tokenization/truncation. Audit reversed
labels, conflicting comparisons, ties, judge/annotator provenance, and systematic
length or formatting shortcuts. Never independently filter one candidate and
leave an incomplete pair. TRL's documented DPO representation is a prompt with
chosen and rejected completions, including a conversational equivalent.
[TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer).

The repository has useful reusable foundations: immutable output directories,
deterministic sampling, NFC utilities, SHA-256 matching, MinHash-LSH candidates
verified by Jaccard/edit similarity, audit products, tokenizer manifests, and
separate SFT adapters. Several documented schema guarantees are still design
contracts rather than end-to-end enforcement:

| Finding | Code evidence | Implication |
| --- | --- | --- |
| Four schema names and required-field lists exist, but no general record validator is implemented there | [datasets/schemas.py](../attention_maps/datasets/schemas.py) | Add executable schema and relationship invariants |
| Workspace records contain only row ID, text, source, and optional scalar label; batch records add provenance/metrics but still flatten content | [eda/workspace.py](../attention_maps/eda/workspace.py), [batch_pipeline/processing.py](../attention_maps/batch_pipeline/processing.py) | Keep the complete structured payload; EDA text should be a derived view |
| Automatic extraction of preference-shaped rows returns the prompt without chosen/rejected responses | [eda/text.py](../attention_maps/eda/text.py) | Preference candidates can collapse to prompt-only content |
| Normalization collapses each line's whitespace, and workspace dedup strips repeated paragraphs | [eda/text.py](../attention_maps/eda/text.py), [eda/workspace.py](../attention_maps/eda/workspace.py) | Indentation and useful repeated instruction text can be lost |
| Deduplication observes text without labels or task identity | [eda/workspace.py](../attention_maps/eda/workspace.py), [batch_pipeline/runner.py](../attention_maps/batch_pipeline/runner.py) | Identical input with conflicting labels silently keeps the first |
| SFT cleaning reuses the prose cleaner per message; preserve mode still removes URLs/markup and applies Devanagari gates | [training/sft_data.py](../attention_maps/training/sft_data.py), [nepali_text.py](../scripts/utils/nepali_text.py) | Numeric, English-only, HTML, and code responses can be rejected or damaged |
| Chat normalization selects the last assistant message and silently discards subsequent messages | [training/sft_data.py](../attention_maps/training/sft_data.py) | Reject malformed conversations or explicitly audit repairs |
| SFT notebooks use seeded row splits without a preceding duplicate/prompt-family grouping stage | [Llama notebook](../notebooks/Llama2_7B_Nepali_MultiDataset_QLoRA.ipynb), [TinyLlama notebook](../notebooks/TinyLlama_Nepali_Alpaca_QLoRA.ipynb) | Evaluation overlap remains possible; the split called test is used for ongoing evaluation |
| UI D2 checks catalog split/confirmation, but batch records/config do not retain and enforce per-record split | [pruning UI](../apps/explorer_tabs/pruning.py), [batch contracts](../attention_maps/batch_pipeline/contracts.py) | Add train-only enforcement in the shared backend before model-aware selection |

The SFT helper also recognizes only user/system/assistant role aliases and
reduces messages to role/content. Function-calling datasets need a separate
adapter preserving tool roles, call IDs, arguments, and schemas. The current
last-turn conversion can be intentional, but is not equivalent to supervising
all assistant turns in a multi-turn conversation.

The pretraining builder deduplicates exact text before deterministic
document-level splitting; it does not provide near-duplicate or parent-document
split isolation. The cross-dataset overlap explorer is a sampled report, not a
persisted full-corpus decontamination stage. These are useful foundations with
different guarantees from final training-data certification.

D2 is a budgeted subset-selection method, not a universal cleaning requirement.
The original paper balances diversity and difficulty; its scope is broader than
this repository's current supported supervised implementation.
[D2 paper](https://arxiv.org/abs/2310.07931). Follow the
[local D2 policy](d2-pruning.md): train-only inputs, aligned embeddings and
difficulty scores, and comparisons with full-data/random/difficulty-only
baselines. A uniform difficulty score is a diagnostic ablation.

Recommended implementation order:

1. Preserve canonical payloads, source IDs, parent/group IDs, task fields, and
   split on every record; derive separate EDA/dedup views.
2. Add executable validators and separate conservative policies for prose,
   conversations, annotated tasks, and preferences.
3. Make paragraph removal and destructive character/markup filtering opt-in
   by purpose. Add label-conflict and complete-pair checks before deletion.
4. Persist duplicate clusters, protect benchmark membership, and enforce
   split/group boundaries in shared backends before selection or augmentation.
5. Add model-specific formatting, truncation, offset, and loss-mask audits.
6. Calibrate quality/LID/near-duplicate thresholds on reviewed Nepali samples,
   then evaluate policy variants under equal training budgets.

For each policy, report retained rows and tokens, reasons for rejection,
per-source/language/class retention, duplicate/conflict counts, split overlap,
truncation and target-token coverage, and schema validity. Manual review should
include both accepted and rejected examples, especially short, mixed-script,
code, and rare-label examples. Thresholds such as 0.80 similarity are starting
settings in this codebase, not established optimum values.

Verification during this review: 48 existing tests passed across SFT data,
Nepali cleaning, schema taxonomy, EDA, batch preprocessing, overlap, dataset
building, and materialized training data. Separate in-memory probes confirmed
role flattening, prompt-only preference extraction, indentation loss,
first-label retention on conflicting labels, silent trailing-user removal,
numeric-answer rejection under the LIMA cleaning settings, and HTML stripping
in preserve mode. Passing tests establish current behavior, not the suitability
of that behavior for every schema. No datasets or training code were changed.
