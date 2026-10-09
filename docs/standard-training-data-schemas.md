# Standard training-data schemas

The project uses seven dataset roles, four training instance schemas, and one
evaluation instance schema. Source adapters may accept other
column names, but materialized training data should be mapped to one of these
contracts. Every split assignment must happen before model-aware selection,
and validation/test records must never participate in pruning.

## Tracker roles and manual classification

| Dataset role | Target instance shape | What to inspect |
| --- | --- | --- |
| Pretraining | `id`, `text`, `split` | Independent unlabelled documents, including continued pretraining in a domain. |
| Instruction SFT | `id`, `messages`, `split` | Instruction/response pairs or conversations with assistant targets. |
| Preference tuning | `id`, `prompt`, `chosen`, `rejected`, `split` | Responses ranked against the same prompt; keep each pair together. |
| Evaluation | `id`, `input`, `reference`, `split` | Held-out inputs, expected answers/labels, and a defined scoring task. |
| Traditional NLP | `id`, `text`, `label`, `task`, `split` plus task annotations | Classification labels, token labels, spans, sentence pairs, or other explicit NLP targets. |
| Domain-specific fine-tuning | Supervised fields above; use the SFT contract for conversations | Domain relevance and explicit targets, such as medical QA or insurance intents. |
| Task-specific fine-tuning | `id`, `text`, `label`, `task`, `split` plus task fields | Inputs paired with targets for a defined downstream task. |

Roles describe intended use and can overlap. A domain-specific instruction
dataset can have both **Instruction SFT** and **Domain-specific fine-tuning**.
A domain by itself does not imply supervision: unlabelled domain documents fit
**Pretraining**. Traditional NLP annotations need adapters that retain native
tokens, spans or paired inputs; choosing a role does not flatten those records.
For supervised D2, an adapter must also supply the example-level labels required
by that algorithm. Do not decide suitability from a filename alone.

Inspect records in **Source sample**, then use **Metadata → Dataset tracker
annotations** to select the CSV entry and edit **Dataset role**, **Primary task**,
and **Data Acquisition Method**. Multiple roles are stored as comma-separated
canonical labels in `Dataset Role (LLM Lifecycle)`. Older labels such as
`Pre-training`, `Domain-specific Finetuning`, and `Task-specific supervised`
remain readable aliases. Internal role and training-schema keys remain stable.

**Primary task** describes the objective (for example, language modeling,
translation, sentiment analysis, or preference ranking). **Data Acquisition
Method** describes how records were collected or produced. The UI recognizes
`Data Acquisition Method`, `Data Acquision Method`, and the older `Dataset
Generation Method` header without renaming it. Task and method dropdowns offer
suggestions and accept custom values when needed; unreviewed/unknown cells can
remain blank.

**Save selected columns to tracker CSV** updates the chosen entry in
`llm_dataset_tracker - To-go-datasets.csv`. It preserves the spreadsheet title
rows, notes, other columns, and other dataset rows. It does not transform the
dataset's instance schema. Exact source URLs suggest matches; duplicated URLs
require an explicit row choice. Concurrent changes to the cells being saved
require **Reload tracker CSV**, while changes to other cells are preserved.
Reload also picks up replacement CSV exports. A single saved role initializes
WORKSPACE; for multiple roles, choose which one to use for the processing run.

## 1. Pretraining corpus

One record is one independently selectable document.

```json
{
  "id": "stable-document-id",
  "text": "नेपाली पाठ",
  "split": "train",
  "source": "publisher-or-corpus",
  "domain": "news",
  "language": "ne",
  "license": "license-id",
  "metadata": {}
}
```

Required fields are `id`, `text`, and `split`. Text must be non-empty after
normalization. `id` must remain stable across reruns. D2 is not enabled for
this schema in the current implementation.

## 2. Instruction fine-tuning

One record is one complete conversation. Messages must stay together during
sampling, deduplication, and split assignment.

```json
{
  "id": "stable-example-id",
  "messages": [
    {"role": "system", "content": "नेपालीमा उत्तर दिनुहोस्।"},
    {"role": "user", "content": "प्रश्न"},
    {"role": "assistant", "content": "उत्तर"}
  ],
  "split": "train",
  "source": "dataset-name",
  "domain": "general",
  "language": "ne",
  "license": "license-id",
  "metadata": {}
}
```

Required fields are `id`, `messages`, and `split`. The final training message
must be an assistant response and role order must be valid. D2 is not enabled
for this schema in the current implementation.

## 3. Task-specific supervised

This schema covers both traditional NLP tasks and domain-specific supervised
fine-tuning. One record is one labelled example.

The WORKSPACE **Dataset role** dropdown offers **Traditional NLP** and
**Domain-specific fine-tuning** as explicit categories alongside the general
**Task-specific fine-tuning** role. Both categories use this supervised instance schema.
They select `traditional_nlp` and `domain_specific_finetuning`, respectively,
when optional D2 pruning is enabled. Sampling and deduplication exports record
the chosen `dataset_role` separately from the canonical `training_schema`.

```json
{
  "id": "stable-example-id",
  "text": "यो चलचित्र राम्रो छ।",
  "label": "positive",
  "task": "sentiment-classification",
  "split": "train",
  "source": "movie-reviews",
  "domain": "entertainment",
  "language": "ne",
  "license": "license-id",
  "metadata": {}
}
```

Required fields are `id`, `text`, `label`, `task`, and `split`. Labels must use
one documented vocabulary per task. Sentence pairs or token annotations may be
stored in task-specific fields, but adapters must produce one text
representation, one example label, and one stable ID for D2.

This is the only schema currently eligible for D2. The supported use cases are
`traditional_nlp` and `domain_specific_finetuning`. Correct-label confidence
variability is the recommended difficulty signal. Uniform difficulty remains
available only as a diagnostic ablation.

## 4. Preference tuning

One record is one atomic prompt/preference group. Chosen and rejected responses
must never be sampled or split independently.

```json
{
  "id": "stable-preference-id",
  "prompt": "प्रश्न",
  "chosen": "रुचाइएको उत्तर",
  "rejected": "अस्वीकृत उत्तर",
  "split": "train",
  "source": "preference-collection",
  "domain": "general",
  "language": "ne",
  "annotator_count": 3,
  "metadata": {}
}
```

Required fields are `id`, `prompt`, `chosen`, `rejected`, and `split`. A chosen
response must not equal its rejected response after normalization. D2 is not
enabled for this schema because pair-level representations and reliable
preference difficulty signals have not been validated here.

## 5. Evaluation

One record is one evaluation input with its expected reference or label.

```json
{
  "id": "stable-evaluation-id",
  "input": "प्रश्न",
  "reference": "अपेक्षित उत्तर",
  "split": "test",
  "task": "question-answering",
  "language": "ne",
  "metadata": {}
}
```

Required fields are `id`, `input`, `reference`, and `split`. Multiple-choice
options, acceptable references, scoring rules and task-specific fields should
be retained. Evaluation is a usage role, not a training objective; assigning
it in the tracker does not establish that a dataset is uncontaminated or that
its splits are independent.

## Shared invariants

- IDs are unique and stable within a dataset version.
- `split` is one of `train`, `validation`, or `test`.
- Selection and augmentation operate only on `train`.
- Source, domain, language, and license provenance is retained when available.
- Related records and duplicate clusters cannot cross protected evaluation
  boundaries.
- Schema or label-vocabulary changes create a new dataset version.
