# Standard training-data schemas

The project uses four canonical schemas. Source adapters may accept other
column names, but materialized training data should be mapped to one of these
contracts. Every split assignment must happen before model-aware selection,
and validation/test records must never participate in pruning.

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

## Shared invariants

- IDs are unique and stable within a dataset version.
- `split` is one of `train`, `validation`, or `test`.
- Selection and augmentation operate only on `train`.
- Source, domain, language, and license provenance is retained when available.
- Related records and duplicate clusters cannot cross protected evaluation
  boundaries.
- Schema or label-vocabulary changes create a new dataset version.
