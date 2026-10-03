# How Language coverage and Script are determined

Inspection analyzes the **selected dataset portion**: chosen files or Hugging Face configuration/split/shards, then any row filter. It samples complete instances from that portion. For pretraining, an instance is one document, and only its mapped text is analyzed; IDs and source metadata are excluded.

## Taxonomy and decision order

```text
Selected portion → sampled records
  └─ Language evidence: record column → HF config/split name → fastText
      └─ Coverage: Nepali-only | English-only | Nepali-English | Multilingual
          └─ Nepali covered? If yes, count writing systems in eligible text:
              Devanagari | Romanized | Devanagari+romanized | Nepali+English
                  └─ One coverage and one script vote per sampling run
                      → final displayed categories
```

Missing or unsupported evidence stays in separate counters; the UI does not force it into a category. Script is conditional on Nepali coverage.

## Sampling and record percentages

Let `N` be selected source rows, `f` the sample fraction, `K` the number of runs, and `U` the number of **distinct** rows selected by at least one run. The current [dataset YAML](../../configs/datasets/dataset.yaml) uses `f = 0.20`, `K = 5`.

```text
Rows per run      = ceil(f × N)
Language share(c) = 100 × distinct sampled records in category c / U
Script share(c)   = 100 × distinct Nepali-eligible sampled records in c
                    / distinct Nepali-eligible sampled records
```

Each run samples without replacement; runs can overlap. The source is scanned once, and an overlapping record is analyzed once but credited to each run that selected it. **Sample and classify** counts source rows scanned, not rows classified. No-evidence and outside-category records remain in the percentage denominators.

## Language coverage

Evidence priority for each record:

1. **Language column** (`language`, `language_code`, `lang`, `language_pair`, etc.). `ne`/`nep`/`npi` mean Nepali; `en`/`eng` mean English. A pair can name both.
2. **Named Hugging Face partition**, if the record has no column label. Split `nep` supplies Nepali; generic `train` does not. A dataset-card language tag is not used.
3. **fastText**, only when neither source above supplies evidence. It predicts one dominant language per record; low-confidence or too-short predictions remain unlabelled. The model reads at most 4,000 characters with the current YAML; script counting reads the full text.

Kaggle/local files have no Hugging Face partition fallback. A single-label fastText prediction cannot establish a bilingual or multilingual *record*.

| Languages found for a record or retained for a run | Coverage category |
| --- | --- |
| `{ne}` | Nepali-only |
| `{en}` | English-only |
| `{ne, en}` | Bilingual (Nepali-English) |
| Two or more including another language | Multilingual |

For a **run-level vote** based on columns or detector predictions, let `S` be sampled records and `M` records with a column label or accepted prediction. A language can count once per record; pairs can make language shares sum above 100%.

```text
Labelled fraction       = M / S                       must be ≥ 0.80
Share of language x     = labelled records naming x / M
Retain language x only if its share is               ≥ 0.05
If only one x is retained, its share must be         ≥ 0.95
```

Hugging Face partition fallback contributes to the **per-record category percentages**, but not `M`. If a run has no column labels or detector attempts, an explicit language-named partition directly supplies its run label; these record-ratio thresholds are not presented as measured. The earlier screenshot's 100% Nepali-only record share came from split `nep`, not text-language verification of every document. The current `default/train` dataset has no language-named split, so its language evidence must come from columns or fastText.

## Script: only when Nepali is covered

Nepali-labelled/predicted records, plus unlabelled records in a Nepali-named Hugging Face partition, are eligible. Explicitly non-Nepali records are not made eligible by that partition. Count Unicode Devanagari letters/marks (`D`), Latin letters/marks (`L`), and other-script letters (`O`) in the **full eligible text**; ignore digits, punctuation, and spaces. Set `T = D + L + O`.

```text
Other share = O / T                 must be ≤ 0.05
D / (D + L) ≥ 0.90                 → Devanagari
Otherwise, if English is covered  → Mixed (Nepali + English)
Otherwise, L / (D + L) ≥ 0.90      → Romanized
Otherwise                          → Mixed (Devanagari + romanized)
```

When `T = 0`, `D + L = 0`, or Other exceeds 5%, no supported Nepali-only script category is assigned. Each record gets a category from **its own** counts; each run votes using **pooled eligible character counts**. For a run covering both Nepali and English, English-only text also contributes to that pooled decision. Thus a Devanagari run vote can coexist with some mixed or Romanized records. “Romanized” is a script heuristic, not proof that every Latin passage is Nepali.

## Final vote and code references

With `K` runs and minimum agreement `a`, a final label needs:

```text
required votes = max(floor(K / 2) + 1, ceil(a × K))
```

The current `K = 5`, `a = 0.60` require **3 of 5** votes. Inconclusive votes remain in the denominator but cannot establish a category. Nepali presence is voted on separately before the final Script label. Overlapping runs and a shared partition label mean 5/5 agreement shows **consistent decisions**, not independent verification. Streamlit displays the saved script result; it does not reclassify data.

| Code | Responsibility |
| --- | --- |
| [`inspect_dataset.py`](../../scripts/inspect_dataset.py), [`huggingface.py`](../../attention_maps/explorer/huggingface.py) | Select source/configuration/split/shards and count source rows. |
| [`semantic_instances.py`](../../attention_maps/explorer/semantic_instances.py) | Choose complete-instance text and preserve metadata separately. |
| [`sampling_vote.py`](../../attention_maps/explorer/sampling_vote.py) | Exact random selection, per-run evidence, and majority vote. |
| [`language_status.py`](../../attention_maps/explorer/language_status.py), [`language_detection.py`](../../attention_maps/explorer/language_detection.py) | Language evidence, script counts, categories, and optional fastText. |
| [`classification_thresholds.py`](../../attention_maps/explorer/classification_thresholds.py), [`inspection_reports.py`](../../apps/components/inspection_reports.py) | Threshold validation and read-only UI display. |
