# Multi-dataset EDA survey progress

Branch: `feature/multi-dataset-eda`

This is the active work log for the corpus-survey EDA foundation. The broader
project history remains in [project-progress.md](project-progress.md).

## Objective

Build one reproducible pipeline that can profile heterogeneous Nepali corpora,
compare them on consistent metrics, and produce survey-ready derived artifacts
without persisting sampled source text.

## Reference audit

Reviewed `/home/aixi/Nepali-Corpus-EDA` on 2026-09-09. Reused its useful study
questions: length, script composition, vocabulary, provenance, duplicates,
basic quality, and cross-dataset comparison. Its notebooks currently import
core catalog/streaming/profile functions that are absent from its Python
module, so the implementation here is a clean modular foundation rather than a
direct copy.

## Completed

- [x] Create a separate EDA branch.
- [x] Define validated dataset, analysis, survey, and result contracts.
- [x] Add schema extraction for plain text, nested fields, chat messages, and
  instruction/response records.
- [x] Add bounded Hugging Face streaming with deterministic shuffle settings.
- [x] Compute length, Devanagari composition, quality, vocabulary, provenance,
  exact duplicates, and approximate SimHash near-duplicates incrementally.
- [x] Bound plot samples and vocabulary state so analysis memory does not scale
  with raw text volume.
- [x] Isolate per-dataset failures so a survey can continue.
- [x] Write per-dataset JSON/CSV, survey JSON/CSV, a reproducibility manifest,
  and consistent comparison plots.
- [x] Add an initial 17-dataset Himalaya catalog configuration.
- [x] Add unit coverage for schema, metrics, bounds, failure isolation, and
  artifact writing.
- [x] Add log-scaled character/word histograms and sentence/line KDE plots with
  their bounded numeric samples exported as CSV.
- [x] Add configurable top bigram, trigram, and 4-gram tables and horizontal
  bar charts.
- [x] Add configurable seed-term co-occurrence edge tables and network plots.
- [x] Integrate single-dataset EDA into Streamlit with live stage progress,
  per-metric drill-down, plot views, and artifact downloads.
- [x] Normalize analysis text and stopwords to NFC, remove BOMs, normalize
  attached Nepali postpositions for co-occurrence matching, and validate
  Devanagari font glyph coverage for cleaned WordCloud input.
- [x] Move the shared Nepali stopword list to `configs/eda/stopwords.txt` and
  load it consistently from both CLI survey configs and Streamlit.

## In progress

- [ ] Validate schema overrides for every configured dataset against a small
  streamed slice and pin repository revisions for the final survey run.
- [ ] Decide which corpora belong in pretraining, SFT, evaluation, or mixed
  survey strata before interpreting a single combined ranking.

## Next work

- [ ] Add license, dataset-card, and provenance audit fields to the study config.
- [ ] Add stratified/manual review samples using hashes and row identifiers,
  without embedding raw text in committed artifacts.
- [ ] Add language-identification, Unicode corruption, repetition, URL/code,
  and sensitive-content diagnostic plugins.
- [ ] Add bootstrap confidence intervals so dataset comparisons do not present
  sample estimates as exact population values.
- [ ] Add a cross-dataset Streamlit queue for launching and comparing several
  selected corpora in one unattended survey run.

## Runbook

Smoke-test one dataset with a small temporary config before a full survey. The
default configuration streams up to 50,000 records per dataset:

```bash
source .venv/bin/activate
python -m attention_maps.eda \
  --config configs/eda/nepali_corpus_survey.json \
  --datasets nepali-corpus-compile \
  --output-dir artifacts/eda/smoke
```

Run all configured datasets:

```bash
python -m attention_maps.eda \
  --config configs/eda/nepali_corpus_survey.json \
  --output-dir artifacts/eda/nepali-corpus-survey
```

`artifacts/` is ignored by Git. The manifest explicitly records that raw text
was not persisted.
