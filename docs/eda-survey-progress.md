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
  normalized SHA-256 exact duplicates, MinHash-LSH near duplicates, and
  repeated-paragraph evidence incrementally.
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
- [x] Add an in-app EDA and cleaning notes tab documenting current metrics,
  duplicate screening, the planned auditable cleaning flow, and open decisions.
- [x] Document the implemented local pretraining language gate and preprocessing
  order, source-specific defaults, rejection reasons, provenance outputs, and
  its distinction from read-only EDA and visualization-only token filtering.
- [x] Separate compressed Hugging Face file size from decoded memory size in
  the explorer and surface its bounded streaming/sample limits for large data.
- [x] Classify sampled dataset text as Devanagari, mixed Devanagari/Romanized,
  Romanized, Latin, other-script, or mixed Nepali/English; expose the evidence
  in Survey EDA and preserve input/output categories in cleaning metadata.
- [x] Derive EDA sample sizes from population percentages, run five repeated
  folds by default, report expected unique coverage and across-fold stability,
  and expose full-population mode for UI-safe corpora.
- [x] Add three-stage duplicate diagnostics with explicit normalization and
  hashing settings, per-fold UI metrics, CSV evidence, and a figure.
- [x] Add a per-dataset Streamlit WORKSPACE with separately gated sampling,
  NFC-normalization, and materialized multi-stage deduplication buttons; persist
  timestamped clean JSONL/audit manifests and unlock EDA only for clean data.
- [x] Focus the Streamlit explorer on four tabs (WORKSPACE, Source sample,
  Inference, and Metadata), consolidate all inference/evaluation tools behind
  one selector, and expose one fixed EDA section only after preprocessing.
- [x] Add independent provider, source/adaptation lineage, purpose hierarchy,
  and tags to the dataset explorer; classify the 17 known Himalaya AI datasets.
- [x] Make dataset-purpose choices dependent on the selected provider/lineage,
  exposing only purpose mappings that exist in the human-curated catalog.
- [x] Add a reusable synthetic-dataset family/variant registry and pipeline card
  shared by both apps; register LIMA original/translated as the first family.
- [x] Split the 5,044-line explorer into reusable domain modules and
  feature-oriented Streamlit renderers; enforce a 1,000-line source-module cap.
- [x] Split the 1,119-line generator into reusable generation services and keep
  both Streamlit composition roots below the enforced module-size cap.
- [x] Align the README and architecture notes around the three project paths:
  dataset exploration, synthetic generation, and planned model/attention work.

## In progress

- [ ] Validate schema overrides for every configured dataset against a small
  streamed slice and pin repository revisions for the final survey run.
- [ ] Decide which corpora belong in pretraining, SFT, evaluation, or mixed
  survey strata before interpreting a single combined ranking.
- [ ] Generalize cleaning beyond the implemented local PDF/news/lyrics cleaner
  and add reviewed near-deduplication materialization; current Survey EDA
  duplicate metrics remain read-only screening evidence.

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
- [ ] Register additional synthetic dataset families only after their source,
  transformation, output schema, purpose, and provenance rules are reviewed.

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
