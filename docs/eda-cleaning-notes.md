# Dataset EDA, cleaning, and deduplication notes

## Current status

The **Survey EDA** workflow is diagnostic and read-only with respect to source
datasets. It calculates evidence and writes derived reports, but it does not
delete, rewrite, or language-filter source rows. A source-specific cleaning
pipeline is already implemented for the local PDF, news, and lyrics
pretraining sources. A general cleaning and reviewed near-deduplication
pipeline for every catalog dataset is still planned.

The same notes are rendered in the Streamlit **EDA & cleaning notes** tab so
the methodology stays beside the metrics. This Markdown file is the source of
truth; changes should be made here rather than duplicated in UI code.

## Language filtration and preprocessing actually implemented

Three related paths exist, and their outputs must not be confused:

| Path | Changes source text? | Purpose |
|---|---:|---|
| Survey EDA | No | Normalize a temporary analysis copy and report quality evidence |
| Pretraining source cleaner | Writes a new cleaned dataset | Filter and standardize the local PDF, news, and lyrics sources |
| WordCloud/token display | No | Produce visualization-only Devanagari tokens and stopword-filtered frequencies |

### Survey EDA: diagnostic normalization, not row filtration

For each extracted logical document, Survey EDA applies NFC normalization,
removes a byte-order mark, and collapses whitespace on an in-memory analysis
copy. Its regex tokenizer recognizes Devanagari, Latin, and numeric word-like
units. A row is **usable** when text can be extracted, the normalized text is
non-empty, and at least one regex token exists.

The configured minimum words and Devanagari ratio are quality thresholds only.
The Streamlit defaults are five words and a `0.70` Devanagari ratio. A row below
either threshold remains in the EDA length, token, n-gram, duplicate, and other
usable-row metrics; it simply does not increment `quality_pass_rows` or
`devanagari_clean_rows`. Running Survey EDA therefore does not create a cleaned
dataset.

The EDA Devanagari ratio is the share of Unicode letters and combining marks in
the U+0900–U+097F block. Whitespace, digits, and punctuation do not contribute
to that ratio.

### Pretraining source cleaner: materialized filtering

`scripts/clean_pretraining_sources.py` uses the shared implementation in
`scripts/utils/nepali_text.py`. It currently supports the local Nepali PDF,
news, and music-lyrics sources and writes new sharded Parquet outputs rather
than changing raw inputs. Processing occurs in this order:

1. Reject missing, non-string, or blank text as `empty`.
2. Normalize Unicode to NFC by default; NFKC is available explicitly.
3. Remove script/style payloads, HTML comments/tags, URLs, and email addresses,
   and decode HTML entities.
4. Remove unsafe Unicode control/format characters while preserving newlines,
   tabs, ZWNJ, and ZWJ needed by Devanagari conjuncts.
5. Calculate the Devanagari-letter ratio before strict character removal. Its
   denominator is Unicode **letters only**, so spaces, punctuation, symbols,
   and digits do not dilute it.
6. Reject rows below the configured ratio as `low_ratio`, or below the minimum
   Devanagari-letter count as `few_devanagari`.
7. In the CLI's default `strict` mode, retain Devanagari characters, ASCII
   digits, whitespace, approved punctuation, ZWNJ, and ZWJ. Remove unsupported
   characters, isolated punctuation tokens, and repeated separators. The
   alternative `preserve` mode retains legitimate mixed-script content after
   the row passes the language gate.
8. Normalize intra-line whitespace and repeated blank lines, then reject empty
   or too-short results as `empty` or `short`.

The current command defaults are:

| Source | Minimum Devanagari-letter ratio | Minimum Devanagari letters | Minimum visible characters |
|---|---:|---:|---:|
| PDF | 0.80 | 10 | 20 |
| News | 0.80 | 10 | 20 |
| Lyrics | 0.10 | 10 | 10 |

The lyrics threshold is intentionally more permissive for short, mixed-script
song content. These values are operational defaults, not research-validated
universal thresholds; retention and rejection counts must be reviewed per
source before promotion.

Accepted rows preserve `source`, `source_id`, `language`, URL, raw-text SHA-256,
cleaned-text SHA-256, and cleaning statistics in metadata. Each source also
receives a manifest containing the exact configuration, output files, counts,
rejection reasons, and character retention. The next build stage can perform
deterministic source sampling, exact cleaned-text SHA-256 deduplication, stable
train/validation/test assignment, and schema standardization. Cleaning itself
does not tokenize, split, combine, or near-deduplicate the sources.

### Important limitation of the language gate

This is a **Devanagari-script heuristic**, not a Nepali language classifier.
Hindi, Sanskrit, Marathi, or other Devanagari text can pass it, while Romanized
Nepali can fail it. Code-switching may also be damaged by `strict` mode. A
future language-identification stage should combine script ratio with a
validated Nepali classifier and manual samples; it must record confidence and
quarantine uncertain rows rather than silently deleting them.

Stopword removal and suffix stripping are used for WordCloud/co-occurrence
analysis only. They are not applied to materialized training text, because
removing function words or changing word forms would alter the language-model
training distribution.

## Dataset-purpose taxonomy and provider lineage

Dataset ownership and dataset purpose are independent facets. The catalog uses
the following purpose hierarchy roots:

- **Pretraining corpus**: general, web, news, document, romanized, or aggregated
  text used for next-token language-model training.
- **Instruction fine-tuning**: general instruction following, function calling,
  structured output, translated instruction data, or compiled SFT mixtures.
- **Preference tuning**: chosen/rejected comparisons, rankings, reward-model
  data, or other preference-optimization inputs. This bucket is reserved even
  when no current dataset has been confidently assigned to it.
- **Task-specific fine-tuning**: supervised data for a bounded task such as
  proofreading, sentiment, or hate-speech classification.
- **Tokenizer development**: text selected specifically to train or assess a
  tokenizer.
- **Evaluation / benchmark**: held-out measurement data that must not be mixed
  into model training.
- **Unclassified — pending review**: the safe default when purpose has not been
  verified.

The catalog tracks three provider roles separately:

- **Provider**: organization publishing the selected dataset artifact.
- **Source provider**: publisher of the upstream dataset from which it derives.
- **Adapted by**: organization/model provider responsible for a transformation.

For example, the translated Nepali LIMA view is published by the local project,
originates from GAIR/LIMA, and was adapted through a Google-hosted model
pipeline. Filtering by Google can find this adaptation without incorrectly
claiming Google published the upstream LIMA dataset. Himalaya AI datasets are
classified per dataset; the provider name never determines the purpose
automatically. Arkios and preference-tuning buckets remain available for
future human-curated mappings without changing the taxonomy.

Synthetic pipeline artifacts are intentionally outside the ordinary source
catalog. The explorer and generator share one registry of pipeline families
and materialized variants. LIMA translation is the first registered family;
future families must declare their input, transformation, output, purpose, and
provenance before they are exposed in either application.

## Population, sample, and usable rows

- **Population rows**: all rows in the selected dataset or filtered split.
- **Requested rows**: the maximum number of rows requested for an EDA run.
- **Examined rows**: rows actually presented to the analyzer.
- **Usable rows**: examined rows from which non-empty text and at least one
  regex token were extracted.
- **Coverage**: `examined rows / population rows × 100`.

When requested and examined rows equal the population, the complete selected
population was analyzed. Otherwise, reported frequencies and distributions are
sample estimates. Hugging Face filtered-dataset byte sizes can be estimates
even when the filtered row count is known.

## Core EDA logic

1. Extract configured text fields from plain, nested, chat, or instruction
   records. Multiple selected fields are joined into one logical document.
2. Apply NFC Unicode normalization, remove byte-order marks, and normalize
   whitespace. The original source row is not overwritten.
3. Calculate document characters, regex-token counts, sentence-token lengths,
   physical-line character lengths, and Devanagari composition.
4. Track exact and near duplicates, bounded token/n-gram frequencies,
   configured seed-term co-occurrences, and row-level provenance.
5. Retain deterministic bounded numeric samples for percentiles, histograms,
   and KDE plots; means, maxima, and counters use every examined usable row.
6. Export summary JSON, evidence CSVs, figures, run configuration, and runtime
   metadata. Raw sampled documents are not embedded in EDA artifacts.

### Metric interpretation

- **Regex token**: a word-like Devanagari/Latin/numeric unit, not a model
  tokenizer token.
- **Normalized character**: a Unicode character counted after NFC and
  whitespace normalization.
- **KDE density**: a smoothed distribution shape. Its height is not a row count
  or percentage; probability is represented by area under the curve.
- **N-grams**: surface-form sequences of 2, 3, or 4 regex tokens. Surface forms
  are intentionally not stemmed.
- **Co-occurrence edge**: a seed term and normalized neighboring token observed
  within the configured window. A pairing contributes at most once per
  document.

## Existing duplicate screening

### Exact duplicates

The analyzer hashes the NFC-normalized, whitespace-collapsed document with a
128-bit BLAKE2 digest. The first digest occurrence is treated as the original;
later occurrences increment `exact_duplicate_rows`.

### Near duplicates

Every non-exact document receives a deterministic 64-bit SimHash built from
case-folded regex-token frequencies. Eight bands retrieve candidates, and the
default Hamming-distance threshold is three bits. A matching later document
increments `near_duplicate_rows`.

The displayed duplicate rate is:

```text
(exact_duplicate_rows + near_duplicate_rows) / usable_rows × 100
```

SimHash is screening evidence, not an automatic deletion instruction. It can
consider reordered documents similar because it emphasizes token frequencies.

## Planned cleaning and deduplication pipeline

```text
immutable source rows
  → schema/text extraction
  → canonical comparison text + stable row ID
  → quality flags
  → exact duplicate clusters
  → near-duplicate candidates
  → pair verification and cluster review
  → deterministic keeper policy
  → cleaned data + quarantine/audit ledger + manifest
```

### 1. Preserve identity and provenance

Assign every row a stable ID derived from dataset, split, source identifier,
and row position. Preserve original text, source metadata, license information,
and the input revision. Cleaning transforms operate on a separate canonical
comparison field.

### 2. Normalize without silently damaging content

Apply NFC, BOM/control-character cleanup, newline policy, and whitespace
normalization. Devanagari-only filtering is appropriate for a Nepali
WordCloud, but must not be applied globally because it would destroy English,
numbers, URLs, code, and mixed-language evidence.

### 3. Produce quality flags before removal

Flag empty or extremely short text, abnormal document/line length, low script
ratio, replacement characters, repeated spans, excessive URLs/code, and other
survey-approved diagnostics. Initially quarantine questionable rows rather
than deleting them.

### 4. Cluster exact duplicates

Group rows by canonical-text digest. Select one keeper deterministically using
an explicit priority order: trusted provenance/license, completeness, desired
dataset stratum, then stable row ID. Record every removed row's digest, keeper
ID, source, and reason.

### 5. Verify near duplicates

Use SimHash only for candidate generation. Before removal, verify candidate
pairs with interpretable measures such as token-set Jaccard overlap, length
ratio, and optionally normalized edit similarity. Thresholds must be tuned on a
manually labeled Nepali sample. Avoid blindly merging transitive chains where
A resembles B and B resembles C but A does not resemble C.

### 6. Deduplicate in two passes

Run within-dataset deduplication first, followed by cross-dataset comparison.
Cross-dataset keeper priority must follow the survey's source quality, license,
and intended-use policy rather than input iteration order.

### 7. Materialize auditable outputs

Write cleaned shards separately from immutable inputs. Produce a manifest with
input revisions, configuration, counts per reason, thresholds, code commit,
and output hashes. Produce a quarantine/rejection ledger that maps removed row
IDs to keeper IDs and reasons. Never overwrite raw data in place.

### 8. Validate before promotion

Compare before/after row counts, length/script distributions, n-grams, source
coverage, and evaluation-set contamination indicators. Manually review samples
from every removal category. Promote a cleaned dataset only after the dry-run
report is approved.

## Decisions still required

- Survey strata and cross-dataset source priority.
- Minimum quality thresholds per pretraining, SFT, and evaluation dataset.
- Manually validated near-duplicate thresholds.
- License compatibility and required provenance fields.
- Quarantine retention period and approval process.
