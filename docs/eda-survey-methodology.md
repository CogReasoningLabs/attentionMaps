# Corpus survey EDA methodology

## Pipeline contract

```text
survey JSON
    │
    ├── dataset identity, revision, split, schema overrides
    └── shared sampling and metric policy
             │
             ▼
bounded streaming source
             │
             ▼
schema adapter ──> Unicode/token metrics ──> bounded accumulators
                                             │
                    ┌────────────────────────┴───────────────────────┐
                    ▼                                                ▼
          per-dataset profile                              survey comparison
          JSON + derived CSV                              CSV + manifest + plots
```

The analyzer never persists raw sampled text. It retains SHA-256 digests,
MinHash signatures/LSH buckets, normalized paragraph digests, bounded
token/source/pattern counters, and deterministic reservoirs of numeric metrics
for quantiles and plots.

Survey normalization is not a cleaning or language-filtering stage. It creates
an in-memory NFC-normalized analysis copy, removes byte-order marks, and
normalizes whitespace. Rows below the configured word-count or Devanagari-ratio
threshold remain in all usable-row diagnostics; those thresholds only control
the `quality_pass_rows` and `devanagari_clean_rows` counters. The separate local
pretraining cleaner, its source-specific defaults, and its limitations are
documented in [eda-cleaning-notes.md](eda-cleaning-notes.md).

## Core measurements

- Document length: normalized Unicode character count and regex token count.
- Document structure: sentence length in regex tokens (split on danda,
  punctuation, and newlines) and physical line length in normalized characters.
- Script composition: share of Unicode letters/marks in the Devanagari block.
- Basic quality: at least the configured token count and Devanagari ratio.
- Lexical profile: bounded token frequencies and observed type-token ratio.
- Phrase profile: bounded frequency tables and horizontal bar charts for each
  configured n-gram order (2, 3, and 4 by default).
- Term relationships: a seed-term network over neighbors in a configurable
  token window. An edge count is the number of documents containing that local
  pairing; repeated occurrences in the same document count once.
- Provenance: configured or automatically detected source/domain fields.
- Duplicates: normalized SHA-256 exact matching, character-shingle MinHash-LSH
  near-document screening, and repeated normalized paragraph evidence.

Near-duplicate counts are screening estimates, not ground truth. Type-token
ratio becomes a documented lower bound when the vocabulary cap is reached.
Quantiles and distribution plots come from deterministic reservoir samples;
means, maxima, counts, and rates are computed over every usable streamed row.
Pattern analysis is capped at a configurable number of leading tokens per
document. The summary and warnings expose documents affected by that cap.
Input text and stopwords are normalized to NFC and stripped of byte-order marks
before matching. Co-occurrence tokens use a conservative one-step Nepali
postposition stripper so forms such as `नेपालको`, `मन्त्रालयबाट`, and
`कार्यालयमा` connect to their configured base seed terms. Phrase-frequency
tables preserve surface forms rather than silently stemming the corpus.
The canonical Nepali list lives at `configs/eda/stopwords.txt`. The survey JSON
resolves that file relative to itself, and Streamlit loads the same resource for
EDA co-occurrence and WordCloud controls.

Streamlit derives each fold size from a population percentage, runs five
independently seeded folds by default, and reports expected overlap-adjusted
coverage. These are repeated EDA samples rather than model cross-validation
folds. The fold table is the primary check on estimate stability; a percentage
alone is not evidence that rare subpopulations are represented.

Sentence and line KDE curves use a bounded deterministic sample. The report
implements Gaussian KDE directly with NumPy, trims the displayed range at the
sampled 99th percentile, and keeps the underlying sample CSV so plots can be
reproduced or restyled. Document-size histograms use a logarithmic x-axis to
make very short files and extreme merged documents visible together.

The default seed list is only a starting hypothesis. Survey runs should set
`cooccurrence_terms` per research question and record that configuration in the
manifest; an empty list intentionally produces an empty network rather than
selecting terms after seeing the data.

## Survey interpretation

The catalog mixes pretraining corpora, instruction datasets, and evaluation
sets. Their metrics are comparable as descriptive data diagnostics, but a
single quality ranking would erase their different purposes. Final reporting
should stratify by intended use and pair automated metrics with dataset-card,
license, provenance, and manual-content review.

For publication, pin every dataset `revision` to an immutable commit and retain
the generated `survey_manifest.json`, the study config, environment lock, and
analysis commit SHA.
