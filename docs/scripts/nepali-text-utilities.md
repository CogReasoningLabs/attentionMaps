# `scripts/utils/nepali_text.py`

## Purpose

Internal reusable text-cleaning library shared by the cleaning pipeline,
processed-dataset builder, EDA notebooks, and tests. It has no command-line
interface and performs no file I/O.

Users normally run `scripts/clean_pretraining_sources.py` instead of importing
this module directly.

## Public configuration

### `CleaningConfig`

```python
from scripts.utils.nepali_text import CleaningConfig

config = CleaningConfig(
    min_devanagari_ratio=0.80,
    min_devanagari_letters=10,
    min_characters=20,
    mode="strict",
    normalization="NFC",
)
```

| Field | Library default | Meaning |
|---|---:|---|
| `min_devanagari_ratio` | `0.80` | Minimum Devanagari share among Unicode letters |
| `min_devanagari_letters` | `10` | Minimum absolute Devanagari-letter count |
| `min_characters` | `20` | Minimum non-whitespace characters after cleaning |
| `mode` | `preserve` | `preserve` mixed-script text or `strict` Nepali-safe characters |
| `normalization` | `NFC` | Unicode normalization: `NFC` or `NFKC` |

The library default for `mode` is `preserve` to make direct API use
conservative. The dataset cleaning command explicitly defaults to `strict` for
the current Nepali-only research corpus.

Invalid ratios, non-positive content limits, unsupported modes, and unsupported
normalization forms raise `ValueError` when the configuration is created.

## Main functions

### `clean_text_with_result(text, config)`

Runs the full document-cleaning policy and returns `CleaningResult`:

```python
from scripts.utils.nepali_text import CleaningConfig, clean_text_with_result

result = clean_text_with_result(
    "<p>नेपाल सुन्दर देश हो। English</p>",
    CleaningConfig(
        min_devanagari_ratio=0.50,
        min_devanagari_letters=1,
        min_characters=1,
        mode="strict",
    ),
)

if result.accepted:
    print(result.text)
else:
    print(result.reason)
```

`CleaningResult` contains:

| Field | Meaning |
|---|---|
| `text` | Cleaned text, or an empty string when rejected |
| `reason` | `None`, `empty`, `low_ratio`, `few_devanagari`, or `short` |
| `devanagari_ratio` | Input ratio measured after markup/web/control removal and before strict filtering |
| `devanagari_letters` | Input Devanagari-letter count at the language gate |
| `accepted` | Convenience property equivalent to `reason is None` |

This is the preferred function when a caller needs rejection reasons and
quality metadata.

### `clean_text(text, config)`

Convenience wrapper returning only cleaned text. Rejected input becomes `""`.
Use it for simple transformations; use `clean_text_with_result` for corpus
audits.

### `devanagari_letter_stats(text)`

Returns `(ratio, devanagari_letter_count)`. The ratio denominator includes
Unicode letters only, excluding digits, spaces, symbols, and punctuation.

This prevents dates and formatting-heavy documents from appearing less Nepali
merely because they contain many non-letter characters.

### `is_devanagari(character)`

Recognizes characters in the main Devanagari block plus supported extended
Devanagari blocks. It is a script/block check, not a language classifier.

### Artifact-removal helpers

- `remove_markup_artifacts(text)` removes HTML comments/tags and complete
  script/style payloads, then decodes HTML entities.
- `remove_web_artifacts(text)` removes full URLs and email addresses before
  character filtering can leave orphan punctuation.
- `remove_unsafe_controls(text)` removes control/format debris while preserving
  newlines, tabs, ZWNJ, and ZWJ.
- `normalize_whitespace(text)` normalizes horizontal Unicode whitespace and
  collapses repeated blank lines while preserving document line boundaries.

### `is_allowed_strict_character(character)`

Used by strict mode to retain:

- supported Devanagari characters;
- whitespace;
- ASCII digits;
- configured Nepali-compatible punctuation;
- ZWNJ and ZWJ used in conjunct behavior.

It removes Latin letters and unsupported symbols only after the document has
passed the input language gate.

## Cleaning order

```text
validate input
→ Unicode NFC/NFKC normalization
→ remove HTML/script/style and decode entities
→ remove URLs/emails
→ remove unsafe controls
→ calculate Devanagari language statistics
→ reject low-ratio/few-letter content
→ strict filtering or mixed-script preservation
→ punctuation-debris cleanup
→ whitespace normalization
→ reject short output
```

The ratio is intentionally calculated before strict Latin removal. Otherwise,
an English-heavy document with a few Nepali words could be stripped down and
incorrectly appear to be pure Nepali.
