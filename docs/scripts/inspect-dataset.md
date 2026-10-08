# Dataset inspection and Devanagari filtering

The standalone script owns the inputs and processing. Streamlit displays the
single saved CSV report, including language coverage, script, selected source sizes,
file extensions, batch parsing requirements, and filtering counts. Opening or
refreshing the report never reruns processing.

## One shared sheet for manual dataset runs

Every successful inspection now appends one row to
`artifacts/dataset_inspection/history.csv` by default. Open it in Excel or
LibreOffice, or browse it in Streamlit under **Script results → History sheet**.
Run the script manually for each source, changing its inputs each time:

```bash
# Kaggle: one selected file.
venv/bin/python scripts/inspect_dataset.py \
  --provider kaggle --dataset hsebarp/oscar-corpus-nepali \
  --dataset-file ne_dedup.txt --text-column text

# Hugging Face: replace the example identifiers with values from --list.
venv/bin/python scripts/inspect_dataset.py \
  --provider huggingface --dataset OWNER/DATASET \
  --config CONFIG_NAME --split train --text-column text

# Local file: the same shared history sheet.
venv/bin/python scripts/inspect_dataset.py \
  --local data/example.jsonl --text-column text
```

The standard report is **one self-contained CSV**:
`artifacts/dataset_inspection/history.csv`. The shared dataset YAML sets
`output` to that path. Every successful inspection appends one row; earlier runs
remain in the same file. Repeating a dataset preserves its previous seed,
settings, votes, and decision. A run means one script invocation, which can
contain multiple independent sampling votes.

The CSV records `Started at`, `Completed at`, and `Processing seconds` for each
successful inspection. Processing time spans command setup, source discovery/loading,
sampling, classification, and report construction; it excludes writing the CSV and
printing the report. Historical rows have blank timing cells.

The CSV records provider, dataset, configuration, split, version/revision,
selected files, formats, size, sampling scope, Language Coverage, Script, pooled
Devanagari percentage, vote totals, every sampling run's evidence, and the final
decision explanation. Filtered-output labels and counts have separate columns.
Devanagari percentage measures sampled letters/marks, not the percentage of
records classified as Nepali. A Unicode CSV with a BOM preserves Nepali text in
spreadsheet applications.

The `Report JSON` column stores each complete report **inside the CSV row**.
There are no separate per-run JSON snapshots or persistent lock files. Copying
this CSV alone is sufficient for Streamlit to display the full saved evidence.
Leave its header and structured-data columns intact so the script can append
and the viewer can read the votes. Concurrent completions serialize CSV writes;
the file is replaced atomically after the new row is ready.

`--list` prints discovery metadata to the terminal and does not write a report.
Failed analysis does not add a completed run. Formats-only and legacy
single-sample runs are recorded without invented sampling votes.

`--output PATH.csv` chooses the one CSV destination. `--history-sheet` remains
an alias for existing commands; it never creates a second report. A JSON output
path is rejected. YAML paths are relative to that file. Without any configured
output, the default CSV is anchored to the project, so launching from another
working directory uses the same sheet. `--no-history` without an explicit output
prints results to the terminal without saving them.

In Streamlit, choose an **Inspection run** from the history table. Separate
**Language coverage** and **Script** cards show the stored final label, vote
totals, each run's label/evidence, and the reason for the decision. A known label
must satisfy `vote_min_agreement` and receive more than half of all votes
(default: at least 3 of 5); ties and abstention majorities leave the summary category blank, but the proportions and evidence are still saved. Hugging Face partition labels are shared evidence, so
unanimous votes do not independently verify that label. The view
loads saved reports without accessing datasets or repeating analysis.
To clean up `artifacts/dataset_inspection/history.csv`, select an **Inspection run**
and click **Delete selected run**. This immediately removes only that run's CSV
entry, keeping all other entries exactly as saved. Source dataset files are kept.
`Report file` can open the latest entry in a saved CSV. You can
launch directly into history with:

```bash
DATASET_INSPECTION_HISTORY=artifacts/dataset_inspection/history.csv \
  venv/bin/streamlit run apps/dataset_explorer.py
```

## Configure and run the script

Edit the single [dataset configuration](../../configs/datasets/dataset.yaml)
for each dataset, then run:

```bash
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml
```

In an interactive terminal, the script shows a separate progress bar for local
file reading, row selection, sampling/classification, and optional filtering.
Known totals show percentage, rate, and ETA; unknown totals show completed rows
and rate until counting finishes. The bars use stderr, so stdout remains valid
JSON and Streamlit still reads only the completed CSV. Piped/noninteractive runs
keep concise periodic status messages instead of terminal control characters.

Use the same settings file for all providers:

| Source | Settings to edit |
| --- | --- |
| Hugging Face | `provider: huggingface`, `dataset`, `revision`, `config`, `split`, optional `shards`; keep `dataset_file` and `local` null. |
| Kaggle | `provider: kaggle`, `dataset: owner/dataset`, `dataset_file`; set `revision`, `config`, `split`, `shards`, and `local` to null. |
| Local | Set `local` to a file/Parquet directory; set `dataset`, `dataset_file`, `revision`, `config`, `split`, and `shards` to null. |

To inspect the same split across every available Hugging Face subset, quote the
wildcard as `config: "*"` and provide an exact `split`. For Updesh's Nepali data:

```yaml
provider: huggingface
dataset: microsoft/Updesh_beta
config: "*"
split: npi_Deva
shards: null
dataset_file: null
local: null
training_schema: instruction_finetuning
field_mapping: {messages: messages}
field_parsers: {}
task_name: null
text_columns: null
row_filters: {}
```

The inspector discovers subsets at one pinned revision and runs the existing
analysis independently on each subset that contains the requested split.
Each completed subset appends its own history row, with its actual configuration
name and shared batch provenance; percentages and votes are not averaged across
subsets. Missing splits are listed as skipped. A subset that fails validation
or processing is reported as failed, other selected subsets are still attempted,
and the command exits nonzero. Existing results remain intact. Standard output
contains a batch summary and the complete successful reports. No failed subset
is represented as a successful inspection. Failure names and reasons are repeated
after the final summary so they remain visible below the full JSON output.

Add `--list` to the normal settings command to preview the selected and skipped
subsets without inspecting records or writing history. Wildcard selection
requires `shards: null` and cannot share a filtered export file; choose a single
configuration for a Devanagari export. An ordinary configuration name retains
the previous single-subset behavior.

For conversation schemas, analysis includes every message's content, including
English system prompts. A Nepali split supplies language-partition evidence;
it does not prove that all conversation text is Nepali. The configured invalid
instance policy still applies, including to empty message content.

`MBZUAI/Bactrian-X` uses a legacy Hugging Face builder script that current
`datasets` versions reject. The inspector reads the selected pinned
`data/<config>.json.gz` file directly as a JSON array; use `config: ne`,
`split: train`, and `training_schema: instruction_finetuning`. Its source
script notes that some outputs are empty, so `invalid_instance_policy: skip`
records and excludes incomplete instruction/answer pairs.

Before analyzing a new dataset, set `training_schema` in the same YAML to one of
`pretraining`, `instruction_finetuning`, `task_specific_supervised`,
`preference_tuning`, or `evaluation`. The first four are training instance
contracts; `evaluation` is a separate reference-based test example contract.
A batch contains complete instances: one document, one conversation, one
labelled example, one preference group, or one evaluation input/reference pair.
`field_mapping` maps canonical fields to source paths (for example,
`{text: payload.body, label: payload.category}`); set `task_name` when a
supervised source lacks a task column. Mapped document text normally uses the
`string` parser: the source value must be a string. To use a list of paragraph
strings as one document, configure:

```yaml
field_mapping: {text: paragraphs}
field_parsers: {text: join_strings}
```

`join_strings` accepts a string or an ordered list of strings and joins list
items with blank lines. It rejects numbers, objects, and mixed lists rather
than silently converting them. `field_parsers` is saved with the instance
definition and also works with `--field-parser text=join_strings`. The same
parser names are available in embedding settings. For TXT, `text_record_unit: line` treats
each non-empty line as a document. Use `blank_line` only when empty lines really
separate complete documents. For JSONL, CSV, Parquet, and Hugging Face, each
source row is an instance. A wrong category or missing mapped field fails the
run when that instance is analyzed; no completed report is appended.

`batch_size` controls source reading where the loader supports batches and the
number of selected complete instances sent to each analysis worker. For
Hugging Face and local/staged Kaggle Parquet, the same value controls source
batches; text/CSV/JSONL readers stream rows. `concurrency` sets the worker
count. The schema changes the content examined for language/script: the whole
document, all conversation messages, the supervised input text, all three
preference branches, or both evaluation input and reference. Labels, task names, and role markers are excluded from
language/script proportions. The report saves the chosen instance definition,
and Streamlit shows its category and boundary. If clustering the same dataset,
set the matching `training_schema` in `configs/embeddings.yaml` as well.

Set `text_columns` appropriately for document or supervised source text; null
enables automatic text-field detection. Conversation and preference instances
use all schema fields, regardless of `text_columns`. Manual `languages`
declarations are no longer used as evidence.
Instruction and conversation records must contain complete user/assistant
turns. By default, any incomplete selected instance stops the run with its
source-row index. For a source with a few known incomplete pairs, set
`invalid_instance_policy: skip` (or pass `--invalid-instance-policy skip`).
The inspector then validates the entire selected portion once, records counts
and example source rows for excluded instances, and draws exact random samples
only from the remaining valid instances. This costs an extra source pass.
The report and history sheet show valid/skipped counts; Streamlit also shows
the reasons. The policy currently applies to percentage sampling without
Devanagari export filtering. It does not synthesize missing answers.

For a reference-based evaluation source, select the held-out split and map
`input` and `reference` to its source columns. For the proofreader dataset:

```yaml
provider: huggingface
dataset: himalaya-ai/nepali-proofreader
config: default
split: test
training_schema: evaluation
field_mapping: {input: corrupted, reference: clean}
task_name: ocr_proofreading
```

Run:

```bash
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml
```

This classifies the text in both fields and records the source split; it does
**not** score OCR correction accuracy. The
dataset also has a `train` split for fine-tuning, so the evaluation role comes
from the selected `test` split and intended use, not the repository name.
Other text benchmarks can reuse this contract by changing the two mapped
paths; datasets without a text reference or with non-text inputs need an
additional adapter.

Edit this file in place; history keeps earlier runs without separate YAMLs.

Settings files accept YAML or JSON. Paths inside them are relative to the
settings file; paths passed as CLI flags are relative to your current directory.
CLI flags override settings, including replacing lists such as shards and text
columns. The project `.env` is loaded for both providers. Hugging Face uses
`HF_TOKEN` / `HF_token`; KaggleHub uses its standard credentials, including
`KAGGLE_API_TOKEN` or a configured `kaggle.json`. Credentials are not stored in
reports. See [KaggleHub authentication](https://github.com/Kaggle/kagglehub#authenticate).

Discover the actual configuration, split, and shard names before configuring a run:

```bash
python scripts/inspect_dataset.py --dataset owner/dataset --list
python scripts/inspect_dataset.py --dataset owner/dataset --config ne --list
```

Without `--config` or `--split`, the script chooses automatically only when
there is exactly one option. `--revision` accepts a branch, tag, or commit;
the report and all reads use the resolved commit. Repeat `--shard` to combine
any subset of physical files. A null `shards` setting selects all shards;
an empty list is rejected.

You can also run entirely with CLI inputs:

```bash
# Inspect a selected split or add --shard NAME from --list.
python scripts/inspect_dataset.py --dataset owner/dataset \
  --config ne --split train --sample-fraction 0.2 --sampling-runs 5 \
  --output artifacts/dataset_inspection/history.csv

# Inspect local Parquet, JSON, JSONL, CSV, TXT, or XLSX.
# A local directory means all Parquet files below it.
python scripts/inspect_dataset.py --local data/example.jsonl \
  --text-column text \
  --output artifacts/dataset_inspection/history.csv
```

Repeat `--text-column` for instruction/answer fields or select a conversation
column. Dotted fields, such as `payload.messages`, are supported. The report records
the effective settings, selected files/revision, timestamp, and evidence.

## Select a language subset before sampling

A multilingual source is valid. **Choose the portion of interest before any
sampling, script filtering, or language/script threshold calculation.** The
classification thresholds do not select that portion automatically. The processing order is:
**provider → configuration/split/shards or file → exact row filters → random samples → language/script votes**.
All percentage denominators and votes describe that selected portion. For example,
if a corpus has 1,000,000 rows but the chosen configuration/split/shards and row
filters retain 10,000 rows, a 20% run samples 2,000 of those 10,000 rows. The
remaining 990,000 rows contribute no language/script evidence and cannot enter
the Devanagari export. When row selection requires scanning a multilingual shard,
those excluded rows may still be read to check their labels, but they are not
sampled or classified. The optional whole-record Devanagari export scans the
selected portion independently of sampling; it never operates on excluded rows.

A language-specific Hugging Face configuration such as `ne`, `nep_Deva`, or
`en-ne` already selects a subset; use its exact configuration name from `--list`.
For a multilingual configuration, filter its language metadata columns instead.
Do not put a row label in `config` unless that configuration actually exists.

The shared YAML currently selects Aya's Nepali rows using the same `npi`
language code as the built-in Aya explorer:

```yaml
provider: huggingface
dataset: CohereLabs/aya_dataset
config: default
split: train
row_filters:
  language_code: [npi]
sampling_method: random
sample_fraction: 0.2
sampling_runs: 5
```

Run it with:

```bash
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml
```

Other selection shapes (choose the one matching the dataset's actual schema):

```yaml
# Accept either exact label from a language field:
row_filters: {language: [ne, nep-Deva]}

# Select records labelled as an English–Nepali pair:
row_filters: {language_pair: [en-ne]}

# Require both columns to match:
row_filters: {source_language: [en], target_language: [ne]}

# Disable row filtering when the configuration/file already selects the language:
row_filters: {}
```

Matching is exact and case-sensitive: `ne`, `npi`, and `nep-Deva` are different
source labels, even though language analysis can normalize them to Nepali.
There is **OR within a column and AND between columns**. Nested paths such as
`metadata.language_code` work; list-valued labels match if any allowed value is
present. Entire matching records are retained, including both sides of a
translation pair. This does not extract Nepali sentences from multilingual text.
For a pair, choose both text fields (for example `text_columns: [source, target]`)
if both sides should contribute to script analysis. Automatic text selection
excludes the recognized language metadata fields and explicit row-filter columns.

CLI filters replace all YAML row filters for that run:

```bash
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml \
  --row-filter language_code=npi

# Example for a dataset with two language columns:
venv/bin/python scripts/inspect_dataset.py --provider huggingface \
  --dataset OWNER/DATASET --config default --split train \
  --row-filter source_language=en --row-filter target_language=ne

# Inspect formats only, without applying the YAML's row filters:
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml \
  --formats-only --no-row-filters
```

Repeat `--row-filter language=ne --row-filter language=nep-Deva` to allow both
labels in one column. These same filters work for Kaggle and local structured
files. A plain TXT corpus has no language metadata column; choose a
language-specific file/configuration instead. A file name is not language evidence.
`languages` / `--language` is deprecated: accepted for old command compatibility,
but ignored with a warning. The analyzer recognizes
`language`, `language_code`, `lang`, `lang_code`, `language_pair`,
`source_language`, `target_language`, `source_lang`, and `target_lang` as language
evidence columns. Other metadata filters still select rows but do not themselves
supply language evidence.
Selecting a label is metadata-based selection, not validation by a language model.

The script first counts matching records, then samples only those rows. For
100,000 source rows with 10,000 matching Nepali rows, each 20% run analyzes
2,000 matching rows. Five runs can overlap; expected unique coverage is about
67.2% of the matching subset, not guaranteed full coverage. Filtering requires
reading the selected source, and the sampling pass traverses it again; a small
sample fraction reduces text analysis, not necessarily source I/O. A missing
filter column fails early; no matches fails with example labels seen, and neither
failure appends a result. Optional `max_records` limits the subsequent Devanagari
scan to the first matching rows, not the first rows of the multilingual source.

The same CSV now records **Row filters**, **Rows before language selection**, and
**Rows after language selection**. Older CSVs remain readable and gain these
columns on the next successful append while preserving their reports. Streamlit's
**Language selection before sampling** card displays these saved counts and
rules; the language/script vote cards describe the matching subset. Source file
bytes still describe the source files, not the storage size of filtered rows.
Existing historical runs are unchanged; run the script again to obtain filtered
results. `cluster_dataset.py run --source-settings configs/datasets/dataset.yaml`
also honors these row filters when reading records for embedding.

## Classification thresholds in the single dataset YAML

Edit these values in `configs/datasets/dataset.yaml`. Ratios are fractions, so
`0.05` means 5%. They classify each sample; they do not change the source row
filters or the optional Devanagari export threshold.

```yaml
language_min_ratio: 0.05
language_dominance_ratio: 0.95
language_min_labelled_ratio: 0.80
script_dominance_ratio: 0.95
script_max_other_ratio: 0.05
vote_min_agreement: 0.60
```

**Language Coverage uses record ratios, never character ratios.** First require
language-column labels or accepted text-model predictions on at least
`language_min_labelled_ratio` of sampled records. For each language, divide the number of labelled records mentioning it
by the number of records with a metadata label or accepted prediction. Count a language at most once
per record, even if several columns repeat it. A bilingual record contributes
once to both languages, so the ratios need not sum to 100%.

Include languages with a ratio **at least** `language_min_ratio`. If exactly one
language remains, it must also reach `language_dominance_ratio`; otherwise the
run has no supported category. With defaults, 96% Nepali-labelled records and 4%
English-labelled records yield Nepali-only; 95%/5% yields Bilingual because English
meets the inclusive 5% cutoff. Multiple individually small minorities cannot
produce a single-language label below the 95% dominance threshold. If metadata labels plus accepted predictions
cover less than 80% of the sample, that run abstains; the report still records every sampled record and its category proportions.

These thresholds describe the **labelled sample**. Nepali-only now means the
thresholds were met, not that every record or word was independently verified as
Nepali. If there are no column language labels at all, the selected HF language
configuration/split remains categorical evidence: do not invent per-record
percentages for that source. Manual declarations and script measurements are
still excluded as language evidence.

**Script uses character ratios within eligible Nepali/English records.** Compute
`Other / (Devanagari + Latin + Other)` first. Values **above**
`script_max_other_ratio` need review and cannot be published as a Script category. At or below the tolerance, ignore those
other-script characters for the dominance calculation:

- `Devanagari / (Devanagari + Latin) >= script_dominance_ratio`: Devanagari.
- Otherwise, if English also meets language thresholds: Mixed (Nepali + English).
- Otherwise, `Latin / (Devanagari + Latin) >= script_dominance_ratio`: Romanized.
- Otherwise: Mixed (Devanagari + romanized).

An empty denominator or missing eligible text gives no supported script category; the report still records the evidence gap. These are Unicode letters/marks; digits, spaces and punctuation are
excluded. For example, 97% Devanagari, 2% Latin and 1% Other passes the 5% noise
allowance; Devanagari is then `97 / 99 = 97.98%` of supported characters and meets
the 95% dominance threshold. A lone unexpected character no longer invalidates a
large sample. A 60%/39%/1% sample yields mixed, assuming Nepali-only coverage.

**Voting is a separate threshold.** Every run gets one vote after applying the
language and script rules above. A positive decision requires at least
`max(floor(runs / 2) + 1, ceil(vote_min_agreement * runs))` votes. Thus 0.60 requires
3 of 5, while 0.80 requires 4 of 5. Abstaining runs remain in the denominator and
cannot themselves establish a supported script. Nepali presence uses the same vote
threshold before a final script classification is accepted.

Defaults are editable analysis choices, not calibrated accuracy guarantees.
Dominance thresholds must exceed 0.5 and be at most 1; `language_min_ratio` must
be greater than 0 and at most 1. Label completeness and other-script tolerance
allow 0..1; vote agreement allows 0.5..1 while always requiring a strict majority.
Invalid values fail before source loading. CLI flags override YAML values:

```bash
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml \
  --script-max-other-ratio 0.02 --script-dominance-ratio 0.98 \
  --language-min-ratio 0.03 --vote-min-agreement 0.80
```

The single CSV stores all effective thresholds, labelled counts, language ratios,
script ratios and per-run reasons inside its report/vote columns. Streamlit's
**Classification thresholds and measured ratios** expander displays these saved
values. Existing reports keep their original policy; rerun inspection to append
new threshold-based results. The same rules apply to legacy fixed-size samples
and the analysis of an optional filtered output.

## Repeated random sampling and voting

The default is **five independent random samples of 20% of the selected
population**, for Hugging Face, Kaggle, and local sources. Configure it in YAML:

```yaml
sampling_method: random
sample_fraction: 0.2
sampling_runs: 5
concurrency: 6
batch_size: 1024
seed: 42
```

Or override settings for a periodic run:

```bash
python scripts/inspect_dataset.py \
  --settings configs/datasets/dataset.yaml \
  --sample-fraction 0.2 --sampling-runs 5 --concurrency 6 --batch-size 1024 --seed 42
```

Each run selects exactly `ceil(sample_fraction * population_rows)` record
positions uniformly, without replacement within that run. Independent runs may
select the same records. Run seeds are derived from the base seed and run number;
the same seed, source ordering, and settings reproduce the samples. Changing
`--seed` produces new selections.

All runs share one streaming traversal when the population count is known.
Each selector draws with probability `remaining_sample_rows / remaining_rows`,
so every record position is eligible, including records in later shards. If row
counts are unavailable, the script counts the selected records first, then makes
the sampling pass. Local metadata inspection can also scan the source before
sampling. **The source is traversed completely, but language/script analysis is
performed only on selected records.** Remote streaming can therefore read the
entire selected source even at 20%. A count mismatch aborts the report instead
of publishing incorrect coverage.

Every character in the chosen text fields of every selected record is analyzed;
percentage mode has no character cap. Overlapping records are analyzed once and
their evidence is added to each run that selected them. The sampler retains
aggregate counters, not 20% of the raw dataset in memory. With
`concurrency: 6`, selected records are classified in six worker processes in
batches of `batch_size: 1024`; at most 12 batches are queued at once. For a
batch with no language-column or Hugging Face partition labels, fastText predicts
the eligible record texts together in one model call, while retaining one result
per record. Batches containing metadata labels use the per-record fallback.
The same `batch_size` applies to Hugging Face's native streaming row batches
and PyArrow row batches for local or Kaggle-staged Parquet files. Kaggle first
stages the selected file; CSV, JSONL, TXT, and XLSX readers still yield records
sequentially before selected records are grouped for worker analysis.
The parent still reads the source and makes all random selections in source
order, so changing concurrency or batch size does not change sampled rows,
votes, or percentages. Local TXT/CSV/JSONL metadata counting is also serial.
These serial stages and process-transfer overhead can limit speedup from more
workers. Each worker loads its own fastText model if text detection is needed,
increasing RAM use.
`concurrency: 1` runs in the original single-process mode. These five
overlapping random runs are not disjoint K-fold cross-validation. Existing
local JSON array parsing still loads the document into memory.

Language coverage and script each receive one vote per run. A label must win
**more than half of all runs** and `vote_min_agreement` (default 0.60, or 3/5); a tie, a mere plurality, or an
abstention majority leaves the summary category blank. The command still appends
a report. Abstaining runs remain in the vote denominator. Every report saves each
run's evidence, vote counts, agreement, and category proportions across unique
selected positions. Pooled evidence stays
visible because a majority result can hide a minority language or script.

Historical research results retain their saved categories, percentages, counts,
thresholds, and votes. The viewer does not backfill the new `Other` category into
earlier runs: their outside-category evidence stays under its original label.
Appending a run preserves existing CSV cell values and embedded reports; newly
introduced CSV columns remain blank for earlier rows. To apply current analysis
rules to an earlier dataset, run a new inspection, which receives a separate run ID.
Adding the remainder categories changes reporting only: with the same source,
settings, seed, and detector, existing percentages, counters, decisions, and votes
stay the same. `Other` and `Unknown` are percentage buckets, not new summary or
voting labels. The existing outside-category and no-evidence counters are retained
as diagnostics; they describe the same records as these two buckets and must not
be added a second time.

The CSV includes **Language category %** (each category's share of all unique
sampled records) and **Nepali script category %** (each category's share of
Nepali-eligible unique sampled records). Missing language evidence and unsupported
single-language labels remain in the language denominator and are reported as
`Other` (outside the supported categories) and `Unknown` (no accepted evidence).
Missing or unsupported script evidence receives the same two percentage buckets in the
Nepali-eligible denominator. Records without Nepali evidence are outside the
conditional script denominator. These are record percentages; the existing
Devanagari % diagnostic is a character percentage. Overlap between runs is counted
once in the pooled proportions. Counts sum exactly to each denominator, so the
unrounded shares sum to 100%. The display includes a Total row and explains any
two-decimal rounding difference without changing the existing percentages. An
empty denominator has no defined percentage total. Older reports expose their
saved gap counters under the original labels without rewriting the report.
A **Multilingual** record proportion counts
records carrying multiple languages; a multilingual dataset can consist entirely
of single-language records. The current fastText fallback predicts one
dominant language per record, so bilingual/multilingual record shares are only
identified when source labels name multiple languages; a zero share does not
rule out code-switching in the text. Language vote thresholds affect only the
optional summary label, not which sampled records enter the proportions.
Per-record script categories use the configured script dominance and other-script
tolerance; records that do not meet them remain in the script denominator as an
evidence gap.

Coverage counts distinct source record positions, not summed sample sizes or
distinct text values. Expected unique coverage is `1 - (1 - k/N)^n`, where `k`
is the rounded sample size, `N` is the population, and `n` is the run count.

### Inspecting actual classified examples in Streamlit

New percentage-sampling runs save up to 10 random examples per language and
script category in `language_status.record_examples`. Streamlit's **Inspect
actual data examples** section lets you choose Language or Script, a category,
and 5 or 10 examples. Categories with fewer records show all available examples.
The definitions expander explains each category using that run's saved thresholds.

Examples come from the exact unique records analyzed, using their already
computed evidence. Selection uses a separate deterministic hash priority based
on the inspection seed and population position. It consumes no sampling random
numbers, makes no additional detector calls, and produces the same examples
regardless of worker count or batch order. Viewing examples never reads the
source again or changes the saved report, percentages, or votes.

Each example shows the analyzed text, language and script categories, evidence
origin, accepted language codes, detector rejection reason/score when available,
and raw script character counts and ratios. Population positions are one-based
within the selected sampling population after row filters and invalid-instance
selection; they are not necessarily spreadsheet row numbers. Text previews are
limited to 8,000 characters and marked when truncated; script counts still cover
the full analyzed text. Language detector truncation is identified separately.

`Unknown` language examples lack accepted language evidence, such as inputs
below the minimum letter count or confidence cutoff. Their raw script counts
can still be 100% Devanagari. Without Nepali evidence, their conditional Script
category is `Not applicable`, and they remain outside the Nepali-script
percentage denominator. The UI displays this as **Excluded: no accepted Nepali
evidence**, and shows **Observed writing system: Devanagari (100.00%)** separately
when that is what the saved character counts show. Exclusion from the
Nepali-specific classification is not a judgment of text quality or a failure
to recognize its writing system. The saved labels and numerical results do not
change. `Unknown` within that denominator instead means Nepali
evidence exists but the record contains no eligible letters/marks.

Earlier reports without saved text keep their original results and display a
notice to run a new inspection to capture examples. Legacy `sample_size` mode
and metadata-only runs do not save this example collection.
Five 20% samples cover approximately **67.2%**, not 100%, on average. The report
and Streamlit show the **actual measured coverage** as well as this expectation.
Use `sample_fraction: 1` for complete coverage; repeating full samples adds no
new evidence.

The population is the selected configuration, split, and shards/file. During a
limited filtering run, source sampling uses only the scanned prefix and is
labelled accordingly. Retained output gets separate samples drawn from all
retained records, with its own population and coverage.

Existing settings with `sample_size: 100`, or explicit `--sample-size 100`, keep
the legacy single-sample behavior and its 100,000-character cap. In a settings
file, remove `sample_size` when enabling `sample_fraction`; they are mutually
exclusive. An explicit CLI `--sample-fraction` clears a legacy settings-file
sample size. Conversely, `--sample-size` switches back to one legacy run.

Voting improves sampling consistency, but does not introduce a language detector:
a shared Hugging Face language partition label can make every run agree without
independent language identification.
Agreement is not a probability that the label is correct. See the evidence rules
below. Selection is separated from evidence aggregation and voting in
`attention_maps/explorer/sampling_vote.py`; `random` is currently the only
implemented strategy. Embedding-based sampling is not implemented yet.

## Kaggle language, script, and file-format inspection

Use the same script with an explicit provider:

```bash
# List every file and its size/extension without downloading the data.
python scripts/inspect_dataset.py --provider kaggle \
  --dataset hsebarp/oscar-corpus-nepali --list

# Download/cache only the selected file, then inspect language and script.
python scripts/inspect_dataset.py --provider kaggle \
  --dataset hsebarp/oscar-corpus-nepali --dataset-file ne_dedup.txt \
  --text-column text --sample-fraction 0.2 --sampling-runs 5 \
  --output artifacts/dataset_inspection/history.csv

# Or set provider: kaggle and its source fields in the shared dataset YAML.
python scripts/inspect_dataset.py \
  --settings configs/datasets/dataset.yaml

# Formats and sizes only: no data-file download.
python scripts/inspect_dataset.py --provider kaggle \
  --dataset hsebarp/oscar-corpus-nepali --formats-only \
  --output artifacts/dataset_inspection/history.csv
```

`dataset` is the Kaggle `owner/dataset` handle. Add `/versions/N` to pin a
specific version; otherwise discovery resolves the current version and uses it
for all file metadata and downloads. `dataset_file` is an exact internal path
from `--list`. There is no Kaggle configuration/split selector: choose the file
that contains the required partition instead. `config`, `split`, `revision`,
and `shards` apply to Hugging Face. An explicit CLI provider switch clears
inapplicable settings-file defaults; explicit incompatible CLI flags fail.

When exactly one supported file exists, it can be selected automatically.
Otherwise normal inspection requires `--dataset-file` and never picks the first
file silently. `--formats-only` without a file catalogs all remote files. `--list`
and `--formats-only` fetch metadata only; normal language/script inspection
caches the selected file first. Large CSV/JSONL/TXT files are scanned for metadata
and repeated random sampling even with a small sample fraction.

Parquet, JSON, JSONL/NDJSON, CSV, TXT, and XLSX use the shared local inspection
readers. XLSX inspection reads the first worksheet; choose its text column with
`--text-column`. Archives and other unsupported formats can be cataloged with
`--formats-only`, then staged/converted separately. Filtering uses the same
supported formats and thresholds as local inspection, including the first
worksheet of XLSX files.

Both providers produce the same saved-report schema. Kaggle reports additionally
record the provider, resolved version, selected file names, cached file paths,
and source signatures. The Streamlit **Script results** view reads those saved
language/script/format values without contacting either provider. For Kaggle,
use `DATASET_INSPECTION_REPORT=artifacts/dataset_inspection/history.csv`.
Kaggle language evidence comes from actual dataset-column labels, or the enabled
text-language fallback when labels are absent. The shared YAML enables fastText
for plain TXT as well as structured files. A dataset title, filename, manual
declaration, or Devanagari script alone does not establish Nepali. Uncertain
predictions remain unlabelled and can prevent the inspector from publishing a language or script category.

## File extensions and batch parsing

Every new report includes `inventory.file_formats`, describing the **selected
physical files**. This is separate from `inventory.format`, which selects the
source loader (for example, `huggingface`). Each file records its extension,
normalized record format, compression wrappers, archive container, size,
matching local batch reader, and any preparation required.

The summary includes format/extension groups with file counts and bytes, flags
for mixed formats and mixed extensions, and a per-file list. `.ndjson` and
`.jsonl` use the same JSON Lines parser but retain their distinct extensions.
`.jsonl.gz` records JSON Lines plus gzip; archive contents remain unknown until
inspected. Signed URL query parameters do not become file extensions. Filtered
outputs also have their own format metadata (currently uncompressed JSONL).

Streamlit displays the saved **Source file formats**, **Formats and batch parsing
requirements**, and **Per-file format details**. Hugging Face discovery with
`--config NAME --list` includes format summaries for each split. Changing the
selected shards changes the reported formats; the first shard does not stand
in for the whole selection.

To catalog files without parsing or sampling their records:

```bash
python scripts/inspect_dataset.py --dataset owner/dataset \
  --config ne --split train --formats-only --output artifacts/dataset_inspection/history.csv

python scripts/inspect_dataset.py --local data/my_dataset \
  --formats-only --output artifacts/dataset_inspection/history.csv
```

You can also set `formats_only: true` in YAML. For a local directory, this mode
lists **all files recursively**, including mixed/unsupported extensions and
non-data files. It excludes its own report and the supplied settings file.
Normal local directory inspection still selects only Parquet files. Format-only
inspection does not open/decompress archives or count records; row counts are
unknown unless already available from Hub split metadata. It cannot be combined
with filtering settings. Reports use the same Streamlit saved-report viewer.

Current **local/staged-file batch loader** behavior:

| File type | Parsing requirement |
| --- | --- |
| `.parquet` | Read Parquet record batches. |
| `.jsonl`, `.ndjson` | Decode each non-empty line as a JSON value; collect batches. |
| `.json` | Read the JSON document and select its record array/object; currently loads the document into memory. |
| `.csv` | Read comma-delimited records with headers. |
| `.txt` | Read the whole file as **one record**; the inspector instead treats non-empty lines as records. |
| `.jsonl.gz`, other outer compression | Decompress first; compressed extensions are not directly supported by this local batch reader. |
| `.zip` | Inspect/extract members and use their supported record formats; a ZIP extension does not identify member formats. |
| `.tar.gz`, other archives | Unpack and inspect members before choosing a record reader. |
| `.tsv`, `.xlsx`, `.arrow`, other recognized formats | A suitable reader/conversion is required by the current local batch loader. |
| Missing/unrecognized extension | Identify content before selecting a parser. |

Hugging Face streaming uses its configured `datasets` loader; the local batch
support labels do not limit what that provider can stream. Extension detection
is **filename evidence**, not content/schema validation or automatic conversion.
Internal Parquet compression cannot be inferred from its extension. Check the
schema and JSON record structure before using a format group for batch loading.

## Optional Devanagari filtering

Enable `min_devanagari_ratio` and `filtered_output` in your settings file, or run:

```bash
python scripts/inspect_dataset.py --local data/example.jsonl \
  --text-column text --min-devanagari-ratio 0.8 \
  --filtered-output artifacts/dataset_inspection/devanagari.jsonl \
  --output artifacts/dataset_inspection/history.csv
```

This keeps **whole records**, preserving their text and metadata. It does not
remove Latin characters from accepted text. The shared
`attention_maps.eda.text.devanagari_ratio` function measures the fraction of
Unicode letters/marks in the Devanagari block across the chosen fields. A record
qualifies when its ratio is at least the threshold and it contains Devanagari
letters/marks. Empty or number-only text is dropped, including at threshold zero.
This selects a script; Devanagari is also used by languages other than Nepali.

The filter streams all records from the selected configuration, split, and
shards unless `--max-records N` is provided. A limited run is explicitly marked
`limited_prefix`; its counts never represent the entire source. Sampling settings
only control language/script evidence, **not** which records the filter checks.
Repeated percentage sampling analyzes the scanned source and retained output as
separate populations. The optional legacy `sample_size` mode uses one reservoir
sample from each population (0–1,000 records).

Filtering supports Hugging Face, Parquet, JSON, JSONL, CSV, TXT, and XLSX. Parquet is
read in batches; line-oriented formats stream. Local JSON arrays are loaded
in memory, matching the existing inspector. Local metadata inspection may scan
the file before filtering even when `max_records` is set. Output is UTF-8
JSONL; non-JSON scalar types such as dates are represented as strings.

The report contains separate source and filtered-output language/script evidence,
records scanned/kept/dropped, filter threshold, scan scope, and output path/size.
Input files are protected from output-path collisions. An existing filtered file
requires `--overwrite` or a new destination; interrupted reads do not publish
partial files. The CSV is replaced atomically so Streamlit sees complete rows.
Periodic runs append to the same standard CSV.

## Display the saved output in Streamlit

```bash
DATASET_INSPECTION_REPORT=artifacts/dataset_inspection/history.csv \
  python scripts/run_dataset_explorer.py
```

Alternatively, start the explorer normally, select **View → Script results**,
and select the saved inspection run. **Reload report** reads the CSV again.
The view shows the exact saved categories and counts without category overrides,
source downloads, filtering, or resampling. Configure the next run in your
settings file and rerun the script. A `--list` response is discovery metadata,
not an inspection report.

**Dataset explorer** remains available for interactive exploration. Select
**Hugging Face**, enter its ID/revision, and click **Load Hugging Face source**,
then choose **Dataset configuration**, **Dataset split**, and **Dataset shards**.
Choose any combination, including only a later shard. Source samples, workspace
sampling, overlap, and EDA use those same files. Workspace identities and
manifests include the revision and selected shards.

After loading a script report, its source language/script cards also appear in
the explorer when the dataset, configuration, split, revision, and shards match.
Local reports must match source file paths, sizes, and modification times.
Changing the selection does not carry over evidence from another source.

## Size semantics

File sizes describe the physical files in the selected configuration/split.
Declared split metadata supplies full-split row counts and decoded-size
estimates. For full non-Parquet splits without those counts, Dataset Viewer
metadata is accepted only when its `X-Revision` matches the selected commit
and the result is complete. A subset of Parquet shards uses range-addressed
Parquet footers for its own counts; uncompressed sizes are estimates of memory
footprint. Whole Parquet tables are not downloaded to calculate these sizes.

For non-Parquet shard subsets without row metadata, inventory row counts and
decoded size are **Unknown**. Repeated percentage sampling performs a counting
pass and stores the resulting population in its sampling evidence. A completed
filtering scan separately records the number of records actually scanned. Percentage-based WORKSPACE sampling requires a known
population. Files hosted outside the repository may have unknown physical sizes.
Large splits without declared row counts can require one footer request per file.

The implementation uses Hugging Face's [configuration and data-file loading
APIs](https://huggingface.co/docs/datasets/package_reference/loading_methods) and
[filesystem interface](https://huggingface.co/docs/huggingface_hub/package_reference/hf_file_system).

## Text-language fallback for unlabelled datasets

The shared `configs/datasets/dataset.yaml` enables the local fastText detector.
It works with **Kaggle, Hugging Face, and local files**, after file/partition/row
selection and random sample selection. It classifies sampled records; it does
not automatically filter an unlabelled multilingual corpus down to Nepali.

```yaml
language_detection: fasttext
language_detection_model: ../../artifacts/language_models/lid.176.bin
language_detection_min_confidence: 0.60
language_detection_min_letters: 5
language_detection_max_characters: 4000
```

Run from the repository root:

```bash
venv/bin/python -m pip install 'fasttext>=0.9.3,<0.10'
venv/bin/python scripts/inspect_dataset.py --settings configs/datasets/dataset.yaml
```

The first applicable prediction downloads the official full model to the
configured cache; later runs reuse it. No dataset text is sent to a language
service. Set the model path to an existing local fastText `.bin`/`.ftz` model for
offline use. Recognized missing `lid.176.ftz` and `lid.176.bin` paths download from
the official fastText server; other missing model paths fail explicitly.
Metadata-only runs and records covered by metadata do not load the model.

[fastText's official language-identification models](https://fasttext.cc/docs/en/language-identification.html)
support 176 languages, including Nepali (`ne`), Hindi (`hi`), and English (`en`).
The configured `.bin` model is about 126 MB. The compressed `.ftz` model is about
917 KB. On the selected Kaggle corpus at a 0.80 score cutoff, the compressed
model accepted 31% of records while the full model accepted 99.9% of 10,004
records spaced across that corpus. For the current local short-line `ne.txt`,
20,040 spaced records gave 92.4% accepted predictions with the YAML's 0.60 score
cutoff and five-letter minimum; the previous full local run accepted only 64.1%
with a 0.80 cutoff and 20-letter minimum. These comparisons measure model
acceptance, not manually verified language accuracy. The model is licensed
CC-BY-SA 3.0.

**The prediction-score cutoff is separate from the existing thresholds.** For
each unlabelled sampled record, normalize whitespace and predict from at most
the first 4,000 characters of the selected text fields. The current YAML leaves
a record unlabelled if it has fewer than five alphabetic characters or a model
score below 0.60. Bare CLI defaults remain 20 letters and 0.80; set these
per dataset after reviewing representative records. The score is a model output,
not a calibrated probability of correctness.

Then apply the existing `language_min_labelled_ratio`, `language_min_ratio`,
`language_dominance_ratio`, script thresholds, and `vote_min_agreement` unchanged.
For example, 80 accepted predictions out of 100 otherwise unlabelled records
meet the 80% evidence gate; language ratios use those 80 records. With 79 accepted
predictions the run abstains; its evidence and category proportions are still saved. Rejected predictions are not counted as
English, Nepali, or an extra language. Nepali must still be covered before a
script category is assigned. Repeated sampling still analyzes all characters
in the eligible sampled text for script, regardless of the model input cap.
Legacy `--sample-size` mode retains its existing 100,000-character script budget.

Repeated sampling predicts once per **unique sampled position** and reuses the
result in each run containing it. Each CSV run saves model identity/hash/version,
score cutoff, accepted/rejected counts, rejection reasons, truncated-input counts,
predicted-language counts and every vote. Streamlit displays those saved values
under the language card and the **Text language detector and rejected predictions**
expander without loading a model or recomputing analysis. Existing CSV entries
remain unchanged; rerun the script to append a new result.

This is a dominant-language baseline, not span-level language detection. It can
miss short English spans inside Nepali records or confuse romanized Nepali with
other Latin-script languages. Even valid Nepali text can fall below the score
cutoff. Validate model performance on representative records before changing
that cutoff; repeated voting cannot remove systematic detector errors.

Use `--language-detection none` to retain metadata-only behavior. Bare CLI calls
without the shared YAML default to `none`; enable it with
`--language-detection fasttext`. All five detector settings also accept CLI
flags with hyphenated names.

## Language/script evidence

Language evidence uses metadata first, with an optional text-model fallback:

1. **Dataset columns:** language labels read from the sampled records. The report
   saves the exact column names and raw label counts (for example
   `language_code: {npi: 4}` in one voting run).
2. **The selected Hugging Face language configuration/split:** the exact selected
   value and its interpreted languages are saved, such as `ne`, `nep-Deva`,
   `eng_Latn-npi_Deva`, `en-ne`, or `train_ne`. Generic `default`, `train`, and
   `test` names do not supply language evidence. Recognition currently covers
   Nepali/English aliases and their Devanagari/Latin tags; unrecognized partition
   names supply no inferred language.
3. **Text language detector:** when a record has no column language label and
   there is no recognized HF language partition, an enabled local fastText model
   predicts its dominant language. Accepted predictions have their own origin;
   they are never written into the source data or reported as column metadata.

Sampled column labels take priority. The Hugging Face selection is the fallback
when no language labels are present in the sample. If the sources have different
language sets, both are saved, a discrepancy is flagged, and the column evidence
is used. Without metadata, an enabled text detector supplies predictions; if it
is disabled or insufficient predictions pass its score cutoff, the run cannot supply a
supported category. Repository/card language tags, manual `languages` declarations,
dataset names, filenames, filter settings without observed labels, and script
percentages are never language evidence. The positive Language Coverage categories are
**Nepali-only**, **Bilingual (Nepali-English)**, **Multilingual**, and **English-only**.
When no category wins, the CSV summary cell is blank; the report and measured
proportions are still appended.

Each run and the pooled sample save `language_evidence_sources`,
`language_evidence_conflict`, and the evidence policy. Streamlit's **Language evidence
origins** expander displays the stored origins; discrepant sources produce a warning.
Older CSV runs remain unchanged and are labelled historical in the UI. Rerun the
script to append a result under the current policy. All evidence remains inside
the one CSV, including each voting run's provenance.

**Script is conditional on Nepali being covered**, rather than a general script label
for any dataset. Each sample first determines whether its language evidence includes
`ne`, including when the overall category is Multilingual.

| Evidence and eligible sampled text | Script |
| --- | --- |
| Known language set excludes Nepali | No Script category applies |
| Language evidence is missing | No summary category; proportions remain recorded |
| Nepali covered; Devanagari reaches dominance threshold within noise tolerance | Devanagari |
| Nepali covered; Latin reaches dominance threshold within noise tolerance, no English coverage | Romanized |
| Nepali covered; neither script dominates, within noise tolerance, no English coverage | Mixed (Devanagari + romanized) |
| Nepali and English covered; Devanagari does not dominate, within noise tolerance | Mixed (Nepali + English) |
| Other-script ratio exceeds tolerance | No summary category; proportions remain recorded |
| Nepali covered but no eligible letters/marks | No summary category; proportions remain recorded |

For **Nepali-only**, the only positive script classifications are **Devanagari**,
**Romanized**, and **Mixed (Devanagari + romanized)**. Unsupported writing systems
above the configured tolerance or missing letters produce an abstention, not a fourth
script class. These abstentions remain in the voting denominator; if the final
vote is inconclusive, the category summary cell is blank and the report still records the proportions. `Other`
may still appear in raw Unicode character counts as a diagnostic. Historical
reports retain their original votes and Report JSON; the CSV category summary
cells are blank and Streamlit displays an em dash instead of treating an old
inconclusive value as a category.

Eligible text comes from Nepali-labelled records. Unlabelled records can use an
explicit Nepali Hugging Face configuration/split. English-labelled records also
contribute when both Nepali and English are covered. Other-language records do
not contribute: Hindi Devanagari cannot turn a romanized Nepali sample into a
mixed-script sample. A record labelled with several languages is analyzed as a
whole across its selected text fields; the script does not identify language
spans within that record.

`nepali_covered`, `script_reason`, `script_policy`, `script_scope`, and
`script_analysis_counts`/`script_analysis_percentages` store this conditional
analysis for every vote and the pooled sample. The raw `script_counts` and
`script_percentages` still describe all sampled text for diagnostics. Accordingly,
the existing CSV **Devanagari %** remains a raw source-character percentage.
Streamlit separates these diagnostics from the **Nepali script analysis scope**
and shows the reason alongside the Script card.

Every run applies the Nepali condition before script voting. Nepali presence also
requires the configured agreement and a strict majority across runs; a pooled sample containing one Nepali row
does not establish majority coverage. If Nepali is absent in most runs, Script is
without a Script category; if its presence is inconclusive, the summary cell is blank but the report is saved.
The language-category vote and conditional script vote are preserved separately.
Old reports are identified as historical; rerun the script to apply this rule.

Devanagari dominance describes **eligible sampled letters/marks** under the
configured tolerance, not absolute purity or all records in the dataset. Numbers and punctuation do not determine the script. Romanization and
Nepali–English mixed labels remain heuristics based on language evidence and Unicode;
Latin names or URLs can trigger a mixed label when they exceed the dominance allowance, and individual spans are not
verified by a language-identification model. The optional whole-record Devanagari
export still uses its configured character-ratio threshold; use language row
filters first when that output should contain Nepali records only.

The optional legacy fixed-size mode uses reservoir sampling for local
CSV/JSONL/text and a bounded streaming shuffle for Hub sources when filtering
is disabled. Repeated percentage sampling instead considers the entire selected
population with uniform inclusion probability in each run.
