# Dataset sources used in this research workspace

The definitions in [`configs/datasets/sources.yaml`](../configs/datasets/sources.yaml)
come from the earlier dataset discussion, saved inspection reports in
`artifacts/dataset_inspection/history.csv`, and the saved Kaggle OSCAR embedding
run. There are **40 research source selections plus one demonstration source**:
33 Hugging Face selections, 3 Kaggle files, and 5 local entries including the demo.
A selection is one specific file, subset, split, or translation direction; these
counts do not represent 40 unrelated datasets.

Each source ID records the provider, dataset identifier or local path, instance
schema, field mappings, and applicable filters. Hugging Face revisions are pinned
to the saved inspections. Those inspections selected every shard of the stated
split, so the registry leaves `shards` unset. Kaggle handles retain version 1.
Local paths identify existing files, not immutable snapshots; moving or replacing
one requires checking its provenance before another research run.

## Hugging Face

The selection column is `configuration / split`. Exact revision hashes and field
names are in the YAML. `pretraining` means one text document per instance;
`instruction_finetuning` retains the conversation; `task_specific_supervised`
retains both input and label; `evaluation` retains input and reference.

| Source ID | Dataset | Selection | Instance definition |
| --- | --- | --- | --- |
| `gorkhapatra-nepali` | `Aananda-giri/gorkhapatra-nepali-epaper` | `default / train` | Pretraining: `text` |
| `sangraha-verified-nep` | `ai4bharat/sangraha` | `verified / nep` | Pretraining: `text`; ID from `doc_id` |
| `iriis-nepali-text` | `IRIIS-RESEARCH/Nepali-Text-Corpus` | `default / train` | Pretraining: `Article`; ID from `index` |
| `himalaya-nepali-pdf` | `himalaya-ai/nepali_pdf_corpus` | `default / train` | Pretraining: `text`, `id` |
| `himalaya-nepali-news` | `himalaya-ai/nepali-news-corpus` | `default / train` | Pretraining: `text`, `id` |
| `himalaya-nepali-proofreader-test` | `himalaya-ai/nepali-proofreader` | `default / test` | Evaluation: `corrupted` input, `clean` reference; task `ocr_proofreading` |
| `boredoom-nepali-corpus` | `Boredoom17/Nepali-Corpus` | `default / train` | Pretraining: `text` |
| `alpaca-nepali-sft` | `Saugatkafley/alpaca-nepali-sft` | `default / train` | Instruction fine-tuning: standard `instruction`, `input`, `output` fields; preparation needed |
| `hermes-function-calling-nepali` | `himalaya-ai/hermes-function-calling-nepali` | `default / ne_deva` | Instruction fine-tuning: `conversations` mapped to `messages` |
| `bactrian-x-ne` | `MBZUAI/Bactrian-X` | `ne / train` | Instruction fine-tuning: standard `instruction`, `input`, `output` fields |
| `culturax-ne` | `uonlp/CulturaX` | `ne / train` | Pretraining: `text` |
| `c4-ne` | `allenai/c4` | `ne / train` | Pretraining: `text` |
| `toxi-text-ne` | `FredZhang7/toxi-text-3M` | `default / train`; `lang == ne` | Supervised: `text`, label `is_toxic`; task `toxicity_classification` |
| `nllb-ne-text1` | `slone/nllb-200-10M-sample` | `default / train`; `lang1 == npi_Deva` | Supervised: input `text1`, label `text2`; task `translation` |
| `nllb-ne-text2` | `slone/nllb-200-10M-sample` | `default / train`; `lang2 == npi_Deva` | Supervised: input `text2`, label `text1`; task `translation` |
| `multi-wiki-qa-ne` | `alexandrainst/multi-wiki-qa` | `ne / train` | Supervised: `context` + `question`, label `answers.text`; task `question_answering` |

The two NLLB selections are separate. Filters on different columns are combined
with AND, so putting both language conditions into one entry would change the
population. Multi Wiki QA uses both input columns; setting `field_mapping.text`
would override that multi-column input.

### All 17 Updesh subsets

Every entry below uses `microsoft/Updesh_beta`, split `npi_Deva`, revision
`73ed1f8e0850656150a3d97261eeab749cc6a7b2`, schema
`instruction_finetuning`, and `field_mapping: {messages: messages}`.
Each ID produces its own embedding run; the registry does not merge subsets.

| Source ID | Configuration |
| --- | --- |
| `updesh-ne-analytical-reasoning` | `analytical_reasoning` |
| `updesh-ne-brain-teaser` | `brain_teaser` |
| `updesh-ne-causal-reasoning` | `causal_reasoning` |
| `updesh-ne-creative-writing` | `creative_writing` |
| `updesh-ne-cultural-multihop-reasoning` | `cultural_multihop_reasoning` |
| `updesh-ne-dialog-gen` | `dialog_gen` |
| `updesh-ne-fermi` | `fermi` |
| `updesh-ne-fs-cot-flow` | `fs_cot_flow` |
| `updesh-ne-logical-reasoning` | `logical_reasoning` |
| `updesh-ne-math` | `math` |
| `updesh-ne-mcq` | `mcq` |
| `updesh-ne-multihop-reasoning` | `multihop_reasoning` |
| `updesh-ne-rc` | `rc` |
| `updesh-ne-summarization` | `summarization` |
| `updesh-ne-text-classification` | `text_classification` |
| `updesh-ne-translation-enxx` | `translation_enxx` |
| `updesh-ne-translation-xxen` | `translation_xxen` |

## Kaggle

| Source ID | Dataset handle | File | Instance definition |
| --- | --- | --- | --- |
| `oscar-nepali-kaggle` | `hsebarp/oscar-corpus-nepali/versions/1` | `ne_dedup.txt` | Pretraining: one nonempty line per instance, `text`; declared language `ne` |
| `nepali-hate-speech-brb` | `mohanbhandari/nepali-hate-speech-collection/versions/1` | `brb.xlsx` | Supervised: `Cleaned`, label `Category` |
| `nepali-hate-speech-lexicon` | `mohanbhandari/nepali-hate-speech-collection/versions/1` | `Nepali hate speech.xlsx` | Supervised: `NormNep`, label `Class` |

Both workbooks use task name `offensive_term_category`, matching their saved
inspection settings. This replaces the initial registry placeholder
`hate_speech_classification` for future runs. Task names enter supervised embedding
text, so retaining the recorded name matters. Previous run artifacts remain as
saved and retain their original definitions.

## Local files and downloads discussed earlier

Relative paths below are relative to the registry in `configs/datasets/`.

| Source ID | Local path | Definition and availability when registered |
| --- | --- | --- |
| `multiparacrawl-ne-v7-1` | `../../artifacts/datasets/multiparacrawl-v7.1/ne.txt` | Present. Pretraining, one nonempty Nepali sentence per line; downloaded OPUS MultiParaCrawl v7.1 monolingual portion associated with `Helsinki-NLP/multi_para_crawl`. This selection contains no paired translation targets. |
| `local-ne-text` | `../../ne.txt` | Present. Pretraining, one nonempty line per instance, matching the saved inspection. Original source unverified; do not identify it as either OSCAR collection from the filename alone. |
| `clean-nepberta-text` | `/home/olive/Downloads/clean_nepberta_data/clean_date_categories.csv` | Present. Pretraining, `text` column per CSV row, matching the saved inspection. |
| `oscar-mini-ne` | `../../artifacts/datasets/oscar-mini/ne.txt` | Planned download of `nthngdy/oscar-mini`, Nepali. Pretraining with blank-line-separated documents. This path does not exist yet. |
| `semantic-demo` | `semantic-demo.jsonl` | Present. Small synthetic demonstration corpus, `text` per JSONL record. Excluded from the 40 research selections. |

OSCAR mini and MultiParaCrawl are registered under their local loading route
because their original Hugging Face repositories use legacy dataset scripts.
The registered provider describes how this pipeline reads data; the upstream
dataset identity is recorded above.

For OSCAR mini, download and decompress the pinned
[`data/ne.gz`](https://huggingface.co/datasets/nthngdy/oscar-mini/resolve/8279d43fc305c5248886d841cb49bd8380456ec9/data/ne.gz)
into its registered local path. Preserve blank lines: the
[original loader](https://huggingface.co/datasets/nthngdy/oscar-mini/blob/8279d43fc305c5248886d841cb49bd8380456ec9/oscar-mini.py)
groups consecutive nonempty lines into one document. The
[MultiParaCrawl v7.1 Nepali file](https://object.pouta.csc.fi/OPUS-MultiParaCrawl/v7.1/mono/ne.txt.gz)
uses sentence lines. These two record boundaries must remain distinct.

## Known preparation requirements

These definitions describe the original sources. The embedding command currently
rejects malformed instances; it does not inherit the inspection command's
`invalid_instance_policy: skip`. The saved full inspections recorded:

| Selection | Source rows | Skipped during inspection | Eligible inspection rows |
| --- | ---: | ---: | ---: |
| `alpaca-nepali-sft` | 52,005 | 42 (39 empty assistant responses, 3 empty instructions) | 51,963 |
| `updesh-ne-fs-cot-flow` | 23,985 | 4 empty assistant responses | 23,981 |
| `updesh-ne-rc` | 49,634 | 10 (2 empty user messages, 8 empty assistant responses) | 49,624 |
| `updesh-ne-text-classification` | 49,153 | 2 empty assistant responses | 49,151 |

For these four sources, a full embedding run needs an explicitly prepared and
separately registered validated copy. A limited prefix can succeed if the bad
records occur later; it is not proof that the full source is valid. Bactrian-X's
saved inspection used skip policy but reported zero skipped records.

## Select a source

List all definitions without downloading datasets or loading models:

```bash
venv/bin/python scripts/cluster_dataset.py sources
```

Select one definition, keep model/run options in `configs/embeddings.yaml`, and
use an unused output directory:

```bash
venv/bin/python scripts/cluster_dataset.py run \
  --settings configs/embeddings.yaml \
  --source-id multi-wiki-qa-ne \
  --output-dir artifacts/dataset_embeddings/multi-wiki-qa-ne-run01
```

Registration and listing do not generate embeddings. The configured
`max_records` still determines whether a run covers a prefix or the full source.
Source IDs currently apply to the embedding command and results UI; the
inspection command continues to use `configs/datasets/dataset.yaml`.

This catalog expansion changes only source configuration and documentation.
It does not change language/script decisions, thresholds, schema validation,
sampling, embedding calculations, or saved artifacts. Older runs whose source
definition differs remain selectable as saved datasets in the UI.
