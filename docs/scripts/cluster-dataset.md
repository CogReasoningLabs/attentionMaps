# Corpus embeddings, similarity, clustering, and B1–B3 sampling

The standalone `scripts/cluster_dataset.py` command owns dataset selection, model
loading, embedding, clustering, and sample exports. Streamlit reads completed
artifacts, selects saved datasets/models, compares any two stored records, and
shows an interactive 3D projection. Opening the UI never downloads a model or
reruns clustering.

## Reusable datasets from three providers

Maintain datasets in [`configs/datasets/sources.yaml`](../../configs/datasets/sources.yaml).
Its `sources` mapping has three groups: `huggingface`, `kaggle`, and `local`.
Every entry has a unique, stable ID and its own complete source/schema selection:

```yaml
version: 1
sources:
  huggingface:
    news-ne:
      label: Nepali news
      dataset: owner/news
      config: ne
      split: train
      training_schema: pretraining
      field_mapping: {text: text}
      text_columns: [text]
  kaggle:
    comments-ne:
      dataset: owner/comments/versions/1
      dataset_file: comments.csv
      training_schema: task_specific_supervised
      field_mapping: {text: comment, label: category}
      task_name: classification
  local:
    multiparacrawl-ne-v7-1:
      local: ../../artifacts/datasets/multiparacrawl-v7.1/ne.txt
      training_schema: pretraining
      text_record_unit: line
      field_mapping: {text: text}
      text_columns: [text]
```

The remote IDs above are structural examples. The checked-in registry defines
the datasets from the earlier discussion and saved research history, including
both NLLB directions, all 17 Nepali Updesh subsets, the Kaggle files, and local
corpora. See the [source catalog](../dataset-source-registry.md) for every ID,
field mapping, and known preparation requirement, including OSCAR mini's pending
download. The demonstration corpus remains a separate entry.
Local paths are relative to the registry file. Hugging Face entries may include
`revision`, `config`, `split`, and `shards`. Kaggle entries require `dataset_file`;
different files in the same account/dataset get different source IDs. Optional
`row_filters`, field parsers, task names, and instance boundaries belong to each
entry. Register a separate ID for each exact subset; wildcard batch inspection
does not imply combined embedding runs.

List sources without network access or model loading:

```bash
venv/bin/python scripts/cluster_dataset.py sources
```

Select a registered source while retaining shared model and clustering settings:

```bash
venv/bin/python scripts/cluster_dataset.py run \
  --settings configs/embeddings.yaml \
  --source-id multiparacrawl-ne-v7-1 \
  --model nepali-bert \
  --output-dir artifacts/dataset_embeddings/multiparacrawl-ne-bert-run01
```

Alternatively, set `source_id` in `configs/embeddings.yaml`. `source_registry`
selects the registry file (relative to the embedding settings file), and
`--source-registry` overrides it relative to the working directory. With no source
ID, the existing inline source or `--source-settings` behavior is retained.
A selected ID replaces **all** inline source/schema fields, including stale
filters and language declarations. Do not combine it with direct source/schema
CLI flags; create a separate entry for another selection. Model, seed, prefix
limit, batch sizes, cluster settings, and output directory remain independently
configurable. Full option names are required for `run`.

Streamlit's **Embedding results** view now starts with **Source type** and
**Dataset**, including registered sources with no completed runs. Expand
**Generate embeddings for this dataset** to choose a model and unused run name;
the UI shows the exact command and configured processing scope. Run that command
from the project root, then click **Refresh saved runs** to visualize the result.
Edit the registry YAML to add more entries. Selecting a source does not download
it or run an embedding model.

New reports save the source ID, registry definition, and its signature alongside
the resolved source provenance. The registration does not change embedding text,
corpus fingerprints, or numerical processing. Existing reports and vectors are
not rewritten. Older runs remain selectable as saved datasets within their
provider, even if the original source file or registry is unavailable. Changing
a registry definition keeps its previous runs in that saved-dataset list;
they are not silently assigned to the new definition. Changing just a display
label does not change the definition signature.

## Reuse embeddings for different cluster counts and word clouds

Each script invocation computes **one requested cluster count**. For an existing
embedding run, change `--clusters` manually; there is no need to load the dataset
or embedding model again:

```bash
venv/bin/python scripts/cluster_dataset.py recluster \
  --run artifacts/dataset_embeddings/trial-1000-schema --clusters 5

venv/bin/python scripts/cluster_dataset.py recluster \
  --run artifacts/dataset_embeddings/trial-1000-schema --clusters 10

venv/bin/python scripts/cluster_dataset.py recluster \
  --run artifacts/dataset_embeddings/trial-1000-schema --clusters 20
```

Replace the run path with your saved embedding directory. These commands save
`clusterings/k5-seed42`, `clusterings/k10-seed42`, and `clusterings/k20-seed42`
inside that directory. Existing results are never overwritten; use `--name
another-trial` or a different `--seed` to save another experiment. Optional
`--cluster-batch-size` and `--cluster-epochs` control clustering. The requested
count must be between 1 and the number of embedded records. Repeated embeddings
can leave some clusters empty.

The saved document vectors and records are shared with the parent run, while
each result has its own assignments, centroids, distances, report, and word
clouds. The original PCA coordinates and transform are copied unchanged, so
positions stay fixed when comparing cluster counts. Cluster IDs are arbitrary;
cluster 0 at K=5 does not necessarily correspond to cluster 0 at K=20. Reusing a
1,000-record run still clusters only those 1,000 records. Keep the parent run
when moving or sharing these results. Existing pair similarity remains unchanged.
The `sample` subcommand continues to use the parent run's original clustering.

In Streamlit, open **Embedding results**, select the embedding run, then choose
**Saved clustering result**. Switch **Visualization dimensions** between **3D**
and **2D**. The 2D view uses saved PC1/PC2; 3D adds PC3. Both views show the same
records, assignments, and projected centroids, and report the variance retained
in the displayed dimensions. No script rerun is needed for existing saved runs.
The metrics, plot colors/centroids, cluster browser,
and word cloud all follow that result. Select **Browse cluster** to see its
cloud, exact top-word counts, and record previews. The UI reads saved artifacts;
it never performs embedding, clustering, PCA fitting, or word-frequency counting.
New `run` invocations also save word clouds for their original clustering.
Older runs remain readable. To add missing clouds to their **existing cluster
assignments**, without re-embedding or reclustering, run:

```bash
venv/bin/python scripts/cluster_dataset.py wordclouds \
  --run artifacts/dataset_embeddings/trial-1000-schema
```

This fills the original result's `wordclouds/` directory. For a saved variant,
add `--clustering k20-seed42`. The command preserves existing embeddings,
assignments, projections, and reports. It refuses to overwrite saved clouds
unless `--refresh` is passed. After editing `configs/eda/stopwords.txt`, regenerate
clouds for the selected result:

```bash
venv/bin/python scripts/cluster_dataset.py wordclouds \
  --run artifacts/dataset_embeddings/trial-1000-schema \
  --clustering k5-seed42 --refresh
```

Omit `--clustering` to refresh the original result. New images and counts are
built in a temporary directory before replacing the previous clouds; a failed
generation preserves the existing images. The UI compares the saved stopword
list with the current configuration and shows a refresh command when they differ.
Refresh Streamlit after the command finishes. When clouds are missing, the UI displays
the exact command for the currently selected result.

Word frequencies are counted over **all records assigned to each cluster**, not
only the 25 visible records or the plotted sample. Counts use canonical content:
full document/input text, all conversation content, or all three preference
branches. IDs, provenance, supervised labels/tasks, and added role/field markers
are excluded. Legacy runs use their stored embedding text. Unicode letters and
combining marks preserve Nepali vowel signs; text is NFC-normalized and
lowercased. Numbers, punctuation, and the existing English stopword list plus
`configs/eda/stopwords.txt` are excluded. Edit that file before running the script
to customize Nepali stopwords; the exact list is stored with each result. There is no stemming or automatic semantic keyword extraction.
A word cloud is a frequency summary, not proof of a cluster's topic or quality.

Each occupied cluster saves an image containing up to 80 frequent words and up
to 200 exact term counts in `wordclouds/summary.json`. Vocabulary aggregation uses
a temporary SQLite table to avoid holding the entire corpus vocabulary in RAM.
Empty clusters or clusters containing only excluded words show an explanatory
message. Noto Devanagari fonts are auto-detected on Linux. You may specify a font
with `--wordcloud-font /path/to/NotoSansDevanagari-Regular.ttf`, or set
`wordcloud_font` in the central embedding YAML for `run`. If no Devanagari font
is available, Nepali word counts are still saved and displayed with instructions
for generating an image after installing a suitable font. The existing
`wordcloud` dependency in `requirements.txt` generates the saved PNGs.

## Central configuration for every model

Edit [`configs/embeddings.yaml`](../../configs/embeddings.yaml). It is the shared
configuration for dataset selection, model choice, embedding options, and
clustering; no separate Gemma or NepBERTa config is needed.

```bash
venv/bin/python scripts/cluster_dataset.py run --settings configs/embeddings.yaml
```

Select the **datasource provider first**: remote sources require an explicit
`provider: huggingface` or `provider: kaggle`, in the central/source settings or
as `--provider`. The provider is independent of the embedding model. The script
prints the selected datasource before discovery/downloads and rejects fields
belonging to the other provider. Local paths identify their source automatically.

| Datasource | Source fields |
| --- | --- |
| Kaggle | `provider: kaggle`, `dataset: owner/dataset`, `dataset_file: file.txt`; Hugging Face fields must be null. |
| Hugging Face | `provider: huggingface`, `dataset: owner/dataset`, `config`, `split`, `revision`, `shards`; `dataset_file` must be null. |
| Local | `provider: local`, `local: path/to/data`; `dataset` and remote selection fields must be null. |

Change `model` to `nepali-bert`, `nepberta`, or `embeddinggemma`. Leave `model_id`
and `max_length` null to use the chosen preset's checkpoint and supported chunk
limit. Original NepBERTa still requires the separate runtime described below.
Set `output_dir` to a new directory for each experiment. The config's `max_records` value
controls the current trial size, and `clusters` controls the requested groups.

CLI arguments override the central file, including replacing list fields:

```bash
venv/bin/python scripts/cluster_dataset.py run \
  --settings configs/embeddings.yaml --model embeddinggemma \
  --output-dir artifacts/dataset_embeddings/gemma-run
```

Paths inside YAML/JSON are relative to the settings file; CLI paths are relative
to the working directory. `--settings` is the embedding configuration file;
`--config` retains its existing meaning of Hugging Face dataset configuration.
`--source-settings` can still reference an inspection YAML for source selection.
An explicit CLI `--source-settings` replaces the central file's source fields;
additional CLI source flags override that referenced source file. Changing the
model through the CLI clears an inherited `model_id` override. Completed reports
save the effective configuration for Streamlit's exact-settings view.

## Research and model choices

| CLI preset | Checkpoint | Use in this workflow |
| --- | --- | --- |
| `nepali-bert` | `Shushant/nepaliBERT` | Nepali masked-language-model baseline, attention-masked mean pooling. |
| `nepberta` | `NepBERTa/NepBERTa` | Original NepBERTa baseline, using its TensorFlow checkpoint in a separate environment. |
| `embeddinggemma` | `google/embeddinggemma-300m` | Task-trained multilingual embedding model; candidate teacher, with Nepali superiority still unverified. |

NepaliBERT's card describes masked-language pretraining on Nepali news. It does
not establish a sentence-similarity benchmark, so this implementation treats
mean pooling as a baseline to evaluate. [Publisher's model card](https://huggingface.co/Shushant/nepaliBERT).

NepBERTa is a BERT-based Nepali model evaluated on downstream Nepali tasks.
Those task results do not establish the quality of an unfine-tuned sentence
embedding space. Its original repository currently publishes `tf_model.h5`,
requiring the optional TensorFlow/Transformers 4 setup below. A similarly named
community checkpoint is not automatically equivalent to the authors' model.
[NepBERTa paper](https://aclanthology.org/2022.aacl-short.34/),
[original checkpoint files](https://huggingface.co/NepBERTa/NepBERTa/tree/main).

EmbeddingGemma is a 300M multilingual embedding model with a 2K input context
and 768-dimensional output. Google supplies distinct clustering and sentence
similarity prompts. This implementation uses the published SentenceTransformer
pipeline, including its pooling/projection/normalization layers, with float32;
the model card excludes float16 activations. Its multilingual results motivate
the comparison but do not prove an advantage on this Nepali corpus.
[Google model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card),
[checkpoint and precision instructions](https://huggingface.co/google/embeddinggemma-300m).

## Install and try a small corpus

In the existing project environment:

```bash
venv/bin/python -m pip install -r requirements-embeddings.txt
venv/bin/python scripts/cluster_dataset.py models

venv/bin/python scripts/cluster_dataset.py run \
  --local configs/datasets/semantic-demo.jsonl --text-column text \
  --model nepali-bert --clusters 3 --batch-size 4 \
  --output-dir artifacts/dataset_embeddings/my-nepali-bert-demo
```

Use a new output directory for each run. The script will not overwrite a saved
experiment. A completed run includes the resolved model commit, exact source
selection, source signatures where available, record counts, and a corpus
fingerprint. `--model-id` can override a repository/local path within a preset's
backend; `--model-revision` pins the model separately from the dataset revision.

The included 12-record corpus is a functional demonstration with duplicates and
three manually written topics, not a semantic benchmark.

## Whole Hugging Face or Kaggle datasets

```bash
# Hugging Face: all selected records, including any requested later shard.
venv/bin/python scripts/cluster_dataset.py run \
  --provider huggingface --dataset owner/dataset \
  --config ne --split train --shard data/train-00001-of-00008.parquet \
  --text-column text --model nepali-bert --clusters 50 \
  --output-dir artifacts/dataset_embeddings/hf-nepali-bert

# Reuse the current provider, source, and text fields from the shared dataset YAML.
venv/bin/python scripts/cluster_dataset.py run \
  --source-settings configs/datasets/dataset.yaml \
  --model nepali-bert --clusters 50 \
  --output-dir artifacts/dataset_embeddings/selected-nepali-bert

# Kaggle source selection with CLI inputs.
venv/bin/python scripts/cluster_dataset.py run \
  --provider kaggle --dataset hsebarp/oscar-corpus-nepali \
  --dataset-file ne_dedup.txt --text-column text --language ne \
  --model embeddinggemma --embedding-task clustering --clusters 50 \
  --output-dir artifacts/dataset_embeddings/kaggle-gemma
```

Omit `--shard` to include all shards of the selected split; repeat it to select
several. Discover available choices using `scripts/inspect_dataset.py --list`.
Kaggle supports versioned `owner/dataset/versions/N` handles and exact file paths.
Local sources use `--local`: Parquet directories or supported individual Parquet,
JSON, JSONL, CSV, TXT, and XLSX files. Text files use explicit line or blank-line instance boundaries;
XLSX uses its first worksheet. Repeat `--text-column` for multiple fields, including
dotted conversation fields.

`--source-settings` reuses only source, language declaration, and text-field
settings. It does not inherit inspection's 20% sample fraction: **embedding and
clustering cover every non-empty record in the selected source by default**.
Enabled filtering settings are rejected; first run the filter and then use its
JSONL output as a local source. Empty-text records are counted and skipped.

For a deliberate small prefix trial, add `--max-records 1000`; the report marks
this `limited_prefix`. Choose `--clusters` no greater than the number of non-empty
records. Remove that limit for the full dataset.

## EmbeddingGemma access and NepBERTa dependencies

EmbeddingGemma is gated on Hugging Face. Accept Google's terms on its
[official model page](https://huggingface.co/google/embeddinggemma-300m), then
authenticate using `hf auth login` or set `HF_TOKEN` in the project `.env`.
The program reads existing credentials; it does not accept terms on your behalf.

For original NepBERTa, keep its TensorFlow dependencies separate from the main
Transformers 5 environment:

```bash
python3.12 -m venv venv-nepberta-embeddings
venv-nepberta-embeddings/bin/pip install -r requirements-nepberta.txt \
  numpy PyYAML pyarrow datasets kagglehub python-dotenv scikit-learn sentencepiece

venv-nepberta-embeddings/bin/python scripts/cluster_dataset.py run \
  --local configs/datasets/semantic-demo.jsonl --text-column text \
  --model nepberta --device cpu --clusters 3 \
  --output-dir artifacts/dataset_embeddings/nepberta-demo
```

The UI can open those artifacts from the main environment; it needs no
TensorFlow runtime. The existing fill-mask command remains separate.

## One complete data instance per embedding

The script now uses the [four existing training schemas](../standard-training-data-schemas.md)
to define the atomic unit. Set `training_schema` in the central YAML:

| Schema | One instance | Text used for its embedding |
| --- | --- | --- |
| `pretraining` | One document | The full document text. |
| `instruction_finetuning` | One complete conversation | Every message, in order, with system/user/assistant role markers. |
| `task_specific_supervised` | One labelled example | Task, full input text, and label, with field markers. |
| `preference_tuning` | One prompt/preference group | Prompt, chosen and rejected together, with branch markers. |
| `evaluation` | One input/reference example | Input and reference together, with field markers. |

`auto` infers these from canonical columns, conversations, or
instruction/input/output fields. Explicit selection is preferred for research
runs. Changing the schema changes the embedding representation and requires a
new run; old vectors are never reinterpreted using the current YAML.

IDs, split, source, domain, language, license and metadata are retained for
inspection and excluded from the vector input. Original source records are also
stored intact, including extra fields outside the canonical schema. Missing IDs
get stable source-derived IDs; duplicate IDs fail the run. Missing splits remain
null, clearly identified as unassigned. This analysis adapter does not create
training splits or certify raw input as ready for training. Assign and validate
splits before downstream training or pruning as required by the schema contracts.

For mapped supervised data:

```yaml
training_schema: task_specific_supervised
field_mapping:
  text: review.body
  label: category
  id: example_id
task_name: sentiment-classification   # Used if no task field is present.
```

For sentence-pair tasks, select all input fields through `text_columns`, e.g.
`[premise, hypothesis]`, without a `field_mapping.text` override. Labels, including
numeric zero, stay attached to the example. Including the label/task makes this
an example-level similarity score, not a label-blind input-only score.

For SFT, use canonical `messages`, `conversations` (human/gpt aliases are mapped),
or instruction/input/output fields. Nested conversations can be selected with
`field_mapping: {messages: payload.turns}`. For a document stored as an ordered
list of paragraph strings, use `field_mapping: {text: paragraphs}` with
`field_parsers: {text: join_strings}` in `configs/embeddings.yaml`. The parser
joins paragraphs with blank lines and rejects non-string list items; the
raw source record remains available for inspection. A scalar text column uses
the default `string` behavior. Both `--field-map text=paragraphs` and
`--field-parser text=join_strings` can override the YAML for one run.
Complete user/assistant turns and an
ending assistant response are required. `text_columns` does not discard turns
or preference branches for structured schemas. Preference chosen and rejected
must be non-empty and different.

CLI equivalents include `--training-schema`, repeatable
`--field-map CANONICAL=SOURCE.PATH`, and `--task-name`.

### TXT boundaries are a separate choice

A schema does not reveal missing document boundaries. Set `text_record_unit`:

- `line`: one non-empty physical line per instance (the previous behavior).
- `blank_line`: collect all lines up to a blank separator into one document;
  `max_records` then counts those documents, not their physical lines.

`blank_line` applies only to local/staged TXT inputs. Use it only when the file
really has that boundary convention; without separators, the whole file becomes
one record. For JSON/JSONL/CSV/Parquet or Hub rows, keep whole documents in their
text field and whole conversations/preferences in each source row. A title-only
line cannot be reconstructed into a full article without original content and
reliable grouping metadata. Prefer a document-preserving source in that case.

The comparison view now shows the complete canonical instance, full original
record, and the exact text used to generate its embedding. Both instances can
be downloaded as JSON without text-preview truncation. Old reports remain
readable with their full saved records and unchanged cosine scores; they are
labelled as legacy source-row embeddings, not retroactively schema-aware.

## Embedding and clustering semantics

Each complete instance becomes one vector using the schema representation above. BERT variants use the last
hidden layer's mean over attended, non-special tokens. Gemma uses its published
embedding stack. Long documents are processed in non-overlapping token chunks
that fit the chosen model limit; decoded chunks are retokenized and checked
before inference. Normalized chunk vectors are averaged with source-token-count
weights, then normalized again. This document aggregation is our implementation
choice, not a claim of trained long-document support. The report records chunk
counts and the effective limit. `--max-length` changes chunk size, not coverage.

For Gemma, `--embedding-task clustering` uses `task: clustering | query: `.
`--embedding-task similarity` uses `task: sentence similarity | query: `.
All pair scores use the vectors from the saved task; the UI never secretly
re-embeds pairs with a different prompt. For a prompt ablation, create a second
run on the identical corpus with the other task.

Embeddings are float32 disk-backed arrays. MiniBatchKMeans visits all embedded
records on every configured epoch, with initialization drawn across the corpus.
Cluster assignment uses original normalized vectors with Euclidean k-means;
this is not spherical k-means. `--cluster-epochs` defaults to 3 and
`--cluster-batch-size` to 1024. Repeated vectors may yield fewer occupied clusters
than requested; both counts are reported. [MiniBatchKMeans reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html).

IncrementalPCA fits all vectors and produces three coordinates per record for
display. Clustering and pair similarity never use those projected coordinates.
Variance retained in 3D is shown because visual distances can distort original
similarities. [IncrementalPCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html).

## Compare any two records and open Streamlit

```bash
venv/bin/python scripts/cluster_dataset.py pair \
  --run artifacts/dataset_embeddings/my-nepali-bert-demo --row-a 0 --row-b 1

# Repeat --run to compare the same pair across identical-corpus model runs.
venv/bin/python scripts/cluster_dataset.py compare \
  --run artifacts/dataset_embeddings/kaggle-nepali-bert \
  --run artifacts/dataset_embeddings/kaggle-gemma --row-a 0 --row-b 100

DATASET_EMBEDDING_RUNS=artifacts/dataset_embeddings \
  venv/bin/python scripts/run_dataset_explorer.py
```

Alternatively select **View → Embedding results**. Select the dataset, model,
and saved run, then any two zero-based **embedded record IDs**. Original source
row numbers are displayed separately because empty records are skipped. Browse
cluster pages to find record IDs and inspect text. The 3D plot shows a seeded,
uniform subset (10,000 points by default), while every embedded record remains
available for pair queries and cluster browsing. The plot limit does not limit
clustering. The optional centroid overlay includes every occupied cluster even
when the plotted record sample misses a small cluster. Changing `N` requires a new script run with `--clusters N`.

Cross-model comparison requires identical corpus fingerprints: same ordered
records and selected text. Cosine is computed in the original space as
`dot(a, b) / (norm(a) * norm(b))`. It ranges from -1 to 1 and is neither a
probability nor directly calibrated between models.

## B1–B3 sampling and language/script voting

The referenced paper distinguishes uniform sampling (B1), inverse-density
sampling in embedding space (B2), and semantic deduplication inside clusters
(B3). B2 uses a p-stable hash sketch and inverse-propensity selection; B3 uses
clustering before removing semantically similar examples.
[How to Train Data-Efficient LLMs, §3.2–3.3 and Appendix B](https://arxiv.org/html/2402.09668v1).

This implementation provides:

| Method | Implemented selection |
| --- | --- |
| `random` / B1 | Exactly `ceil(fraction * N)` uniformly chosen IDs without replacement per run. |
| `density` / B2 | Gaussian p-stable projections `floor((a·x+b)/width) mod bins`; two corpus passes count/query hash buckets; weights proportional to the inverse mean bucket count; weighted selection without replacement. |
| `semdedup` / B3 | Order each cluster farthest from its centroid first; drop a row if its cosine with any earlier cluster member is greater than the threshold; sample the surviving IDs. |

B2 defaults to a smaller 64×2048 sketch and width 0.5 for practical trials.
The screenshot's 1000×20000 setting is available with `--density-rows 1000
--density-bins 20000`. Counters here are int64; that sketch consumes about 160 MB
plus projections and working batches. Our encoder, normalization, finite hash
range, and sketch settings differ from the paper's experiment; results are not
a reproduction of its published scores.

B3 uses blockwise pair comparisons without allocating a whole cluster's square
similarity matrix. It compares against all earlier members, including previously
removed ones, matching the ordered maximum-similarity rule. It uses a strict
`cosine > threshold` removal criterion and cannot detect duplicates split across
different clusters. The preference for farther-from-centroid examples follows
the authors' default hard-example ordering. [SemDeDup source](https://github.com/facebookresearch/SemDeDup/blob/main/semdedup.py).

```bash
venv/bin/python scripts/cluster_dataset.py sample \
  --run artifacts/dataset_embeddings/kaggle-nepali-bert \
  --method random --sample-fraction 0.2 --sampling-runs 5 \
  --output-dir artifacts/dataset_sampling/b1

venv/bin/python scripts/cluster_dataset.py sample \
  --run artifacts/dataset_embeddings/kaggle-nepali-bert \
  --method density --sample-fraction 0.2 --sampling-runs 5 \
  --density-rows 64 --density-bins 2048 --density-width 0.5 \
  --output-dir artifacts/dataset_sampling/b2

venv/bin/python scripts/cluster_dataset.py sample \
  --run artifacts/dataset_embeddings/kaggle-nepali-bert \
  --method semdedup --similarity-threshold 0.95 \
  --sample-fraction 0.2 --sampling-runs 5 \
  --output-dir artifacts/dataset_sampling/b3
```

`N` here is the non-empty embedded population. Different runs may overlap.
Source ordering, embeddings, clustering, and seeds determine reproducibility;
these random samples need not match the older inspection sampler's exact IDs.
For B3, if too few survivors remain, every survivor is exported and the shortfall
is reported. Duplicates are never added back to reach the requested fraction.
Use fraction 1 and one run to export all deduplicated survivors.

Each sampling run exports `sample-N.jsonl` with original whole records and
`sample-N.npy` with embedded record IDs. The sampling report uses the shared
Unicode/language-evidence logic and strict-majority voting, analyzes all selected
text, and counts unique coverage once across overlapping runs. It can be loaded
under **B1–B3 sampling results** in the embedding view. B2/B3 intentionally change
the sampling distribution; their language coverage is not an unbiased estimate
of corpus prevalence. Existing language declarations are not replaced by a
trained language detector. No `1-(1-f)^n` coverage expectation is claimed for
weighted or deduplicated samples.

## Artifacts, scaling, and evaluation

- `report.json`: source/model identity, scope, counts, clustering settings, and projection statistics.
- `embeddings.f32`: normalized vectors; shape/dtype in the report.
- `records.sqlite`: indexed embedding text, full canonical instances, original records, stable instance IDs, source row IDs, and chunk counts.
- `labels.npy`, `centroids.npy`, `centroid_distance.npy`: clustering in original space.
- `coordinates.npy`, `projection.npz`: saved 3D visualization coordinates and PCA parameters.

Embedding storage costs `records × dimensions × 4` bytes: one million 768d
vectors need about 3.07 GB, plus stored text/metadata and other arrays. Processing
uses bounded vector batches; local JSON array parsing still loads its document
into memory. Scalar ID/score arrays use memory proportional to record count.
B3's runtime remains quadratic within each cluster despite bounded matrix
memory; choose enough clusters and inspect size distributions before a large
deduplication run. There is no resume support yet: interrupted runs remove their
temporary artifacts and can be rerun into the same unused destination.

To test the proposed teacher hypothesis, hold the corpus, text fields, source
revision, and pair IDs fixed. Create human-rated Nepali paraphrase, unrelated,
negation, same-topic/different-fact, romanized, and mixed-language pairs. Compare
rank correlation with those judgments and retrieval/duplicate decisions, and
audit cluster coherence blind to model name. Tune deduplication thresholds per
model on held-out examples. The saved cosine silhouette is a bounded intrinsic
diagnostic, not proof of semantic accuracy or improved pretraining performance.
