# D2 data pruning

This project implements D2 Pruning as an optional, model-aware coreset
selection stage after normalization, quality filtering, exact/near
deduplication, and boilerplate removal. It never rewrites the source dataset or
the full `clean-data/` output.

The supported record contracts are defined in
[Standard training-data schemas](standard-training-data-schemas.md).

Primary references:

- Maharana, Yadav, and Bansal,
  [D2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data
  Pruning](https://arxiv.org/abs/2310.07931)
- [Authors' research implementation](https://github.com/adymaharana/d2pruning)

## Why this is preprocessing, not cleaning

Cleaning removes or quarantines malformed, empty, unsafe, or policy-rejected
records. Deduplication removes verified copies. D2 makes a different decision:
given a fixed training budget, it selects a representative subset using model
embeddings and optional model-derived difficulty scores. A record excluded from
a D2 coreset is not necessarily bad data.

The batch sequence is therefore:

```text
immutable input
  -> sampling
  -> normalization and quality filtering
  -> exact/verified-near deduplication
  -> repeated-boilerplate removal
  -> full immutable clean-data shards
  -> optional D2 selection
  -> D2 coreset + scores + manifest
  -> comparative EDA and packaging
```

## Algorithm implemented

Each clean document is a graph node. In the equations below, `i` is the current
document and `j` is one of its nearest neighboring documents.

1. Embed every document and connect it to its `k` nearest neighbors using
   Euclidean distance. The directed k-nearest-neighbor relation is symmetrized.
2. Initialize node `i` with difficulty score `x_i`.
3. Perform one forward message-passing step:

   ```text
   forward_i = x_i + sum(exp(-gamma_forward * distance(i, j)^2) * x_j)
   ```

4. Select the node with the highest current score.
5. Down-weight each unselected neighbor `j` of selected node `i`:

   ```text
   score_j = score_j
             - exp(-gamma_reverse * distance(i, j)^2) * score_i
   ```

6. Repeat steps 4-5 until the requested retention budget is filled.

The implementation uses deterministic index-based tie breaking and a lazy
max-heap for selection. The public Python API is:

```python
from attention_maps.pruning import D2PruningConfig, select_d2_coreset

result = select_d2_coreset(
    embeddings,
    difficulty_scores,
    D2PruningConfig(
        use_case="traditional_nlp",
        retention_fraction=0.50,
        n_neighbors=10,
        gamma_forward=1.0,
        gamma_reverse=0.8,
        graph_backend="exact",
    ),
)
```

`retention_fraction` is the fraction kept. Thus, `0.50` means a 50% pruning
rate. This naming avoids confusing “percentage retained” with “percentage
removed.”

## Difficulty modes

### Uniform difficulty

`difficulty_mode: uniform` assigns every node a value of one. Selection is then
driven by embedding-space density and reverse-message diversity. It is retained
only as a diagnostic ablation for the two supported supervised use cases. It is
not authorization to apply this implementation to pretraining or SFT corpora.

### Supervised NLP difficulty

For supervised NLP, the paper uses the variability of correct-label confidence
across training checkpoints/epochs and a classifier's `[CLS]` representation.
The helper accepts an `(epochs, examples)` confidence matrix:

```python
from attention_maps.pruning import confidence_variability

difficulty_scores = confidence_variability(correct_label_probabilities)
```

External supervised scores must be supplied as an aligned `.npz` artifact:

```python
import numpy as np

np.savez(
    "difficulty.npz",
    scores=difficulty_scores,
    doc_ids=np.asarray(clean_doc_ids),
)
```

The Streamlit workspace can also calculate these scores from confidence
history directly. Its upload uses an `(epochs, documents)` array:

```python
np.savez(
    "confidence-history.npz",
    confidences=correct_label_probabilities,
    doc_ids=np.asarray(clean_workspace_row_ids),
)
```

The pipeline validates every `doc_id` and refuses to continue if score and
embedding order differ from the clean corpus. Difficulty values must be finite
and non-negative.

## Embeddings

By default, an enabled run extracts `[CLS]` embeddings with
`FacebookAI/xlm-roberta-base`. This is an operational starting point, not a
claim that it is the best Nepali representation model. Pin
`embedding_revision` to an immutable commit before recording research results.

`embedding_pooling` may be `cls` or `mean`. `normalize_embeddings` is disabled
by default so the paper's Euclidean representation is not silently changed.

Precomputed embeddings use this format:

```python
np.savez(
    "embeddings.npz",
    embeddings=embedding_matrix,
    doc_ids=np.asarray(clean_doc_ids),
)
```

## Batch configuration

Add a `pruning` section to the Drive/local batch pipeline configuration:

```yaml
input:
  local_path: ../../data/cleaned/reviews.jsonl
  text_columns: [text]
  source_columns: [source]
  label_column: label

pruning:
  enabled: true
  use_case: traditional_nlp
  retention_fraction: 0.50
  n_neighbors: 10
  gamma_forward: 1.0
  gamma_reverse: 0.8
  graph_backend: auto
  exact_block_size: 1024
  max_exact_records: 10000
  label_balanced: true
  embedding_model_id: FacebookAI/xlm-roberta-base
  embedding_revision: REPLACE_WITH_COMMIT
  embedding_pooling: cls
  embedding_batch_size: 32
  embedding_max_length: 256
  embedding_device: auto
  normalize_embeddings: false
  local_files_only: false
  embeddings_path: null
  difficulty_mode: external
  difficulty_scores_path: ../../artifacts/reviews-difficulty.npz
```

Run the existing pipeline command:

```bash
.venv/bin/python scripts/run_drive_preprocessing_pipeline.py \
  --config configs/eda/drive_preprocessing.example.yaml
```

When `label_balanced` is enabled, `input.label_column` must be populated on every
retained clean row. Selection is performed within each label and budgets are
assigned proportionally with a deterministic largest-remainder rule.

## Streamlit workspace

Launch the existing explorer:

```bash
.venv/bin/python scripts/run_dataset_explorer.py
```

In the `WORKSPACE` tab:

1. Select text, provenance, and label fields.
2. Run sampling, NFC normalization, and deduplication.
3. Set the canonical schema to **Task-specific supervised**.
4. Choose **Traditional NLP task** or **Domain-specific fine-tuning**.
5. Confirm that the workspace contains training-split records only.
6. Upload correct-label confidence history or precomputed scores. Uniform
   difficulty is available only for an explicit ablation.
7. Generate XLM-R embeddings in the UI or upload aligned embeddings.
8. Set retention, `k`, graph backend, class balancing, and gamma values, then
   run Step 4.

The UI reports live progress for embedding generation, graph construction,
forward message passing, reverse-message selection, and artifact writing. It
then displays retained count, class retention, and a diagnostic randomized-PCA
view of selected versus pruned examples. D2 operates on full embeddings, not
the displayed two-dimensional projection.

Every upload must contain `doc_ids` in exact clean-workspace order. The UI
rejects misaligned arrays and blocks D2 on validation/test catalog splits.

## Outputs

An enabled run adds the following under `output/d2/`:

```text
d2/
  coreset/part-*.parquet
  d2_scores.parquet
  d2_manifest.json
  embeddings.npy          # when embeddings were generated by the pipeline
```

`d2_scores.parquet` contains `doc_id`, input difficulty, score after forward
message passing, final score, selection status, selection rank, and score at
selection. The manifest records configuration, graph backend and edge counts,
difficulty statistics, requested/actual retention, and artifact locations.

The top-level run manifest points to these artifacts. If EDA is enabled, both
the full clean dataset and D2 coreset are analyzed. The Zip64 package includes
the full clean data and D2 outputs.

An interactive workspace run writes an immutable timestamped folder below the
Step 3 workspace directory:

```text
d2/<timestamp>/
  d2_coreset.jsonl
  d2_scores.csv
  d2_manifest.json
  embeddings.npy          # when generated in the UI
  difficulty_input.npz    # when confidence/scores were uploaded
  embeddings_input.npz    # when embeddings were uploaded
```

## Graph backends and resource limits

- `exact` computes exact squared-L2 neighbors in bounded query blocks. It does
  not allocate an all-pairs distance matrix, but still performs quadratic work.
- `faiss` uses an optional `faiss-cpu` installation with `IndexFlatL2`.
- `auto` uses exact NumPy search through `max_exact_records`, then requires
  FAISS. It fails clearly instead of starting an accidental large quadratic
  run.

The embedding matrix is stored as a NumPy memory-mapped file during model
extraction. Graph edges are stored as sparse compressed rows. The full cleaned
text remains in Parquet and is streamed again when artifacts are materialized.

## Dataset policy

- D2 is currently enabled only for `traditional_nlp` and
  `domain_specific_finetuning`, both represented by the task-specific
  supervised schema.
- Start the supervised evaluation with the Nepali movie-review sentiment
  dataset because it is closest to the paper's IMDb experiment.
- Apply supervised D2 to hate-speech tweets only after label/schema validation.
- Do not enable this implementation for pretraining, instruction fine-tuning,
  or preference tuning datasets.
- Never run pruning on evaluation or test datasets. Split protection and
  duplicate-cluster isolation must happen before difficulty training and D2
  selection.

For each study, compare full data, seeded random retention, difficulty-only
retention, uniform-score graph selection, and D2 at the same budget. Tune `k`
and `gamma_reverse` on validation only and keep model training steps fixed
across pruning rates.

## Known boundaries

- The repository exposes the D2 algorithm and confidence-variability helper,
  but does not yet contain a general classifier-training-dynamics runner.
  Supervised difficulty is therefore an explicit, validated input artifact.
- Uniform difficulty does not reproduce the paper's supervised NLP method.
- Model embeddings can encode model and dataset bias. Model/tokenizer revisions
  and pooling choices are part of the dataset version and must be audited.
- D2 exclusion is a budget decision, not evidence that a document is low
  quality or unsafe.
