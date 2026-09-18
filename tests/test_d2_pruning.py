import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from attention_maps.batch_pipeline.processing import (
    ParquetShardWriter,
    PreparedRecord,
    iter_parquet_records,
)
from attention_maps.eda.workspace import WorkspaceDocument
from attention_maps.pruning import (
    D2PruningConfig,
    build_knn_graph,
    confidence_variability,
    select_d2_coreset,
)
from attention_maps.pruning.pipeline import run_d2_pruning_on_parquet
from attention_maps.pruning.workspace import (
    difficulty_from_confidence_artifact,
    run_workspace_d2,
)


class D2GraphTests(unittest.TestCase):
    def test_restricts_d2_to_approved_supervised_use_cases(self):
        with self.assertRaisesRegex(ValueError, "pruning.use_case"):
            D2PruningConfig(use_case="pretraining")

    def test_builds_a_symmetric_graph_with_squared_l2_distances(self):
        graph = build_knn_graph(
            np.array([[0.0], [2.0], [5.0]], dtype=np.float32),
            1,
            backend="exact",
            block_size=2,
        )

        first_neighbors, first_distances = graph.neighbors(0)
        second_neighbors, _ = graph.neighbors(1)
        third_neighbors, third_distances = graph.neighbors(2)

        self.assertEqual(first_neighbors.tolist(), [1])
        self.assertEqual(first_distances.tolist(), [4.0])
        self.assertEqual(second_neighbors.tolist(), [0, 2])
        self.assertEqual(third_neighbors.tolist(), [1])
        self.assertEqual(third_distances.tolist(), [9.0])
        self.assertEqual(graph.directed_neighbor_edges, 3)
        self.assertEqual(graph.undirected_edges, 2)

    def test_reverse_message_passing_avoids_a_redundant_neighbor(self):
        progress = []
        result = select_d2_coreset(
            np.array([[0.0], [0.1], [10.0]], dtype=np.float32),
            np.ones(3),
            D2PruningConfig(
                retention_fraction=2 / 3,
                n_neighbors=1,
                gamma_forward=1.0,
                gamma_reverse=0.0,
                graph_backend="exact",
            ),
            progress=lambda stage, completed, total: progress.append(
                (stage, completed, total)
            ),
        )

        self.assertEqual(result.selected_count, 2)
        self.assertIn(2, result.selected_indices)
        self.assertEqual(len(set(result.selected_indices).intersection({0, 1})), 1)
        self.assertEqual(
            {"graph", "forward", "selection"},
            {stage for stage, _, _ in progress},
        )

    def test_forward_scores_follow_the_paper_rbf_equation(self):
        result = select_d2_coreset(
            np.array([[0.0], [2.0]], dtype=np.float32),
            np.array([1.0, 3.0]),
            D2PruningConfig(
                retention_fraction=0.5,
                n_neighbors=1,
                gamma_forward=1.0,
                graph_backend="exact",
            ),
        )

        weight = np.exp(-4.0)
        self.assertAlmostEqual(result.forward_scores[0], 1.0 + 3.0 * weight)
        self.assertAlmostEqual(result.forward_scores[1], 3.0 + weight)

    def test_label_balancing_preserves_proportional_class_budget(self):
        result = select_d2_coreset(
            np.arange(12, dtype=np.float32).reshape(6, 2),
            np.ones(6),
            D2PruningConfig(
                retention_fraction=0.5,
                n_neighbors=1,
                graph_backend="exact",
                label_balanced=True,
            ),
            labels=np.array(["a", "a", "a", "a", "b", "b"]),
        )

        selected_labels = np.array(["a", "a", "a", "a", "b", "b"])[
            list(result.selected_indices)
        ]
        self.assertEqual(result.selected_count, 3)
        self.assertEqual(np.sum(selected_labels == "a"), 2)
        self.assertEqual(np.sum(selected_labels == "b"), 1)

    def test_computes_correct_label_confidence_variability(self):
        values = confidence_variability(
            np.array([[0.1, 0.5], [0.3, 0.5], [0.5, 0.5]])
        )

        self.assertAlmostEqual(values[0], np.std([0.1, 0.3, 0.5]))
        self.assertEqual(values[1], 0.0)


class D2PipelineTests(unittest.TestCase):
    def test_workspace_run_uses_aligned_confidence_and_writes_ui_artifacts(self):
        documents = tuple(
            WorkspaceDocument(
                row_id=f"doc-{index}",
                text=f"नेपाली उदाहरण {index}",
                source="fixture",
                label="positive" if index < 2 else "negative",
            )
            for index in range(4)
        )
        confidence_buffer = io.BytesIO()
        np.savez(
            confidence_buffer,
            confidences=np.array(
                [
                    [0.7, 0.8, 0.6, 0.9],
                    [0.9, 0.7, 0.8, 0.8],
                    [0.8, 0.9, 0.7, 0.7],
                ]
            ),
            doc_ids=np.array([document.row_id for document in documents]),
        )
        difficulty = difficulty_from_confidence_artifact(
            confidence_buffer.getvalue(),
            [document.row_id for document in documents],
        )
        embedding_values = np.array(
            [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
            dtype=np.float32,
        )
        embedding_buffer = io.BytesIO()
        np.savez(
            embedding_buffer,
            embeddings=embedding_values,
            doc_ids=np.array([document.row_id for document in documents]),
        )
        updates = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "d2-run"
            result = run_workspace_d2(
                documents,
                D2PruningConfig(
                    enabled=True,
                    retention_fraction=0.5,
                    n_neighbors=1,
                    graph_backend="exact",
                    label_balanced=True,
                    difficulty_mode="external",
                    difficulty_scores_path=Path("ui-upload.npz"),
                ),
                output,
                difficulty,
                embeddings=embedding_values,
                difficulty_artifact=confidence_buffer.getvalue(),
                embeddings_artifact=embedding_buffer.getvalue(),
                progress=lambda stage, completed, total: updates.append(
                    (stage, completed, total)
                ),
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            coreset_lines = result.coreset_path.read_text(encoding="utf-8").splitlines()
            copied_inputs = (
                (output / "difficulty_input.npz").is_file(),
                (output / "embeddings_input.npz").is_file(),
            )

        self.assertEqual(len(result.selected_documents), 2)
        self.assertEqual(len(coreset_lines), 2)
        self.assertEqual(len(result.projection), 4)
        self.assertEqual(
            {document.label for document in result.selected_documents},
            {"positive", "negative"},
        )
        self.assertFalse(manifest["source_dataset_modified"])
        self.assertEqual(manifest["use_case"], "traditional_nlp")
        self.assertTrue(all(copied_inputs))
        self.assertIn("label_groups", {stage for stage, _, _ in updates})
        self.assertIn("artifacts", {stage for stage, _, _ in updates})

    def test_materializes_coreset_scores_and_manifest_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_writer = ParquetShardWriter(root / "clean-data", 10)
            records = []
            for index, text in enumerate(("पहिलो", "दोस्रो", "तेस्रो", "चौथो")):
                record = PreparedRecord(
                    doc_id=f"doc-{index}",
                    text=text,
                    source="fixture",
                    source_file="fixture.jsonl",
                    source_row=index,
                    text_sha256=f"hash-{index}",
                    token_count=1,
                    character_count=len(text),
                    devanagari_ratio=1.0,
                )
                records.append(record)
                source_writer.add(record)
            clean_paths = tuple(source_writer.close())
            source_bytes = clean_paths[0].read_bytes()
            embedding_path = root / "embeddings.npz"
            np.savez(
                embedding_path,
                embeddings=np.array([[0.0], [0.1], [5.0], [10.0]], dtype=np.float32),
                doc_ids=np.array([record.doc_id for record in records]),
            )

            result = run_d2_pruning_on_parquet(
                clean_paths,
                len(records),
                root / "d2",
                D2PruningConfig(
                    enabled=True,
                    retention_fraction=0.5,
                    n_neighbors=1,
                    graph_backend="exact",
                    embeddings_path=embedding_path,
                ),
                shard_rows=10,
                batch_size=2,
            )
            selected = list(iter_parquet_records(result.coreset_paths, 10))
            scores = list(iter_parquet_records((result.scores_path,), 10))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            source_unchanged = clean_paths[0].read_bytes() == source_bytes

        self.assertEqual(len(selected), 2)
        self.assertEqual(len(scores), 4)
        self.assertEqual(sum(bool(row["selected"]) for row in scores), 2)
        self.assertEqual(manifest["method"], "D2 Pruning")
        self.assertEqual(manifest["selected_documents"], 2)
        self.assertFalse(manifest["source_dataset_modified"])
        self.assertTrue(source_unchanged)


if __name__ == "__main__":
    unittest.main()
