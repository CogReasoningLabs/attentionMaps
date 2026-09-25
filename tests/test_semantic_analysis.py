import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from attention_maps.explorer.inspection import file_signatures, inspect_dataset
from attention_maps.explorer.semantic_artifacts import build_run, load_run, open_embeddings, pair_similarity
from attention_maps.explorer.semantic_models import normalize_rows, text_chunks
from attention_maps.explorer.semantic_sampling import density_scores, select_indices, semdedup_survivors, curate_run
from scripts.cluster_dataset import parser, main


class FixtureEncoder:
    dimension = 4
    metadata = {"id": "test/fixture", "revision": "test", "preset": "fixture", "dimension": 4,
                "task": "clustering", "prompt": "", "role": "test fixture"}

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(4)
            vector[ord(text[0]) % 4] = 1
            vectors.append(vector)
        return np.array(vectors, dtype="float32"), np.ones(len(texts), dtype=int)


def make_run(root, *, max_records=None):
    source = root / "source.jsonl"
    rows = [{"text": value, "language": "ne", "extra": i} for i, value in enumerate(["A", "A", "B", "C", "", "D", "B", "C"])]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    inventory = inspect_dataset(file_signatures((source,)))
    args = parser().parse_args(["run", "--local", str(source), "--clusters", "3", "--cluster-batch-size", "3",
                                "--batch-size", "2", "--output-dir", str(root / "run")])
    args.max_records = max_records
    with contextlib.redirect_stderr(io.StringIO()):
        return build_run(inventory, ["text"], FixtureEncoder(), args)


class SemanticMathTests(unittest.TestCase):
    def test_chunking_covers_all_tokens_and_obeys_prompt_limit(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False, **kwargs):
                return ([0] if add_special_tokens else []) + [ord(x) for x in text] + ([0] if add_special_tokens else [])

            def decode(self, ids, **kwargs):
                return "".join(chr(x) for x in ids)

        tokenizer = Tokenizer()
        source = "नेपालमा विज्ञानको अध्ययन " * 20
        chunks = list(text_chunks(tokenizer, source, 32, "task: "))
        self.assertEqual("".join(text for text, _ in chunks), source)
        self.assertEqual(sum(weight for _, weight in chunks), len(source))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(tokenizer.encode("task: " + text, add_special_tokens=True)) <= 32 for text, _ in chunks))

    def test_normalization_rejects_invalid_vectors(self):
        np.testing.assert_allclose(normalize_rows([[3, 4]]), [[0.6, 0.8]])
        for value in ([[0, 0]], [[float("nan"), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                normalize_rows(value)

    def test_density_sketch_and_inverse_sampling_favor_sparse_regions(self):
        vectors = np.array([[1.0, 0.0]] * 80 + [[0.0, 1.0]] * 20, dtype="float32")
        with tempfile.TemporaryDirectory() as directory:
            scores = density_scores(vectors, Path(directory), rows=128, bins=4096, width=0.1, batch_size=17)
            self.assertGreater(float(scores[0]), float(scores[-1]))
            selections = [select_indices(100, 10, seed, scores=scores) for seed in range(100)]
            rare = sum(np.count_nonzero(ids >= 80) for ids in selections)
            self.assertGreater(rare, 350)  # Uniform B1 expectation is only 200 / 1000.
            first = select_indices(100, 20, 42, scores=scores)
            np.testing.assert_array_equal(first, select_indices(100, 20, 42, scores=scores))
            self.assertEqual(len(np.unique(first)), 20)

    def test_semdedup_is_cluster_local_and_prefers_farther_example(self):
        vectors = np.array([[1, 0], [1, 0], [0, 1], [1, 0]], dtype="float32")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            survivors = semdedup_survivors(vectors, np.array([0, 0, 0, 1]), np.array([0.1, 0.9, 0.2, 0.1]), root,
                                          threshold=0.95, block_size=1)
            np.testing.assert_array_equal(survivors, [1, 2, 3])
            self.assertEqual(np.load(root / "semdedup_neighbor.npy")[0], 1)
            all_rows = semdedup_survivors(vectors, np.array([0, 0, 0, 1]), np.ones(4), root, threshold=1, block_size=2)
            np.testing.assert_array_equal(all_rows, np.arange(4))

    def test_semdedup_compares_all_earlier_points_not_only_survivors(self):
        angle = np.deg2rad([0, 20, 40])
        vectors = np.column_stack((np.cos(angle), np.sin(angle))).astype("float32")
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            survivors = semdedup_survivors(vectors, np.zeros(3), np.array([3, 2, 1]), Path(directory), threshold=0.9, block_size=2)
            np.testing.assert_array_equal(survivors, [0])


class SemanticPipelineTests(unittest.TestCase):
    def test_full_corpus_artifacts_similarity_and_source_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report = make_run(Path(directory))
            self.assertEqual(report["source_records_scanned"], 8)
            self.assertEqual(report["embedded_records"], 7)
            self.assertEqual(report["empty_records_skipped"], 1)
            self.assertEqual(report["scope"], "complete_selection")
            self.assertEqual(sum(report["clustering"]["cluster_sizes"]), 7)
            self.assertEqual(np.load(root / "coordinates.npy").shape, (7, 3))
            self.assertEqual(open_embeddings(*load_run(root)).shape, (7, 4))
            self.assertAlmostEqual(pair_similarity(root, 0, 1)["cosine_similarity"], 1)
            self.assertAlmostEqual(pair_similarity(root, 0, 2)["cosine_similarity"], 0)
            self.assertEqual(pair_similarity(root, 4, 6)["first"]["source_row"], 5)
            for a, b in ((-1, 2), (0, 7)):
                with self.assertRaises(ValueError):
                    pair_similarity(root, a, b)
            (root / "embeddings.f32").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_run(root)

    def test_prefix_is_explicit_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report = make_run(Path(directory), max_records=4)
            self.assertEqual(report["scope"], "limited_prefix")
            self.assertEqual(report["embedded_records"], 4)
            original = (root / "report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                make_run(Path(directory))
            self.assertEqual((root / "report.json").read_bytes(), original)

    def test_sampling_exports_whole_records_and_votes_for_all_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, report = make_run(base)
            for method in ("random", "density", "semdedup"):
                with self.subTest(method=method):
                    args = parser().parse_args(["sample", "--run", str(root), "--method", method,
                        "--sample-fraction", "1", "--sampling-runs", "2", "--output-dir", str(base / method)])
                    with contextlib.redirect_stderr(io.StringIO()):
                        output, result = curate_run(args)
                    exported = [json.loads(line) for line in (output / "sample-1.jsonl").read_text().splitlines()]
                    self.assertTrue(all("extra" in item for item in exported))
                    sample = result["language_status"]["sampling"]
                    self.assertEqual(sample["population_rows"], 7)
                    self.assertEqual(sample["unique_sampled_records"], len(exported))
                    self.assertEqual(result["language_status"]["language_coverage"], "Nepali-only")
                    self.assertEqual(result["language_status"]["voting"]["language_coverage"]["winning_votes"], 2)
                    if method == "semdedup":
                        self.assertLess(len(exported), 7)
                        self.assertEqual(len(exported), result["eligible_records"])

    def test_failed_cluster_run_does_not_publish_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("attention_maps.explorer.semantic_artifacts.cluster_embeddings", side_effect=ValueError("test failure")):
                with self.assertRaisesRegex(ValueError, "test failure"):
                    make_run(Path(directory))
            self.assertFalse((Path(directory) / "run").exists())
            self.assertFalse(list(Path(directory).glob(".run-*")))

    def test_pair_cli_never_loads_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            with patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model load")), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["pair", "--run", str(root), "--row-a", "0", "--row-b", "1"]), 0)
            self.assertEqual(json.loads(output.getvalue())["cosine_similarity"], 1)

    def test_streamlit_reads_saved_run_without_inference_or_source(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            (Path(directory) / "source.jsonl").unlink()
            with patch.dict(os.environ, {"DATASET_EMBEDDING_RUNS": str(root)}), \
                 patch("attention_maps.explorer.semantic_models.DocumentEncoder", side_effect=AssertionError("No inference")), \
                 patch("attention_maps.explorer.semantic_artifacts.cluster_embeddings", side_effect=AssertionError("No clustering")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(next(x.value for x in app.metric if x.label == "Cosine similarity"), "1.000000")
                self.assertTrue(app.get("plotly_chart"))
                next(x for x in app.number_input if x.label == "Record B ID").set_value(2).run()
                self.assertFalse(app.exception)
                self.assertEqual(next(x.value for x in app.metric if x.label == "Cosine similarity"), "0.000000")


if __name__ == "__main__":
    unittest.main()
