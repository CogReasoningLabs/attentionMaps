import contextlib
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from attention_maps.explorer.semantic_variants import discover_clusterings, load_clustering
from attention_maps.explorer.semantic_wordclouds import content_text, configured_stopwords, save_wordclouds, words
from apps.components.semantic_results import plot_records
from scripts.cluster_dataset import main
from tests.test_semantic_analysis import make_run


class ClusterVariantTests(unittest.TestCase):
    def recluster(self, root, clusters, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["recluster", "--run", str(root), "--clusters", str(clusters), *extra])

    def test_recluster_reuses_embeddings_and_projection_without_source_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report = make_run(Path(directory))
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
            (Path(directory) / "source.jsonl").unlink()
            with patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model")), \
                 patch("scripts.cluster_dataset.resolve_source", side_effect=AssertionError("No source")), \
                 patch("sklearn.decomposition.IncrementalPCA", side_effect=AssertionError("Reuse projection")):
                self.assertEqual(self.recluster(root, 2), 0)
                self.assertEqual(self.recluster(root, 1), 0)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
            self.assertEqual(len(discover_clusterings(root)), 3)
            for k in (1, 2):
                path = root / "clusterings" / f"k{k}-seed42"
                result = load_clustering(root, report, path)
                self.assertEqual(result["clustering"]["requested_clusters"], k)
                self.assertFalse((path / "embeddings.f32").exists())
                for filename in ("projection.npz", "coordinates.npy"):
                    self.assertEqual((root / filename).read_bytes(), (path / filename).read_bytes())
                summary = json.loads((path / "wordclouds/summary.json").read_text())
                self.assertEqual(sum(x["records"] for x in summary["clusters"].values()), 7)
                cloud = summary["clusters"]["0"]
                if k == 1:
                    self.assertEqual(dict(cloud["top_words"]), {"b": 2, "c": 2, "d": 1})
                    self.assertTrue((path / "wordclouds" / cloud["image"]).is_file())
                points = plot_records(root, report, clustering_root=path)
                np.testing.assert_array_equal(points["cluster"], np.load(path / "labels.npy").astype(str))

    def test_invalid_or_duplicate_results_preserve_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report = make_run(Path(directory))
            self.assertEqual(self.recluster(root, 2), 0)
            path = root / "clusterings/k2-seed42"
            saved = (path / "report.json").read_bytes()
            for k, extra in ((2, []), (8, []), (2, ["--name", "../escape"]), (2, ["--seed", "-1"])):
                self.assertEqual(self.recluster(root, k, *extra), 1)
            self.assertEqual((path / "report.json").read_bytes(), saved)
            self.assertEqual(self.recluster(root, 2, "--name", "another-trial"), 0)
            wrong = dict(report, corpus_fingerprint="wrong corpus")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_clustering(root, wrong, path)
            with patch("attention_maps.explorer.semantic_variants.save_wordclouds", side_effect=ValueError("failed")):
                self.assertEqual(self.recluster(root, 1), 1)
            self.assertFalse((root / "clusterings/k1-seed42").exists())
            self.assertFalse(list((root / "clusterings").glob(".k1-*")))

    def test_backfill_original_and_variant_preserves_existing_assignments(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            self.assertEqual(self.recluster(root, 2), 0)
            (Path(directory) / "source.jsonl").unlink()
            for target, extra in ((root, []), (root / "clusterings/k2-seed42", ["--clustering", "k2-seed42"])):
                shutil.rmtree(target / "wordclouds")
                before = {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()}
                command = ["wordclouds", "--run", str(root), *extra]
                with patch("attention_maps.explorer.semantic_variants.cluster_embeddings", side_effect=AssertionError("No clustering")), \
                     patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model")), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(command), 0)
                    self.assertEqual(main(command), 1)  # Never overwrite completed clouds.
                self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir() if p.is_file()})
                self.assertTrue((target / "wordclouds/summary.json").is_file())
                self.assertTrue(list((target / "wordclouds").glob("cluster-*.png")))
            with patch.dict(os.environ, {"DATASET_EMBEDDING_RUNS": str(root)}):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                summary = json.loads((root / "wordclouds/summary.json").read_text())
                cluster = next(int(key) for key, item in summary["clusters"].items() if item["image"])
                next(x for x in app.selectbox if x.label == "Browse cluster").set_value(cluster).run()
                self.assertTrue(any(x.label == "Download cluster word cloud" for x in app.get("download_button")))
                self.assertFalse(any("no saved word clouds" in x.value for x in app.info))

    def test_backfill_failure_does_not_publish_partial_clouds(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            shutil.rmtree(root / "wordclouds")
            with patch("attention_maps.explorer.semantic_variants.save_wordclouds", side_effect=ValueError("failed")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["wordclouds", "--run", str(root)]), 1)
            self.assertFalse((root / "wordclouds").exists())
            self.assertFalse(list(root.glob(".wordclouds-*")))

    def test_screenshot_stopwords_are_excluded_and_topic_words_remain(self):
        excluded = "त नि छु छौं छौँ छन छिन् थिइन् होस् उसले उनले उनको उहाँले मैले गरेका गरेर गरी भन्दै भनेर उक्त सो रुपमा अनी रे गर्नुहोस"
        topic = "नेपाल नेपाली सरकार सरकारले खेल विद्यालय कृषि काम"
        self.assertEqual(list(words(excluded + " " + topic)), topic.split())
        self.assertEqual(list(words("﻿मैले, उनले। गरेकी\nनेपाल")), ["नेपाल"])

    def test_refresh_applies_changed_stopwords_without_reclustering(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            original = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
            updated_stopwords = configured_stopwords() | {"b"}
            with patch("attention_maps.explorer.semantic_wordclouds.configured_stopwords", return_value=updated_stopwords), \
                 patch.dict(os.environ, {"DATASET_EMBEDDING_RUNS": str(root)}), \
                 patch("attention_maps.explorer.semantic_variants.cluster_embeddings", side_effect=AssertionError("No clustering")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
                self.assertTrue(any("different stopword list" in x.value for x in app.warning))
                self.assertTrue(any("--refresh" in x.value for x in app.code))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["wordclouds", "--run", str(root), "--refresh"]), 0)
                app.run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertFalse(any("different stopword list" in x.value for x in app.warning))
            self.assertEqual(original, {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()})
            summary = json.loads((root / "wordclouds/summary.json").read_text())
            self.assertTrue(all(not (set(dict(item["top_words"])) & updated_stopwords) for item in summary["clusters"].values()))

    def test_failed_refresh_restores_previous_clouds(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            before = {p.name: p.read_bytes() for p in (root / "wordclouds").iterdir()}
            rename = Path.rename

            def fail_publication(path, target):
                if path.name == "wordclouds" and path.parent.name.startswith(".wordclouds-"):
                    raise OSError("simulated publication failure")
                return rename(path, target)

            with patch.object(Path, "rename", fail_publication), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["wordclouds", "--run", str(root), "--refresh"]), 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "wordclouds").iterdir()})
            self.assertFalse(list(root.glob(".wordclouds-*")))

    def test_nepali_tokens_and_schema_content_exclude_metadata_and_roles(self):
        self.assertEqual(list(words("नेपालमा कृषि, खेती। र को १२३ Hello HELLO")), ["नेपालमा", "कृषि", "खेती", "hello", "hello"])
        conversation = {"messages": [{"role": "user", "content": "कृषि"}, {"role": "assistant", "content": "खेती"}], "id": "secret"}
        self.assertEqual(content_text(conversation, "instruction_finetuning", "unused"), "कृषि\nखेती")
        self.assertEqual(content_text({"text": "खेती", "label": "category", "task": "topic"}, "task_specific_supervised", ""), "खेती")
        self.assertEqual(content_text({"prompt": "कृषि", "chosen": [{"role": "assistant", "content": "खेती"}], "rejected": "खेल"}, "preference_tuning", ""), "कृषि\nखेती\nखेल")

    def test_word_counts_cover_all_records_not_only_the_first_page(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = sqlite3.connect(root / "records.sqlite")
            connection.execute("CREATE TABLE records (embedding_id INTEGER, text TEXT)")
            connection.executemany("INSERT INTO records VALUES (?, ?)", [(i, "नेपाल कृषि" if i < 29 else "अन्तिम खेती") for i in range(30)])
            connection.commit()
            connection.close()
            with patch("attention_maps.explorer.semantic_wordclouds.find_font", return_value=None):
                save_wordclouds(root, root, np.zeros(30, dtype=int))
            summary = json.loads((root / "wordclouds/summary.json").read_text())
            cloud = summary["clusters"]["0"]
            self.assertEqual(cloud["records"], 30)
            self.assertEqual(cloud["counted_tokens"], 60)
            self.assertEqual(dict(cloud["top_words"]), {"नेपाल": 29, "कृषि": 29, "अन्तिम": 1, "खेती": 1})
            self.assertIsNone(cloud["image"])
            self.assertIn("Devanagari", cloud["note"])

    def test_streamlit_switches_saved_cluster_counts_without_computation(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            root, _ = make_run(Path(directory))
            self.assertEqual(self.recluster(root, 2), 0)
            self.assertEqual(self.recluster(root, 1), 0)
            (Path(directory) / "source.jsonl").unlink()
            with patch.dict(os.environ, {"DATASET_EMBEDDING_RUNS": str(root)}), \
                 patch("attention_maps.explorer.semantic_variants.cluster_embeddings", side_effect=AssertionError("No clustering")), \
                 patch("attention_maps.explorer.semantic_wordclouds.save_wordclouds", side_effect=AssertionError("No counting")), \
                 patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No embedding")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
                for k in (2, 1):
                    next(x for x in app.selectbox if x.label == "Saved clustering result").set_value(root / "clusterings" / f"k{k}-seed42").run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertEqual(next(x.value for x in app.metric if x.label == "Requested clusters"), str(k))
                    self.assertTrue(app.get("plotly_chart"))
                    for view, trace_type in (("2D", "scatter"), ("3D", "scatter3d")):
                        next(x for x in app.radio if x.label == "Visualization dimensions").set_value(view).run()
                        self.assertFalse(app.exception)
                        self.assertFalse(app.error)
                        figure = json.loads(app.get("plotly_chart")[0].proto.spec)
                        self.assertTrue(all(trace["type"] == trace_type for trace in figure["data"]))
                        self.assertEqual(figure["data"][-1]["name"], "Cluster centroids")
                        self.assertTrue(any(x.value.startswith(f"Variance retained in {view}:") for x in app.caption))
                    summary = json.loads((root / "clusterings" / f"k{k}-seed42" / "wordclouds/summary.json").read_text())
                    cluster = next(int(key) for key, item in summary["clusters"].items() if item["image"])
                    next(x for x in app.selectbox if x.label == "Browse cluster").set_value(cluster).run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any(item.label == "Download cluster word cloud" for item in app.get("download_button")))
                    self.assertEqual(next(x.value for x in app.metric if x.label == "Cosine similarity"), "1.000000")


if __name__ == "__main__":
    unittest.main()
