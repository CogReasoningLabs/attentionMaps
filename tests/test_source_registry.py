"""Registered source selection preserves existing numerical results and artifacts."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from attention_maps.explorer.semantic_source import source_settings
from attention_maps.explorer.source_registry import load_source_registry, registered_report
from scripts.cluster_dataset import main, parse_arguments
from tests.test_semantic_analysis import FixtureEncoder


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / "sources.yaml"
        self.central = self.root / "embeddings.yaml"
        self.source = self.root / "data.jsonl"
        self.source.write_text(''.join(json.dumps({"text": value}) + '\n' for value in ["A", "B", "C", "D"]))
        self.document = {"version": 1, "sources": {
            "huggingface": {"hf-news": {"label": "News", "dataset": "owner/news", "config": "ne", "split": "train",
                                           "revision": "commit", "training_schema": "pretraining", "text_columns": ["body"]}},
            "kaggle": {"kg-comments": {"dataset": "owner/comments/versions/2", "dataset_file": "brb.xlsx",
                                       "training_schema": "task_specific_supervised", "task_name": "classification",
                                       "field_mapping": {"text": "Cleaned", "label": "Category"}}},
            "local": {"local-text": {"label": "Local text", "local": "data.jsonl", "training_schema": "pretraining",
                                      "text_columns": ["text"]}},
        }}
        self.write_registry()
        self.central.write_text("provider: huggingface\ndataset: old/chat\nconfig: old\nsplit: test\n"
                                "languages: [en]\nrow_filters: {language: [en]}\ntraining_schema: instruction_finetuning\n"
                                "field_mapping: {messages: turns}\nfield_parsers: {text: join_strings}\n"
                                "text_record_unit: blank_line\ntext_columns: [stale]\nmodel: nepali-bert\n"
                                "batch_size: 2\nclusters: 2\ncluster_batch_size: 4\ncluster_epochs: 1\nseed: 42\n"
                                "source_registry: sources.yaml\noutput_dir: runs/registered\n")

    def write_registry(self):
        self.registry.write_text(yaml.safe_dump(self.document, sort_keys=False), encoding="utf-8")

    def args(self, identifier="local-text", *extra):
        return parse_arguments(["run", "--settings", str(self.central), "--source-id", identifier, *extra])

    def test_all_three_providers_resolve_complete_definitions_and_preserve_models(self):
        for identifier, provider in (("hf-news", "huggingface"), ("kg-comments", "kaggle"), ("local-text", "local")):
            with self.subTest(source=identifier):
                args = self.args(identifier, "--model", "embeddinggemma", "--clusters", "3")
                selection = source_settings(args)
                self.assertEqual(selection["provider"], provider)
                self.assertEqual(args.model, "embeddinggemma")
                self.assertEqual(args.clusters, 3)
                self.assertEqual(selection["row_filters"], {})
                self.assertIsNone(selection["languages"])
                self.assertEqual(args.field_parsers, {})
                self.assertEqual(args.text_record_unit, "line")
                self.assertNotIn("messages", args.field_mapping)
        self.assertEqual(source_settings(self.args("local-text"))["local"], str(self.source))
        self.assertIsNone(source_settings(self.args("local-text"))["dataset"])
        self.assertEqual(source_settings(self.args("hf-news"))["revision"], "commit")
        self.assertEqual(source_settings(self.args("kg-comments"))["dataset_file"], "brb.xlsx")
        self.assertEqual(self.args("kg-comments").task_name, "classification")

    def test_yaml_selected_id_and_cli_registry_paths(self):
        self.central.write_text(self.central.read_text() + "source_id: local-text\n")
        args = parse_arguments(["run", "--settings", str(self.central)])
        self.assertEqual(source_settings(args)["local"], str(self.source))
        args = parse_arguments(["run", "--source-registry", str(self.registry), "--source-id", "local-text",
                                "--output-dir", str(self.root / "separate")])
        self.assertEqual(source_settings(args)["provider"], "local")

    def test_invalid_and_ambiguous_selections_fail_before_model_or_source_access(self):
        with patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model access")), \
             patch("scripts.cluster_dataset.resolve_source", side_effect=AssertionError("No source access")):
            for flags in (["--source-id", "missing"], ["--source-id", "local-text", "--dataset", "other/data"],
                          ["--source-id", "local-text", "--field-map", "text:old"]):
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["run", "--settings", str(self.central), *flags]), 1)
        bad_entries = [
            ("local", "local-text", {"split": "train"}),
            ("local", "local-text", {"provider": "kaggle"}),
            ("local", "local-text", {"model": "nepberta"}),
            ("kaggle", "kg-comments", {"dataset_file": None}),
            ("huggingface", "hf-news", {"config": "*"}),
            ("huggingface", "hf-news", {"local": "file.txt"}),
        ]
        original = copy.deepcopy(self.document)
        for provider, identifier, changed in bad_entries:
            with self.subTest(changed=changed):
                self.document = copy.deepcopy(original)
                self.document["sources"][provider][identifier].update(changed)
                self.write_registry()
                with self.assertRaises(ValueError):
                    load_source_registry(self.registry)

    def test_duplicate_ids_and_yaml_keys_are_rejected(self):
        self.document["sources"]["local"]["hf-news"] = {"local": "data.jsonl"}
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "Duplicate source ID"):
            load_source_registry(self.registry)
        self.registry.write_text("version: 1\nsources:\n  local:\n    repeated: {local: a.txt}\n    repeated: {local: b.txt}\n")
        with self.assertRaisesRegex(ValueError, "Duplicate source registry key"):
            load_source_registry(self.registry)

    def test_listing_never_downloads_data_or_loads_models(self):
        with patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model access")), \
             patch("scripts.cluster_dataset.resolve_source", side_effect=AssertionError("No source access")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["sources", "--source-registry", str(self.registry)]), 0)
        self.assertEqual({item["provider"] for item in json.loads(output.getvalue())}, {"huggingface", "kaggle", "local"})

    def test_registered_run_matches_direct_run_and_preserves_old_artifacts(self):
        baseline, registered = self.root / "baseline", self.root / "runs/registered"
        with patch("scripts.cluster_dataset.DocumentEncoder", return_value=FixtureEncoder()), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--provider", "local", "--local", str(self.source), "--training-schema", "pretraining",
                                   "--text-column", "text", "--batch-size", "2", "--clusters", "2", "--cluster-batch-size", "4",
                                   "--cluster-epochs", "1", "--output-dir", str(baseline)]), 0)
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.rglob('*') if p.is_file()}
            self.assertEqual(main(["run", "--settings", str(self.central), "--source-id", "local-text"]), 0)
        old = json.loads((baseline / "report.json").read_text())
        new = json.loads((registered / "report.json").read_text())
        self.assertEqual(old["corpus_fingerprint"], new["corpus_fingerprint"])
        self.assertEqual(old["instance_definition"], new["instance_definition"])
        self.assertEqual((baseline / "embeddings.f32").read_bytes(), (registered / "embeddings.f32").read_bytes())
        np.testing.assert_array_equal(np.load(baseline / "labels.npy"), np.load(registered / "labels.npy"))
        np.testing.assert_array_equal(np.load(baseline / "coordinates.npy"), np.load(registered / "coordinates.npy"))
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline.rglob('*') if p.is_file()})
        entry = load_source_registry(self.registry)["local-text"]
        self.assertTrue(registered_report(entry, new))
        self.assertFalse(registered_report(entry, old))
        self._check_saved_registry_ui(registered, entry)
        self.document["sources"]["local"]["local-text"]["text_record_unit"] = "blank_line"
        self.write_registry()
        self.assertFalse(registered_report(load_source_registry(self.registry)["local-text"], new))

    def _check_saved_registry_ui(self, registered, entry):
        from streamlit.testing.v1 import AppTest
        self.source.unlink()  # Viewing a saved run must not require the original dataset.
        app_path = self.root / "saved_app.py"
        app_path.write_text("import streamlit as st\nfrom apps.components.semantic_results import render_semantic_results\n"
                            f"render_semantic_results(st, default_root={str(registered)!r})\n")
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in registered.rglob('*') if p.is_file()}
        with patch.dict(os.environ, {"DATASET_SOURCE_REGISTRY": str(self.registry)}), \
             patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No UI inference")):
            app = AppTest.from_file(str(app_path), default_timeout=25).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(x.value for x in app.selectbox if x.label == "Dataset"), entry["id"])
            self.assertTrue(app.get("plotly_chart"))
            self.assertTrue(any(x.label == "Cosine similarity" for x in app.metric))
            self.registry.unlink()
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.get("plotly_chart"))
            self.assertTrue(any("Cannot read source registry" in x.value for x in app.warning))
        self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in registered.rglob('*') if p.is_file()})

    def test_streamlit_selects_unprocessed_sources_and_produces_working_commands(self):
        from streamlit.testing.v1 import AppTest
        app_path = self.root / "app.py"
        app_path.write_text("import streamlit as st\nfrom apps.components.semantic_results import render_semantic_results\n"
                            f"render_semantic_results(st, default_root={str(self.root / 'runs')!r})\n")
        with patch.dict(os.environ, {"DATASET_SOURCE_REGISTRY": str(self.registry)}), \
             patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No UI inference")), \
             patch("attention_maps.explorer.semantic_source.resolve_source", side_effect=AssertionError("No UI downloads")):
            app = AppTest.from_file(str(app_path), default_timeout=25).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            provider = next(x for x in app.selectbox if x.label == "Source type")
            self.assertEqual(provider.options, ["Hugging Face", "Kaggle", "Local"])
            for kind, identifier in (("local", "local-text"), ("kaggle", "kg-comments"), ("huggingface", "hf-news")):
                next(x for x in app.selectbox if x.label == "Source type").set_value(kind).run()
                self.assertFalse(app.exception)
                command = next(x.value for x in app.code if "--source-id" in x.value)
                args = parse_arguments(shlex.split(command)[2:])
                self.assertEqual(args.source_id, identifier)
                self.assertEqual(source_settings(args)["provider"], kind)
            next(x for x in app.selectbox if x.label == "Model for new run").set_value("embeddinggemma").run()
            command = next(x.value for x in app.code if "--source-id" in x.value)
            self.assertEqual(parse_arguments(shlex.split(command)[2:]).model, "embeddinggemma")
            self.assertFalse((self.root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
