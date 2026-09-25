from attention_maps.explorer.inspection_runs import load_inspection_report
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.explorer.catalog import DatasetSpec
from attention_maps.explorer.huggingface import (
    discover_huggingface_dataset, inspect_huggingface_configuration,
    inspect_huggingface_selection, selected_source_spec,
)
from attention_maps.explorer.inspection import sample_huggingface_rows
from attention_maps.explorer.language_status import analyze_language_status, inventory_language_status
from tests.inspection_helpers import isolated_main as main


class LanguageStatusTests(unittest.TestCase):
    def test_script_does_not_invent_a_language(self):
        result = analyze_language_status([{"text": "नेपाल सुन्दर छ।"}])
        self.assertEqual(result["script"], "Unknown")
        self.assertEqual(result["language_coverage"], "Unknown")

    def test_configuration_overrides_repository_wide_languages(self):
        result = inventory_language_status(
            {"format": "huggingface", "dataset_config": "ne", "declared_languages": ["ne", "en", "hi"]},
            [{"text": "नेपाल सुन्दर छ।"}],
        )
        self.assertEqual(result["language_coverage"], "Nepali-only")
        self.assertEqual(result["language_basis"], "Hugging Face configuration/split")

    def test_bilingual_labels_and_nested_conversation_script(self):
        records = [
            {"language": "npi", "messages": [{"role": "user", "content": "नेपाल सुन्दर छ।"}]},
            {"language": "eng", "messages": [{"role": "assistant", "content": "Hello world"}]},
        ]
        result = analyze_language_status(records, text_columns=("messages",))
        self.assertEqual(result["language_coverage"], "Bilingual (Nepali-English)")
        self.assertEqual(result["script"], "Mixed (Nepali + English)")
        self.assertEqual(result["observed_language_counts"], {"ne": 1, "en": 1})

    def test_romanized_requires_nepali_evidence_and_sample_is_bounded(self):
        records = [{"text": "namaste " * 100}]
        unknown = analyze_language_status(records, max_characters=20)
        declared = analyze_language_status(records, declared_languages=("ne",), max_characters=20)
        self.assertEqual(unknown["script"], "Unknown")
        self.assertEqual(declared["script"], "Unknown")
        self.assertEqual(declared["language_coverage"], "Unknown")
        labelled = analyze_language_status([{**records[0], "language_code": "npi"}], max_characters=20)
        self.assertEqual(labelled["script"], "Romanized")
        self.assertEqual(declared["sampled_characters"], 20)
        empty = analyze_language_status([{"text": "123 !"}])
        self.assertEqual(empty["script"], "Unknown")


class HuggingFaceSelectionTests(unittest.TestCase):
    def test_discovery_pins_revision_and_configuration_lists_real_splits(self):
        info = SimpleNamespace(
            sha="frozen-commit", card_data=SimpleNamespace(to_dict=lambda: {"language": ["ne", "en"]}),
            siblings=[SimpleNamespace(rfilename="ne/train-0.parquet", size=100),
                      SimpleNamespace(rfilename="ne/test-0.parquet", size=50),
                      SimpleNamespace(rfilename="en/train.parquet", size=900)],
        )
        builder = SimpleNamespace(
            config=SimpleNamespace(data_files={
                "train": ["hf://datasets/owner/data@frozen-commit/ne/train-0.parquet"],
                "test": ["hf://datasets/owner/data@frozen-commit/ne/test-0.parquet"],
            }),
            info=SimpleNamespace(features={"text": "string"}, splits={
                "train": SimpleNamespace(num_examples=10, num_bytes=300),
                "test": SimpleNamespace(num_examples=5, num_bytes=150),
            }),
        )
        with patch("huggingface_hub.HfApi.dataset_info", return_value=info), \
             patch("datasets.get_dataset_config_names", return_value=["ne", "en"]) as names, \
             patch("datasets.load_dataset_builder", return_value=builder) as loader:
            catalog = discover_huggingface_dataset("owner/data", revision="release")
            configuration = inspect_huggingface_configuration(catalog, "ne")
        self.assertEqual(names.call_args.kwargs["revision"], "frozen-commit")
        self.assertEqual(loader.call_args.kwargs["revision"], "frozen-commit")
        self.assertEqual(configuration["splits"]["test"]["file_formats"]["extensions"], [".parquet"])
        inventory = inspect_huggingface_selection(configuration, "test")
        self.assertEqual(inventory["rows"], 5)
        self.assertEqual(inventory["hub_file_bytes"], 50)
        self.assertEqual(inventory["memory_bytes"], 150)
        self.assertEqual(inventory["dataset_shards"], builder.config.data_files["test"])

    def test_second_shard_has_its_own_counts_and_is_the_only_sampled_file(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.parquet", root / "second.parquet"
            pq.write_table(pa.table({"text": ["from first"]}), first)
            pq.write_table(pa.table({"text": ["second A", "second B"]}), second)
            configuration = self.configuration(first, second)
            inventory = inspect_huggingface_selection(configuration, "train", shards=[str(second)])
            self.assertEqual(inventory["rows"], 2)
            self.assertEqual(inventory["hub_file_bytes"], second.stat().st_size)
            self.assertEqual(inventory["files"], 1)
            # Actual datasets streaming reads the selected local Parquet only.
            inventory["dataset_id"] = "parquet"
            inventory["dataset_revision"] = None
            records = sample_huggingface_rows(inventory, 10, 42, ("text",))
            self.assertEqual({row["text"] for row in records}, {"second A", "second B"})

    @staticmethod
    def configuration(first, second):
        return {
            "dataset_id": "owner/data", "config": "default", "revision": "frozen",
            "languages": ["en"], "schema": [{"column": "text", "type": "string"}],
            "splits": {"train": {
                "rows": 3, "memory_bytes": 100,
                "shards": [{"path": str(path), "name": path.name, "bytes": path.stat().st_size}
                           for path in (first, second)],
            }},
        }

    def test_empty_or_foreign_shards_fail_and_selection_changes_workspace_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.parquet", Path(directory) / "b.parquet"
            first.touch(); second.touch()
            configuration = self.configuration(first, second)
            spec = DatasetSpec("source", "Source", "dataset", (), format="huggingface", dataset_id="owner/data")
            one = selected_source_spec(spec, configuration, "train", [str(first)])
            two = selected_source_spec(spec, configuration, "train", [str(second)])
            self.assertNotEqual(one.key, two.key)
            self.assertEqual(two.dataset_shards, (str(second),))
            for selection in ([], ["unknown.parquet"]):
                with self.assertRaises(ValueError):
                    inspect_huggingface_selection(configuration, "train", shards=selection)

    def test_non_parquet_subset_does_not_reuse_full_split_counts(self):
        configuration = {
            "dataset_id": "owner/data", "config": "default", "revision": "frozen",
            "schema": [{"column": "text", "type": "string"}],
            "splits": {"train": {"rows": 1000, "memory_bytes": 9999, "shards": [
                {"path": "first.jsonl", "name": "first.jsonl", "bytes": 100},
                {"path": "second.jsonl", "name": "second.jsonl", "bytes": 200},
            ]}},
        }
        inventory = inspect_huggingface_selection(configuration, "train", shards=["second.jsonl"])
        self.assertIsNone(inventory["rows"])
        self.assertIsNone(inventory["memory_bytes"])
        self.assertEqual(inventory["hub_file_bytes"], 200)

    def test_viewer_counts_require_the_exact_selected_revision(self):
        from attention_maps.explorer.huggingface import _viewer_split_metadata

        config = {"dataset_id": "owner/data", "config": "ne", "revision": "selected"}
        payload = {"size": {"splits": [{"config": "ne", "split": "train", "num_rows": 123}]}}
        for revision, expected in (("different", None), ("selected", 123)):
            response = io.BytesIO(json.dumps(payload).encode())
            response.headers = {"X-Revision": revision}
            with patch("attention_maps.explorer.huggingface.urllib.request.urlopen", return_value=response):
                result = _viewer_split_metadata(config, "train", "private-token")
            self.assertEqual(None if result is None else result["num_rows"], expected)
        payload["partial"] = True
        response = io.BytesIO(json.dumps(payload).encode())
        response.headers = {"X-Revision": "selected"}
        with patch("attention_maps.explorer.huggingface.urllib.request.urlopen", return_value=response):
            self.assertIsNone(_viewer_split_metadata(config, "train", None))

    def test_filtered_sampler_passes_the_selected_shards(self):
        stream = SimpleNamespace(shuffle=lambda **kwargs: SimpleNamespace(take=lambda count: [
            {"text": "नेपाली", "language_code": "npi"},
        ]))
        inventory = {
            "dataset_id": "owner/data", "dataset_config": "ne", "dataset_split": "test",
            "dataset_revision": "frozen", "dataset_shards": ["second.parquet"],
            "columns": ["text", "language_code"], "filter_column": "language_code", "filter_value": "npi",
        }
        with patch("datasets.load_dataset", return_value=stream) as loader:
            rows = sample_huggingface_rows(inventory, 1, 42, ("text",))
        self.assertEqual(rows[0]["text"], "नेपाली")
        self.assertEqual(loader.call_args.kwargs["data_files"], {"test": ["second.parquet"]})
        self.assertEqual(loader.call_args.kwargs["revision"], "frozen")


class InspectionCLITests(unittest.TestCase):
    def test_local_script_writes_language_report_without_streamlit(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "report.csv"
            source.write_text(json.dumps({"text": "नेपाल सुन्दर छ।", "language": "ne"}) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(["--local", str(source), "--output", str(output)])
            report = load_inspection_report(output)
        self.assertEqual(status, 0)
        self.assertEqual(report["language_status"]["language_coverage"], "Nepali-only")
        self.assertEqual(report["language_status"]["script"], "Devanagari")

    def test_ambiguous_configuration_requires_explicit_choice(self):
        with patch("scripts.inspect_dataset.discover_huggingface_dataset", return_value={"configs": ["ne", "en"]}), \
             contextlib.redirect_stderr(io.StringIO()) as error:
            status = main(["--dataset", "owner/data"])
        self.assertEqual(status, 1)
        self.assertIn("Choose --config", error.getvalue())


if __name__ == "__main__":
    unittest.main()
