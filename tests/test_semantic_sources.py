import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer.semantic_source import source_settings, resolve_source
from scripts.cluster_dataset import parser


class SemanticSourceTests(unittest.TestCase):
    def args(self, *flags):
        return parser().parse_args(["run", "--output-dir", "/tmp/unused-semantic-test", *flags])

    def test_huggingface_keeps_config_split_revision_and_later_shard(self):
        settings = source_settings(self.args("--provider", "huggingface", "--dataset", "owner/data", "--config", "ne", "--split", "test",
                                            "--revision", "commit", "--shard", "second.parquet", "--text-column", "text"))
        catalog = {"configs": ["ne", "en"], "revision": "commit"}
        configuration = {"splits": {"test": {"shards": [{"name": "first.parquet", "path": "first"},
                                                          {"name": "second.parquet", "path": "second"}]}}}
        inventory = {"columns": ["text"], "schema": []}
        with patch("attention_maps.explorer.semantic_source.discover_huggingface_dataset", return_value=catalog) as discover, \
             patch("attention_maps.explorer.semantic_source.inspect_huggingface_configuration", return_value=configuration) as configure, \
             patch("attention_maps.explorer.semantic_source.inspect_huggingface_selection", return_value=inventory) as selection:
            resolved, fields = resolve_source(settings, token="test-token")
        discover.assert_called_once_with("owner/data", revision="commit", token="test-token")
        configure.assert_called_once_with(catalog, "ne", token="test-token")
        selection.assert_called_once_with(configuration, "test", shards=["second"], token="test-token")
        self.assertEqual(fields, ["text"])

    def test_kaggle_stages_exact_selected_file_and_version(self):
        settings = source_settings(self.args("--provider", "kaggle", "--dataset", "owner/data/versions/3",
                                            "--dataset-file", "later.jsonl", "--text-column", "text"))
        catalog = {"dataset_handle": "owner/data/versions/3"}
        selected = [{"name": "later.jsonl"}]
        with patch("attention_maps.explorer.semantic_source.discover_kaggle_dataset", return_value=catalog) as discover, \
             patch("attention_maps.explorer.semantic_source.kaggle_selection", return_value=selected) as select, \
             patch("attention_maps.explorer.semantic_source.stage_kaggle_selection", return_value=Path("cached.jsonl")) as stage, \
             patch("attention_maps.explorer.semantic_source.inspect_kaggle_selection", return_value={"columns": ["text"], "schema": []}):
            resolve_source(settings)
        discover.assert_called_once_with("owner/data/versions/3")
        select.assert_called_once_with(catalog, "later.jsonl", metadata_only=False)
        stage.assert_called_once_with(catalog, selected)

    def test_source_settings_reuse_selection_and_cli_lists_replace_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("provider: huggingface\ndataset: owner/data\nconfig: ne\nsplit: train\ntext_columns: [old]\nshards: [first]\nsample_fraction: 0.2\nsampling_runs: 5\n")
            settings = source_settings(self.args("--source-settings", str(path), "--text-column", "text", "--shard", "second"))
            self.assertEqual(settings["text_columns"], ["text"])
            self.assertEqual(settings["shards"], ["second"])
            self.assertNotIn("sample_fraction", settings)
            settings = source_settings(self.args("--source-settings", str(path), "--local", "new.jsonl"))
            self.assertIsNone(settings["dataset"])
            self.assertIsNone(settings["shards"])
            self.assertEqual(settings["provider"], "local")

    def test_remote_provider_is_required_but_local_sources_remain_automatic(self):
        with self.assertRaisesRegex(ValueError, "explicit provider"):
            source_settings(self.args("--dataset", "owner/data"))
        self.assertEqual(source_settings(self.args("--local", "data.jsonl"))["provider"], "local")

    def test_incompatible_provider_fields_fail_before_loading(self):
        for flags in (("--provider", "kaggle", "--dataset", "owner/data", "--split", "train"),
                      ("--provider", "huggingface", "--dataset", "owner/data", "--dataset-file", "file.txt")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                source_settings(self.args(*flags))

    def test_filter_settings_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("local: data.jsonl\nmin_devanagari_ratio: 0.8\nfiltered_output: kept.jsonl\n")
            with self.assertRaisesRegex(ValueError, "filtering first"):
                source_settings(self.args("--source-settings", str(path)))


if __name__ == "__main__":
    unittest.main()
