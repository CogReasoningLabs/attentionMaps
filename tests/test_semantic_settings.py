import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer.semantic_settings import load_embedding_settings
from attention_maps.explorer.semantic_source import source_settings
from scripts.cluster_dataset import main, parse_arguments
from tests.test_semantic_analysis import FixtureEncoder


class EmbeddingSettingsTests(unittest.TestCase):
    def test_one_config_supports_all_model_presets(self):
        path = Path(__file__).resolve().parents[1] / "configs/embeddings.yaml"
        for model in ("nepali-bert", "nepberta", "embeddinggemma"):
            args = parse_arguments(["run", "--settings", str(path), "--model", model])
            self.assertEqual(args.model, model)
            self.assertIsNone(args.max_length)
            self.assertIsNone(args.model_id)
            self.assertEqual(args.batch_size, load_embedding_settings(path)["batch_size"])
            self.assertEqual(args.clusters, 50)
            self.assertEqual(source_settings(args)["dataset"], "hsebarp/oscar-corpus-nepali")

    def test_relative_paths_are_relative_to_config_and_cli_values_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "settings.yaml"
            config.write_text("local: source.jsonl\noutput_dir: runs/base\nmodel: nepali-bert\nbatch_size: 4\ntext_columns: [old]\n")
            args = parse_arguments(["run", "--settings", str(config), "--batch-size", "2", "--text-column", "text", "--text-column", "answer"])
            self.assertEqual(args.local, root / "source.jsonl")
            self.assertEqual(args.output_dir, root / "runs/base")
            self.assertEqual(args.text_columns, ["text", "answer"])
            self.assertEqual(args.batch_size, 2)
            args = parse_arguments(["run", "--settings", str(config), "--output-dir", "cli-output"])
            self.assertEqual(args.output_dir, Path("cli-output"))

    def test_model_and_source_switches_clear_incompatible_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "settings.yaml"
            config.write_text("dataset: owner/data\nprovider: huggingface\nconfig: ne\nsplit: train\nshards: [first]\nmodel_id: custom/bert\noutput_dir: result\n")
            args = parse_arguments(["run", "--settings", str(config), "--model", "embeddinggemma", "--provider", "kaggle", "--dataset-file", "later.csv"])
            resolved = source_settings(args)
            self.assertIsNone(args.model_id)
            self.assertIsNone(resolved["config"])
            self.assertIsNone(resolved["shards"])
            self.assertEqual(resolved["dataset_file"], "later.csv")
            unchanged = parse_arguments(["run", "--settings", str(config), "--provider", "huggingface", "--model", "nepali-bert"])
            self.assertEqual(unchanged.shards, ["first"])
            self.assertEqual(unchanged.model_id, "custom/bert")
            args = parse_arguments(["run", "--settings", str(config), "--local", "input.jsonl"])
            self.assertIsNone(source_settings(args)["dataset"])
            args = parse_arguments(["run", "--settings", str(config), "--config", "en", "--shard", "second"])
            self.assertEqual(args.config, "en")
            self.assertEqual(args.shards, ["second"])

    def test_source_settings_override_replaces_central_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            central, source = root / "central.yaml", root / "source.yaml"
            central.write_text("provider: kaggle\ndataset: owner/old\ndataset_file: old.txt\ntext_columns: [old]\noutput_dir: result\n")
            source.write_text("local: new.jsonl\ntext_columns: [text]\n")
            args = parse_arguments(["run", "--settings", str(central), "--source-settings", str(source)])
            resolved = source_settings(args)
            self.assertEqual(resolved["provider"], "local")
            self.assertEqual(resolved["local"], str(root / "new.jsonl"))
            self.assertEqual(resolved["text_columns"], ["text"])

    def test_bad_config_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.yaml"
            for content in ("batchsize: 8", "batch_size: true", "clusters: 0", "seed: -1", "model: unknown",
                            "max_length: 8", "text_columns: text", "output_dir: 3", "model_revision: null", "[a,b]"):
                with self.subTest(content=content):
                    config.write_text(content)
                    with self.assertRaises(ValueError):
                        load_embedding_settings(config)
                    with patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No model load")), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(main(["run", "--settings", str(config)]), 1)

    def test_config_drives_complete_pipeline_and_is_saved_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.jsonl").write_text(''.join(json.dumps({"text": text}) + '\n' for text in ["A", "B", "C", "D"]))
            config = root / "embedding.yaml"
            config.write_text("local: source.jsonl\noutput_dir: result\nmodel: nepali-bert\ntext_columns: [text]\nclusters: 2\nbatch_size: 2\n")
            with patch("scripts.cluster_dataset.DocumentEncoder", return_value=FixtureEncoder()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--settings", str(config)]), 0)
            report = json.loads((root / "result/report.json").read_text())
            self.assertEqual(report["embedded_records"], 4)
            self.assertEqual(report["settings"]["clusters"], 2)
            self.assertEqual(report["settings"]["batch_size"], 2)
            self.assertEqual(report["settings"]["model"], "nepali-bert")
            self.assertEqual(report["settings"]["local"], str(root / "source.jsonl"))
            self.assertEqual(report["settings"]["settings_file"], str(config))


if __name__ == "__main__":
    unittest.main()
