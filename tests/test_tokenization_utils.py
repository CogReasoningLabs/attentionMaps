from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.tokenization.pipeline import (
    BPEConfig,
    InputConfig,
    OutputConfig,
    SpecialTokenConfig,
    TokenizationConfig,
    TokenizerSourceConfig,
    load_tokenization_config,
    run_tokenization,
)


class TokenizationPipelineTests(unittest.TestCase):
    def _write_processed_splits(self, root: Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq

        dataset_dir = root / "processed"
        dataset_dir.mkdir()
        schema = pa.schema(
            [
                ("doc_id", pa.string()),
                ("text", pa.string()),
                ("source", pa.string()),
                ("text_sha256", pa.string()),
            ]
        )
        texts = {
            "train": ["नेपाल सुन्दर देश हो।", "नेपाली भाषा हाम्रो भाषा हो।"] * 10,
            "validation": ["काठमाडौं नेपालको राजधानी हो।"],
            "test": ["हिमाल र पहाड नेपालमा छन्।"],
        }
        for split, documents in texts.items():
            rows = [
                {
                    "doc_id": f"{split}-{index}",
                    "text": text,
                    "source": "fixture",
                    "text_sha256": f"hash-{split}-{index}",
                }
                for index, text in enumerate(documents)
            ]
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, dataset_dir / f"{split}.parquet")
        (dataset_dir / "build_manifest.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        return dataset_dir

    def test_train_and_tokenize_all_splits(self) -> None:
        from tokenizers import Tokenizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = self._write_processed_splits(root)
            output_dir = root / "tokenized"
            config = TokenizationConfig(
                input=InputConfig(dataset_dir=dataset_dir),
                tokenizer=BPEConfig(
                    vocab_size=128,
                    min_frequency=1,
                    max_chars_per_segment=100,
                    special_tokens=SpecialTokenConfig(additional=("<|eot_id|>",)),
                ),
                output=OutputConfig(
                    output_dir=output_dir,
                    batch_size=4,
                    rows_per_shard=7,
                ),
            )

            manifest = run_tokenization(config, show_progress=False)

            self.assertEqual(manifest["training_split_only"], "train")
            self.assertEqual(manifest["split_stats"]["train"]["documents"], 20)
            self.assertEqual(manifest["split_stats"]["validation"]["documents"], 1)
            self.assertEqual(manifest["split_stats"]["test"]["documents"], 1)
            self.assertTrue((output_dir / "tokenization_manifest.json").is_file())
            self.assertTrue((output_dir / "tokenizer" / "tokenizer.json").is_file())
            self.assertEqual(len(list((output_dir / "train").glob("*.parquet"))), 3)

            tokenizer = Tokenizer.from_file(
                str(output_dir / "tokenizer" / "tokenizer.json")
            )
            encoding = tokenizer.encode("नेपाल सुन्दर देश हो।")
            self.assertEqual(
                encoding.ids[0], tokenizer.token_to_id("<|begin_of_text|>")
            )
            self.assertEqual(
                encoding.ids[-1], tokenizer.token_to_id("<|end_of_text|>")
            )

    def test_yaml_paths_are_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_processed_splits(root)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "tokenizer.yaml"
            config_path.write_text(
                """\
schema_version: 1
input:
  dataset_dir: ../processed
tokenizer:
  vocab_size: 128
  min_frequency: 1
output:
  output_dir: ../tokenized
""",
                encoding="utf-8",
            )

            config = load_tokenization_config(config_path)

            self.assertEqual(config.input.dataset_dir, (root / "processed").resolve())
            self.assertEqual(config.output.output_dir, (root / "tokenized").resolve())

    def test_existing_tokenizer_skips_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = self._write_processed_splits(root)
            trained_output = root / "trained"
            config = TokenizationConfig(
                input=InputConfig(dataset_dir=dataset_dir),
                tokenizer=BPEConfig(vocab_size=128, min_frequency=1),
                output=OutputConfig(output_dir=trained_output, batch_size=4),
            )
            run_tokenization(config, show_progress=False)

            encoded_output = root / "encoded_existing"
            encode_config = TokenizationConfig(
                input=config.input,
                tokenizer=config.tokenizer,
                output=OutputConfig(output_dir=encoded_output, batch_size=4),
            )
            with patch(
                "attention_maps.tokenization.pipeline.train_bpe_tokenizer",
                side_effect=AssertionError("training must not run"),
            ):
                manifest = run_tokenization(
                    encode_config,
                    show_progress=False,
                    existing_tokenizer_dir=trained_output / "tokenizer",
                )

            self.assertEqual(
                manifest["tokenizer"]["origin"]["mode"],
                "local",
            )
            self.assertIsNone(manifest["training_split_only"])
            self.assertEqual(manifest["split_stats"]["train"]["documents"], 20)

    def test_local_source_preserves_tokenizer_ids_and_bytes(self) -> None:
        from tokenizers import Tokenizer

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = self._write_processed_splits(root)
            trained_output = root / "trained"
            training_config = TokenizationConfig(
                input=InputConfig(dataset_dir=dataset_dir),
                tokenizer=BPEConfig(vocab_size=128, min_frequency=1),
                output=OutputConfig(output_dir=trained_output, batch_size=4),
            )
            run_tokenization(training_config, show_progress=False)

            source_dir = trained_output / "tokenizer"
            encoded_output = root / "encoded_local"
            local_config = TokenizationConfig(
                input=training_config.input,
                tokenizer=BPEConfig(
                    source=TokenizerSourceConfig(type="local", path=source_dir),
                    special_tokens=training_config.tokenizer.special_tokens,
                ),
                output=OutputConfig(output_dir=encoded_output, batch_size=4),
            )
            manifest = run_tokenization(local_config, show_progress=False)

            source_bytes = (source_dir / "tokenizer.json").read_bytes()
            output_bytes = (encoded_output / "tokenizer" / "tokenizer.json").read_bytes()
            source_tokenizer = Tokenizer.from_str(source_bytes.decode("utf-8"))
            output_tokenizer = Tokenizer.from_str(output_bytes.decode("utf-8"))
            sample = "नेपाल सुन्दर देश हो।"
            self.assertEqual(source_bytes, output_bytes)
            self.assertEqual(
                source_tokenizer.encode(sample, add_special_tokens=False).ids,
                output_tokenizer.encode(sample, add_special_tokens=False).ids,
            )
            self.assertEqual(manifest["tokenizer"]["origin"]["mode"], "local")

    def test_huggingface_source_uses_pinned_snapshot_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = self._write_processed_splits(root)
            trained_output = root / "trained"
            training_config = TokenizationConfig(
                input=InputConfig(dataset_dir=dataset_dir),
                tokenizer=BPEConfig(vocab_size=128, min_frequency=1),
                output=OutputConfig(output_dir=trained_output, batch_size=4),
            )
            run_tokenization(training_config, show_progress=False)

            encoded_output = root / "encoded_hub"
            hub_config = TokenizationConfig(
                input=training_config.input,
                tokenizer=BPEConfig(
                    source=TokenizerSourceConfig(
                        type="huggingface",
                        repository="organization/tokenizer",
                        revision="abc123",
                    ),
                    special_tokens=training_config.tokenizer.special_tokens,
                ),
                output=OutputConfig(output_dir=encoded_output, batch_size=4),
            )
            snapshot = trained_output / "tokenizer"
            with patch(
                "huggingface_hub.snapshot_download",
                return_value=str(snapshot),
            ), patch(
                "attention_maps.tokenization.pipeline.train_bpe_tokenizer",
                side_effect=AssertionError("training must not run"),
            ):
                manifest = run_tokenization(hub_config, show_progress=False)

            origin = manifest["tokenizer"]["origin"]
            self.assertEqual(origin["mode"], "huggingface")
            self.assertEqual(origin["repository"], "organization/tokenizer")
            self.assertEqual(origin["requested_revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
