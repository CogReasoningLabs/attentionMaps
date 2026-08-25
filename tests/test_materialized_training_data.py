from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from attention_maps.tokenization.models import (
    HuggingFaceBPETokenizer,
    tokenizer_from_state,
)
from attention_maps.tokenization.pipeline import (
    BPEConfig,
    InputConfig,
    OutputConfig,
    TokenizationConfig,
    run_tokenization,
)
from attention_maps.training.cli import parse_args
from attention_maps.training.configuration import load_decoder_training_config
from attention_maps.training.tokenized_data import (
    PackedSequenceDataset,
    create_packed_dataloaders,
    load_materialized_tokenized_corpus,
)


class MaterializedTrainingDataTests(unittest.TestCase):
    def _tokenized_fixture(self, root: Path) -> Path:
        processed = root / "processed"
        processed.mkdir()
        schema = pa.schema(
            [
                ("doc_id", pa.string()),
                ("text", pa.string()),
                ("source", pa.string()),
                ("text_sha256", pa.string()),
            ]
        )
        documents = {
            "train": ["नेपाल सुन्दर देश हो।", "नेपाली भाषा हाम्रो भाषा हो।"] * 8,
            "validation": ["काठमाडौं नेपालको राजधानी हो।"] * 3,
            "test": ["हिमाल र पहाड नेपालमा छन्।"] * 3,
        }
        for split, texts in documents.items():
            table = pa.Table.from_pylist(
                [
                    {
                        "doc_id": f"{split}-{index}",
                        "text": text,
                        "source": "fixture",
                        "text_sha256": f"hash-{split}-{index}",
                    }
                    for index, text in enumerate(texts)
                ],
                schema=schema,
            )
            pq.write_table(table, processed / f"{split}.parquet")
        tokenized = root / "tokenized"
        run_tokenization(
            TokenizationConfig(
                input=InputConfig(dataset_dir=processed),
                tokenizer=BPEConfig(vocab_size=128, min_frequency=1),
                output=OutputConfig(output_dir=tokenized, batch_size=4),
            ),
            show_progress=False,
        )
        return tokenized

    def test_fixed_blocks_are_shifted_by_one_token(self) -> None:
        token_ids = torch.arange(13, dtype=torch.long)
        dataset = PackedSequenceDataset(token_ids, sequence_length=4)

        self.assertEqual(len(dataset), 3)
        x, y = dataset[1]
        self.assertEqual(x.tolist(), [4, 5, 6, 7])
        self.assertEqual(y.tolist(), [5, 6, 7, 8])

    def test_materialized_corpus_and_tokenizer_are_checkpoint_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenized = self._tokenized_fixture(root)

            encoded = load_materialized_tokenized_corpus(tokenized, batch_size=4)
            restored = tokenizer_from_state(encoded.tokenizer.to_state())
            loaders = create_packed_dataloaders(
                encoded,
                sequence_length=4,
                batch_size=2,
                seed=42,
                drop_last_batch=False,
            )

            self.assertIsInstance(encoded.tokenizer, HuggingFaceBPETokenizer)
            self.assertEqual(restored.vocab_size, encoded.tokenizer.vocab_size)
            self.assertGreater(loaders.blocks["train"], 0)
            x, y = next(iter(loaders.train))
            self.assertEqual(tuple(x.shape), (2, 4))
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_bundled_training_yaml_supplies_cli_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/training/nepali_decoder_small.yaml"
        config = load_decoder_training_config(path)
        args = parse_args(["--training-config", str(path), "--max-steps", "1"])

        self.assertEqual(config.data.sequence_length, 512)
        self.assertEqual(args.seq_len, 512)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_steps, 1)
        self.assertEqual(args.tokenized_dir, str(config.data.tokenized_dir))


if __name__ == "__main__":
    unittest.main()
