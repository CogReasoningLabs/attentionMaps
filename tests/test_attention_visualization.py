from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from attention_maps.visualization.attention import (
    compute_global_attention_scale,
    format_token_labels,
    plot_attention,
    save_all_attention_maps,
)


class AttentionVisualizationTests(unittest.TestCase):
    def test_formats_nepali_bpe_and_special_token_labels(self) -> None:
        labels = format_token_labels(
            ["<|begin_of_text|>", "नेपालको</w>", "\n\n", "<|end_of_text|>"],
            {
                "pad": None,
                "unk": "<|unk|>",
                "bos": "<|begin_of_text|>",
                "eos": "<|end_of_text|>",
            },
        )
        self.assertEqual(labels, ["<BOS>", "नेपालको", "\\n\\n", "<EOS>"])

    def test_global_scale_uses_all_layers(self) -> None:
        attentions = [
            torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]]),
            torch.tensor([[[[0.5, 0.6], [0.7, 0.8]]]]),
        ]
        vmin, vmax = compute_global_attention_scale(attentions, percentile=100)
        self.assertEqual(vmin, 0.0)
        self.assertAlmostEqual(vmax, 0.8)

    def test_single_map_rejects_token_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "token labels"):
            plot_attention(
                tokens=["नेपाल"],
                attention=torch.eye(2),
                save_path="unused.png",
                title="invalid",
            )

    def test_all_heads_are_saved_in_layer_directories(self) -> None:
        tokens = ["नेपाल", "सुन्दर"]
        attention = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.4, 0.6]],
                    [[1.0, 0.0], [0.2, 0.8]],
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            paths = save_all_attention_maps(
                tokens=tokens,
                attentions=[attention],
                output_dir=output_dir,
                variant_name="softmax",
            )

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue((output_dir / "layer_00" / "head_00.png").is_file())
            self.assertTrue((output_dir / "layer_00" / "head_01.png").is_file())


if __name__ == "__main__":
    unittest.main()
