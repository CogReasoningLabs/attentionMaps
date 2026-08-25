from __future__ import annotations

import unittest

import torch

from attention_maps.config import ModelConfig
from attention_maps.inference.generate import final_attention_context_ids
from attention_maps.training.model import TinyTransformerLM


class GenerationAttentionContextTests(unittest.TestCase):
    def test_generation_retains_only_final_step_attention(self) -> None:
        model = TinyTransformerLM(
            ModelConfig(
                vocab_size=16,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                max_seq_len=8,
                dropout=0.0,
                use_moe=False,
            )
        ).eval()
        output = model.generate(
            torch.tensor([[2, 4]], dtype=torch.long),
            max_new_tokens=2,
            do_sample=False,
            return_attention=True,
        )

        self.assertNotIn("step_attentions", output)
        self.assertEqual(len(output["last_step_attentions"]), 1)
        self.assertEqual(
            tuple(output["last_step_attentions"][0].shape),
            (1, 2, 3, 3),
        )

    def test_excludes_token_sampled_after_attention_pass(self) -> None:
        self.assertEqual(
            final_attention_context_ids([10, 11, 12, 13, 14], context_length=4),
            [10, 11, 12, 13],
        )

    def test_uses_cropped_final_context(self) -> None:
        self.assertEqual(
            final_attention_context_ids([1, 2, 3, 4, 5, 6], context_length=3),
            [3, 4, 5],
        )


if __name__ == "__main__":
    unittest.main()
