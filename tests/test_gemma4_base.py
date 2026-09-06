import unittest
from types import SimpleNamespace

import torch

from attention_maps.inference.comparison import DecodingConfig
from attention_maps.inference.gemma4_base import (
    Gemma4BaseBundle,
    generate_gemma4_base,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self):
        self.text = ""

    def __call__(self, text, **kwargs):
        self.text = text
        self.arguments = kwargs
        return {"input_ids": torch.tensor([[10, 11]])}

    def decode(self, token_ids, **kwargs):
        self.decoded_ids = token_ids.tolist()
        self.decode_arguments = kwargs
        return "base completion"


class FakeModel:
    def __init__(self):
        self.embedding = SimpleNamespace(weight=torch.zeros(1))

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        self.generation = kwargs
        return torch.tensor([[10, 11, 12, 13]])


class Gemma4BaseTests(unittest.TestCase):
    def test_generates_a_plain_base_model_continuation(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        bundle = Gemma4BaseBundle(
            model,
            tokenizer,
            torch,
            "google/gemma-4-E2B",
            "main",
            "cpu",
            "float32",
        )

        output = generate_gemma4_base(
            bundle,
            "Translate this sentence.",
            DecodingConfig(max_new_tokens=32, temperature=0),
            "Continue the supplied text.",
        )

        self.assertEqual(output, "base completion")
        self.assertEqual(
            tokenizer.text,
            "Continue the supplied text.\n\nTranslate this sentence.",
        )
        self.assertEqual(tokenizer.decoded_ids, [12, 13])
        self.assertFalse(model.generation["do_sample"])
        self.assertNotIn("temperature", model.generation)


if __name__ == "__main__":
    unittest.main()
