import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from attention_maps.explorer.semantic_models import DocumentEncoder, PROMPTS


class SemanticModelTests(unittest.TestCase):
    def test_bert_pooling_excludes_padding_and_special_tokens(self):
        import torch

        encoder = DocumentEncoder.__new__(DocumentEncoder)
        encoder.backend = "bert"
        encoder.device = "cpu"
        encoder.torch = torch
        encoder.tokenizer = MagicMock(return_value={
            "input_ids": torch.tensor([[101, 12, 13, 102, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]),
        })
        encoder.model = MagicMock(return_value=SimpleNamespace(last_hidden_state=torch.tensor([
            [[100.0, 0], [1, 0], [0, 1], [0, 100], [100, 100]]
        ])))
        result = encoder._encode_chunks(["text"])
        np.testing.assert_allclose(result, [[2 ** -0.5, 2 ** -0.5]], rtol=1e-6)
        self.assertNotIn("special_tokens_mask", encoder.model.call_args.kwargs)
        self.assertFalse(encoder.tokenizer.call_args.kwargs["truncation"])

    def test_gemma_uses_published_sentence_transformer_and_explicit_task_prompt(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            for task in ("clustering", "similarity"):
                with self.subTest(task=task):
                    model = MagicMock()
                    model.max_seq_length = 2048
                    model.get_sentence_embedding_dimension.return_value = 768
                    model.encode.return_value = np.array([[3, 4]], dtype="float32")
                    args = SimpleNamespace(model="embeddinggemma", model_id=directory, model_revision="main",
                                           embedding_task=task, batch_size=8, device="cpu", max_length=512)
                    with patch("sentence_transformers.SentenceTransformer", return_value=model) as constructor:
                        encoder = DocumentEncoder(args)
                    self.assertEqual(constructor.call_args.kwargs["model_kwargs"]["torch_dtype"], torch.float32)
                    self.assertFalse(constructor.call_args.kwargs["trust_remote_code"])
                    self.assertEqual(model.max_seq_length, 512)
                    np.testing.assert_allclose(encoder._encode_chunks(["नेपाल"]), [[0.6, 0.8]])
                    self.assertEqual(model.encode.call_args.kwargs["prompt"], PROMPTS[task])
                    self.assertEqual(encoder.metadata["prompt"], PROMPTS[task])

    def test_record_embedding_combines_all_chunk_batches_with_token_weights(self):
        encoder = DocumentEncoder.__new__(DocumentEncoder)
        encoder.dimension = 2
        encoder.batch_size = 2
        encoder.max_length = 32
        encoder.prompt = ""
        encoder.tokenizer = None
        encoder._encode_chunks = lambda texts: np.array([[1, 0] if text == "first" else [0, 1] for text in texts], dtype="float32")
        with patch("attention_maps.explorer.semantic_models.text_chunks", return_value=iter([("first", 4), ("second", 2), ("third", 1)])):
            vectors, counts = encoder.encode(["long record"])
        np.testing.assert_allclose(vectors, [[0.8, 0.6]])
        np.testing.assert_array_equal(counts, [3])


if __name__ == "__main__":
    unittest.main()
