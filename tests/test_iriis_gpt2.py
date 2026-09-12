import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from attention_maps.inference.comparison import DecodingConfig
from attention_maps.inference.iriis_gpt2 import (
    IRIIS_GPT2_SPECS,
    IRIISGPT2Backend,
    IRIISGPT2Bundle,
    load_iriis_gpt2,
)


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, item):
        if isinstance(item, tuple):
            _, positions = item
            return self.values[positions]
        return self.values[item]


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return {"input_ids": FakeTensor([10, 11])}

    def decode(self, ids, **kwargs):
        self.decoded = ids
        self.decode_kwargs = kwargs
        return "नेपाली उत्तर"


class FakeTorch:
    def __init__(self):
        self.cuda = Mock()
        self.inference_mode = nullcontext
        self.manual_seed = Mock()


class IRIISGPT2Tests(unittest.TestCase):
    def test_registers_both_pinned_models(self):
        self.assertEqual(len(IRIIS_GPT2_SPECS), 2)
        self.assertTrue(all(len(spec.revision) == 40 for spec in IRIIS_GPT2_SPECS))
        self.assertTrue(IRIIS_GPT2_SPECS[0].instruction_tuned)
        self.assertFalse(IRIIS_GPT2_SPECS[1].instruction_tuned)

    def test_loader_aligns_stale_model_special_ids_with_tokenizer(self):
        tokenizer = FakeTokenizer()
        model = Mock()
        model.to.return_value = model
        model.config = SimpleNamespace(
            bos_token_id=50256, eos_token_id=50256, pad_token_id=None
        )
        model.generation_config = SimpleNamespace(
            bos_token_id=50256, eos_token_id=50256, pad_token_id=None
        )
        with patch(
            "attention_maps.inference.iriis_gpt2._runtime",
            return_value=("cpu", "float32", "float32"),
        ), patch(
            "huggingface_hub.snapshot_download", return_value="/snapshot"
        ) as snapshot, patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
        ), patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
        ):
            bundle = load_iriis_gpt2(IRIIS_GPT2_SPECS[0], token="secret")

        snapshot.assert_called_once_with(
            repo_id=IRIIS_GPT2_SPECS[0].model_id,
            revision=IRIIS_GPT2_SPECS[0].revision,
            local_files_only=False,
            token="secret",
        )
        self.assertEqual(model.config.bos_token_id, 0)
        self.assertEqual(model.config.eos_token_id, 2)
        self.assertEqual(model.config.pad_token_id, 1)
        self.assertEqual(bundle.context_length, 512)

    def test_backend_generates_only_the_completion(self):
        tokenizer = FakeTokenizer()
        model = Mock()
        model.generate.return_value = FakeTensor([10, 11, 20, 21])
        torch = FakeTorch()
        bundle = IRIISGPT2Bundle(
            IRIIS_GPT2_SPECS[0], model, tokenizer, torch, "cpu", "float32", 512
        )

        output = IRIISGPT2Backend(bundle).generate(
            "प्रश्न",
            DecodingConfig(
                temperature=0.4,
                top_p=0.9,
                top_k=25,
                max_new_tokens=32,
                seed=7,
            ),
            "नेपालीमा उत्तर दिनुहोस्।",
        )

        self.assertEqual(output, "नेपाली उत्तर")
        self.assertIn("नेपालीमा उत्तर दिनुहोस्", tokenizer.prompt)
        self.assertEqual(tokenizer.kwargs["max_length"], 480)
        generation = model.generate.call_args.kwargs
        self.assertEqual(generation["temperature"], 0.4)
        self.assertEqual(generation["top_k"], 25)
        self.assertEqual(tokenizer.decoded, [20, 21])


if __name__ == "__main__":
    unittest.main()
