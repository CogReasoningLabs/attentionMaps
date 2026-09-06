import unittest
from unittest.mock import patch

from attention_maps.inference.arkios import ArkiosBackend, ArkiosBundle
from attention_maps.inference.comparison import DecodingConfig
from attention_maps.inference.himalayagpt import (
    DEFAULT_HIMALAYAGPT_REVISION,
    HimalayaGPTBackend,
    HimalayaGPTBundle,
    build_himalayagpt_prompt_ids,
)
from scripts.nepali_inference_compare import build_backends, parse_args


class FakeHimalayaTokenizer:
    _special_to_id = {
        "<|bos|>": 1,
        "<|user_start|>": 2,
        "<|user_end|>": 3,
        "<|assistant_start|>": 4,
    }
    unk_token_id = 0
    unk_token = "<unk>"

    def __call__(self, text, **kwargs):
        self.text = text
        return {"input_ids": [10, 11]}

    def convert_tokens_to_ids(self, token):
        return self._special_to_id.get(token, 0)


class AdditionalLocalModelTests(unittest.TestCase):
    def test_himalayagpt_uses_pinned_revision_and_reference_framing(self):
        tokenizer = FakeHimalayaTokenizer()

        ids = build_himalayagpt_prompt_ids(
            tokenizer,
            "नेपालको राजधानी के हो?",
            system_prompt="छोटो उत्तर देऊ।",
            vocab_size=32_768,
        )

        self.assertEqual(ids, [1, 2, 10, 11, 3, 4])
        self.assertIn("छोटो उत्तर देऊ।", tokenizer.text)
        self.assertEqual(
            DEFAULT_HIMALAYAGPT_REVISION,
            "10ef5130093789db77b186fc37e524754848bb3a",
        )

    def test_model_backends_delegate_to_their_own_modules(self):
        arkios_bundle = ArkiosBundle(
            None, None, None, "arkios", "main", "cpu", "float32"
        )
        himalaya_bundle = HimalayaGPTBundle(
            None, None, None, "himalaya", "pinned", "cpu", "float32"
        )
        with patch(
            "attention_maps.inference.arkios.generate_arkios",
            return_value="arkios output",
        ), patch(
            "attention_maps.inference.himalayagpt.generate_himalayagpt",
            return_value="himalaya output",
        ):
            self.assertEqual(
                ArkiosBackend(arkios_bundle).generate("prompt", DecodingConfig()),
                "arkios output",
            )
            self.assertEqual(
                HimalayaGPTBackend(himalaya_bundle).generate(
                    "prompt", DecodingConfig()
                ),
                "himalaya output",
            )

    def test_batch_cli_builds_arkios_and_himalayagpt_backends(self):
        args = parse_args(["--models", "arkios", "himalayagpt"])
        arkios_bundle = ArkiosBundle(
            None, None, None, args.arkios_model, args.arkios_revision, "cpu", "float32"
        )
        himalaya_bundle = HimalayaGPTBundle(
            None,
            None,
            None,
            args.himalayagpt_model,
            args.himalayagpt_revision,
            "cpu",
            "float32",
        )
        with patch(
            "scripts.nepali_inference_compare.load_arkios",
            return_value=arkios_bundle,
        ), patch(
            "scripts.nepali_inference_compare.load_himalayagpt",
            return_value=himalaya_bundle,
        ):
            backends = build_backends(args)

        self.assertEqual(
            {type(backend) for backend in backends},
            {ArkiosBackend, HimalayaGPTBackend},
        )


if __name__ == "__main__":
    unittest.main()
