import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.inference.local_comparison import (
    LocalAdapterSpec,
    LocalDecodingConfig,
    LocalInferenceError,
    LocalModelPair,
    build_local_decoding_grid,
    discover_local_adapters,
    format_local_prompt,
)
from scripts.nepali_inference_compare import build_backends, parse_args


class FakeChatTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "FORMATTED"


class LocalAdapterDiscoveryTests(unittest.TestCase):
    def test_discovers_complete_adapter_and_base_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            complete = root / "tinyllama-nepali-alpaca-qlora"
            complete.mkdir()
            (complete / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": (
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                        )
                    }
                ),
                encoding="utf-8",
            )
            (complete / "adapter_model.safetensors").touch()
            incomplete = root / "missing-weights"
            incomplete.mkdir()
            (incomplete / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "gpt2"}), encoding="utf-8"
            )

            specs = discover_local_adapters(root)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].key, "tinyllama-nepali-alpaca-qlora")
            self.assertEqual(
                specs[0].base_model_id,
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )

    def test_batch_comparison_accepts_both_local_models(self):
        args = parse_args(
            ["--models", "local-gpt2", "local-tinyllama", "--local-files-only"]
        )
        specs = [
            LocalAdapterSpec(
                key="gpt2-alpaca-nepali-lora",
                label="GPT-2",
                path=Path("gpt2"),
                base_model_id="gpt2",
            ),
            LocalAdapterSpec(
                key="tinyllama-nepali-alpaca-qlora",
                label="TinyLlama",
                path=Path("tinyllama"),
                base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ),
        ]

        with patch(
            "scripts.nepali_inference_compare.discover_local_adapters",
            return_value=specs,
        ), patch(
            "scripts.nepali_inference_compare.load_local_model_pair",
            side_effect=lambda spec, **kwargs: LocalModelPair(
                spec, None, None, None, "cpu", "float32"
            ),
        ):
            backends = build_backends(args)

        self.assertEqual(
            {backend.label for backend in backends},
            {
                "local-finetuned:gpt2-alpaca-nepali-lora",
                "local-finetuned:tinyllama-nepali-alpaca-qlora",
            },
        )


class LocalDecodingTests(unittest.TestCase):
    def test_builds_cartesian_grid_and_supports_greedy(self):
        configs = build_local_decoding_grid(
            [0, 0.7],
            [0.9],
            [40, 50],
            max_new_tokens=64,
            repetition_penalty=1.05,
            seed=7,
        )

        self.assertEqual(len(configs), 4)
        self.assertFalse(configs[0].do_sample)
        self.assertIn("greedy", configs[0].name)
        self.assertTrue(configs[2].do_sample)

    def test_rejects_invalid_generation_values(self):
        with self.assertRaises(LocalInferenceError):
            LocalDecodingConfig(temperature=-0.1)

    def test_formats_chat_and_plain_prompts(self):
        tokenizer = FakeChatTokenizer()
        formatted = format_local_prompt(tokenizer, "नमस्ते", "नेपालीमा उत्तर देऊ")

        self.assertEqual(formatted, "FORMATTED")
        self.assertEqual(tokenizer.messages[0]["role"], "system")
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

        class PlainTokenizer:
            chat_template = None

        self.assertEqual(
            format_local_prompt(PlainTokenizer(), "question", "system"),
            "system\n\nquestion",
        )


if __name__ == "__main__":
    unittest.main()
