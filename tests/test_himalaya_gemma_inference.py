from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from attention_maps.inference.himalaya_gemma import (
    DEFAULT_ADAPTER_ID,
    ModelBundle,
    build_messages,
    generation_kwargs,
    parse_args,
    run_inference,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.template_call = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = (messages, kwargs)
        return {
            "input_ids": torch.tensor([[2, 10, 11]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        self.decode_call = (token_ids.tolist(), kwargs)
        return "यो परीक्षण उत्तर हो।"


class _FakeModel:
    config = SimpleNamespace(
        text_config=SimpleNamespace(max_position_embeddings=1024)
    )

    def __init__(self) -> None:
        self.generate_call = None

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.empty(1))

    def generate(self, **kwargs):
        self.generate_call = kwargs
        return torch.tensor([[2, 10, 11, 90, 91]])


class HimalayaGemmaInferenceTests(unittest.TestCase):
    def test_cli_defaults_are_model_specific(self) -> None:
        args = parse_args([])
        self.assertEqual(args.adapter_id, DEFAULT_ADAPTER_ID)
        self.assertEqual(args.quantization, "none")
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.do_sample)
        self.assertFalse(args.cpu_offload)
        self.assertIsNone(args.device_map)

    def test_cpu_offload_memory_options_are_parsed(self) -> None:
        args = parse_args(
            [
                "प्रश्न",
                "--quantization",
                "8bit",
                "--cpu-offload",
                "--gpu-max-memory",
                "5GiB",
                "--cpu-max-memory",
                "24GiB",
            ]
        )
        self.assertTrue(args.cpu_offload)
        self.assertEqual(args.gpu_max_memory, "5GiB")
        self.assertEqual(args.cpu_max_memory, "24GiB")

    def test_rejects_unsupported_four_bit_cpu_offload(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                ["प्रश्न", "--quantization", "4bit", "--cpu-offload"]
            )

    def test_build_messages_adds_optional_system_prompt(self) -> None:
        self.assertEqual(
            build_messages(" प्रश्न ", " सहायक "),
            [
                {"role": "system", "content": "सहायक"},
                {"role": "user", "content": "प्रश्न"},
            ],
        )
        self.assertEqual(
            build_messages("प्रश्न", None),
            [{"role": "user", "content": "प्रश्न"}],
        )

    def test_sampling_parameters_are_only_passed_when_sampling(self) -> None:
        greedy = generation_kwargs(parse_args(["प्रश्न"]))
        self.assertNotIn("temperature", greedy)
        sampled = generation_kwargs(parse_args(["प्रश्न", "--do-sample"]))
        self.assertEqual(sampled["temperature"], 0.7)
        self.assertEqual(sampled["top_p"], 0.9)

    def test_inference_uses_chat_template_and_decodes_completion_only(self) -> None:
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        bundle = ModelBundle(
            model=model,
            tokenizer=tokenizer,
            base_model_id="himalaya-ai/himalaya-gemma-4-e2b-it",
            device="cpu",
            dtype="torch.bfloat16",
        )
        args = parse_args(["नेपालीमा उत्तर दिनुहोस्", "--enable-thinking"])

        with patch(
            "attention_maps.inference.himalaya_gemma.load_model",
            return_value=bundle,
        ):
            result = run_inference(args)

        self.assertEqual(result["completion"], "यो परीक्षण उत्तर हो।")
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(tokenizer.decode_call[0], [90, 91])
        self.assertTrue(tokenizer.template_call[1]["enable_thinking"])

    def test_output_parent_can_be_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            args = parse_args(["प्रश्न", "--output", str(output)])
            self.assertEqual(args.output, output)


if __name__ == "__main__":
    unittest.main()
