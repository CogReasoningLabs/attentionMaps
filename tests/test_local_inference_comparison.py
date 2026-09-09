import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from attention_maps.inference.local_comparison import (
    LocalAdapterSpec,
    LocalDecodingConfig,
    LocalInferenceError,
    LocalModelPair,
    build_local_decoding_grid,
    discover_local_adapters,
    format_local_prompt,
    load_local_model_pair,
    local_model_context_limit,
    local_model_quantization,
)
from scripts.nepali_inference_compare import build_backends, parse_args


class FakeChatTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "FORMATTED"


class LocalAdapterDiscoveryTests(unittest.TestCase):
    def test_auto_quantization_uses_four_bit_only_for_large_models(self):
        self.assertEqual(
            local_model_quantization("meta-llama/Llama-2-7b-chat-hf"),
            "4bit",
        )
        self.assertEqual(
            local_model_quantization("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            "none",
        )
        self.assertEqual(local_model_quantization("gpt2"), "none")
        self.assertEqual(local_model_quantization("org/model-13b", "8bit"), "8bit")

    def test_large_auto_quantization_refuses_cpu_before_loading_weights(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def is_bf16_supported():
                return False

        class FakeTorch:
            cuda = FakeCuda()
            float32 = "float32"
            float16 = "float16"
            bfloat16 = "bfloat16"

        spec = LocalAdapterSpec(
            key="ckpt-504-llama7b",
            label="Llama 2 7B",
            path=Path("llama7b"),
            base_model_id="meta-llama/Llama-2-7b-chat-hf",
        )
        model_loader = Mock()
        with patch(
            "attention_maps.inference.local_comparison._require_dependencies",
            return_value=(FakeTorch(), model_loader, Mock(), Mock()),
        ), self.assertRaisesRegex(LocalInferenceError, "requires a CUDA GPU"):
            load_local_model_pair(spec)

        model_loader.from_pretrained.assert_not_called()

    def test_four_bit_loading_enables_gpu_cpu_offload(self):
        spec = LocalAdapterSpec(
            key="ckpt-504-llama7b",
            label="Llama 2 7B",
            path=Path("llama7b"),
            base_model_id="meta-llama/Llama-2-7b-chat-hf",
        )
        fake_torch = Mock()
        model_loader = Mock()
        tokenizer_loader = Mock()
        peft_loader = Mock()
        tokenizer = Mock(pad_token_id=0)
        base_model = Mock()
        base_model.config.pad_token_id = 0
        base_model.hf_device_map = {
            "model.layers.0": 0,
            "model.layers.31": "disk",
        }
        base_model.load_adapter = Mock()
        model_loader.from_pretrained.return_value = base_model
        tokenizer_loader.from_pretrained.return_value = tokenizer
        peft_loader.from_pretrained.return_value = base_model

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "attention_maps.inference.local_comparison._require_dependencies",
            return_value=(
                fake_torch,
                model_loader,
                tokenizer_loader,
                peft_loader,
            ),
        ), patch(
            "attention_maps.inference.local_comparison._resolve_runtime",
            return_value=("cuda", "bfloat16", "bfloat16"),
        ), patch(
            "attention_maps.inference.local_comparison._bitsandbytes_config_class"
        ) as config_class:
            bits_config = Mock()
            config_class.return_value = bits_config
            quantization_config = bits_config.return_value
            bundle = load_local_model_pair(
                spec,
                quantization="4bit",
                offload_folder=temporary_directory,
            )

        bits_config.assert_called_once_with(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype="bfloat16",
        )
        load_kwargs = model_loader.from_pretrained.call_args.kwargs
        self.assertIs(load_kwargs["quantization_config"], quantization_config)
        self.assertEqual(load_kwargs["device_map"], "auto")
        self.assertEqual(load_kwargs["offload_folder"], str(Path(temporary_directory).resolve()))
        self.assertTrue(load_kwargs["offload_state_dict"])
        peft_loader.from_pretrained.assert_not_called()
        base_model.load_adapter.assert_called_once()
        adapter_kwargs = base_model.load_adapter.call_args.kwargs
        self.assertNotIn("local_files_only", adapter_kwargs)
        self.assertTrue(adapter_kwargs["low_cpu_mem_usage"])
        self.assertTrue(adapter_kwargs["use_safetensors"])
        self.assertEqual(
            adapter_kwargs["load_config"].disk_offload_folder,
            str(Path(temporary_directory).resolve()),
        )
        self.assertEqual(
            adapter_kwargs["load_config"].device_map,
            base_model.hf_device_map,
        )
        base_model.to.assert_not_called()
        self.assertEqual(bundle.quantization, "4bit")

    def test_base_only_loading_does_not_construct_peft_adapter(self):
        spec = LocalAdapterSpec(
            key="ckpt-504-llama7b",
            label="Llama 2 7B",
            path=Path("llama7b"),
            base_model_id="meta-llama/Llama-2-7b-chat-hf",
        )
        fake_torch = Mock()
        model_loader = Mock()
        tokenizer_loader = Mock()
        peft_loader = Mock()
        tokenizer = Mock(pad_token_id=0)
        base_model = Mock()
        base_model.config.pad_token_id = 0
        model_loader.from_pretrained.return_value = base_model
        tokenizer_loader.from_pretrained.return_value = tokenizer

        with patch(
            "attention_maps.inference.local_comparison._require_dependencies",
            return_value=(
                fake_torch,
                model_loader,
                tokenizer_loader,
                peft_loader,
            ),
        ), patch(
            "attention_maps.inference.local_comparison._resolve_runtime",
            return_value=("cpu", "float32", "float32"),
        ):
            bundle = load_local_model_pair(
                spec,
                quantization="none",
                load_adapter=False,
            )

        peft_loader.from_pretrained.assert_not_called()
        base_model.to.assert_called_once_with("cpu")
        self.assertIs(bundle.model, base_model)
        self.assertFalse(bundle.adapter_loaded)

    def test_cli_exposes_local_quantization(self):
        self.assertEqual(
            parse_args(
                [
                    "--models",
                    "local-llama7b",
                    "--local-quantization",
                    "4bit",
                ]
            ).local_quantization,
            "4bit",
        )

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

    def test_batch_comparison_accepts_all_named_local_adapters(self):
        args = parse_args(
            [
                "--models",
                "local-gpt2",
                "local-tinyllama",
                "local-llama7b",
                "--local-files-only",
            ]
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
            LocalAdapterSpec(
                key="ckpt-504-llama7b",
                label="Llama 2 7B",
                path=Path("llama7b"),
                base_model_id="meta-llama/Llama-2-7b-chat-hf",
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
                "local-finetuned:ckpt-504-llama7b",
            },
        )

    def test_discovers_llama7b_checkpoint_with_friendly_name_and_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "ckpt-504-llama7b"
            checkpoint.mkdir()
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": (
                            "meta-llama/Llama-2-7b-chat-hf"
                        )
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").touch()

            spec = discover_local_adapters(root)[0]

            self.assertEqual(spec.key, "ckpt-504-llama7b")
            self.assertEqual(
                spec.label,
                "Llama 2 7B Chat · Nepali Multi-Dataset QLoRA",
            )
            self.assertEqual(local_model_context_limit(spec.base_model_id), 4_096)


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
