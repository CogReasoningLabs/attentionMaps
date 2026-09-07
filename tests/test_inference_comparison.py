import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
    GoogleGenAIBackend,
    HuggingFaceBackend,
    LocalPeftBackend,
    OpenAIBackend,
    build_decoding_grid,
    build_prompt,
    comparison_csv,
    run_comparison,
)
from attention_maps.inference.local_comparison import LocalAdapterSpec, LocalModelPair


class FakeBackend:
    label = "fake:model"

    def generate(self, prompt, config, system_prompt=None):
        return f"{system_prompt} | {prompt} @ {config.temperature}"


class FailingBackend:
    label = "fake:failing"

    def generate(self, prompt, config, system_prompt=None):
        raise RuntimeError("provider unavailable")


class FakeMessage:
    content = "नमस्ते"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeOpenAIResponse:
    output_text = "नेपाली उत्तर"


class FakeOpenAIResponses:
    def __init__(self):
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        return FakeOpenAIResponse()


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeOpenAIResponses()


class FakeInferenceClient:
    def __init__(self):
        self.arguments = None

    def chat_completion(self, **kwargs):
        self.arguments = kwargs
        return FakeResponse()


class UnsupportedInferenceClient:
    def chat_completion(self, **kwargs):
        raise RuntimeError("model_not_supported: not supported by any provider")


class FakeInteraction:
    output_text = "नेपाली उत्तर"
    status = "completed"


class FakeUsage:
    total_thought_tokens = 250
    total_output_tokens = 6


class IncompleteInteraction:
    output_text = "अधुरो उत्"
    status = "incomplete"
    usage = FakeUsage()


class FakeInteractions:
    def __init__(self, error=None, response=None):
        self.arguments = None
        self.error = error
        self.response = response or FakeInteraction()

    def create(self, **kwargs):
        self.arguments = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeGenerateContentResponse:
    text = "जेम्मा उत्तर"


class FakeModels:
    def __init__(self, error=None, response=None):
        self.arguments = None
        self.error = error
        self.response = response or FakeGenerateContentResponse()

    def generate_content(self, **kwargs):
        self.arguments = kwargs
        if self.error:
            raise self.error
        return self.response


class FakeGoogleClient:
    def __init__(
        self,
        interaction_error=None,
        interaction_response=None,
        model_error=None,
        model_response=None,
    ):
        self.interactions = FakeInteractions(interaction_error, interaction_response)
        self.models = FakeModels(model_error, model_response)


class DecodingConfigurationTests(unittest.TestCase):
    def test_builds_cartesian_decoding_grid(self):
        configs = build_decoding_grid(
            [0.2, 0.8],
            [0.9],
            [20, 40],
            max_new_tokens=50,
            seed=7,
        )

        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[0].max_new_tokens, 50)
        self.assertEqual(configs[0].seed, 7)

    def test_rejects_invalid_decoding_values(self):
        with self.assertRaises(ComparisonConfigurationError):
            DecodingConfig(temperature=2.1)
        with self.assertRaises(ComparisonConfigurationError):
            DecodingConfig(top_p=0)
        with self.assertRaises(ComparisonConfigurationError):
            DecodingConfig(top_k=0)

    def test_prompt_requires_text_placeholder(self):
        self.assertEqual(build_prompt("उत्तर: {text}", "नेपाल"), "उत्तर: नेपाल")
        with self.assertRaises(ComparisonConfigurationError):
            build_prompt("उत्तर दिनुहोस्", "नेपाल")
        with self.assertRaises(ComparisonConfigurationError):
            build_prompt("उत्तर: {text}", "  ")

    def test_prompt_only_mode_does_not_require_source_placeholder(self):
        self.assertEqual(build_prompt("नेपालबारे लेख्नुहोस्।"), "नेपालबारे लेख्नुहोस्।")


class ComparisonRunnerTests(unittest.TestCase):
    def test_serializes_results_to_csv(self):
        results = run_comparison(
            [FakeBackend()],
            ["नेपाल"],
            [DecodingConfig()],
            prompt_template="लेख्नुहोस्: {text}",
        )

        exported = comparison_csv(results)

        self.assertIn("model,decoding,sample_index", exported)
        self.assertIn("fake:model", exported)

    def test_runs_all_backends_and_isolates_provider_errors(self):
        results = run_comparison(
            [FakeBackend(), FailingBackend()],
            ["नेपाल"],
            [DecodingConfig()],
            prompt_template="लेख्नुहोस्: {text}",
            system_prompt="नेपालीमा उत्तर दिनुहोस्।",
        )

        self.assertEqual(len(results), 2)
        self.assertIn("लेख्नुहोस्: नेपाल", results[0].output)
        self.assertEqual(results[0].system_prompt, "नेपालीमा उत्तर दिनुहोस्।")
        self.assertIsNone(results[0].error)
        self.assertIn("provider unavailable", results[1].error)

    def test_huggingface_backend_forwards_top_k_as_provider_option(self):
        backend = object.__new__(HuggingFaceBackend)
        backend.model_id = "fake/model"
        backend.provider = "auto"
        backend.label = "huggingface:fake/model"
        backend._client = FakeInferenceClient()

        output = backend.generate(
            "नेपाल",
            DecodingConfig(top_k=25, seed=9),
            "नेपालीमा उत्तर दिनुहोस्।",
        )

        self.assertEqual(output, "नमस्ते")
        self.assertEqual(backend._client.arguments["extra_body"], {"top_k": 25})
        self.assertEqual(backend._client.arguments["seed"], 9)
        self.assertEqual(
            backend._client.arguments["messages"][0],
            {"role": "system", "content": "नेपालीमा उत्तर दिनुहोस्।"},
        )

    def test_openai_backend_uses_responses_instructions_and_temperature(self):
        backend = object.__new__(OpenAIBackend)
        backend.model_id = "gpt-4.1-mini"
        backend.label = "openai:gpt-4.1-mini"
        backend._client = FakeOpenAIClient()

        output = backend.generate(
            "नेपाल",
            DecodingConfig(temperature=0.4, top_p=0.9, max_new_tokens=80),
            "नेपालीमा उत्तर दिनुहोस्।",
        )

        arguments = backend._client.responses.arguments
        self.assertEqual(output, "नेपाली उत्तर")
        self.assertEqual(arguments["input"], "नेपाल")
        self.assertEqual(arguments["instructions"], "नेपालीमा उत्तर दिनुहोस्।")
        self.assertEqual(arguments["temperature"], 0.4)
        self.assertNotIn("top_p", arguments)
        self.assertFalse(arguments["store"])

    def test_openai_backend_omits_sampling_for_reasoning_models(self):
        backend = object.__new__(OpenAIBackend)
        backend.model_id = "gpt-5.6-terra"
        backend.label = "openai:gpt-5.6-terra"
        backend._client = FakeOpenAIClient()

        backend.generate("नेपाल", DecodingConfig())

        arguments = backend._client.responses.arguments
        self.assertNotIn("temperature", arguments)
        self.assertNotIn("top_p", arguments)

    def test_local_peft_backend_uses_shared_decoding_configuration(self):
        spec = LocalAdapterSpec(
            key="gpt2-alpaca-nepali-lora",
            label="GPT-2 · Nepali Alpaca LoRA",
            path=Path("finetuned_models/gpt2-alpaca-nepali-lora"),
            base_model_id="gpt2",
        )
        bundle = LocalModelPair(
            spec=spec,
            model=None,
            tokenizer=None,
            torch=None,
            device="cpu",
            dtype="float32",
        )
        backend = LocalPeftBackend(bundle)

        with patch(
            "attention_maps.inference.local_comparison.generate_local_text",
            return_value="स्थानीय उत्तर",
        ) as generate:
            output = backend.generate(
                "नेपाल",
                DecodingConfig(
                    temperature=0.4,
                    top_p=0.9,
                    top_k=25,
                    max_new_tokens=80,
                    seed=11,
                ),
                "नेपालीमा उत्तर दिनुहोस्।",
            )

        self.assertEqual(output, "स्थानीय उत्तर")
        local_config = generate.call_args.args[2]
        self.assertEqual(local_config.temperature, 0.4)
        self.assertEqual(local_config.max_new_tokens, 80)
        self.assertTrue(generate.call_args.kwargs["use_adapter"])

    def test_local_peft_backend_can_evaluate_the_base_model(self):
        spec = LocalAdapterSpec(
            key="tinyllama-nepali-alpaca-qlora",
            label="TinyLlama",
            path=Path("finetuned_models/tinyllama-nepali-alpaca-qlora"),
            base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )
        bundle = LocalModelPair(spec, None, None, None, "cpu", "float32")
        backend = LocalPeftBackend(bundle, use_adapter=False)

        with patch(
            "attention_maps.inference.local_comparison.generate_local_text",
            return_value="आधार उत्तर",
        ) as generate:
            backend.generate("Translate this", DecodingConfig())

        self.assertEqual(
            backend.label, "local-base:tinyllama-nepali-alpaca-qlora"
        )
        self.assertFalse(generate.call_args.kwargs["use_adapter"])

    def test_huggingface_backend_explains_unsupported_model(self):
        backend = object.__new__(HuggingFaceBackend)
        backend.model_id = "fake/unsupported"
        backend.provider = "auto"
        backend.label = "huggingface:fake/unsupported"
        backend._client = UnsupportedInferenceClient()

        with self.assertRaisesRegex(RuntimeError, "enabled Inference Provider"):
            backend.generate("नेपाल", DecodingConfig())

    def test_gemini_backend_uses_stateless_interactions_api(self):
        backend = object.__new__(GoogleGenAIBackend)
        backend.model_id = "gemini-3.6-flash"
        backend.is_gemma = False
        backend.label = "gemini:gemini-3.6-flash"
        backend._client = FakeGoogleClient()

        output = backend.generate(
            "नेपाल",
            DecodingConfig(
                temperature=0.4,
                top_p=0.9,
                top_k=30,
                max_new_tokens=80,
                seed=11,
            ),
            "नेपालीमा उत्तर दिनुहोस्।",
        )

        arguments = backend._client.interactions.arguments
        self.assertEqual(output, "नेपाली उत्तर")
        self.assertEqual(arguments["model"], "gemini-3.6-flash")
        self.assertEqual(arguments["input"], "नेपाल")
        self.assertFalse(arguments["store"])
        self.assertEqual(arguments["generation_config"]["temperature"], 0.4)
        self.assertEqual(arguments["generation_config"]["top_k"], 30)
        self.assertEqual(arguments["generation_config"]["thinking_level"], "minimal")

        self.assertEqual(arguments["system_instruction"], "नेपालीमा उत्तर दिनुहोस्।")

    def test_gemini_backend_rejects_incomplete_response_with_token_diagnostics(self):
        backend = object.__new__(GoogleGenAIBackend)
        backend.model_id = "gemini-3.6-flash"
        backend.is_gemma = False
        backend.label = "google-gemini:gemini-3.6-flash"
        backend._client = FakeGoogleClient(
            interaction_response=IncompleteInteraction()
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "status=incomplete, thought_tokens=250, output_tokens=6",
        ):
            backend.generate("नेपाल", DecodingConfig())

    def test_gemini_backend_explains_retired_model(self):
        backend = object.__new__(GoogleGenAIBackend)
        backend.model_id = "gemini-2.5-flash"
        backend.is_gemma = False
        backend.label = "gemini:gemini-2.5-flash"
        backend._client = FakeGoogleClient(
            interaction_error=RuntimeError(
                "This model is no longer available to new users"
            )
        )

        with self.assertRaisesRegex(RuntimeError, "gemini-3.6-flash"):
            backend.generate("नेपाल", DecodingConfig())

    def test_google_gemma_backend_explains_retired_model(self):
        backend = object.__new__(GoogleGenAIBackend)
        backend.model_id = "gemma-old-it"
        backend.is_gemma = True
        backend.label = "google-gemma:gemma-old-it"
        backend._client = FakeGoogleClient(
            model_error=RuntimeError(
                "This model is no longer available to new users"
            )
        )

        with self.assertRaisesRegex(RuntimeError, "gemma-4-26b-a4b-it"):
            backend.generate("नेपाल", DecodingConfig())

    def test_google_gemma_uses_generate_content_with_minimal_thinking(self):
        backend = object.__new__(GoogleGenAIBackend)
        backend.model_id = "gemma-4-26b-a4b-it"
        backend.is_gemma = True
        backend.label = "google-gemma:gemma-4-26b-a4b-it"
        backend._client = FakeGoogleClient()

        output = backend.generate(
            "नेपाल",
            DecodingConfig(
                temperature=0.4,
                top_p=0.9,
                top_k=30,
                max_new_tokens=80,
                seed=11,
            ),
            "छोटो उत्तर दिनुहोस्।",
        )

        arguments = backend._client.models.arguments
        self.assertEqual(output, "जेम्मा उत्तर")
        self.assertEqual(arguments["model"], "gemma-4-26b-a4b-it")
        self.assertEqual(arguments["contents"], "नेपाल")
        self.assertEqual(arguments["config"].temperature, 0.4)
        self.assertEqual(arguments["config"].max_output_tokens, 80)
        self.assertEqual(
            arguments["config"].system_instruction, "छोटो उत्तर दिनुहोस्।"
        )
        thinking_level = arguments["config"].thinking_config.thinking_level
        self.assertEqual(
            str(getattr(thinking_level, "value", thinking_level)).lower(), "minimal"
        )


if __name__ == "__main__":
    unittest.main()
