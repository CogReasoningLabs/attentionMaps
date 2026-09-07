"""Reusable remote-model inference comparison for CLI and UI entry points."""

from __future__ import annotations

import csv
import io
import time
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Protocol, Sequence


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GEMINI_FLASH_LITE_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GOOGLE_GEMMA_MODEL = "gemma-4-26b-a4b-it"


class ComparisonConfigurationError(ValueError):
    """Raised when a comparison cannot be configured safely."""


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = 40
    max_new_tokens: int = 4_096
    seed: int = 42
    thinking_level: str = "minimal"

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ComparisonConfigurationError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ComparisonConfigurationError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ComparisonConfigurationError("top_k must be positive or omitted")
        if self.max_new_tokens <= 0:
            raise ComparisonConfigurationError("max_new_tokens must be positive")
        if self.seed < 0:
            raise ComparisonConfigurationError("seed must be non-negative")
        if self.thinking_level not in {"minimal", "low", "medium", "high"}:
            raise ComparisonConfigurationError(
                "thinking_level must be minimal, low, medium, or high"
            )

    @property
    def name(self) -> str:
        top_k = "off" if self.top_k is None else str(self.top_k)
        return (
            f"T{self.temperature:g}_P{self.top_p:g}_K{top_k}_"
            f"TH{self.thinking_level}_S{self.seed}"
        )


@dataclass(frozen=True)
class ComparisonResult:
    model: str
    decoding: str
    sample_index: int
    prompt: str
    output: str
    latency_seconds: float
    system_prompt: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ModelBackend(Protocol):
    label: str

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str: ...


class GoogleGenAIBackend:
    """Google Gen AI backend using the documented endpoint for each model family."""

    def __init__(self, model_id: str, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as error:
            raise ComparisonConfigurationError(
                "Google inference requires `google-genai`; install the project requirements."
            ) from error

        if not api_key:
            raise ComparisonConfigurationError(
                "Google inference requires the GEMINI_API_KEY environment variable."
            )
        if not model_id.strip():
            raise ComparisonConfigurationError("Google model ID cannot be empty")

        self.model_id = model_id.strip()
        normalized_model_id = self.model_id.removeprefix("models/")
        self.is_gemma = normalized_model_id.startswith("gemma-")
        family = "gemma" if self.is_gemma else "gemini"
        self.label = f"google-{family}:{self.model_id}"
        self._client = genai.Client(api_key=api_key)
        if not hasattr(self._client, "interactions"):
            raise ComparisonConfigurationError(
                "Google Interactions API support requires `google-genai>=2.3.0`; "
                "upgrade the project requirements."
            )

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        try:
            if self.is_gemma:
                text = self._generate_gemma(prompt, config, system_prompt)
            else:
                text = self._generate_gemini(prompt, config, system_prompt)
        except Exception as error:
            message = str(error)
            if "no longer available" in message:
                replacement = (
                    DEFAULT_GOOGLE_GEMMA_MODEL
                    if self.is_gemma
                    else DEFAULT_GEMINI_MODEL
                )
                raise RuntimeError(
                    f"Google model {self.model_id!r} is no longer available for "
                    f"this account. Select {replacement!r} or another currently "
                    "supported model."
                ) from error
            raise
        if not text:
            raise RuntimeError("Google model returned an empty text response")
        return text.strip()

    def _generate_gemini(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None,
    ) -> str:
        request = {
            "model": self.model_id,
            "input": prompt,
            "generation_config": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_output_tokens": config.max_new_tokens,
                "seed": config.seed,
                "thinking_level": config.thinking_level,
            },
            "store": False,
        }
        if system_prompt and system_prompt.strip():
            request["system_instruction"] = system_prompt.strip()
        response = self._client.interactions.create(**request)
        text = getattr(response, "output_text", None)
        status = self._enum_value(getattr(response, "status", None))
        if status and status != "completed":
            raise RuntimeError(self._incomplete_response_message(response, "Gemini"))
        if not text:
            raise RuntimeError(self._empty_response_message(response, "Gemini"))
        return str(text)

    def _generate_gemma(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None,
    ) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_new_tokens,
                seed=config.seed,
                system_instruction=system_prompt.strip() if system_prompt else None,
                # Gemma 4 can spend the output budget on internal thought tokens.
                thinking_config=types.ThinkingConfig(
                    thinking_level=config.thinking_level
                ),
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(self._empty_response_message(response, "Gemma"))
        return str(text)

    @classmethod
    def _incomplete_response_message(cls, response: object, family: str) -> str:
        status = cls._enum_value(getattr(response, "status", None)) or "unknown"
        usage = getattr(response, "usage", None)
        thought_tokens = getattr(usage, "total_thought_tokens", None)
        output_tokens = getattr(usage, "total_output_tokens", None)
        token_details = []
        if thought_tokens is not None:
            token_details.append(f"thought_tokens={thought_tokens}")
        if output_tokens is not None:
            token_details.append(f"output_tokens={output_tokens}")
        suffix = f", {', '.join(token_details)}" if token_details else ""
        return (
            f"Google {family} generation was incomplete: status={status}{suffix}. "
            "Increase maximum new tokens or lower the thinking level."
        )

    @staticmethod
    def _enum_value(value: object) -> str:
        normalized = getattr(value, "value", value)
        return str(normalized).lower() if normalized is not None else ""

    @staticmethod
    def _empty_response_message(response: object, family: str) -> str:
        status = getattr(response, "status", None)
        errors = getattr(response, "errors", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            getattr(candidates[0], "finish_reason", None) if candidates else None
        )
        details = [
            f"status={status}" if status else "",
            f"finish_reason={finish_reason}" if finish_reason else "",
            f"errors={errors}" if errors else "",
        ]
        detail_text = ", ".join(detail for detail in details if detail)
        suffix = f" ({detail_text})" if detail_text else ""
        return f"Google {family} returned no text{suffix}"


class HuggingFaceBackend:
    """Generic Hugging Face Inference Providers chat-completion backend."""

    def __init__(
        self,
        model_id: str,
        token: str | None = None,
        provider: str = "auto",
    ):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as error:
            raise ComparisonConfigurationError(
                "Hugging Face inference requires `huggingface_hub`."
            ) from error

        if not token:
            raise ComparisonConfigurationError(
                "Hugging Face inference requires a token or the HF_TOKEN environment variable."
            )
        if not model_id.strip():
            raise ComparisonConfigurationError("Hugging Face model ID cannot be empty")

        self.model_id = model_id.strip()
        self.provider = provider.strip() or "auto"
        self.label = f"huggingface:{self.model_id}"
        self._client = InferenceClient(
            model=self.model_id,
            token=token,
            provider=self.provider,
        )

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        extra_body = {"top_k": config.top_k} if config.top_k is not None else None
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat_completion(
                messages=messages,
                temperature=max(config.temperature, 1e-7),
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                seed=config.seed,
                extra_body=extra_body,
            )
        except Exception as error:
            message = str(error)
            if (
                "model_not_supported" in message
                or "not supported by any provider" in message
            ):
                raise RuntimeError(
                    f"{self.model_id!r} is not available through the selected "
                    f"Hugging Face provider ({self.provider!r}). Choose a model "
                    "with an enabled Inference Provider or change the provider."
                ) from error
            raise
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Hugging Face provider returned no message content")
        return str(content).strip()


class OpenAIBackend:
    """OpenAI Responses API backend for dataset-generation workflows."""

    def __init__(self, model_id: str, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ComparisonConfigurationError(
                "OpenAI inference requires `openai`; install the project requirements."
            ) from error
        if not api_key:
            raise ComparisonConfigurationError(
                "OpenAI inference requires the OPENAI_API_KEY environment variable."
            )
        if not model_id.strip():
            raise ComparisonConfigurationError("OpenAI model ID cannot be empty")
        self.model_id = model_id.strip()
        self.label = f"openai:{self.model_id}"
        self._client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        input_messages = []
        if system_prompt and system_prompt.strip():
            input_messages.append({"role": "system", "content": system_prompt.strip()})
        input_messages.append({"role": "user", "content": prompt})
        request: dict[str, object] = {
            "model": self.model_id,
            "input": input_messages,
            "max_output_tokens": config.max_new_tokens,
            "store": False,
        }
        # Some reasoning models do not accept sampling parameters.  Keeping the
        # request minimal for deterministic temperature-zero runs is portable.
        if config.temperature > 0:
            request["temperature"] = config.temperature
            request["top_p"] = config.top_p
        response = self._client.responses.create(**request)
        output = getattr(response, "output_text", None)
        if not output:
            raise RuntimeError("OpenAI returned an empty text response")
        return str(output).strip()


class LocalPeftBackend:
    """Adapt either side of a loaded base/PEFT pair to the shared interface."""

    def __init__(self, bundle: object, *, use_adapter: bool = True):
        from attention_maps.inference.local_comparison import LocalModelPair

        if not isinstance(bundle, LocalModelPair):
            raise ComparisonConfigurationError(
                "Local PEFT comparison requires a loaded LocalModelPair"
            )
        self.bundle = bundle
        self.use_adapter = use_adapter
        variant = "finetuned" if use_adapter else "base"
        self.label = f"local-{variant}:{bundle.spec.key}"

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        from attention_maps.inference.local_comparison import (
            LocalDecodingConfig,
            generate_local_text,
        )

        local_config = LocalDecodingConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k or 50,
            max_new_tokens=config.max_new_tokens,
            repetition_penalty=1.05,
            seed=config.seed,
        )
        return generate_local_text(
            self.bundle,
            prompt,
            local_config,
            system_prompt=system_prompt or "",
            use_adapter=self.use_adapter,
        )


def build_decoding_grid(
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    top_ks: Sequence[int | None],
    *,
    max_new_tokens: int,
    seed: int,
    thinking_level: str = "minimal",
) -> list[DecodingConfig]:
    """Return the Cartesian product of user-selected decoding values."""

    if not temperatures or not top_ps or not top_ks:
        raise ComparisonConfigurationError("decoding value lists cannot be empty")
    return [
        DecodingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            seed=seed,
            thinking_level=thinking_level,
        )
        for temperature, top_p, top_k in product(temperatures, top_ps, top_ks)
    ]


def build_prompt(template: str, text: str | None = None) -> str:
    if not isinstance(template, str) or not template.strip():
        raise ComparisonConfigurationError("user prompt cannot be empty")
    if text is None:
        return template.replace("{text}", "").strip()
    if not isinstance(text, str) or not text.strip():
        raise ComparisonConfigurationError("source text cannot be empty")
    if "{text}" not in template:
        raise ComparisonConfigurationError(
            "prompt template must contain the `{text}` placeholder"
        )
    try:
        prompt = template.format(text=text)
    except (KeyError, ValueError) as error:
        raise ComparisonConfigurationError(f"invalid prompt template: {error}") from error
    if not prompt.strip():
        raise ComparisonConfigurationError("formatted prompt cannot be empty")
    return prompt


def comparison_csv(results: Sequence[ComparisonResult]) -> str:
    """Serialize comparison results for CLI and UI exports."""

    if not results:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(results[0].as_dict()))
    writer.writeheader()
    writer.writerows(result.as_dict() for result in results)
    return output.getvalue()


def run_comparison(
    backends: Sequence[ModelBackend],
    source_texts: Iterable[str | None],
    decoding_configs: Sequence[DecodingConfig],
    *,
    prompt_template: str,
    system_prompt: str = "",
) -> list[ComparisonResult]:
    """Run every backend/config/text combination while isolating call failures."""

    texts = [text.strip() if isinstance(text, str) else None for text in source_texts]
    if not backends:
        raise ComparisonConfigurationError("select at least one model backend")
    if not texts:
        raise ComparisonConfigurationError("provide at least one prompt input")
    if not decoding_configs:
        raise ComparisonConfigurationError("provide at least one decoding configuration")

    results: list[ComparisonResult] = []
    for sample_index, text in enumerate(texts):
        prompt = build_prompt(prompt_template, text)
        for backend in backends:
            for config in decoding_configs:
                started = time.perf_counter()
                try:
                    output = backend.generate(prompt, config, system_prompt)
                    error = None
                except Exception as generation_error:  # isolate remote API failures
                    output = ""
                    error = f"{type(generation_error).__name__}: {generation_error}"
                results.append(
                    ComparisonResult(
                        model=backend.label,
                        decoding=config.name,
                        sample_index=sample_index,
                        prompt=prompt,
                        output=output,
                        latency_seconds=round(time.perf_counter() - started, 3),
                        system_prompt=system_prompt.strip(),
                        error=error,
                    )
                )
    return results
