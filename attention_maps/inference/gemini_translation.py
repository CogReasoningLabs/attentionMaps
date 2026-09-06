"""Gemini teacher inference matching the LIMA translation pipeline."""

from __future__ import annotations

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
)


LIMA_TEACHER_MODEL = "gemini-3.5-flash-lite"
LIMA_TEACHER_TEMPERATURE = 0.2
LIMA_TEACHER_TOP_P = 0.95
LIMA_TEACHER_TOP_K = 40


class GeminiTranslationBackend:
    """Use the same Gemini chat request style as the LIMA translation notebook."""

    def __init__(self, api_key: str, model_id: str = LIMA_TEACHER_MODEL):
        if not api_key:
            raise ComparisonConfigurationError(
                "Gemini teacher evaluation requires GEMINI_API_KEY"
            )
        try:
            from google import genai
        except ImportError as error:
            raise ComparisonConfigurationError(
                "Gemini teacher evaluation requires `google-genai`"
            ) from error
        self.model_id = model_id
        self.label = f"lima-teacher:{model_id}"
        self._client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        from google.genai import types

        if not prompt.strip():
            raise ComparisonConfigurationError("translation prompt cannot be empty")
        request_prompt = prompt
        if system_prompt and system_prompt.strip():
            request_prompt = f"{system_prompt.strip()}\n\n{prompt}"
        chat = self._client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_output_tokens=config.max_new_tokens,
            ),
        )
        response = chat.send_message(request_prompt)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini translation teacher returned no text")
        return str(text).strip()

