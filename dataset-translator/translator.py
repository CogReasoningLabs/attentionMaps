"""Provider-independent translation and LIMA loading utilities."""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

LIMA_DATASET = "GAIR/lima"
LIMA_TRAIN_URL = "https://huggingface.co/datasets/GAIR/lima/resolve/main/train.jsonl?download=true"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMMA_MODEL = "gemma-4-31b-it"
DEFAULT_PROMPT_NAME = "Faithful translation"
DEFAULT_PROMPT_TEMPLATE = """Translate the following instruction-following conversation into {target_language}.

Rules:
- Translate BOTH the HUMAN and ASSISTANT turns completely. Do not leave any turn in the source language.
- Keep the speaker labels (HUMAN:, ASSISTANT:) exactly as-is, untranslated.
- Write ONLY in {target_language}. Do not add English glosses, translations, or parenthetical originals next to translated words.
- Proper nouns, brand/product names, and specialized technical terms with no standard {target_language} equivalent may remain in English. Ordinary vocabulary must be translated.
- Preserve all markdown formatting, line breaks, and structure exactly.
- Inside code blocks, translate only user-facing strings and natural-language comments; preserve identifiers, syntax, file paths, and URLs.
- Preserve numbers without changing their values.
- Translate naturally and idiomatically for {target_language}; preserve the original tone.
- Do not answer the conversation, add commentary, or wrap output in code fences.
- Use only correct {target_language} script. Do not mix in unrelated scripts or languages.

Return only the translation, with both turns fully translated.

Conversation:
{source}"""


@dataclass(frozen=True)
class GenerationSettings:
    provider: str
    model: str
    target_language: str
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 4096


def load_lima_sample(sample_size: int = 5, seed: int = 42) -> list[dict[str, Any]]:
    """Reservoir-sample gated LIMA JSONL without executing its legacy dataset script.

    Recent versions of ``datasets`` intentionally no longer execute repository
    loading scripts such as LIMA's ``lima.py``.  The canonical data file is a
    small JSONL file, so stream it directly with the user's Hugging Face token.
    """
    token = os.getenv("HF_token")
    if not token:
        raise RuntimeError("HF_token is missing from .env. LIMA requires authenticated access.")

    request = Request(LIMA_TRAIN_URL, headers={"Authorization": f"Bearer {token}", "User-Agent": "lima-translation-workbench/1.0"})
    rng, sample, records_seen = random.Random(seed), [], 0
    try:
        with urlopen(request, timeout=90) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line.decode("utf-8"))
                records_seen += 1
                if len(sample) < sample_size:
                    sample.append(record)
                else:
                    replacement_index = rng.randrange(records_seen)
                    if replacement_index < sample_size:
                        sample[replacement_index] = record
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise RuntimeError("Hugging Face denied access to GAIR/lima. Confirm HF_token is valid and has accepted the dataset's access terms.") from exc
        raise RuntimeError(f"Hugging Face returned HTTP {exc.code} while downloading LIMA.") from exc
    except URLError as exc:
        raise RuntimeError("Could not reach Hugging Face to download LIMA. Check your network connection.") from exc

    if not sample:
        raise RuntimeError("LIMA's train.jsonl was empty or could not be parsed.")
    return sample


def conversation_to_text(record: dict[str, Any]) -> str:
    """Make LIMA's conversation list readable while retaining each speaker."""
    rendered_turns = []
    for index, turn in enumerate(record.get("conversations", [])):
        # The original LIMA JSONL stores alternating strings. Some converted
        # dataset versions instead expose {"from": ..., "value": ...} turns.
        if isinstance(turn, Mapping):
            speaker = str(turn.get("from", "unknown")).upper()
            content = str(turn.get("value", turn.get("content", "")))
        else:
            speaker = "HUMAN" if index % 2 == 0 else "ASSISTANT"
            content = str(turn)
        rendered_turns.append(f"{speaker}: {content}")
    return "\n\n".join(rendered_turns)


def build_prompt(source: str, target_language: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    """Render only the two supported placeholders, leaving literal braces intact."""
    if "{source}" not in template:
        raise ValueError("The prompt template must include the {source} placeholder.")
    return template.replace("{target_language}", target_language).replace("{source}", source)


def _clean_model_text(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:text|markdown)?\s*\n?(.*?)\n?```\s*", text, re.S | re.I)
    return match.group(1).strip() if match else text.strip()


def translate(source: str, settings: GenerationSettings, prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> dict[str, Any]:
    """Generate one translation and return comparable timing/usage metadata."""
    started = time.perf_counter()
    prompt = build_prompt(source, settings.target_language, prompt_template)
    if settings.provider == "Google Gemini API":
        text, usage = _translate_google(prompt, settings)
    elif settings.provider == "OpenAI API":
        text, usage = _translate_openai(prompt, settings)
    else:
        raise ValueError(f"Unsupported provider: {settings.provider}")
    return {"translation": _clean_model_text(text), "elapsed_seconds": round(time.perf_counter() - started, 3), "settings": asdict(settings), "usage": usage, "rendered_prompt": prompt}


def _translate_google(prompt: str, settings: GenerationSettings) -> tuple[str, dict[str, Any]]:
    from google import genai
    from google.genai import types

    api_key = os.getenv("Gemini_api")
    if not api_key:
        raise RuntimeError("Gemini_api is missing from .env.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=settings.model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k, max_output_tokens=settings.max_output_tokens),
    )
    if not response.text:
        raise RuntimeError("The model returned no text (it may have been blocked by a safety filter).")
    metadata = getattr(response, "usage_metadata", None)
    usage = {"prompt_tokens": getattr(metadata, "prompt_token_count", None), "output_tokens": getattr(metadata, "candidates_token_count", None), "total_tokens": getattr(metadata, "total_token_count", None)}
    return response.text, usage


def _translate_openai(prompt: str, settings: GenerationSettings) -> tuple[str, dict[str, Any]]:
    """Ready for later use: add OPENAI_API_KEY and select this provider in the UI."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")
    response = OpenAI(api_key=api_key).chat.completions.create(
        model=settings.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.temperature, top_p=settings.top_p, max_tokens=settings.max_output_tokens,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("The OpenAI model returned no text.")
    usage_data = response.usage
    usage = {"prompt_tokens": getattr(usage_data, "prompt_tokens", None), "output_tokens": getattr(usage_data, "completion_tokens", None), "total_tokens": getattr(usage_data, "total_tokens", None)}
    return text, usage


def results_json(records: list[dict[str, Any]]) -> str:
    return json.dumps(records, ensure_ascii=False, indent=2)
