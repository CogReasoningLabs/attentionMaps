"""Text-only local inference for Google's pre-trained Gemma 4 E2B model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
)


DEFAULT_GEMMA4_BASE_MODEL_ID = "google/gemma-4-E2B"
DEFAULT_GEMMA4_BASE_REVISION = "main"
GEMMA4_BASE_CONTEXT_LENGTH = 128 * 1_024


@dataclass
class Gemma4BaseBundle:
    model: Any
    tokenizer: Any
    torch: Any
    model_id: str
    revision: str
    device: str
    dtype: str


def _runtime(torch: Any, device: str, dtype: str) -> tuple[str, Any, str]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ComparisonConfigurationError(
            "CUDA was selected for Gemma 4 E2B, but PyTorch found no CUDA GPU"
        )
    if device not in {"cpu", "cuda"}:
        raise ComparisonConfigurationError(
            "Gemma 4 E2B device must be auto, cpu, or cuda"
        )

    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype == "auto":
        dtype = (
            "bfloat16"
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else "float16" if device == "cuda" else "float32"
        )
    if dtype not in dtypes:
        raise ComparisonConfigurationError(
            "Gemma 4 E2B dtype must be auto, float32, float16, or bfloat16"
        )
    if device == "cpu" and dtype == "float16":
        raise ComparisonConfigurationError(
            "Gemma 4 E2B float16 inference requires CUDA"
        )
    return device, dtypes[dtype], dtype


def load_gemma4_base(
    *,
    model_id: str = DEFAULT_GEMMA4_BASE_MODEL_ID,
    revision: str = DEFAULT_GEMMA4_BASE_REVISION,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    token: str | None = None,
) -> Gemma4BaseBundle:
    """Load a Hub snapshot with Gemma 4's multimodal Transformers classes."""

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMultimodalLM, AutoTokenizer
    except ImportError as error:
        raise ComparisonConfigurationError(
            "Gemma 4 E2B requires torch, transformers>=5.15, accelerate, and "
            "huggingface_hub"
        ) from error

    requested_device = device
    resolved_device, torch_dtype, dtype_name = _runtime(torch, device, dtype)
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
            token=token or None,
        )
        # The full AutoProcessor initializes image/audio components and therefore
        # imports torchvision. This backend accepts text only, so the tokenizer is
        # both sufficient and keeps the Streamlit environment text-only.
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            snapshot,
            dtype=torch_dtype,
            device_map="auto" if requested_device == "auto" else resolved_device,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    except Exception as error:
        raise ComparisonConfigurationError(
            "Could not load Gemma 4 E2B. Confirm access to "
            f"{model_id!r}, available disk/RAM/VRAM, and HF_TOKEN if the Hub "
            f"requests authentication: {error}"
        ) from error

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return Gemma4BaseBundle(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        dtype=dtype_name,
    )


def _input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, TypeError):
        return model.device


def generate_gemma4_base(
    bundle: Gemma4BaseBundle,
    prompt: str,
    config: DecodingConfig,
    system_prompt: str = "",
) -> str:
    """Generate a direct text continuation from the pre-trained base model."""

    if not prompt.strip():
        raise ComparisonConfigurationError("Gemma 4 E2B prompt cannot be empty")
    if config.max_new_tokens >= GEMMA4_BASE_CONTEXT_LENGTH:
        raise ComparisonConfigurationError(
            f"Gemma 4 E2B max_new_tokens must be below "
            f"{GEMMA4_BASE_CONTEXT_LENGTH:,}"
        )

    full_prompt = prompt.strip()
    if system_prompt.strip():
        full_prompt = f"{system_prompt.strip()}\n\n{full_prompt}"
    maximum_input = GEMMA4_BASE_CONTEXT_LENGTH - config.max_new_tokens
    inputs = bundle.tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_input,
    )
    input_device = _input_device(bundle.model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    tokenizer = bundle.tokenizer
    generation: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.temperature > 0:
        generation.update(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k or 64,
        )

    bundle.torch.manual_seed(config.seed)
    if bundle.device == "cuda":
        bundle.torch.cuda.manual_seed_all(config.seed)
    with bundle.torch.inference_mode():
        output_ids = bundle.model.generate(**inputs, **generation)
    completion_ids = output_ids[0, input_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


class Gemma4BaseBackend:
    """Shared-comparison adapter for the Hugging Face Gemma 4 E2B base model."""

    def __init__(self, bundle: Gemma4BaseBundle):
        self.bundle = bundle
        self.label = f"gemma4-base:{bundle.model_id}"

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        return generate_gemma4_base(
            self.bundle, prompt, config, system_prompt or ""
        )
