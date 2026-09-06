"""Local inference for the Arkios bilingual English/Nepali chat model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
)


DEFAULT_ARKIOS_MODEL_ID = "sajalregmi4/arkios-1b-chat"
DEFAULT_ARKIOS_REVISION = "main"
ARKIOS_CONTEXT_LENGTH = 4_096


@dataclass
class ArkiosBundle:
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
            "CUDA was selected for Arkios, but PyTorch found no CUDA GPU"
        )
    if device not in {"cpu", "cuda"}:
        raise ComparisonConfigurationError("Arkios device must be auto, cpu, or cuda")
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
            "Arkios dtype must be auto, float32, float16, or bfloat16"
        )
    if device == "cpu" and dtype == "float16":
        raise ComparisonConfigurationError("Arkios float16 inference requires CUDA")
    return device, dtypes[dtype], dtype


def load_arkios(
    *,
    model_id: str = DEFAULT_ARKIOS_MODEL_ID,
    revision: str = DEFAULT_ARKIOS_REVISION,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> ArkiosBundle:
    """Load Arkios from a pinned/cached Hugging Face snapshot."""

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ComparisonConfigurationError(
            "Arkios requires torch, transformers, accelerate, and huggingface_hub"
        ) from error

    resolved_device, torch_dtype, dtype_name = _runtime(torch, device, dtype)
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch_dtype,
            local_files_only=True,
        ).to(resolved_device)
    except Exception as error:
        raise ComparisonConfigurationError(f"Could not load Arkios: {error}") from error
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return ArkiosBundle(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        dtype=dtype_name,
    )


def generate_arkios(
    bundle: ArkiosBundle,
    prompt: str,
    config: DecodingConfig,
    system_prompt: str = "",
) -> str:
    """Generate with Arkios using the model card's published chat template."""

    if not prompt.strip():
        raise ComparisonConfigurationError("Arkios prompt cannot be empty")
    if config.max_new_tokens >= ARKIOS_CONTEXT_LENGTH:
        raise ComparisonConfigurationError(
            f"Arkios max_new_tokens must be below {ARKIOS_CONTEXT_LENGTH:,}"
        )
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    tokenizer = bundle.tokenizer
    maximum_input = ARKIOS_CONTEXT_LENGTH - config.max_new_tokens
    formatted = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_input,
    )
    model_device = next(bundle.model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
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
            top_k=config.top_k or 50,
        )
    bundle.torch.manual_seed(config.seed)
    if bundle.device == "cuda":
        bundle.torch.cuda.manual_seed_all(config.seed)
    with bundle.torch.inference_mode():
        output_ids = bundle.model.generate(**inputs, **generation)
    completion_ids = output_ids[0, input_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


class ArkiosBackend:
    """Shared-comparison adapter for Arkios chat inference."""

    def __init__(self, bundle: ArkiosBundle):
        self.bundle = bundle
        self.label = f"arkios-chat:{bundle.model_id}"

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        return generate_arkios(self.bundle, prompt, config, system_prompt or "")
