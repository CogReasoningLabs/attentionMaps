"""Local inference for IRIIS Nepali GPT-2 base and instruction checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
)


IRIIS_GPT2_CONTEXT_LENGTH = 512


@dataclass(frozen=True)
class IRIISGPT2Spec:
    key: str
    label: str
    model_id: str
    revision: str
    instruction_tuned: bool


IRIIS_GPT2_SPECS = (
    IRIISGPT2Spec(
        key="iriis-gpt2-instruct-nepali-124m",
        label="IRIIS GPT-2 Instruct Nepali 124M",
        model_id="IRIIS-RESEARCH/GPT2Instruct_Nepali_124M",
        revision="831727de04d61cdeb3046871d1f641de27df8d5c",
        instruction_tuned=True,
    ),
    IRIISGPT2Spec(
        key="iriis-gpt2-nepali-124m",
        label="IRIIS GPT-2 Nepali 124M Base",
        model_id="IRIIS-RESEARCH/GPT2_Nepali_124M",
        revision="47d2e0d80a4521c66848bd48aea118e911678182",
        instruction_tuned=False,
    ),
)
IRIIS_GPT2_SPEC_BY_KEY = {spec.key: spec for spec in IRIIS_GPT2_SPECS}
IRIIS_GPT2_BACKEND_OPTIONS = {
    f"{spec.label} · Hugging Face local": spec for spec in IRIIS_GPT2_SPECS
}


@dataclass
class IRIISGPT2Bundle:
    spec: IRIISGPT2Spec
    model: Any
    tokenizer: Any
    torch: Any
    device: str
    dtype: str
    context_length: int


def iriis_gpt2_spec(key: str) -> IRIISGPT2Spec:
    try:
        return IRIIS_GPT2_SPEC_BY_KEY[key]
    except KeyError as error:
        raise ComparisonConfigurationError(
            f"Unknown IRIIS GPT-2 model: {key}"
        ) from error


def _runtime(torch: Any, device: str, dtype: str) -> tuple[str, Any, str]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ComparisonConfigurationError(
            "CUDA was selected for IRIIS GPT-2, but PyTorch found no CUDA GPU"
        )
    if device not in {"cpu", "cuda"}:
        raise ComparisonConfigurationError(
            "IRIIS GPT-2 device must be auto, cpu, or cuda"
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
            "IRIIS GPT-2 dtype must be auto, float32, float16, or bfloat16"
        )
    if device == "cpu" and dtype == "float16":
        raise ComparisonConfigurationError(
            "IRIIS GPT-2 float16 inference requires CUDA"
        )
    return device, dtypes[dtype], dtype


def load_iriis_gpt2(
    spec: IRIISGPT2Spec,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    token: str | None = None,
) -> IRIISGPT2Bundle:
    """Load one immutable IRIIS GPT-2 snapshot for local generation."""

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ComparisonConfigurationError(
            "IRIIS GPT-2 requires torch, transformers, and huggingface_hub"
        ) from error

    resolved_device, torch_dtype, dtype_name = _runtime(torch, device, dtype)
    try:
        snapshot = snapshot_download(
            repo_id=spec.model_id,
            revision=spec.revision,
            local_files_only=local_files_only,
            token=token or None,
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch_dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(resolved_device)
    except Exception as error:
        mode = "local cache" if local_files_only else "local cache or Hugging Face Hub"
        raise ComparisonConfigurationError(
            f"Could not load {spec.label} from the {mode}: {error}"
        ) from error

    # The repositories' config files retain GPT-2's original token ID 50256,
    # while their custom tokenizers use BOS=0 and EOS=2. Always trust the
    # tokenizer shipped in the same immutable snapshot.
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return IRIISGPT2Bundle(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=resolved_device,
        dtype=dtype_name,
        context_length=IRIIS_GPT2_CONTEXT_LENGTH,
    )


def generate_iriis_gpt2(
    bundle: IRIISGPT2Bundle,
    prompt: str,
    config: DecodingConfig,
    system_prompt: str = "",
) -> str:
    """Generate a continuation using the checkpoint's bundled tokenizer."""

    prompt = prompt.strip()
    if not prompt:
        raise ComparisonConfigurationError("IRIIS GPT-2 prompt cannot be empty")
    if config.max_new_tokens >= bundle.context_length:
        raise ComparisonConfigurationError(
            f"IRIIS GPT-2 max_new_tokens must be below {bundle.context_length:,}"
        )
    if system_prompt.strip():
        prompt = f"{system_prompt.strip()}\n\n{prompt}"
    maximum_input = bundle.context_length - config.max_new_tokens
    inputs = bundle.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_input,
        add_special_tokens=True,
    )
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    generation: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "repetition_penalty": 1.05,
        "pad_token_id": bundle.tokenizer.pad_token_id,
        "eos_token_id": bundle.tokenizer.eos_token_id,
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
    return bundle.tokenizer.decode(
        output_ids[0, input_length:], skip_special_tokens=True
    ).strip()


class IRIISGPT2Backend:
    """Shared-comparison adapter for one IRIIS GPT-2 checkpoint."""

    def __init__(self, bundle: IRIISGPT2Bundle):
        self.bundle = bundle
        self.label = f"iriis-gpt2:{bundle.spec.key}"

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        return generate_iriis_gpt2(
            self.bundle, prompt, config, system_prompt or ""
        )
