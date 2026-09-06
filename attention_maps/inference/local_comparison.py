"""Local Hugging Face base-versus-PEFT inference comparisons."""

from __future__ import annotations

import csv
import io
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


class LocalInferenceError(ValueError):
    """Raised when a local comparison cannot be configured or loaded."""


@dataclass(frozen=True)
class LocalAdapterSpec:
    """A local PEFT adapter and the base model declared by its config."""

    key: str
    label: str
    path: Path
    base_model_id: str


@dataclass(frozen=True)
class LocalDecodingConfig:
    """Generation controls shared by the base and finetuned variants."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise LocalInferenceError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise LocalInferenceError("top_p must be in (0, 1]")
        if self.top_k <= 0:
            raise LocalInferenceError("top_k must be positive")
        if self.max_new_tokens <= 0:
            raise LocalInferenceError("max_new_tokens must be positive")
        if self.repetition_penalty <= 0:
            raise LocalInferenceError("repetition_penalty must be positive")
        if self.seed < 0:
            raise LocalInferenceError("seed must be non-negative")

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0

    @property
    def name(self) -> str:
        mode = f"T{self.temperature:g}_P{self.top_p:g}_K{self.top_k}"
        return (
            f"{mode}_R{self.repetition_penalty:g}_S{self.seed}"
            if self.do_sample
            else f"greedy_R{self.repetition_penalty:g}_S{self.seed}"
        )


@dataclass(frozen=True)
class LocalComparisonResult:
    adapter: str
    base_model: str
    variant: str
    decoding: str
    prompt: str
    formatted_prompt: str
    output: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LocalModelPair:
    """One base model with a local PEFT adapter attached."""

    spec: LocalAdapterSpec
    model: Any
    tokenizer: Any
    torch: Any
    device: str
    dtype: str


FRIENDLY_MODEL_NAMES = {
    "gpt2-alpaca-nepali-lora": "GPT-2 · Nepali Alpaca LoRA",
    "tinyllama-nepali-alpaca-qlora": "TinyLlama 1.1B · Nepali Alpaca QLoRA",
}


def discover_local_adapters(root: Path) -> list[LocalAdapterSpec]:
    """Discover complete PEFT adapters immediately beneath ``root``."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    specs: list[LocalAdapterSpec] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = directory / "adapter_config.json"
        has_weights = any(
            (directory / filename).is_file()
            for filename in ("adapter_model.safetensors", "adapter_model.bin")
        )
        if not config_path.is_file() or not has_weights:
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        base_model_id = str(config.get("base_model_name_or_path") or "").strip()
        if not base_model_id:
            continue
        specs.append(
            LocalAdapterSpec(
                key=directory.name,
                label=FRIENDLY_MODEL_NAMES.get(
                    directory.name, directory.name.replace("-", " ").title()
                ),
                path=directory,
                base_model_id=base_model_id,
            )
        )
    return specs


def build_local_decoding_grid(
    temperatures: Iterable[float],
    top_ps: Iterable[float],
    top_ks: Iterable[int],
    *,
    max_new_tokens: int,
    repetition_penalty: float,
    seed: int,
) -> list[LocalDecodingConfig]:
    """Return every requested decoding combination."""

    return [
        LocalDecodingConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        for temperature, top_p, top_k in product(temperatures, top_ps, top_ks)
    ]


def _require_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise LocalInferenceError(
            "Local model comparison requires Transformers, PEFT, Accelerate, and "
            "Safetensors. Install the project requirements again."
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def _resolve_runtime(torch: Any, device: str, dtype: str) -> tuple[str, Any, str]:
    if device not in {"auto", "cpu", "cuda"}:
        raise LocalInferenceError("device must be auto, cpu, or cuda")
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise LocalInferenceError("CUDA was selected, but PyTorch found no CUDA GPU")

    dtype_names = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype == "auto":
        if resolved_device == "cpu":
            resolved_dtype = torch.float32
            dtype = "float32"
        elif torch.cuda.is_bf16_supported():
            resolved_dtype = torch.bfloat16
            dtype = "bfloat16"
        else:
            resolved_dtype = torch.float16
            dtype = "float16"
    elif dtype in dtype_names:
        resolved_dtype = dtype_names[dtype]
    else:
        raise LocalInferenceError(
            "dtype must be auto, float32, float16, or bfloat16"
        )
    if resolved_device == "cpu" and resolved_dtype == torch.float16:
        raise LocalInferenceError("float16 local inference is not supported on CPU")
    return resolved_device, resolved_dtype, dtype


def load_local_model_pair(
    spec: LocalAdapterSpec,
    *,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> LocalModelPair:
    """Load a base causal LM and attach its local inference-only PEFT adapter."""

    torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = _require_dependencies()
    resolved_device, resolved_dtype, dtype_name = _resolve_runtime(
        torch, device, dtype
    )
    try:
        base_model_source = spec.base_model_id
        if local_files_only:
            from huggingface_hub import snapshot_download

            base_model_source = snapshot_download(
                repo_id=spec.base_model_id,
                local_files_only=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.path), local_files_only=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_source,
            dtype=resolved_dtype,
            local_files_only=True if local_files_only else False,
        )
        base_model.to(resolved_device)
        model = PeftModel.from_pretrained(
            base_model,
            str(spec.path),
            is_trainable=False,
            local_files_only=True,
        )
    except Exception as error:
        mode = "local cache" if local_files_only else "local cache or Hugging Face Hub"
        raise LocalInferenceError(
            f"Could not load {spec.label} from the {mode}: {error}"
        ) from error

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return LocalModelPair(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=resolved_device,
        dtype=dtype_name,
    )


def format_local_prompt(tokenizer: Any, prompt: str, system_prompt: str = "") -> str:
    """Use an adapter's chat template when present, otherwise use plain text."""

    prompt = prompt.strip()
    if not prompt:
        raise LocalInferenceError("prompt cannot be empty")
    system_prompt = system_prompt.strip()
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{system_prompt}\n\n{prompt}" if system_prompt else prompt


def _input_device(model: Any) -> Any:
    return next(model.parameters()).device


def _generate_variant(
    bundle: LocalModelPair,
    formatted_prompt: str,
    config: LocalDecodingConfig,
    *,
    use_adapter: bool,
) -> tuple[str, float, int, int]:
    tokenizer = bundle.tokenizer
    torch = bundle.torch
    model = bundle.model
    model_limit = int(
        getattr(model.config, "max_position_embeddings", 0)
        or getattr(tokenizer, "model_max_length", 2048)
        or 2048
    )
    # Some tokenizers use a very large sentinel instead of a real model limit.
    if model_limit > 1_000_000:
        model_limit = 2048
    if config.max_new_tokens >= model_limit:
        raise LocalInferenceError(
            f"max_new_tokens must be below this model's {model_limit:,}-token "
            "context limit"
        )
    maximum_input_tokens = max(1, model_limit - config.max_new_tokens)
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_input_tokens,
    )
    inputs = {key: value.to(_input_device(model)) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.do_sample:
        kwargs.update(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )
    torch.manual_seed(config.seed)
    if bundle.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    adapter_context = nullcontext() if use_adapter else model.disable_adapter()
    started = time.perf_counter()
    with torch.inference_mode(), adapter_context:
        output_ids = model.generate(**inputs, **kwargs)
    latency = time.perf_counter() - started
    completion_ids = output_ids[0, input_tokens:]
    output = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return output, latency, input_tokens, int(completion_ids.shape[-1])


def generate_local_text(
    bundle: LocalModelPair,
    prompt: str,
    config: LocalDecodingConfig,
    *,
    system_prompt: str = "",
    use_adapter: bool = True,
) -> str:
    """Generate one completion from either the base or adapted model."""

    formatted_prompt = format_local_prompt(bundle.tokenizer, prompt, system_prompt)
    output, _, _, _ = _generate_variant(
        bundle,
        formatted_prompt,
        config,
        use_adapter=use_adapter,
    )
    return output


def run_local_comparison(
    bundle: LocalModelPair,
    prompt: str,
    configs: Sequence[LocalDecodingConfig],
    *,
    system_prompt: str = "",
) -> list[LocalComparisonResult]:
    """Generate base and adapted continuations for each decoding setup."""

    formatted_prompt = format_local_prompt(bundle.tokenizer, prompt, system_prompt)
    results: list[LocalComparisonResult] = []
    for config in configs:
        for variant, use_adapter in (("Base", False), ("Finetuned", True)):
            try:
                output, latency, input_tokens, output_tokens = _generate_variant(
                    bundle,
                    formatted_prompt,
                    config,
                    use_adapter=use_adapter,
                )
                error = None
            except Exception as generation_error:
                output = ""
                latency = 0.0
                input_tokens = 0
                output_tokens = 0
                error = str(generation_error)
            results.append(
                LocalComparisonResult(
                    adapter=bundle.spec.label,
                    base_model=bundle.spec.base_model_id,
                    variant=variant,
                    decoding=config.name,
                    prompt=prompt,
                    formatted_prompt=formatted_prompt,
                    output=output,
                    latency_seconds=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=error,
                )
            )
    return results


def local_comparison_csv(results: Sequence[LocalComparisonResult]) -> str:
    output = io.StringIO()
    fields = list(LocalComparisonResult.__dataclass_fields__)
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(result.as_dict() for result in results)
    return output.getvalue()
