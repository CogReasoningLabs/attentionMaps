"""Local inference for the custom-code HimalayaGPT 0.5B instruction model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    DecodingConfig,
)


DEFAULT_HIMALAYAGPT_MODEL_ID = "himalaya-ai/himalayagpt-0.5b-it"
DEFAULT_HIMALAYAGPT_REVISION = "10ef5130093789db77b186fc37e524754848bb3a"
HIMALAYAGPT_CONTEXT_LENGTH = 2_048
SPECIAL_TOKENS = (
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
)
STOP_TOKENS = ("<|assistant_end|>", "<|output_end|>", "<|user_start|>")
HIMALAYAGPT_REPETITION_PENALTY = 1.08


@dataclass
class HimalayaGPTBundle:
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
            "CUDA was selected for HimalayaGPT, but PyTorch found no CUDA GPU"
        )
    if device not in {"cpu", "cuda"}:
        raise ComparisonConfigurationError(
            "HimalayaGPT device must be auto, cpu, or cuda"
        )
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if dtype == "auto":
        dtype = (
            "bfloat16"
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else "float32"
        )
    if dtype == "float16":
        raise ComparisonConfigurationError(
            "HimalayaGPT follows its reference runner and supports float32 or bfloat16"
        )
    if dtype not in dtypes:
        raise ComparisonConfigurationError(
            "HimalayaGPT dtype must be auto, float32, or bfloat16"
        )
    return device, dtypes[dtype], dtype


def load_himalayagpt(
    *,
    model_id: str = DEFAULT_HIMALAYAGPT_MODEL_ID,
    revision: str = DEFAULT_HIMALAYAGPT_REVISION,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> HimalayaGPTBundle:
    """Load the pinned custom model code and weights from Hugging Face."""

    try:
        import tiktoken  # noqa: F401 - required to unpickle tokenizer.pkl
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ComparisonConfigurationError(
            "HimalayaGPT requires torch, transformers, safetensors, "
            "huggingface_hub, and tiktoken. Install requirements.txt in the "
            "same environment that runs Streamlit."
        ) from error

    resolved_device, torch_dtype, dtype_name = _runtime(torch, device, dtype)
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        # The upstream standalone runner defaults to a manual load. This avoids
        # meta-tensor/device-map edge cases in the custom Nanochat architecture.
        model_config = AutoConfig.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_config(
            model_config, trust_remote_code=True
        )
        state_dict = load_file(f"{snapshot}/model.safetensors", device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "HimalayaGPT state dictionary mismatch: "
                f"missing={len(incompatible.missing_keys)} "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
        del state_dict
        model = model.to(device=resolved_device, dtype=torch_dtype)
    except Exception as error:
        raise ComparisonConfigurationError(
            f"Could not load HimalayaGPT: {error}"
        ) from error
    model.eval()
    return HimalayaGPTBundle(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        dtype=dtype_name,
    )


def _special_id(tokenizer: Any, token: str) -> int | None:
    special_map = getattr(tokenizer, "_special_to_id", None)
    if isinstance(special_map, dict) and token in special_map:
        token_id = int(special_map[token])
        return token_id if token_id >= 0 else None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if (
        tokenizer.unk_token_id is not None
        and token_id == tokenizer.unk_token_id
        and token != tokenizer.unk_token
    ):
        return None
    return int(token_id)


def build_himalayagpt_prompt_ids(
    tokenizer: Any, prompt: str, *, system_prompt: str = "", vocab_size: int
) -> list[int]:
    """Apply the model's published user/assistant special-token framing."""

    user_text = prompt.strip()
    if system_prompt.strip():
        user_text = f"{system_prompt.strip()}\n\n{user_text}"
    user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
    chat_tokens = (
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
    )
    if not all(_special_id(tokenizer, token) is not None for token in chat_tokens):
        ids = user_ids
    else:
        required = [
            _special_id(tokenizer, token)
            for token in ("<|bos|>", *chat_tokens)
        ]
        if any(token_id is None for token_id in required):
            raise RuntimeError(
                "HimalayaGPT chat prompt was detected, but a special token is missing"
            )
        bos, user_start, user_end, assistant_start = required
        ids = [bos, user_start, *user_ids, user_end, assistant_start]
    return [min(max(int(token_id), 0), vocab_size - 1) for token_id in ids]


def _strip_special(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def generate_himalayagpt(
    bundle: HimalayaGPTBundle,
    prompt: str,
    config: DecodingConfig,
    system_prompt: str = "",
) -> str:
    """Generate with the model's reference-compatible manual token loop."""

    if not prompt.strip():
        raise ComparisonConfigurationError("HimalayaGPT prompt cannot be empty")
    if config.max_new_tokens >= HIMALAYAGPT_CONTEXT_LENGTH:
        raise ComparisonConfigurationError(
            f"HimalayaGPT max_new_tokens must be below {HIMALAYAGPT_CONTEXT_LENGTH:,}"
        )
    model = bundle.model
    tokenizer = bundle.tokenizer
    torch = bundle.torch
    vocab_size = int(
        getattr(model.config, "padded_vocab_size", None)
        or getattr(model.config, "vocab_size", len(tokenizer))
    )
    prompt_ids = build_himalayagpt_prompt_ids(
        tokenizer,
        prompt,
        system_prompt=system_prompt,
        vocab_size=vocab_size,
    )[: HIMALAYAGPT_CONTEXT_LENGTH - config.max_new_tokens]
    model_device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model_device)
    ids = input_ids
    stop_ids = {
        token_id
        for token_id in (_special_id(tokenizer, token) for token in STOP_TOKENS)
        if token_id is not None
    }
    generator = None
    if config.temperature > 0:
        generator = torch.Generator(device=model_device)
        generator.manual_seed(config.seed)

    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            attention_mask = torch.ones_like(ids)
            logits = model(
                input_ids=ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits[:, -1, :]
            seen_ids = torch.unique(ids[0])
            penalized = logits.clone()
            penalized[:, seen_ids] = (
                penalized[:, seen_ids] / HIMALAYAGPT_REPETITION_PENALTY
            )
            logits = penalized
            if config.top_k:
                top_values, _ = torch.topk(
                    logits, min(config.top_k, logits.shape[-1])
                )
                logits = logits.masked_fill(
                    logits < top_values[:, [-1]], -float("inf")
                )
            if config.temperature > 0:
                probabilities = torch.softmax(logits / config.temperature, dim=-1)
                next_ids = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            if int(next_ids.item()) in stop_ids:
                break

    completion_ids = ids[0, input_ids.shape[-1] :]
    return _strip_special(
        tokenizer.decode(completion_ids, skip_special_tokens=False)
    )


class HimalayaGPTBackend:
    """Shared-comparison adapter for HimalayaGPT instruction inference."""

    def __init__(self, bundle: HimalayaGPTBundle):
        self.bundle = bundle
        self.label = f"himalayagpt-it:{bundle.model_id}"

    def generate(
        self,
        prompt: str,
        config: DecodingConfig,
        system_prompt: str | None = None,
    ) -> str:
        return generate_himalayagpt(
            self.bundle, prompt, config, system_prompt or ""
        )
