"""Text-generation inference for the Himalaya Gemma 4 PEFT adapter."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ADAPTER_ID = "Govsovereign/himalaya-gemma-4-e2b-it-merged-s1-qlora"
DEFAULT_PROMPT = "नेपालका हिमालहरूको महत्त्वबारे छोटकरीमा लेख्नुहोस्।"
DEFAULT_CACHE_DIR = Path("data/cache/huggingface")
DEFAULT_OFFLOAD_DIR = DEFAULT_CACHE_DIR / "offload" / "himalaya_gemma"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    base_model_id: str
    device: str
    dtype: str


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Nepali or English text with the Himalaya Gemma 4 base "
            "model and its Govsovereign QLoRA/PEFT adapter."
        )
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--adapter-id", default=DEFAULT_ADAPTER_ID)
    parser.add_argument(
        "--adapter-revision",
        default="main",
        help="Adapter branch, tag, or commit (default: main)",
    )
    parser.add_argument(
        "--base-model-id",
        default=None,
        help=(
            "Override the base checkpoint. By default it is read from the "
            "adapter_config.json file."
        ),
    )
    parser.add_argument(
        "--base-revision",
        default="main",
        help="Base-model branch, tag, or commit (default: main)",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default=None,
        help=(
            "Placement strategy. Default: single CUDA device normally, auto "
            "when --cpu-offload is enabled."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Weight/compute dtype when the model is not quantized",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="bitsandbytes quantization; 4bit is recommended on limited VRAM",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help=(
            "Allow modules that do not fit in VRAM to run from CPU/disk. "
            "Offloaded quantized weights are held in full precision."
        ),
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=DEFAULT_OFFLOAD_DIR,
        help=f"Temporary disk-offload directory (default: {DEFAULT_OFFLOAD_DIR})",
    )
    parser.add_argument(
        "--gpu-max-memory",
        default=None,
        metavar="SIZE",
        help='Optional Accelerate limit such as "6GiB"',
    )
    parser.add_argument(
        "--cpu-max-memory",
        default=None,
        metavar="SIZE",
        help='Optional CPU offload limit such as "24GiB"',
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the optional thinking mode exposed by this chat template",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Hugging Face cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow repository Python code; not needed with supported Transformers",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve the full Python traceback when inference fails",
    )

    args = parser.parse_args(arguments)
    if not args.prompt.strip():
        parser.error("prompt must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be positive")
    if args.cpu_offload and args.quantization == "4bit":
        parser.error(
            "bitsandbytes CPU offload is not supported reliably for 4-bit "
            "models; use 4-bit without --cpu-offload or use "
            "--quantization 8bit --cpu-offload"
        )
    return args


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    return messages


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import transformers
        from peft import PeftConfig, PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Gemma inference dependencies are missing. Install them with "
            "`.venv/bin/pip install -r requirements-himalaya-gemma.txt`."
        ) from exc

    # Gemma 4 was added after the 4.x release installed by the NepBERTa path.
    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        raise RuntimeError(
            "This Transformers installation does not support Gemma 4 "
            f"(found {transformers.__version__}). Install the isolated Gemma "
            "requirements from requirements-himalaya-gemma.txt."
        )
    return (
        torch,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PeftConfig,
        PeftModel,
    )


def _resolve_device(torch: Any, requested: str) -> str:
    has_cuda = bool(torch.cuda.is_available())
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("--device cuda was requested, but PyTorch found no CUDA GPU")
    if requested == "auto":
        return "cuda" if has_cuda else "cpu"
    return requested


def _quantization_config(
    torch: Any,
    quantization: str,
    dtype: str,
    device: str,
    cpu_offload: bool,
) -> Any | None:
    if quantization == "none":
        return None
    if device != "cuda":
        raise RuntimeError(
            f"--quantization {quantization} requires a CUDA GPU and bitsandbytes"
        )
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("bitsandbytes quantization support is unavailable") from exc

    if dtype == "bfloat16":
        compute_dtype = torch.bfloat16
    elif dtype == "float32":
        compute_dtype = torch.float32
    elif dtype == "float16":
        compute_dtype = torch.float16
    else:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=cpu_offload,
    )


def load_model(args: argparse.Namespace) -> ModelBundle:
    """Load the full base checkpoint, then attach the inference-only adapter."""

    (
        torch,
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        PeftConfig,
        PeftModel,
    ) = _require_dependencies()
    device = _resolve_device(torch, args.device)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    adapter_load_kwargs: dict[str, Any] = {
        "revision": args.adapter_revision,
        "cache_dir": str(args.cache_dir),
        "local_files_only": args.local_files_only,
    }
    peft_config = PeftConfig.from_pretrained(
        args.adapter_id,
        **adapter_load_kwargs,
    )
    base_model_id = args.base_model_id or peft_config.base_model_name_or_path
    if not base_model_id:
        raise RuntimeError(
            "The adapter does not declare base_model_name_or_path; pass "
            "--base-model-id explicitly."
        )

    base_config = AutoConfig.from_pretrained(
        base_model_id,
        revision=args.base_revision,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_id,
        config=base_config,
        trust_remote_code=args.trust_remote_code,
        **adapter_load_kwargs,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _quantization_config(
        torch,
        args.quantization,
        args.dtype,
        device,
        args.cpu_offload,
    )
    if device == "cpu":
        device_map: str | dict[str, str | int] = {"": "cpu"}
    elif args.device_map == "auto" or args.cpu_offload:
        device_map = "auto"
    else:
        # On a single GPU, automatic mapping reserves additional swap space
        # for the largest layer. That can offload modules even when the final
        # quantized checkpoint itself fits, and bitsandbytes then rejects the
        # mixed placement. Direct placement avoids that conservative reserve.
        device_map = {"": 0}

    model_load_kwargs: dict[str, Any] = {
        "revision": args.base_revision,
        "cache_dir": str(args.cache_dir),
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": device_map,
        "attn_implementation": args.attn_implementation,
    }
    max_memory: dict[int | str, str] = {}
    if args.gpu_max_memory is not None:
        max_memory[0] = args.gpu_max_memory
    if args.cpu_max_memory is not None:
        max_memory["cpu"] = args.cpu_max_memory
    if max_memory:
        model_load_kwargs["max_memory"] = max_memory

    if args.cpu_offload:
        if device != "cuda":
            raise RuntimeError("--cpu-offload requires --device cuda or auto with CUDA")
        args.offload_dir.mkdir(parents=True, exist_ok=True)
        model_load_kwargs.update(
            offload_folder=str(args.offload_dir),
            offload_state_dict=True,
            offload_buffers=True,
        )
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    else:
        model_load_kwargs["dtype"] = args.dtype

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=base_config,
        **model_load_kwargs,
    )
    peft_dispatch_kwargs: dict[str, Any] = {}
    if args.cpu_offload:
        # PEFT removes the base model's Accelerate hooks, injects the adapter,
        # and dispatches the combined model a second time. It therefore needs
        # its own copy of the offload directory and memory policy.
        peft_dispatch_kwargs.update(
            device_map="auto",
            offload_dir=str(args.offload_dir),
        )
        if max_memory:
            peft_dispatch_kwargs["max_memory"] = max_memory
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_id,
        revision=args.adapter_revision,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
        is_trainable=False,
        low_cpu_mem_usage=True,
        **peft_dispatch_kwargs,
    )
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        base_model_id=base_model_id,
        device=device,
        dtype=str(getattr(model, "dtype", args.dtype)),
    )


def _model_input_device(model: Any, fallback_device: str) -> Any:
    """Return a real execution device, never an offloaded meta placeholder."""

    import torch

    embedding_device = model.get_input_embeddings().weight.device
    if embedding_device.type != "meta":
        return embedding_device

    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        base_model = getattr(model, "base_model", None)
        device_map = getattr(base_model, "hf_device_map", {})
    for placement in device_map.values():
        if isinstance(placement, int):
            return torch.device(f"cuda:{placement}")
        if isinstance(placement, str) and placement.startswith("cuda"):
            return torch.device(placement)
    return torch.device(fallback_device)


def _maximum_context(model: Any) -> int | None:
    config = model.config
    text_config = getattr(config, "text_config", None)
    return getattr(text_config or config, "max_position_embeddings", None)


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    return kwargs


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bundle = load_model(args)
    messages = build_messages(args.prompt, args.system_prompt)
    inputs = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=args.enable_thinking,
    )
    input_device = _model_input_device(bundle.model, bundle.device)
    inputs = {name: value.to(input_device) for name, value in inputs.items()}
    input_length = int(inputs["input_ids"].shape[-1])
    max_context = _maximum_context(bundle.model)
    if max_context is not None and input_length + args.max_new_tokens > max_context:
        raise ValueError(
            f"Prompt ({input_length} tokens) plus requested completion "
            f"({args.max_new_tokens}) exceeds the model context of {max_context}."
        )

    kwargs = generation_kwargs(args)
    if bundle.tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = bundle.tokenizer.pad_token_id
    if bundle.tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = bundle.model.generate(**inputs, **kwargs)
    completion_ids = output_ids[0, input_length:]
    completion = bundle.tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "adapter_model": args.adapter_id,
        "adapter_revision": args.adapter_revision,
        "base_model": bundle.base_model_id,
        "base_revision": args.base_revision,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "quantization": args.quantization,
        "cpu_offload": args.cpu_offload,
        "offload_dir": str(args.offload_dir) if args.cpu_offload else None,
        "prompt": args.prompt,
        "system_prompt": args.system_prompt,
        "thinking_enabled": args.enable_thinking,
        "prompt_tokens": input_length,
        "completion_tokens": int(completion_ids.numel()),
        "completion": completion,
    }


def print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print("\n=== Completion ===")
    print(result["completion"])
    print(
        "\n"
        f"Model: {result['adapter_model']} on {result['base_model']} | "
        f"device={result['device']} | quantization={result['quantization']}"
    )


def main() -> None:
    args = parse_args()
    try:
        result = run_inference(args)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (OSError, RuntimeError, ValueError) as exc:
        if args.debug:
            raise
        raise SystemExit(f"Error: {exc}") from exc
    print_result(result, as_json=args.json)
    if args.output is not None and not args.json:
        print(f"Saved result to: {args.output}")


if __name__ == "__main__":
    main()
