#!/usr/bin/env python3
"""Compare Nepali generation across hosted and local finetuned models.

The Streamlit dataset explorer provides the interactive version. This command
remains useful for reproducible batch comparisons and CSV export.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.inference.comparison import (
    ComparisonConfigurationError,
    ComparisonResult,
    DEFAULT_GEMINI_FLASH_LITE_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GOOGLE_GEMMA_MODEL,
    GoogleGenAIBackend,
    HuggingFaceBackend,
    LocalPeftBackend,
    build_decoding_grid,
    run_comparison,
)  # noqa: E402
from attention_maps.inference.local_comparison import (  # noqa: E402
    LocalInferenceError,
    discover_local_adapters,
    load_local_model_pair,
)
from attention_maps.inference.arkios import (  # noqa: E402
    DEFAULT_ARKIOS_MODEL_ID,
    DEFAULT_ARKIOS_REVISION,
    ArkiosBackend,
    load_arkios,
)
from attention_maps.inference.himalayagpt import (  # noqa: E402
    DEFAULT_HIMALAYAGPT_MODEL_ID,
    DEFAULT_HIMALAYAGPT_REVISION,
    HimalayaGPTBackend,
    load_himalayagpt,
)


DEFAULT_LOCAL_MODELS_ROOT = PROJECT_ROOT / "finetuned_models"
LOCAL_MODEL_KEYS = {
    "local-gpt2": "gpt2-alpaca-nepali-lora",
    "local-tinyllama": "tinyllama-nepali-alpaca-qlora",
}


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=(
            "gemini",
            "gemini-flash-lite",
            "google-gemma",
            "huggingface",
            "gemma",
            "local-gpt2",
            "local-tinyllama",
            "himalayagpt",
            "arkios",
        ),
        help=(
            "One or more hosted or local backends; legacy 'gemma' is an alias "
            "for 'huggingface'. Local adapter choices load from finetuned_models/; "
            "Arkios and HimalayaGPT load cached or downloaded Hub snapshots."
        ),
    )
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--gemini-flash-lite-model",
        default=DEFAULT_GEMINI_FLASH_LITE_MODEL,
    )
    parser.add_argument(
        "--google-gemma-model",
        default=DEFAULT_GOOGLE_GEMMA_MODEL,
        help="Gemma model served by Google's generate_content API",
    )
    parser.add_argument(
        "--hf-model",
        "--gemma-model",
        dest="hf_model",
        default="google/gemma-2-2b-it",
    )
    parser.add_argument(
        "--hf-provider",
        default="auto",
        help="Hugging Face Inference Provider, or auto",
    )
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--arkios-model", default=DEFAULT_ARKIOS_MODEL_ID)
    parser.add_argument("--arkios-revision", default=DEFAULT_ARKIOS_REVISION)
    parser.add_argument(
        "--himalayagpt-model", default=DEFAULT_HIMALAYAGPT_MODEL_ID
    )
    parser.add_argument(
        "--himalayagpt-revision", default=DEFAULT_HIMALAYAGPT_REVISION
    )
    parser.add_argument(
        "--local-models-root",
        type=Path,
        default=DEFAULT_LOCAL_MODELS_ROOT,
        help="Directory containing local PEFT adapter directories",
    )
    parser.add_argument(
        "--local-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--local-dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download missing base-model weights from Hugging Face",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-file", type=Path)
    source.add_argument("--dataset-name", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.ne")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument(
        "--prompt-template",
        default="निम्न पाठलाई नेपालीमा संक्षेप गर्नुहोस्:\n\n{text}",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system instruction applied separately from the user prompt",
    )

    parser.add_argument("--temperatures", nargs="+", type=float, default=(0.2, 0.9))
    parser.add_argument("--top-p", nargs="+", type=float, default=(0.95,))
    parser.add_argument("--top-k", nargs="+", type=int, default=(40,))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
        help="Google model thinking effort; thinking shares the output-token budget",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nepali_inference_results.csv"),
    )
    return parser.parse_args(arguments)


def load_nepali_prompts(args: argparse.Namespace) -> list[str]:
    if args.num_samples <= 0:
        raise ComparisonConfigurationError("--num-samples must be positive")
    if args.max_chars <= 0:
        raise ComparisonConfigurationError("--max-chars must be positive")

    if args.input_file:
        try:
            lines = [
                line.strip()[: args.max_chars]
                for line in args.input_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as error:
            raise ComparisonConfigurationError(
                f"could not read {args.input_file}: {error}"
            ) from error
        if not lines:
            raise ComparisonConfigurationError(
                f"no non-empty lines found in {args.input_file}"
            )
        return lines[: args.num_samples]

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ComparisonConfigurationError(
            "Hugging Face dataset loading requires the `datasets` package"
        ) from error

    config_name = args.dataset_config or None
    try:
        dataset = load_dataset(
            args.dataset_name,
            config_name,
            split=args.dataset_split,
            streaming=True,
        )
    except Exception as error:
        raise ComparisonConfigurationError(
            f"could not stream dataset {args.dataset_name!r}: {error}"
        ) from error

    texts: list[str] = []
    for row in dataset:
        value = row.get(args.text_column)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip()[: args.max_chars])
        if len(texts) >= args.num_samples:
            break
    if not texts:
        raise ComparisonConfigurationError(
            f"no text found in dataset column {args.text_column!r}"
        )
    return texts


def build_backends(args: argparse.Namespace):
    names = set(args.models)
    backends = []
    if "gemini" in names:
        backends.append(
            GoogleGenAIBackend(
                args.gemini_model,
                args.gemini_api_key or os.getenv("GEMINI_API_KEY"),
            )
        )
    if "gemini-flash-lite" in names:
        backends.append(
            GoogleGenAIBackend(
                args.gemini_flash_lite_model,
                args.gemini_api_key or os.getenv("GEMINI_API_KEY"),
            )
        )
    if "google-gemma" in names:
        backends.append(
            GoogleGenAIBackend(
                args.google_gemma_model,
                args.gemini_api_key or os.getenv("GEMINI_API_KEY"),
            )
        )
    if names.intersection({"huggingface", "gemma"}):
        backends.append(
            HuggingFaceBackend(
                args.hf_model,
                args.hf_token or os.getenv("HF_TOKEN"),
                args.hf_provider,
            )
        )
    local_names = names.intersection(LOCAL_MODEL_KEYS)
    if local_names:
        discovered = {
            spec.key: spec
            for spec in discover_local_adapters(args.local_models_root)
        }
        for local_name in sorted(local_names):
            adapter_key = LOCAL_MODEL_KEYS[local_name]
            spec = discovered.get(adapter_key)
            if spec is None:
                raise ComparisonConfigurationError(
                    f"local adapter {adapter_key!r} was not found under "
                    f"{args.local_models_root}"
                )
            try:
                bundle = load_local_model_pair(
                    spec,
                    device=args.local_device,
                    dtype=args.local_dtype,
                    local_files_only=args.local_files_only,
                )
            except LocalInferenceError as error:
                raise ComparisonConfigurationError(str(error)) from error
            backends.append(LocalPeftBackend(bundle))
    if "himalayagpt" in names:
        backends.append(
            HimalayaGPTBackend(
                load_himalayagpt(
                    model_id=args.himalayagpt_model,
                    revision=args.himalayagpt_revision,
                    device=args.local_device,
                    dtype=args.local_dtype,
                    local_files_only=args.local_files_only,
                )
            )
        )
    if "arkios" in names:
        backends.append(
            ArkiosBackend(
                load_arkios(
                    model_id=args.arkios_model,
                    revision=args.arkios_revision,
                    device=args.local_device,
                    dtype=args.local_dtype,
                    local_files_only=args.local_files_only,
                )
            )
        )
    return backends


def save_results(results: list[ComparisonResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].as_dict()) if results else []
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.as_dict() for result in results)


def print_results(results: list[ComparisonResult]) -> None:
    for result in results:
        content = f"ERROR: {result.error}" if result.error else result.output
        print(
            f"\n[{result.model} | {result.decoding} | "
            f"{result.latency_seconds:.3f}s]\n{content}"
        )


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        prompts = load_nepali_prompts(args)
        configs = build_decoding_grid(
            args.temperatures,
            args.top_p,
            args.top_k,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            thinking_level=args.thinking_level,
        )
        results = run_comparison(
            build_backends(args),
            prompts,
            configs,
            prompt_template=args.prompt_template,
            system_prompt=args.system_prompt,
        )
    except ComparisonConfigurationError as error:
        raise SystemExit(f"Error: {error}") from error

    print_results(results)
    save_results(results, args.output)
    print(f"\nSaved {len(results)} comparison rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
