"""Reusable settings-file inputs for the standalone dataset inspector."""

from __future__ import annotations

from pathlib import Path

from .sampling_vote import SAMPLING_STRATEGIES
from .row_selection import validate_row_filters
from .language_detection import DETECTION_DEFAULTS, detection_settings
from .classification_thresholds import DEFAULT_THRESHOLDS, resolve_thresholds

SETTINGS_KEYS = {
    "row_filters", "history_sheet", "record_history", "provider", "dataset", "dataset_file", "local", "revision", "config", "split", "shards", "sample_size",
    "text_columns", "languages", "seed", "output", "min_devanagari_ratio",
    "max_records", "filtered_output", "overwrite", "formats_only", "sample_fraction", "sampling_runs", "sampling_method",
    "concurrency", "batch_size",
}

SETTINGS_KEYS.update(DEFAULT_THRESHOLDS)
SETTINGS_KEYS.update(DETECTION_DEFAULTS)

def load_inspection_settings(path: Path) -> dict:
    import yaml

    path = path.expanduser().resolve()
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid inspection settings YAML: {error}") from error
    if not isinstance(settings, dict):
        raise ValueError("Inspection settings must be a YAML/JSON mapping")
    unknown = set(settings) - SETTINGS_KEYS
    if unknown:
        raise ValueError(f"Unknown inspection settings: {sorted(map(str, unknown))}")
    for key in ("local", "output", "filtered_output", "history_sheet", "language_detection_model"):
        if settings.get(key) is not None:
            if not isinstance(settings[key], str) or not settings[key].strip():
                raise ValueError(f"{key} must be a non-empty path")
            value = Path(settings[key]).expanduser()
            settings[key] = (path.parent / value).resolve()
    return settings


def validate_inspection_settings(settings: dict, *, listing: bool = False) -> None:
    resolve_thresholds(settings)
    detection_settings(settings)
    validate_row_filters(settings.get("row_filters"))
    if settings.get("output") and Path(settings["output"]).suffix.lower() != ".csv":
        raise ValueError("Inspection output must be a .csv file; per-run JSON reports are no longer written")
    if type(settings.get("record_history", True)) is not bool:
        raise ValueError("record_history must be true or false")
    if settings.get("record_history", True) and "history_sheet" in settings:
        value = settings["history_sheet"]
        if not isinstance(value, (str, Path)) or not str(value).strip() or Path(value).suffix.lower() != ".csv":
            raise ValueError("history_sheet must be a non-empty .csv path")
    if bool(settings.get("dataset")) == bool(settings.get("local")):
        raise ValueError("Choose exactly one dataset or local source")
    if settings.get("provider", "huggingface") not in {"huggingface", "kaggle", "local"}:
        raise ValueError("provider must be huggingface, kaggle, or local")
    for key in ("dataset", "dataset_file", "revision", "config", "split"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a non-empty string")
    for key in ("shards", "text_columns", "languages"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        )):
            raise ValueError(f"{key} must be a non-empty list of strings, or null")
    size = settings["sample_size"]
    if size is not None and (type(size) is not int or not 0 <= size <= 1000):
        raise ValueError("sample_size must be between 0 and 1000")
    fraction = settings.get("sample_fraction")
    if size is not None and fraction is not None:
        raise ValueError("Choose sample_fraction or legacy sample_size, not both")
    if size is None and (type(fraction) not in (int, float) or not 0 < fraction <= 1):
        raise ValueError("sample_fraction must be greater than 0 and at most 1")
    runs = settings.get("sampling_runs")
    if type(runs) is not int or runs < 1:
        raise ValueError("sampling_runs must be a positive integer")
    if size is not None and runs != 1:
        raise ValueError("Use sample_fraction for repeated voting; legacy sample_size requires sampling_runs: 1")
    method = settings.get("sampling_method")
    if not isinstance(method, str) or method not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling_method must be one of: {', '.join(SAMPLING_STRATEGIES)}")
    if type(settings["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key in ("concurrency", "batch_size"):
        value = settings.get(key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings.get("formats_only", False)) is not bool:
        raise ValueError("formats_only must be true or false")
    if type(settings["overwrite"]) is not bool:
        raise ValueError("overwrite must be true or false")
    ratio = settings.get("min_devanagari_ratio")
    if ratio is not None and (type(ratio) not in (int, float) or not 0 <= ratio <= 1):
        raise ValueError("min_devanagari_ratio must be between 0 and 1")
    limit = settings.get("max_records")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("max_records must be a positive integer")
    if settings.get("provider") != "huggingface" and any(settings.get(key) is not None for key in ("config", "split", "revision", "shards")):
        raise ValueError("config, split, revision, and shards apply only to Hugging Face datasets")
    if settings.get("provider") == "local" and not settings.get("local"):
        raise ValueError("provider local requires a local source")
    if settings.get("dataset_file") and settings.get("provider") != "kaggle":
        raise ValueError("dataset_file applies only to Kaggle datasets")
    if listing:
        if not settings.get("dataset"):
            raise ValueError("--list requires a Hugging Face or Kaggle dataset")
        return
    if settings.get("formats_only") and (ratio is not None or limit is not None or settings.get("filtered_output") or settings.get("row_filters")):
        raise ValueError("formats_only cannot be combined with filtering settings")
    if ratio is None and (limit is not None or settings.get("filtered_output")):
        raise ValueError("filtered_output/max_records require min_devanagari_ratio")
    if ratio is not None and not settings.get("filtered_output"):
        raise ValueError("Filtering requires filtered_output (a JSONL file)")
    output = settings.get("filtered_output")
    if output:
        if Path(output).suffix.lower() != ".jsonl":
            raise ValueError("filtered_output must end in .jsonl")
        if Path(output).exists() and not settings["overwrite"]:
            raise ValueError("Filtered output already exists; choose a new path or pass --overwrite")


def validate_output_paths(settings: dict, source_files: list[str]) -> None:
    """Reject report/data collisions before any source data is processed."""
    keys = ["output", "filtered_output"]
    if settings.get("record_history", True) and settings.get("history_sheet") and not settings.get("output"):
        keys.append("history_sheet")
    outputs = [Path(settings[key]).resolve() for key in keys if settings.get(key)]
    if len(outputs) != len(set(outputs)):
        raise ValueError("The report, filtered output, and history sheet must use different paths")
    for output in outputs:
        if output.is_dir():
            raise ValueError(f"Output path is a directory: {output}")
        for source in source_files:
            source_path = Path(source).resolve()
            if output == source_path or (output.exists() and source_path.exists() and output.samefile(source_path)):
                raise ValueError("An output path must not overwrite a source file")
