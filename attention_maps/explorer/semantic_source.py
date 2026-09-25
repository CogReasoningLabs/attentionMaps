"""Resolve the same dataset selections as inspection, without running analysis."""

from pathlib import Path

from .huggingface import discover_huggingface_dataset, inspect_huggingface_configuration, inspect_huggingface_selection
from .inspection import file_signatures, inspect_dataset
from .inspection_settings import load_inspection_settings
from .kaggle_inspection import discover_kaggle_dataset, kaggle_selection, stage_kaggle_selection, inspect_kaggle_selection
from .text import text_columns
from .row_selection import with_row_filters, validate_row_filters

SOURCE_KEYS = ("row_filters", "provider", "dataset", "local", "revision", "config", "split", "shards", "dataset_file", "text_columns", "languages")


def source_settings(args) -> dict:
    saved = load_inspection_settings(args.source_settings) if args.source_settings else {}
    if saved.get("min_devanagari_ratio") is not None:
        raise ValueError("Run inspection filtering first, then use --local with its filtered JSONL output")
    settings = {key: saved.get(key) for key in SOURCE_KEYS}
    for key in SOURCE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if args.provider is not None and args.provider != (saved.get("provider") or "huggingface"):
        for key in ("config", "split", "shards", "revision", "dataset_file"):
            if getattr(args, key, None) is None:
                settings[key] = None
    if args.local is not None:
        settings.update(dataset=None, config=None, split=None, shards=None, revision=None, dataset_file=None)
    if args.dataset is not None:
        settings["local"] = None
    if bool(settings["local"]) == bool(settings["dataset"]):
        raise ValueError("Choose exactly one --local or --dataset, directly or in --source-settings")
    if settings["dataset"] and not settings["provider"]:
        raise ValueError("Remote datasets require an explicit provider: set provider: huggingface or provider: kaggle in YAML, or pass --provider")
    settings["provider"] = "local" if settings["local"] else settings["provider"]
    if settings["provider"] not in {"local", "huggingface", "kaggle"}:
        raise ValueError("Unsupported source provider")
    for key in ("shards", "text_columns", "languages"):
        value = settings[key]
        if value is not None and (not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value)):
            raise ValueError(f"{key} must be a non-empty list of strings")
    for key in ("dataset", "dataset_file", "revision", "config", "split"):
        value = settings[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a non-empty string")
    if settings["provider"] != "huggingface" and any(settings[k] is not None for k in ("config", "split", "shards", "revision")):
        raise ValueError("Configuration, split, shards, and revision apply only to Hugging Face sources")
    if settings["provider"] != "kaggle" and settings["dataset_file"]:
        raise ValueError("dataset_file applies only to Kaggle")
    validate_row_filters(settings.get("row_filters"))
    if settings["local"]:
        settings["local"] = str(Path(settings["local"]).expanduser().resolve())
    return settings


def resolve_source(settings: dict, token: str | None = None, *, validate_text_fields=True) -> tuple[dict, list[str]]:
    if settings["provider"] == "local":
        path = Path(settings["local"])
        paths = tuple(sorted(path.rglob("*.parquet"))) if path.is_dir() else (path,)
        if not paths:
            raise ValueError("Local directories must contain Parquet files")
        signatures = file_signatures(paths)
        inventory = inspect_dataset(signatures)
        inventory.update(source_files=list(map(str, paths)), source_signatures=[list(x) for x in signatures])
    elif settings["provider"] == "kaggle":
        catalog = discover_kaggle_dataset(settings["dataset"])
        selected = kaggle_selection(catalog, settings["dataset_file"], metadata_only=False)
        path = stage_kaggle_selection(catalog, selected)
        inventory = inspect_kaggle_selection(catalog, selected, cached_path=path)
    else:
        catalog = discover_huggingface_dataset(settings["dataset"], revision=settings["revision"], token=token)
        config = settings["config"]
        if config is None:
            if len(catalog["configs"]) != 1:
                raise ValueError("Choose --config; discover choices with scripts/inspect_dataset.py --list")
            config = catalog["configs"][0]
        configuration = inspect_huggingface_configuration(catalog, config, token=token)
        split = settings["split"]
        if split is None:
            if len(configuration["splits"]) != 1:
                raise ValueError("Choose --split; the configuration has multiple splits")
            split = next(iter(configuration["splits"]))
        shards = settings["shards"]
        if shards is not None:
            names = {x["name"]: x["path"] for x in configuration["splits"].get(split, {}).get("shards", [])}
            shards = [names.get(x, x) for x in shards]
        inventory = inspect_huggingface_selection(configuration, split, shards=shards, token=token)
    inventory = with_row_filters(inventory, settings.get("row_filters"))
    fields = settings["text_columns"] or text_columns(inventory["schema"])
    if not settings["text_columns"]:
        fields = [field for field in fields if field not in inventory.get("row_filters", {})]
    if validate_text_fields and (not fields or any(field.split(".")[0] not in inventory["columns"] for field in fields)):
        raise ValueError("Choose valid --text-column fields for embedding")
    if settings.get("languages"):
        inventory["declared_languages"] = settings["languages"]
    return inventory, list(fields)


def verify_source(inventory: dict) -> None:
    signatures = inventory.get("source_signatures")
    if signatures and [list(x) for x in file_signatures(tuple(Path(x[0]) for x in signatures))] != signatures:
        raise ValueError("Source files changed during embedding; result was not published")
