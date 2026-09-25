"""Shared model-independent YAML/JSON settings for embedding and clustering runs."""

from pathlib import Path

from .semantic_models import MODEL_PRESETS
from .semantic_instances import SCHEMAS, FIELD_NAMES
from .semantic_source import SOURCE_KEYS

RUN_KEYS = set(SOURCE_KEYS) | {
    "source_settings", "model", "model_id", "model_revision", "embedding_task",
    "device", "batch_size", "max_length", "max_records", "clusters",
    "cluster_batch_size", "cluster_epochs", "seed", "output_dir",
    "training_schema", "field_mapping", "task_name", "text_record_unit", "wordcloud_font",
}


def load_embedding_settings(path):
    import yaml

    path = Path(path).expanduser().resolve()
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid embedding settings YAML: {error}") from error
    if not isinstance(settings, dict):
        raise ValueError("Embedding settings must be a YAML/JSON mapping")
    unknown = set(settings) - RUN_KEYS
    if unknown:
        raise ValueError(f"Unknown embedding settings: {sorted(map(str, unknown))}")
    choices = {"training_schema": ("auto", *SCHEMAS), "text_record_unit": ("line", "blank_line"), "model": tuple(MODEL_PRESETS), "provider": ("huggingface", "kaggle"),
               "embedding_task": ("clustering", "similarity"), "device": ("auto", "cpu", "cuda")}
    for key, allowed in choices.items():
        if key in settings and not (key == "provider" and settings[key] is None) and settings[key] not in allowed:
            raise ValueError(f"{key} must be one of: {', '.join(allowed)}")
    for key in ("batch_size", "max_length", "max_records", "clusters", "cluster_batch_size", "cluster_epochs", "seed"):
        if key not in settings or (key in {"max_length", "max_records"} and settings[key] is None):
            continue
        minimum = 0 if key == "seed" else (16 if key == "max_length" else 1)
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("shards", "text_columns", "languages"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value)):
            raise ValueError(f"{key} must be a non-empty list of strings or null")
    for key in ("dataset", "dataset_file", "revision", "config", "split", "model_id", "model_revision", "task_name"):
        if key in settings:
            value = settings[key]
            if (value is not None or key == "model_revision") and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{key} must be a non-empty string")
    for key in ("source_settings", "local", "output_dir", "wordcloud_font"):
        value = settings.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty path")
            settings[key] = (path.parent / Path(value).expanduser()).resolve()
    mapping = settings.get("field_mapping")
    if mapping is not None and (not isinstance(mapping, dict) or any(
        key not in FIELD_NAMES or not isinstance(value, str) or not value.strip() for key, value in mapping.items()
    )):
        raise ValueError("field_mapping must map canonical field names to source column paths")
    model_id = settings.get("model_id")
    if model_id and model_id.startswith(("./", "../", "/", "~/")):
        settings["model_id"] = str((path.parent / Path(model_id).expanduser()).resolve())
    if settings.get("local") and settings.get("dataset"):
        raise ValueError("Choose local or dataset in embedding settings, not both")
    return settings


def apply_cli_overrides(defaults, raw):
    """Explicit selectors replace inherited sources/lists rather than appending."""
    defaults = dict(defaults)

    def supplied(flag):
        return any(item == flag or item.startswith(flag + "=") for item in raw)

    def explicit_value(flag):
        for index in range(len(raw) - 1, -1, -1):
            if raw[index].startswith(flag + "="):
                return raw[index].split("=", 1)[1]
            if raw[index] == flag:
                return raw[index + 1] if index + 1 < len(raw) else None
        return None

    if supplied("--model") and explicit_value("--model") != defaults.get("model", "nepali-bert"):
        defaults["model_id"] = None  # Do not carry another preset's repository override.
    if supplied("--provider") and explicit_value("--provider") != (defaults.get("provider") or "huggingface"):
        for key in ("config", "split", "shards", "revision", "dataset_file"):
            defaults[key] = None
    if supplied("--local"):
        for key in ("dataset", "config", "split", "shards", "revision", "dataset_file"):
            defaults[key] = None
    if supplied("--dataset"):
        defaults["local"] = None
    if supplied("--source-settings"):
        for key in SOURCE_KEYS:
            defaults[key] = None
    for flag, key in (("--field-map", "field_mapping"), ("--shard", "shards"), ("--text-column", "text_columns"), ("--language", "languages")):
        if supplied(flag):
            defaults[key] = None
    return defaults


def effective_run_settings(args, source):
    """Persist all effective run inputs without credentials for the report UI."""
    settings = {key: getattr(args, key, None) for key in RUN_KEYS}
    settings.update(source)
    settings["settings_file"] = getattr(args, "settings", None)
    return {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
