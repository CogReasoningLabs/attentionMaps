"""Named source selections shared by the embedding CLI and its results browser."""

import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from .semantic_source import SOURCE_KEYS, source_settings
from .semantic_settings import validate_embedding_settings

DEFAULT_SOURCE_REGISTRY = Path(__file__).resolve().parents[2] / "configs/datasets/sources.yaml"
PROVIDERS = {"huggingface": "Hugging Face", "kaggle": "Kaggle", "local": "Local"}
INSTANCE_KEYS = {"training_schema", "text_record_unit", "field_mapping", "field_parsers", "task_name"}
SELECTION_KEYS = set(SOURCE_KEYS) | INSTANCE_KEYS


def load_source_registry(path=DEFAULT_SOURCE_REGISTRY):
    """Read provider groups without downloading data or loading any model."""
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Source registry keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate source registry key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    path = Path(path).expanduser().resolve()
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid source registry YAML: {error}") from error
    if (not isinstance(document, dict) or set(document) != {"version", "sources"}
            or type(document["version"]) is not int or document["version"] != 1
            or not isinstance(document["sources"], dict)):
        raise ValueError("Source registry requires version: 1 and a sources mapping")
    unknown = set(document["sources"]) - set(PROVIDERS)
    if unknown:
        raise ValueError(f"Unknown source providers: {sorted(unknown)}")
    entries = {}
    for provider, group in document["sources"].items():
        if not isinstance(group, dict):
            raise ValueError(f"sources.{provider} must map source IDs to their settings")
        for identifier, value in group.items():
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
                raise ValueError(f"Invalid source ID {identifier!r}; use lowercase letters, digits, '-' or '_'")
            if identifier in entries:
                raise ValueError(f"Duplicate source ID across providers: {identifier}")
            if not isinstance(value, dict) or set(value) - (SELECTION_KEYS - {"provider"} | {"label"}):
                raise ValueError(f"Source {identifier}: use source/schema fields only; models belong in embeddings.yaml")
            label = value.get("label", identifier)
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Source {identifier}: label must be non-empty text")
            settings = {key: None for key in SELECTION_KEYS}
            settings.update(training_schema="auto", text_record_unit="line", field_mapping={}, field_parsers={}, row_filters={})
            settings.update({key: val for key, val in value.items() if key != "label"}, provider=provider)
            if provider == "local":
                if not settings["local"] or settings["dataset"]:
                    raise ValueError(f"Source {identifier}: local requires a local path and no dataset identifier")
            elif not settings["dataset"] or settings["local"]:
                raise ValueError(f"Source {identifier}: {provider} requires a dataset identifier and no local path")
            if provider == "kaggle" and not settings["dataset_file"]:
                raise ValueError(f"Source {identifier}: Kaggle requires dataset_file to identify the selected file")
            if provider != "huggingface" and any(settings[key] is not None for key in ("config", "split", "shards", "revision")):
                raise ValueError(f"Source {identifier}: config, split, shards, and revision apply only to Hugging Face")
            if provider != "kaggle" and settings["dataset_file"] is not None:
                raise ValueError(f"Source {identifier}: dataset_file applies only to Kaggle")
            if provider == "huggingface" and (settings["config"] == "*" or settings["split"] == "*"):
                raise ValueError(f"Source {identifier}: register one exact configuration/split per embedding source")
            settings = validate_embedding_settings(settings, path)
            # Reuse provider/selection validation, including incompatible fields and filters.
            settings.update(source_settings(SimpleNamespace(source_settings=None, **settings)))
            settings = {key: str(val) if isinstance(val, Path) else val for key, val in settings.items()}
            signature = hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            entries[identifier] = {"id": identifier, "label": label, "provider": provider,
                                   "settings": settings, "signature": signature, "registry": str(path)}
    return entries


def registered_defaults(defaults, raw, *, source_id=None, registry=None):
    """Replace the entire source definition while retaining model/run parameters."""
    identifier = source_id or defaults.get("source_id")
    if not identifier:
        return defaults
    flags = {"--source-settings", "--local", "--dataset", "--provider", "--dataset-file", "--config",
             "--split", "--revision", "--shard", "--text-column", "--language", "--training-schema",
             "--field-map", "--field-parser", "--task-name", "--text-record-unit"}
    if any(arg.split("=", 1)[0] in flags for arg in raw):
        raise ValueError("A source ID selects its complete source/schema definition; use another registry ID for a different selection")
    path = registry or defaults.get("source_registry") or DEFAULT_SOURCE_REGISTRY
    entries = load_source_registry(path)
    if identifier not in entries:
        raise ValueError(f"Unknown source ID {identifier!r}; list available IDs with cluster_dataset.py sources")
    entry = entries[identifier]
    resolved = {key: value for key, value in defaults.items() if key not in SELECTION_KEYS | {"source_settings"}}
    resolved.update(entry["settings"], source_settings=None, source_id=identifier, source_registry=Path(entry["registry"]),
                    source_registration=entry)
    return resolved


def report_provider(report):
    inventory = report.get("inventory", {})
    provider = inventory.get("provider") or report.get("settings", {}).get("provider")
    if provider in PROVIDERS:
        return provider
    kind = inventory.get("format")
    return kind if kind in PROVIDERS else "local"


def registered_report(entry, report):
    """Changed registrations and older runs remain separate, without relabelling history."""
    saved = report.get("settings", {}).get("source_registration") or {}
    return saved.get("id") == entry["id"] and saved.get("signature") == entry["signature"]
