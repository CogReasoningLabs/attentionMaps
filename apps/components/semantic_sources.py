"""Select registered datasets or historical runs without loading source data."""

import os
from pathlib import Path
import re
import shlex

from attention_maps.explorer.semantic_models import MODEL_PRESETS
from attention_maps.explorer.semantic_settings import load_embedding_settings
from attention_maps.explorer.source_registry import (
    DEFAULT_SOURCE_REGISTRY, PROVIDERS, load_source_registry, registered_report, report_provider,
)


def select_source_runs(st, found, directory, dataset_label):
    with st.expander("Dataset source registry"):
        registry_path = st.text_input("Source registry file", value=os.getenv("DATASET_SOURCE_REGISTRY", str(DEFAULT_SOURCE_REGISTRY)))
        st.caption("Maintain named datasets in this YAML under huggingface, kaggle, or local. Each entry stores its identifier, selected files, and schema.")
    try:
        entries = load_source_registry(registry_path)
    except (OSError, ValueError) as error:
        st.warning(f"Cannot read source registry: {error}. Saved runs are still available.")
        entries = {}
    providers = list(PROVIDERS)
    default_provider = report_provider(found[0][1]) if found else providers[0]
    provider = st.selectbox("Source type", providers, index=providers.index(default_provider),
                            format_func=PROVIDERS.get, key="embedding-source-provider")
    options = {}
    assigned = set()
    for identifier, entry in entries.items():
        if entry["provider"] != provider:
            continue
        matches = [(path, report) for path, report in found if registered_report(entry, report)]
        options[identifier] = {"label": f"{entry['label']} · {identifier}", "entry": entry, "runs": matches}
        assigned.update(path for path, _ in matches)
    for path, report in found:
        if report_provider(report) != provider or path in assigned:
            continue
        label = dataset_label(report)
        option = options.setdefault("saved:" + label, {"label": label + " · saved", "entry": None, "runs": []})
        option["runs"].append((path, report))
    if not options:
        st.info(f"No {PROVIDERS[provider]} sources or saved runs yet. Add a dataset to {registry_path}.")
        return []
    keys = list(options)
    ready = next((key for key in keys if options[key]["runs"]), keys[0])
    selected = st.selectbox("Dataset", keys, index=keys.index(ready), format_func=lambda key: options[key]["label"],
                            key=f"embedding-source-dataset:{provider}:{registry_path}")
    option = options[selected]
    if option["entry"]:
        entry = option["entry"]
        settings = entry["settings"]
        st.caption(f"Source ID: {entry['id']} · {settings['dataset'] or settings['local']}")
        with st.expander("Source selection and schema"):
            st.json(settings)
        _render_run_command(st, entry, directory, expanded=not option["runs"])
    if not option["runs"]:
        st.info("No completed embedding runs for this source selection yet. Run the command above, then refresh saved runs.")
    return option["runs"]


def _render_run_command(st, entry, directory, *, expanded):
    with st.expander("Generate embeddings for this dataset", expanded=expanded):
        settings_path = st.text_input("Embedding settings file", value="configs/embeddings.yaml", key="embedding-run-settings")
        try:
            settings = load_embedding_settings(settings_path)
        except (OSError, ValueError) as error:
            st.warning(f"Cannot prepare run command: {error}")
            return
        models = list(MODEL_PRESETS)
        model = st.selectbox("Model for new run", models, index=models.index(settings.get("model", models[0])),
                             key="embedding-new-model")
        st.caption(f"Checkpoint: {settings.get('model_id') if model == settings.get('model', models[0]) and settings.get('model_id') else MODEL_PRESETS[model]['id']}")
        limit = settings.get("max_records")
        scope = f"First {limit:,} records" if limit is not None else "Entire selected dataset"
        st.caption(f"{scope} · {settings.get('clusters', 10)} clusters · seed {settings.get('seed', 42)}. Edit the embedding settings file to change these run parameters.")
        root = Path(directory).expanduser()
        if (root / "report.json").is_file():
            root = root.parent
        name = st.text_input("New embedding run name", value=f"{entry['id']}-{model}-run01",
                             key=f"embedding-new-run:{entry['id']}:{model}")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            st.warning("Use a run name containing letters, digits, dots, hyphens, or underscores.")
            return
        destination = root / name
        if destination.exists():
            st.warning("That run directory already exists. Choose a new name to preserve the saved experiment.")
            return
        if entry["provider"] == "local" and not Path(entry["settings"]["local"]).exists():
            st.warning("The registered local source is currently unavailable. Restore it before running a new embedding job.")
        command = ["venv/bin/python", "scripts/cluster_dataset.py", "run", "--settings", settings_path,
                   "--source-registry", entry["registry"], "--source-id", entry["id"], "--model", model,
                   "--output-dir", str(destination)]
        st.code(shlex.join(command), language="bash")
        st.caption("Run from the project root. This saves a new experiment; model and dataset processing happen in the script.")
