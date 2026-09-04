from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from persistence import load_configurations, load_experiments, load_prompts, save_configurations, save_experiments, save_prompts
from translator import DEFAULT_GEMINI_MODEL, DEFAULT_GEMMA_MODEL, DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_TEMPLATE, GenerationSettings, conversation_to_text, load_lima_sample, results_json, translate

st.set_page_config(page_title="LIMA Translation Workbench", layout="wide")
st.title("LIMA Translation Workbench")
st.caption("Sample five LIMA conversations, translate them, compare variants, and record your assessment.")

st.session_state.setdefault("records", [])
st.session_state.setdefault("sample_seed", 42)
# Keep the editable widget value separate from the seed for the records that
# are currently on screen. Without an explicit widget key, Streamlit can
# retain a prior value across reruns while ``sample_seed`` is updated only
# after loading, making it appear that a changed seed was ignored.
st.session_state.setdefault("sample_seed_input", st.session_state.sample_seed)
if "runs" not in st.session_state:
    st.session_state.runs = load_experiments()
if "prompt_templates" not in st.session_state:
    st.session_state.prompt_templates = load_prompts()
    if not any(prompt.get("name") == DEFAULT_PROMPT_NAME for prompt in st.session_state.prompt_templates):
        st.session_state.prompt_templates.insert(0, {"name": DEFAULT_PROMPT_NAME, "template": DEFAULT_PROMPT_TEMPLATE})
        save_prompts(st.session_state.prompt_templates)
if "experiment_configurations" not in st.session_state:
    st.session_state.experiment_configurations = load_configurations() or [
        {"name": "Flash-Lite balanced", "enabled": True, "provider": "Google Gemini API", "model": DEFAULT_GEMINI_MODEL, "temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 4096},
        {"name": "Gemma balanced", "enabled": True, "provider": "Google Gemini API", "model": DEFAULT_GEMMA_MODEL, "temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 4096},
    ]
    save_configurations(st.session_state.experiment_configurations)
for index, configuration in enumerate(st.session_state.experiment_configurations):
    configuration.setdefault("id", f"configuration-{index + 1}")

with st.sidebar:
    st.header("Dataset")
    seed = st.number_input("Random seed", min_value=0, step=1, key="sample_seed_input")
    if st.button("Load 5 random records", type="primary", use_container_width=True):
        try:
            with st.spinner("Downloading and sampling GAIR/lima…"):
                st.session_state.records = load_lima_sample(5, int(seed))
                st.session_state.sample_seed = int(seed)
            st.success("Loaded 5 records.")
        except Exception as exc:
            st.error(f"Could not load LIMA: {exc}")

    st.divider()
    st.header("Translation settings")
    provider = st.selectbox("Provider", ["Google Gemini API", "OpenAI API"])
    if provider == "Google Gemini API":
        model_choice = st.selectbox("Model", [DEFAULT_GEMINI_MODEL, DEFAULT_GEMMA_MODEL, "Custom…"])
        model = st.text_input("Google model ID", value=DEFAULT_GEMINI_MODEL) if model_choice == "Custom…" else model_choice
        api_ready = bool(os.getenv("Gemini_api"))
    else:
        model, api_ready = st.text_input("OpenAI model ID", value="gpt-4.1-mini"), bool(os.getenv("OPENAI_API_KEY"))
    target_language = st.text_input("Target language", value="Nepali")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.95, 0.01)
    top_k = st.slider("Top-k (Google only)", 1, 100, 40)
    max_output_tokens = st.number_input("Maximum output tokens", 128, 32768, 4096, 128)
    if not api_ready:
        st.warning("Selected provider key is not available in .env.")

    st.divider()
    st.header("Prompt experiment")
    prompt_names = [prompt["name"] for prompt in st.session_state.prompt_templates]
    selected_prompt_name = st.selectbox("Saved prompt", prompt_names)
    selected_prompt = next(prompt for prompt in st.session_state.prompt_templates if prompt["name"] == selected_prompt_name)
    prompt_name = st.text_input("Prompt name", value=selected_prompt_name, key=f"prompt_name_{selected_prompt_name}")
    prompt_template = st.text_area(
        "Prompt template",
        value=selected_prompt["template"],
        height=260,
        key=f"prompt_template_{selected_prompt_name}",
        help="Use {source} for the LIMA conversation and {target_language} for the selected language.",
    )
    if st.button("Save prompt", use_container_width=True):
        prompt_name = prompt_name.strip()
        if not prompt_name:
            st.error("A prompt name is required.")
        elif "{source}" not in prompt_template:
            st.error("The prompt template must include {source}.")
        else:
            saved_prompt = {"name": prompt_name, "template": prompt_template, "updated_at": datetime.now(timezone.utc).isoformat()}
            st.session_state.prompt_templates = [prompt for prompt in st.session_state.prompt_templates if prompt["name"] not in {selected_prompt_name, prompt_name}]
            st.session_state.prompt_templates.append(saved_prompt)
            save_prompts(st.session_state.prompt_templates)
            st.success(f"Saved prompt: {prompt_name}")

    st.divider()
    st.header("Batch execution")
    batch_concurrency = st.slider("Simultaneous requests", 1, 5, 2, help="Increase carefully; your provider's rate limits still apply.")
    show_archived = st.checkbox("Show archived runs in history", value=False)

settings = GenerationSettings(provider, model, target_language, temperature, top_p, top_k, int(max_output_tokens))

st.divider()
show_parameter_sets = st.toggle("Show batch parameter sets", value=False, help="Show or hide the parameter-set editor without changing saved configurations.")
if show_parameter_sets:
    batch_title_col, batch_add_col = st.columns([5, 1])
    with batch_title_col:
        st.subheader("Batch parameter sets")
        st.caption("Configure named variants here. Changes are saved automatically; enable only the variants you want to run.")
    with batch_add_col:
        add_configuration = st.button("Add parameter set", use_container_width=True)

    if add_configuration:
        next_number = len(st.session_state.experiment_configurations) + 1
        st.session_state.experiment_configurations.append({"id": f"configuration-{datetime.now(timezone.utc).timestamp()}", "name": f"New experiment {next_number}", "enabled": True, "provider": "Google Gemini API", "model": DEFAULT_GEMINI_MODEL, "temperature": 0.2, "top_p": 0.95, "top_k": 40, "max_output_tokens": 4096})
        save_configurations(st.session_state.experiment_configurations)
        st.rerun()

    updated_configurations = []
    configuration_to_remove = None
    configuration_to_duplicate = None
    for configuration in st.session_state.experiment_configurations:
        configuration_id = configuration["id"]
        with st.container(border=True):
            heading_col, enabled_col, duplicate_col, remove_col = st.columns([5, 1.2, 1.2, 1])
            with heading_col:
                configuration_name = st.text_input("Experiment name", value=str(configuration.get("name", "")), key=f"configuration_name_{configuration_id}")
            with enabled_col:
                configuration_enabled = st.toggle("Include", value=bool(configuration.get("enabled", True)), key=f"configuration_enabled_{configuration_id}")
            with duplicate_col:
                if st.button("Duplicate", key=f"duplicate_{configuration_id}", use_container_width=True):
                    configuration_to_duplicate = configuration.copy()
            with remove_col:
                if st.button("Remove", key=f"remove_{configuration_id}", use_container_width=True):
                    configuration_to_remove = configuration_id

            parameter_left, parameter_right = st.columns(2)
            with parameter_left:
                provider_options = ["Google Gemini API", "OpenAI API"]
                current_provider = str(configuration.get("provider", "Google Gemini API"))
                configuration_provider = st.selectbox("Provider", provider_options, index=provider_options.index(current_provider) if current_provider in provider_options else 0, key=f"configuration_provider_{configuration_id}")
                default_model = DEFAULT_GEMINI_MODEL if configuration_provider == "Google Gemini API" else "gpt-4.1-mini"
                configuration_model = st.text_input("Model ID", value=str(configuration.get("model", default_model)), key=f"configuration_model_{configuration_id}")
                configuration_temperature = st.slider("Temperature", 0.0, 2.0, float(configuration.get("temperature", 0.2)), 0.05, key=f"configuration_temperature_{configuration_id}")
            with parameter_right:
                configuration_top_p = st.slider("Top-p", 0.0, 1.0, float(configuration.get("top_p", 0.95)), 0.01, key=f"configuration_top_p_{configuration_id}")
                configuration_top_k = st.slider("Top-k", 1, 100, int(configuration.get("top_k", 40)), key=f"configuration_top_k_{configuration_id}", help="Used by Google models.")
                configuration_max_tokens = st.number_input("Maximum output tokens", 128, 32768, int(configuration.get("max_output_tokens", 4096)), 128, key=f"configuration_max_tokens_{configuration_id}")
            updated_configurations.append({"id": configuration_id, "name": configuration_name.strip(), "enabled": configuration_enabled, "provider": configuration_provider, "model": configuration_model.strip(), "temperature": configuration_temperature, "top_p": configuration_top_p, "top_k": configuration_top_k, "max_output_tokens": int(configuration_max_tokens)})

    if configuration_to_remove:
        updated_configurations = [configuration for configuration in updated_configurations if configuration["id"] != configuration_to_remove]
    if configuration_to_duplicate:
        duplicate = configuration_to_duplicate.copy()
        duplicate["id"] = f"configuration-{datetime.now(timezone.utc).timestamp()}"
        duplicate["name"] = f"{duplicate.get('name', 'Experiment')} copy"
        updated_configurations.append(duplicate)
    if updated_configurations != st.session_state.experiment_configurations:
        st.session_state.experiment_configurations = updated_configurations
        save_configurations(updated_configurations)
        if configuration_to_remove or configuration_to_duplicate:
            st.rerun()

if not st.session_state.records:
    st.info("Choose a seed and click **Load 5 random records** to begin.")
else:
    st.write(f"Seed **{st.session_state.sample_seed}** · {len(st.session_state.records)} sampled conversations")
    translate_all = st.button("Translate all 5", disabled=not api_ready)
    selected_indices: list[int] = list(range(len(st.session_state.records))) if translate_all else []
    active_configurations = [config for config in st.session_state.experiment_configurations if config.get("enabled") and str(config.get("name", "")).strip() and str(config.get("model", "")).strip()]
    batch_ready = all(
        (config.get("provider") == "Google Gemini API" and bool(os.getenv("Gemini_api")))
        or (config.get("provider") == "OpenAI API" and bool(os.getenv("OPENAI_API_KEY")))
        for config in active_configurations
    )
    batch_scope = st.radio("Batch target", ["All 5 samples", "One selected record"], horizontal=True)
    if batch_scope == "One selected record":
        batch_record_number = st.selectbox("Record for this batch", range(1, len(st.session_state.records) + 1))
        batch_record_indices = [batch_record_number - 1]
    else:
        batch_record_indices = list(range(len(st.session_state.records)))
    batch_experiment_label = st.text_input("Batch experiment name", value="LIMA parameter comparison", help="This name groups all variants and records from one batch into one JSON file.")
    st.write(f"**{len(active_configurations)}** enabled parameter set(s) · **{len(active_configurations) * len(batch_record_indices)}** batch translations")
    run_batch = st.button("Run enabled parameter sets", type="primary", disabled=not active_configurations or not batch_ready)

    for record_index, record in enumerate(st.session_state.records):
        source = conversation_to_text(record)
        with st.expander(f"Record {record_index + 1}", expanded=record_index == 0):
            source_col, output_col = st.columns(2)
            with source_col:
                st.subheader("Original")
                source_text_tab, source_preview_tab = st.tabs(["Text", "Markdown preview"])
                with source_text_tab:
                    st.text_area("Source", source, height=290, disabled=True, key=f"source_{st.session_state.sample_seed}_{record_index}")
                with source_preview_tab:
                    st.markdown(source)
            with output_col:
                st.subheader("Translations")
                if st.button("Translate this record", key=f"translate_{record_index}", disabled=not api_ready):
                    selected_indices.append(record_index)
                if st.button("Run enabled parameter sets for this record", key=f"batch_record_{record_index}", disabled=not active_configurations or not batch_ready):
                    batch_record_indices = [record_index]
                    run_batch = True
                matching_runs = [r for r in st.session_state.runs if r["record_index"] == record_index and r.get("sample_seed") == st.session_state.sample_seed and not r.get("archived", False)]
                if matching_runs:
                    tabs = st.tabs([f"{r.get('experiment_name', f'Run {i + 1}')}: {r['settings']['model']}" for i, r in enumerate(matching_runs)])
                    for tab, run in zip(tabs, matching_runs):
                        with tab:
                            st.caption(f"Prompt: {run.get('prompt_name', 'Unknown')} · {run['elapsed_seconds']} s · {run['usage'].get('total_tokens') or 'unknown'} total tokens")
                            with st.expander("Parameters used"):
                                st.json(run["settings"])
                            translation_text_tab, translation_preview_tab = st.tabs(["Text", "Markdown preview"])
                            with translation_text_tab:
                                edited_translation = st.text_area("Translation", run["translation"], height=220, key=f"out_{run['id']}")
                                if edited_translation != run["translation"]:
                                    run["translation"] = edited_translation
                                    run["translated_text"] = edited_translation
                                    run["edited_at"] = datetime.now(timezone.utc).isoformat()
                                    save_experiments(st.session_state.runs)
                            with translation_preview_tab:
                                st.markdown(edited_translation)
                            with st.expander("Exact prompt used"):
                                st.code(run.get("rendered_prompt", run.get("prompt_template", "Not recorded for this older run.")), language="markdown")
                            rating = st.select_slider("Your rating", options=["Unrated", 1, 2, 3, 4, 5], value=run.get("rating", "Unrated"), key=f"rating_{run['id']}")
                            run["rating"] = rating
                            run["notes"] = st.text_input("Notes", value=run.get("notes", ""), key=f"notes_{run['id']}")
                            if st.button("Archive this run", key=f"archive_{run['id']}"):
                                run["archived"] = True
                                save_experiments(st.session_state.runs)
                                st.rerun()

    if selected_indices:
        manual_experiment_id = f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        manual_experiment_created_at = datetime.now(timezone.utc).isoformat()
        progress = st.progress(0, text="Starting translations…")
        for position, record_index in enumerate(selected_indices, start=1):
            try:
                with st.spinner(f"Translating record {record_index + 1} with {model}…"):
                    source = conversation_to_text(st.session_state.records[record_index])
                    result = translate(source, settings, prompt_template)
                created_at = datetime.now(timezone.utc).isoformat()
                result.update({"id": f"{datetime.now(timezone.utc).timestamp()}-{record_index}", "experiment_id": manual_experiment_id, "experiment_label": "Manual translation", "experiment_type": "single-record" if len(selected_indices) == 1 else "manual-all-samples", "experiment_created_at": manual_experiment_created_at, "record_index": record_index, "dataset": "GAIR/lima", "sample_seed": st.session_state.sample_seed, "source": source, "original_text": source, "translated_text": result["translation"], "prompt_name": prompt_name.strip() or selected_prompt_name, "prompt_template": prompt_template, "created_at": created_at, "rating": "Unrated", "notes": "", "archived": False})
                st.session_state.runs.append(result)
                save_experiments(st.session_state.runs)
            except Exception as exc:
                st.error(f"Record {record_index + 1} failed: {exc}")
            progress.progress(position / len(selected_indices), text=f"Completed {position} of {len(selected_indices)}")
        st.rerun()

    if run_batch:
        tasks = []
        for configuration in active_configurations:
            try:
                configuration_settings = GenerationSettings(
                    str(configuration["provider"]), str(configuration["model"]), target_language,
                    float(configuration["temperature"]), float(configuration["top_p"]), int(configuration["top_k"]), int(configuration["max_output_tokens"]),
                )
                for record_index in batch_record_indices:
                    tasks.append((record_index, conversation_to_text(st.session_state.records[record_index]), configuration, configuration_settings))
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Invalid parameter set '{configuration.get('name', 'unnamed')}': {exc}")

        batch_id = f"batch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        batch_created_at = datetime.now(timezone.utc).isoformat()
        progress = st.progress(0, text="Starting batch…")
        completed = 0
        with ThreadPoolExecutor(max_workers=batch_concurrency) as executor:
            futures = {executor.submit(translate, source, configuration_settings, prompt_template): (record_index, source, configuration) for record_index, source, configuration, configuration_settings in tasks}
            for future in as_completed(futures):
                record_index, source, configuration = futures[future]
                completed += 1
                try:
                    result = future.result()
                    created_at = datetime.now(timezone.utc).isoformat()
                    result.update({"id": f"{datetime.now(timezone.utc).timestamp()}-{record_index}", "experiment_id": batch_id, "experiment_label": batch_experiment_label.strip() or "LIMA parameter comparison", "experiment_type": "batch", "experiment_created_at": batch_created_at, "record_index": record_index, "dataset": "GAIR/lima", "sample_seed": st.session_state.sample_seed, "source": source, "original_text": source, "translated_text": result["translation"], "prompt_name": prompt_name.strip() or selected_prompt_name, "prompt_template": prompt_template, "experiment_name": str(configuration["name"]), "created_at": created_at, "rating": "Unrated", "notes": "", "archived": False})
                    st.session_state.runs.append(result)
                    save_experiments(st.session_state.runs)
                except Exception as exc:
                    st.error(f"{configuration.get('name', 'Unnamed')} · record {record_index + 1} failed: {exc}")
                progress.progress(completed / len(tasks), text=f"Completed {completed} of {len(tasks)} translations")
        st.rerun()

    active_runs = [run for run in st.session_state.runs if not run.get("archived", False)]
    if active_runs:
        st.divider()
        st.subheader("Experiment summary")
        table = pd.DataFrame([{"Batch": r.get("experiment_label", "Legacy"), "Parameter set": r.get("experiment_name", "Manual"), "Record": r["record_index"] + 1, "Seed": r.get("sample_seed"), "Prompt": r.get("prompt_name", "Unknown"), "Model": r["settings"]["model"], "Temperature": r["settings"]["temperature"], "Top-p": r["settings"]["top_p"], "Top-k": r["settings"].get("top_k"), "Max tokens": r["settings"].get("max_output_tokens"), "Seconds": r["elapsed_seconds"], "Tokens": r["usage"].get("total_tokens"), "Rating": r.get("rating", "Unrated"), "Notes": r.get("notes", "")} for r in active_runs])
        st.dataframe(table, use_container_width=True, hide_index=True)
        experiment_choices = {str(run.get("experiment_id", "legacy-import")): str(run.get("experiment_label", run.get("experiment_name", "Manual"))) for run in active_runs}
        experiments_to_archive = st.multiselect("Archive entire experiment(s)", list(experiment_choices), format_func=lambda experiment_id: experiment_choices[experiment_id], help="Archived results remain saved and can be restored from history.")
        if st.button("Archive selected experiments", disabled=not experiments_to_archive):
            for run in st.session_state.runs:
                if str(run.get("experiment_id", "legacy-import")) in experiments_to_archive:
                    run["archived"] = True
            save_experiments(st.session_state.runs)
            st.rerun()
        save_experiments(st.session_state.runs)
        st.download_button("Download all experiment JSON", results_json(st.session_state.runs), "lima_translation_runs.json", "application/json")

    archived_runs = [run for run in st.session_state.runs if run.get("archived", False)]
    if show_archived and archived_runs:
        with st.expander(f"Archived runs ({len(archived_runs)})"):
            for run in archived_runs:
                st.write(f"{run.get('experiment_name', 'Manual')} · record {run['record_index'] + 1} · {run['settings']['model']} · rating {run.get('rating', 'Unrated')}")
                if st.button("Restore", key=f"restore_{run['id']}"):
                    run["archived"] = False
                    save_experiments(st.session_state.runs)
                    st.rerun()
