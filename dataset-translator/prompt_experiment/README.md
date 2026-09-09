# LIMA Translation Workbench

A Streamlit UI for testing translation prompts and sampling settings on five randomly selected records from [`GAIR/lima`](https://huggingface.co/datasets/GAIR/lima).

## Setup

1. Create a virtual environment and install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Put these values in `.env` (see `.env.example`):

   ```env
   HF_token=...
   Gemini_api=...
   # Optional, for the OpenAI selection in the UI:
   OPENAI_API_KEY=...
   ```

3. Start the workbench:

   ```powershell
   streamlit run app.py
   ```

Choose a seed, load five records, select `gemini-3.5-flash-lite` or `gemma-4-31b-it`, set the target language and sampling controls, then translate one record or all five. Re-run after changing controls to compare outputs side-by-side, rate them, and export the results as JSON.

## Prompt experiments

Use the **Prompt experiment** section in the sidebar to edit the translation template and save it under a name. Templates support `{source}` (required) and `{target_language}`. Each generation permanently records the selected prompt name, its full template, the exact rendered prompt, source text, translation, model parameters, timing, token usage, rating, and notes.

Saved templates are stored in `data/prompts.json`; the complete flat history is stored in `data/experiments.json`. In addition, every batch or manual launch creates its own self-contained file in `data/experiments/<experiment-id>.json`. Each individual experiment file contains its metadata and every run's original text, translated text, prompt name/template/rendered prompt, provider/model/hyperparameters, timing, usage, rating, notes, and archive state. They are retained across Streamlit restarts and are also available in the on-screen experiment summary and JSON export.

## Batch experiments and archive

In **Batch parameter sets**, create named parameter-set cards. Each card has its own enable switch, provider/model, temperature, top-p, top-k, and maximum-token controls, plus duplicate and remove actions. Changes save automatically. Choose **All 5 samples** or **One selected record**, then click **Run enabled parameter sets**. Each result also has a **Parameters used** panel showing the immutable settings that produced it. The per-record **Run enabled parameter sets for this record** button provides the same single-record workflow directly beside the source.

Results save automatically as each request finishes. Ratings and notes save automatically on every app interaction. **Archive this run** removes a result from the main comparison and summary while keeping it in `data/experiments.json`; enable **Show archived runs in history** to review or restore it later. Parameter-set definitions are persisted in `data/experiment_configurations.json`.

`Gemini_api` and `HF_token` are read only from the environment and never shown in the UI or exports. LIMA is gated: accept its access terms on Hugging Face, then use an account token with access. The app streams the repository's canonical `train.jsonl` directly, avoiding its deprecated `lima.py` dataset script.
