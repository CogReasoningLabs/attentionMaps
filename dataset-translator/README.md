# LIMA Dataset Translation

A reproducible Gemini translation pipeline for the [GAIR/LIMA](https://huggingface.co/datasets/GAIR/lima) instruction-following dataset. It downloads the dataset with Hugging Face authentication, translates each two-turn conversation, rotates across Gemini API keys, observes per-key request limits, and saves resumable JSON results.

The default target language is Nepali, but it can be changed through an environment variable.

## What is included

- `lima_gemini_translation.ipynb` — the end-to-end, resumable batch translation workflow.
- `translator.py` — reusable helpers for loading LIMA samples, rendering conversations and prompts, and translating through Gemini or OpenAI.
- `translation_prompt.md` — the faithful-translation prompt and generation settings.
- `requirements.txt` — Python dependencies for the reusable helpers and Streamlit-based workflows.

Generated data, results, quota state, virtual environments, and credentials are intentionally excluded from version control.

## Prerequisites

- Python 3.10 or newer
- A Hugging Face account with access to the GAIR/LIMA dataset and an access token
- One or more Gemini API keys

Install the notebook dependencies:

```bash
pip install google-genai huggingface_hub python-dotenv
```

Or install the project requirements:

```bash
pip install -r requirements.txt
```

## Configure credentials

Create a local `.env` file. It is ignored by Git.

```env
HF_token=your_hugging_face_token
Gemini_api=your_first_gemini_key
Gemini_api_RL=your_second_gemini_key
Gemini_api_CH=your_third_gemini_key
Gemini_api_YN=your_fourth_gemini_key

# Optional
TARGET_LANGUAGE=Nepali
GEMINI_MODEL=gemini-3.5-flash-lite
```

The notebook expects every key listed in `KEY_NAMES` to be set. To use fewer keys, edit that tuple in its configuration cell.

For `translator.py`, use `HF_token` and `Gemini_api`. Its OpenAI helper uses `OPENAI_API_KEY`.

## Run the batch pipeline

1. Open `lima_gemini_translation.ipynb` in Jupyter or VS Code.
2. Review the first configuration cell. Set `MAX_RECORDS` to a small integer for a trial run, or leave it as `None` to process the dataset.
3. Run all cells.

On the first run, the notebook downloads the source data into `lima_records.json`. It writes translated records incrementally to `lima_translations.json` and stores daily/request timing state in `gemini_quota_state.json`, so rerunning it continues from saved progress.

## Translation behavior

The prompt translates both `HUMAN` and `ASSISTANT` turns while preserving labels, Markdown, line breaks, code structure, URLs, and numeric values. It requests a natural translation in the selected target language and strips accidental Markdown code fences from model output.

## Notes

- Respect the LIMA dataset’s access terms and license when downloading or redistributing data.
- Keep `.env` private; never commit access tokens or API keys.
- Tune `REQUESTS_PER_MINUTE` and `REQUESTS_PER_DAY` in the notebook to match the limits of the Gemini plan and model you use.
