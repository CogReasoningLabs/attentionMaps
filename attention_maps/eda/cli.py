"""Command-line entry point for repeatable multi-dataset EDA studies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from attention_maps.eda.contracts import SurveyPlan
from attention_maps.eda.pipeline import run_survey, select_datasets
from attention_maps.eda.reporting import write_survey_report
from attention_maps.eda.text import load_stopwords_file


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and profile multiple text datasets for a corpus survey."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eda/nepali_corpus_survey.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eda/nepali_corpus_survey"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset keys; defaults to every dataset in the study config",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args(arguments)


def load_plan(path: Path) -> SurveyPlan:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read EDA config {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"EDA config is not valid JSON: {error}") from error
    analysis = value.setdefault("analysis", {})
    stopwords_file = analysis.pop("cooccurrence_stopwords_file", None)
    if stopwords_file:
        stopwords_path = (path.resolve().parent / str(stopwords_file)).resolve()
        configured = analysis.get("cooccurrence_stopwords", ())
        analysis["cooccurrence_stopwords"] = list(
            dict.fromkeys((*load_stopwords_file(stopwords_path), *configured))
        )
    return SurveyPlan.from_dict(value)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        plan = load_plan(args.config)
        if args.datasets:
            plan = select_datasets(plan, set(args.datasets))
    except (KeyError, TypeError, ValueError) as error:
        print(f"Configuration error: {error}")
        return 2

    def progress(dataset: str, stage: str, value: int) -> None:
        if stage == "analyzing":
            print(f"[{dataset}] analyzed {value:,} records", flush=True)
        else:
            print(f"[{dataset}] {stage}", flush=True)

    run = run_survey(
        plan,
        token=os.getenv(args.token_env) or None,
        progress=progress,
    )
    written = write_survey_report(run, args.output_dir, plots=not args.no_plots)
    print(
        f"Completed {len(run.profiles)}/{len(plan.datasets)} dataset(s); "
        f"wrote {len(written)} artifact(s) to {args.output_dir.resolve()}"
    )
    for dataset, error in run.failures.items():
        print(f"[{dataset}] {error}")
    return 1 if run.failures else 0
