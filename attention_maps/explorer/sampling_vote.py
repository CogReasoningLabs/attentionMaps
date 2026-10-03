"""Streaming repeated random samples and majority voting over language/script evidence."""

from __future__ import annotations

import random
from collections import Counter
from decimal import Decimal, ROUND_CEILING
from typing import Protocol

from attention_maps.eda.text import extract_text

from .classification_thresholds import resolve_thresholds
from .language_detection import make_language_detector
from .inspection_examples import InspectionExamples, validate_inspection_examples
from .language_status import (
    LANGUAGE_COVERAGE_OPTIONS, LANGUAGE_FIELDS, SCRIPT_OPTIONS, LanguageEvidence, language_context,
    record_language_evidence, status_from_evidence,
)


class SamplingStrategy(Protocol):
    """Selectors decide membership; classification and voting are independent."""

    sample_rows: int

    def select(self, index: int) -> bool: ...


class RandomWithoutReplacement:
    """Select exactly ceil(fraction * N) uniform row positions in a single pass."""

    def __init__(self, population: int, fraction: float, seed: int | str):
        self.population = population
        self.sample_rows = int((Decimal(str(fraction)) * population).to_integral_value(rounding=ROUND_CEILING))
        self.remaining = self.sample_rows
        self.rng = random.Random(seed)

    def select(self, index: int) -> bool:
        remaining_rows = self.population - index
        if remaining_rows <= 0:
            raise ValueError("Source row count changed during sampling; report was not published")
        selected = self.remaining > 0 and (
            self.remaining == remaining_rows or self.rng.randrange(remaining_rows) < self.remaining
        )
        self.remaining -= int(selected)
        return selected


SAMPLING_STRATEGIES = {"random": RandomWithoutReplacement}

_worker_detector = None


def initialize_evidence_worker(settings: dict) -> None:
    """Create one lazy language detector per process, not per batch."""
    global _worker_detector
    _worker_detector = make_language_detector(settings)


def _inspection_text(record, fields, definition, source_row, source_identity, source_split):
    if definition is None:
        return extract_text(record, fields)
    from .semantic_instances import inspection_instance_text
    try:
        return inspection_instance_text(record, definition, source_row=source_row,
                                        source_identity=source_identity, source_split=source_split)
    except ValueError as error:
        raise ValueError(f"Invalid {definition['schema']} instance at source row {source_row}: {error}") from error


def analyze_selected_batch(batch: list[tuple[dict, tuple[int, ...]]], fields: list[str],
                           hf_selection: tuple[dict, ...], thresholds: dict, runs: int,
                           definition=None, source_identity="inspection", source_split=None, example_seed=None):
    """Return mergeable counts without sending individual evidence back to the parent."""
    unique = LanguageEvidence()
    per_run = [LanguageEvidence() for _ in range(runs)]
    examples = InspectionExamples(example_seed) if example_seed is not None else None
    # TXT/other unlabelled batches are common. Predict all their records in
    # one model call, then keep the existing one-record evidence/vote logic.
    unlabelled_batch = (
        _worker_detector is not None and not hf_selection
        and all(not any(record.get(field) not in (None, "") for field in LANGUAGE_FIELDS)
                for record, *_ in batch)
    )
    texts = [
        _inspection_text(item[0], fields, definition, item[2] if len(item) > 2 else position,
                         source_identity, source_split)
        for position, item in enumerate(batch)
    ] if unlabelled_batch or definition is not None else None
    predictions = _worker_detector.detect_many(texts) if unlabelled_batch else None
    for position, item in enumerate(batch):
        record, chosen = item[:2]
        content = texts[position] if texts is not None else extract_text(record, fields)
        prediction_details = {}
        summary = record_language_evidence(
            record, fields, detector=_worker_detector, hf_selection=hf_selection, thresholds=thresholds,
            extracted_text=content, prediction_details=prediction_details,
            precomputed_prediction=predictions[position] if predictions is not None else None,
        )
        if examples is not None:
            examples.observe(summary, content, item[2] + 1, hf_selection=hf_selection, prediction=prediction_details)
        unique.add(summary)
        for index in chosen:
            per_run[index].add(summary)
    metadata = dict(_worker_detector.metadata) if _worker_detector is not None else None
    return (unique, per_run, metadata, examples) if examples is not None else (unique, per_run, metadata)


def majority_vote(labels: list[str], min_agreement: float | None = None) -> dict:
    """Each run gets one vote; Unknown cannot provide a positive conclusion."""
    counts = Counter(labels)
    largest = max(counts.values(), default=0)
    leaders = sorted(label for label, count in counts.items() if count == largest)
    required = len(labels) // 2 + 1
    if min_agreement is not None:
        resolve_thresholds({"vote_min_agreement": min_agreement})
        required = max(required, int((Decimal(str(min_agreement)) * len(labels)).to_integral_value(rounding=ROUND_CEILING)))
    winner = leaders[0] if len(leaders) == 1 and largest >= required else "Unknown"
    decided = winner != "Unknown"
    return {
        "label": winner, "counts": dict(sorted(counts.items())), "runs": len(labels),
        "winning_votes": counts.get(winner, 0) if decided else 0,
        "agreement": counts[winner] / len(labels) if decided else 0.0,
        "decision": "majority" if decided else "inconclusive",
        "leading_labels": leaders,
        **({"min_agreement": min_agreement, "required_votes": required} if min_agreement is not None else {}),
    }


class RepeatedSampleAnalysis:
    """One source traversal, independent samples, and exact unique coverage counts."""

    def __init__(self, inventory: dict, settings: dict, fields: list[str], population: int,
                 *, population_basis: str = "source metadata", instance_definition=None):
        if type(population) is not int or population < 0:
            raise ValueError("Sampling population must be a non-negative integer")
        if type(settings.get("sampling_runs")) is not int or settings["sampling_runs"] < 1:
            raise ValueError("sampling_runs must be a positive integer")
        if type(settings.get("sample_fraction")) not in (int, float) or not 0 < settings["sample_fraction"] <= 1:
            raise ValueError("sample_fraction must be greater than 0 and at most 1")
        for key, default in (("concurrency", 1), ("batch_size", 1024)):
            value = settings.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        self.population = population
        self.population_basis = population_basis
        self.worker_settings = settings
        self.concurrency = settings.get("concurrency", 1)
        self.batch_size = settings.get("batch_size", 1024)
        self.fields = fields
        self.instance_definition = instance_definition
        self.source_identity = inventory.get("dataset_id") or inventory.get("path") or "inspection"
        self.source_split = inventory.get("dataset_split")
        self.thresholds = resolve_thresholds(settings)
        self.detector = make_language_detector(settings)
        self.context = {"detector": self.detector, **language_context(inventory, settings.get("languages") or ()), "thresholds": self.thresholds}
        self.fraction = settings["sample_fraction"]
        self.method = settings.get("sampling_method", "random")
        strategy = SAMPLING_STRATEGIES.get(self.method)
        if strategy is None:
            raise ValueError(f"Unsupported sampling method: {self.method}")
        self.seeds = [f"{settings['seed']}:{index + 1}" for index in range(settings["sampling_runs"])]
        self.selectors: list[SamplingStrategy] = [strategy(population, self.fraction, seed) for seed in self.seeds]
        self.evidence = [LanguageEvidence() for _ in self.seeds]
        self.unique = LanguageEvidence()
        self.examples = InspectionExamples(settings["seed"])
        self.scanned = 0

    def select_record(self, record: dict) -> tuple[dict, tuple[int, ...]] | None:
        chosen = [index for index, selector in enumerate(self.selectors) if selector.select(self.scanned)]
        self.scanned += 1
        if not chosen:
            return None
        return record, tuple(chosen)

    def observe(self, record: dict) -> None:
        selected = self.select_record(record)
        if selected is None:
            return
        # Every character in every selected record is analyzed, once per unique row.
        content = _inspection_text(record, self.fields, self.instance_definition, self.scanned - 1,
                                   self.source_identity, self.source_split)
        prediction_details = {}
        summary = record_language_evidence(record, self.fields, extracted_text=content, detector=self.detector,
                                           hf_selection=self.context["hf_selection"], thresholds=self.thresholds,
                                           prediction_details=prediction_details)
        self.examples.observe(summary, content, self.scanned, hf_selection=self.context["hf_selection"],
                              prediction=prediction_details)
        self.unique.add(summary)
        for index in selected[1]:
            self.evidence[index].add(summary)

    def merge_batch(self, result) -> None:
        unique, per_run, metadata = result[:3]
        if len(result) > 3:
            self.examples.merge(result[3])
        self.unique.add(unique)
        for target, batch_evidence in zip(self.evidence, per_run, strict=True):
            target.add(batch_evidence)
        if self.detector is not None and metadata is not None:
            self.detector.metadata.update(metadata)

    def finish(self) -> dict:
        if self.scanned != self.population:
            raise ValueError(f"Source row count changed: expected {self.population}, read {self.scanned}; report was not published")
        runs = []
        for index, evidence in enumerate(self.evidence):
            if evidence.records != self.selectors[index].sample_rows:
                raise ValueError("Sampling strategy did not return its declared number of records")
            runs.append({"run": index + 1, "seed": self.seeds[index],
                         "sample_fraction_actual": evidence.records / self.population if self.population else 0.0,
                         **status_from_evidence(evidence, **self.context)})
        for run in runs:
            if run["sampled_records"] == 0:
                run.update(language_coverage="Unknown", script="Unknown", language_basis="No sampled records",
                           nepali_covered=None, script_reason="No sampled records")
        votes = {key: majority_vote([run[key] for run in runs], self.thresholds["vote_min_agreement"]) for key in ("language_coverage", "script")}
        combined = status_from_evidence(self.unique, **self.context)
        combined["pooled_language_coverage"] = combined["language_coverage"]
        combined["pooled_script"] = combined["script"]
        combined["pooled_nepali_covered"] = combined["nepali_covered"]
        nepali_vote = majority_vote([
            "Present" if run["nepali_covered"] is True else "Absent" if run["nepali_covered"] is False else "Unknown"
            for run in runs
        ], self.thresholds["vote_min_agreement"])
        combined["nepali_covered"] = {"Present": True, "Absent": False}.get(nepali_vote["label"])
        combined.update({key: result["label"] for key, result in votes.items()})
        combined["script_reason"] = (
            "Nepali coverage is inconclusive across runs; no script category is assigned." if combined["nepali_covered"] is None else
            "Nepali absence meets the configured vote threshold; Script is not applicable." if not combined["nepali_covered"] else
            "Needs review: Nepali is covered, but no supported script category met the vote threshold. See the per-run reasons." if combined["script"] == "Unknown" else
            "Nepali coverage and the conditional script classification meet the configured vote threshold."
        )
        combined["script_basis"] = "Majority vote over Nepali-conditional script labels; percentages describe unique sampled records"
        combined["voting"] = {
            "rule": "strict_majority_with_minimum_agreement", **votes, "nepali_coverage": nepali_vote,
            "note": "Agreement measures run consistency, not probability of correctness. Selected Hugging Face partition labels are shared evidence, not independent language detections.",
        }
        actual_fraction = self.selectors[0].sample_rows / self.population if self.population else 0.0
        combined["sampling"] = {
            "method": self.method, "fraction_requested": self.fraction, "runs": len(runs),
            "concurrency": self.concurrency, "batch_size": self.batch_size,
            "population_rows": self.population, "population_basis": self.population_basis,
            "rows_scanned": self.scanned, "rows_per_run": self.selectors[0].sample_rows,
            "sampled_record_occurrences": sum(evidence.records for evidence in self.evidence),
            "unique_sampled_records": self.unique.records,
            "unique_coverage": self.unique.records / self.population if self.population else 0.0,
            "expected_unique_coverage": 1 - (1 - actual_fraction) ** len(runs),
            "within_run_replacement": False, "overlap_between_runs": True,
            "character_limit_per_run": None, "run_results": runs,
        }
        combined["record_examples"] = self.examples.result()
        return combined


def _validate_nepali_script(status: dict) -> None:
    if not status.get("script_policy"):
        return  # Historical reports retain their original semantics.
    covered, script = status.get("nepali_covered"), status.get("script")
    allowed = {"Devanagari", "Romanized", "Mixed (Devanagari + romanized)",
               "Mixed (Nepali + English)", "Other", "Unknown"}
    if status["script_policy"] in {"nepali_required_v2", "nepali_required_v3"} and status.get("language_coverage") == "Nepali-only":
        allowed = {"Devanagari", "Romanized", "Mixed (Devanagari + romanized)", "Unknown"}
    if (status["script_policy"] not in {"nepali_required_v1", "nepali_required_v2", "nepali_required_v3"}
            or covered is not None and type(covered) is not bool
            or covered is None and script != "Unknown"
            or covered is False and script != "Not applicable"
            or covered is True and script not in allowed):
        raise ValueError("Inspection report's Script is inconsistent with Nepali coverage")


def _validate_category_proportions(status: dict) -> None:
    fields = ("language_category_counts", "language_category_percentages",
              "language_no_evidence_records", "language_outside_categories_records",
              "script_category_counts", "script_category_percentages",
              "script_eligible_records", "script_no_evidence_records",
              "script_outside_categories_records")
    if not any(field in status for field in fields):
        return  # Earlier reports predate record-category proportions.
    if any(field not in status for field in fields):
        raise ValueError("Inspection report has incomplete category proportions")
    sampled = status["sampled_records"]
    eligible = status["script_eligible_records"]
    if type(sampled) is not int or sampled < 0 or type(eligible) is not int or not 0 <= eligible <= sampled:
        raise ValueError("Inspection report has invalid category denominators")
    for prefix, options, denominator, gaps in (
        ("language", LANGUAGE_COVERAGE_OPTIONS, sampled,
         ("language_no_evidence_records", "language_outside_categories_records")),
        ("script", SCRIPT_OPTIONS, eligible,
         ("script_no_evidence_records", "script_outside_categories_records")),
    ):
        counts = status[f"{prefix}_category_counts"]
        percentages = status[f"{prefix}_category_percentages"]
        allowed_options = [set(options), set(options) | {"Other", "Unknown"}]
        if prefix == "language":
            allowed_options.append(set(options) | {"Other"})  # Interim reports.
        valid_options = set(counts) if isinstance(counts, dict) else set()
        complete = "Unknown" in valid_options
        if (not isinstance(counts, dict) or valid_options not in allowed_options
                or not isinstance(percentages, dict) or set(percentages) != valid_options
                or any(type(count) is not int or count < 0 for count in counts.values())
                or any(type(status[gap]) is not int or status[gap] < 0 for gap in gaps)
                or sum(counts.values()) + (0 if complete else sum(status[gap] for gap in gaps)) != denominator
                or complete and (counts["Unknown"] != status[gaps[0]] or counts["Other"] != status[gaps[1]])):
            raise ValueError("Inspection report has inconsistent category counts")
        for name in counts:
            expected = round(counts[name] * 100 / denominator, 2) if denominator else 0.0
            value = percentages[name]
            if type(value) not in (int, float) or abs(value - expected) > 1e-6:
                raise ValueError("Inspection report has inconsistent category percentages")


def validate_sampling_result(status: dict) -> None:
    """Keep malformed saved voting results out of the read-only UI."""
    _validate_nepali_script(status)
    _validate_category_proportions(status)
    validate_inspection_examples(status)
    if status.get("script_policy") == "nepali_required_v3" and not isinstance(status.get("classification_thresholds"), dict):
        raise ValueError("Inspection report is missing classification thresholds")
    thresholds = resolve_thresholds(status.get("classification_thresholds")) if status.get("script_policy") == "nepali_required_v3" else None
    agreement = thresholds["vote_min_agreement"] if thresholds else None
    sampling = status.get("sampling")
    if sampling is None:
        return
    votes = status.get("voting")
    if not isinstance(sampling, dict) or not isinstance(votes, dict):
        raise ValueError("Inspection report has invalid sampling/voting results")
    for key in ("population_rows", "rows_scanned", "rows_per_run", "sampled_record_occurrences", "unique_sampled_records", "runs"):
        if type(sampling.get(key)) is not int or sampling[key] < 0:
            raise ValueError("Inspection report has invalid sampling counts")
    for key in ("concurrency", "batch_size"):
        if key in sampling and (type(sampling[key]) is not int or sampling[key] < 1):
            raise ValueError("Inspection report has invalid parallel sampling settings")
    for key in ("fraction_requested", "unique_coverage", "expected_unique_coverage"):
        if type(sampling.get(key)) not in (int, float) or not 0 <= sampling[key] <= 1:
            raise ValueError("Inspection report has invalid sampling fractions")
    runs = sampling.get("run_results")
    if (sampling["runs"] < 1 or not isinstance(runs, list) or len(runs) != sampling["runs"]
            or not isinstance(sampling.get("population_basis"), str)
            or not isinstance(votes.get("note"), str)
            or not all(isinstance(status.get(key), str) for key in ("pooled_language_coverage", "pooled_script"))):
        raise ValueError("Inspection report has incomplete sampling results")
    for run in runs:
        if (not isinstance(run, dict) or "seed" not in run or type(run.get("run")) is not int
                or type(run.get("sampled_records")) is not int or run["sampled_records"] != sampling["rows_per_run"]
                or type(run.get("sampled_characters")) is not int or run["sampled_characters"] < 0
                or not isinstance(run.get("script_percentages"), dict)
                or not all(isinstance(run.get(key), str) for key in ("language_coverage", "language_basis", "script"))):
            raise ValueError("Inspection report has invalid per-run evidence")
        _validate_nepali_script(run)
        _validate_category_proportions(run)
        if thresholds and run.get("classification_thresholds") != thresholds:
            raise ValueError("Inspection report has inconsistent per-run classification thresholds")
    if status.get("script_policy"):
        expected_presence = majority_vote([
            "Present" if run.get("nepali_covered") is True else "Absent" if run.get("nepali_covered") is False else "Unknown"
            for run in runs
        ], agreement)
        if (votes.get("nepali_coverage") != expected_presence
                or status["nepali_covered"] is not {"Present": True, "Absent": False}.get(expected_presence["label"])):
            raise ValueError("Inspection report has inconsistent Nepali coverage votes")
    if (sampling["rows_scanned"] != sampling["population_rows"]
            or not sampling["rows_per_run"] <= sampling["unique_sampled_records"] <= sampling["population_rows"]
            or sampling["sampled_record_occurrences"] != sampling["runs"] * sampling["rows_per_run"]
            or status["sampled_records"] != sampling["unique_sampled_records"]):
        raise ValueError("Inspection report has inconsistent sampling counts")
    for key in ("language_coverage", "script"):
        expected = majority_vote([run[key] for run in runs], agreement)
        if votes.get(key) != expected or status[key] != expected["label"]:
            raise ValueError("Inspection report's saved vote does not match its run results")
