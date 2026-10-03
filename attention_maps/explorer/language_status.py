"""Evidence-labelled language coverage and Unicode script status."""

from __future__ import annotations

import unicodedata
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Sequence

from attention_maps.eda.text import extract_text
from .classification_thresholds import DEFAULT_THRESHOLDS, resolve_thresholds


LANGUAGE_COVERAGE_OPTIONS = (
    "Nepali-only", "Bilingual (Nepali-English)", "Multilingual", "English-only",
)
SCRIPT_OPTIONS = (
    "Devanagari", "Mixed (Devanagari + romanized)", "Romanized",
    "Mixed (Nepali + English)",
)
LANGUAGE_FIELDS = ("language", "language_code", "lang", "lang_code", "language_pair",
                   "source_language", "target_language", "source_lang", "target_lang")
_ALIASES = {"ne": "ne", "nep": "ne", "npi": "ne", "nepali": "ne",
            "en": "en", "eng": "en", "english": "en"}


def normalize_language(value: Any) -> str:
    return _ALIASES.get(str(value).strip().lower().replace("-", "_").split("_")[0],
                        str(value).strip().lower())


def language_codes(value: Any) -> list[str]:
    """Normalize language pairs and script-qualified codes without losing either side."""
    raw = str(value).strip()
    parts = re.split(r"[-_/+]", raw)
    codes = []
    for part in parts:
        lower = part.lower()
        if codes and lower in {"dev", "deva", "devanagari", "latn", "latin", "cyrl", "arab", "hans", "hant"}:
            continue
        if codes and len(part) == 2 and part.isupper() and lower not in _ALIASES:
            continue  # Region suffix, e.g. en-US.
        if lower in _ALIASES:
            codes.append(_ALIASES[lower])
        elif re.fullmatch(r"[a-z]{2,3}", lower):
            codes.append(lower)
        else:
            return [raw.lower()]
    return list(dict.fromkeys(codes)) or [raw.lower()]


def coverage_label(languages: Sequence[str]) -> str:
    codes = {code for value in languages if str(value).strip() for code in language_codes(value)}
    if not codes:
        return "Unknown"
    if codes == {"ne"}:
        return "Nepali-only"
    if codes == {"en"}:
        return "English-only"
    if codes == {"ne", "en"}:
        return "Bilingual (Nepali-English)"
    return "Multilingual" if len(codes) > 1 else "Other"


def saved_language_options(status: Mapping[str, Any]) -> tuple[str, ...]:
    """Percentage remainder buckets never become supported summary/vote labels."""
    counts = status.get("language_category_counts")
    return tuple(name for name in LANGUAGE_COVERAGE_OPTIONS
                 if not isinstance(counts, dict) or name in counts)


@dataclass
class LanguageEvidence:
    """Mergeable counts; sampled records need not be retained in memory."""

    observed: Counter = field(default_factory=Counter)
    predicted: Counter = field(default_factory=Counter)
    detection_reasons: Counter = field(default_factory=Counter)
    detection_truncated: int = 0
    language_records: Counter = field(default_factory=Counter)
    language_category_records: Counter = field(default_factory=Counter)
    language_no_evidence_records: int = 0
    language_outside_categories_records: int = 0
    script_category_records: Counter = field(default_factory=Counter)
    script_eligible_records: int = 0
    script_no_evidence_records: int = 0
    script_outside_categories_records: int = 0
    labelled_records: int = 0
    column_labels: dict[str, Counter] = field(default_factory=dict)
    scripts: Counter = field(default_factory=Counter)
    nepali_scripts: Counter = field(default_factory=Counter)
    english_only_scripts: Counter = field(default_factory=Counter)
    unlabelled_scripts: Counter = field(default_factory=Counter)
    records: int = 0
    text_records: int = 0
    characters: int = 0

    def add(self, other: "LanguageEvidence") -> None:
        self.observed.update(other.observed)
        self.predicted.update(other.predicted)
        self.detection_reasons.update(other.detection_reasons)
        self.detection_truncated += other.detection_truncated
        self.language_records.update(other.language_records)
        self.language_category_records.update(other.language_category_records)
        self.language_no_evidence_records += other.language_no_evidence_records
        self.language_outside_categories_records += other.language_outside_categories_records
        self.script_category_records.update(other.script_category_records)
        self.script_eligible_records += other.script_eligible_records
        self.script_no_evidence_records += other.script_no_evidence_records
        self.script_outside_categories_records += other.script_outside_categories_records
        self.labelled_records += other.labelled_records
        for column, counts in other.column_labels.items():
            self.column_labels.setdefault(column, Counter()).update(counts)
        self.scripts.update(other.scripts)
        self.nepali_scripts.update(other.nepali_scripts)
        self.english_only_scripts.update(other.english_only_scripts)
        self.unlabelled_scripts.update(other.unlabelled_scripts)
        self.records += other.records
        self.text_records += other.text_records
        self.characters += other.characters


@lru_cache(maxsize=4096)
def _character_script(character: str) -> str | None:
    category = unicodedata.category(character)
    if not category.startswith(("L", "M")):
        return None
    name = unicodedata.name(character, "")
    if "DEVANAGARI" in name:
        return "Devanagari"
    if "LATIN" in name:
        return "Latin"
    return "Other" if category.startswith("L") else None


def _record_script_category(scripts: Counter, languages: set[str], thresholds: Mapping[str, float]) -> str | None:
    """Classify one Nepali-eligible record; None means no supported category."""
    total = sum(scripts.values())
    if not total or scripts.get("Other", 0) / total > thresholds["script_max_other_ratio"]:
        return None
    supported = scripts.get("Devanagari", 0) + scripts.get("Latin", 0)
    if not supported:
        return None
    dominant = thresholds["script_dominance_ratio"]
    if scripts.get("Devanagari", 0) / supported >= dominant:
        return "Devanagari"
    if "en" in languages:
        return "Mixed (Nepali + English)"
    if scripts.get("Latin", 0) / supported >= dominant:
        return "Romanized"
    return "Mixed (Devanagari + romanized)"


def record_language_evidence(
    record: Mapping[str, Any], text_columns: Sequence[str] = (),
    *, max_characters: int | None = None, detector=None, hf_selection: Sequence[dict] = (),
    thresholds: Mapping[str, float] | None = None, extracted_text: str | None = None,
    precomputed_prediction: Mapping[str, Any] | None = None,
    prediction_details: dict | None = None,
) -> LanguageEvidence:
    evidence = LanguageEvidence(records=1)
    for name in LANGUAGE_FIELDS:
        value = record.get(name)
        values = value if isinstance(value, (list, tuple)) else [value]
        for language in values:
            if language is not None and str(language).strip():
                evidence.observed.update(language_codes(language))
                evidence.column_labels.setdefault(name, Counter())[str(language)] += 1
    text = extracted_text if extracted_text is not None else extract_text(record, text_columns)
    if not evidence.observed and not hf_selection and detector is not None:
        prediction = precomputed_prediction if precomputed_prediction is not None else detector.detect(text)
        if prediction_details is not None:
            prediction_details.update(prediction)
        evidence.detection_reasons[prediction['reason']] += 1
        evidence.detection_truncated = int(prediction['truncated'])
        if prediction['language']:
            evidence.predicted.update(language_codes(prediction['language']))
    record_codes = set(evidence.observed) | set(evidence.predicted)
    evidence.language_records.update(record_codes)
    evidence.labelled_records = int(bool(record_codes))
    if max_characters is not None:
        text = text[:max(0, max_characters)]
    evidence.characters = len(text)
    evidence.text_records = int(bool(text.strip()))
    for character, count in Counter(text).items():
        script = _character_script(character)
        if script:
            evidence.scripts[script] += count
    if "ne" in record_codes:
        evidence.nepali_scripts.update(evidence.scripts)
    elif "en" in record_codes:
        evidence.english_only_scripts.update(evidence.scripts)
    elif not record_codes:
        evidence.unlabelled_scripts.update(evidence.scripts)
    # Count every sampled record once. A selected HF language partition supplies
    # categorical fallback only when the record has no column/model label.
    category_codes = record_codes or {code for item in hf_selection for code in item["languages"]}
    if not category_codes:
        evidence.language_no_evidence_records = 1
    else:
        category = coverage_label(category_codes)
        if category in LANGUAGE_COVERAGE_OPTIONS:
            evidence.language_category_records[category] += 1
        else:
            evidence.language_outside_categories_records = 1
    if "ne" in category_codes:
        evidence.script_eligible_records = 1
        category = _record_script_category(evidence.scripts, category_codes, thresholds or DEFAULT_THRESHOLDS)
        if category is not None:
            evidence.script_category_records[category] += 1
        elif evidence.scripts:
            evidence.script_outside_categories_records = 1
        else:
            evidence.script_no_evidence_records = 1
    return evidence


def status_from_evidence(
    evidence: LanguageEvidence, *, declared_languages: Sequence[str] = (),
    language_hint: str | None = None, declaration_basis: str = "Repository/catalog declaration",
    character_limit: int | None = None, hf_selection: Sequence[dict] = (),
    thresholds: Mapping[str, float] | None = None, detector=None,
) -> dict[str, Any]:
    # Legacy declarations/hints remain accepted for API compatibility, but are
    # never evidence. Model predictions are recorded separately from metadata.
    sources = [{"origin": "dataset_column", "column": column, "raw_label_counts": dict(counts)}
               for column, counts in sorted(evidence.column_labels.items())]
    sources.extend({"origin": "huggingface_selection", **item} for item in hf_selection)
    selected_languages = sorted({code for item in hf_selection for code in item["languages"]})
    detection = None
    if detector is not None:
        attempted = sum(evidence.detection_reasons.values())
        accepted = evidence.detection_reasons.get('accepted', 0)
        detection = {**detector.metadata, 'attempted_records': attempted,
                     'accepted_records': accepted, 'rejected_records': attempted - accepted,
                     'reason_counts': dict(evidence.detection_reasons),
                     'truncated_records': evidence.detection_truncated,
                     'predicted_language_counts': dict(evidence.predicted)}
        if attempted:
            sources.append({'origin': 'text_language_detector', **detection})
    thresholds = resolve_thresholds(thresholds)
    labelled_ratio = evidence.labelled_records / evidence.records if evidence.records else 0
    language_ratios = {code: count / evidence.labelled_records
                       for code, count in sorted(evidence.language_records.items())} if evidence.labelled_records else {}
    if evidence.observed or evidence.detection_reasons:
        basis = ("Dataset column labels + text language detector" if evidence.observed and evidence.detection_reasons
                 else "Dataset column labels" if evidence.observed else "Text language detector (fastText)")
        languages = sorted(code for code, ratio in language_ratios.items()
                           if ratio >= thresholds["language_min_ratio"])
        if labelled_ratio < thresholds["language_min_labelled_ratio"]:
            languages = []
            language_reason = (f"Only {labelled_ratio:.2%} of sampled records have language labels or accepted predictions; "
                               f"need at least {thresholds['language_min_labelled_ratio']:.2%}.")
        elif len(languages) == 1 and language_ratios[languages[0]] < thresholds["language_dominance_ratio"]:
            language_reason = (f"Needs review: the sole significant language is present in only {language_ratios[languages[0]]:.2%} "
                               f"of labelled records; single-language coverage requires {thresholds['language_dominance_ratio']:.2%}.")
            languages = []
        else:
            language_reason = (f"Include languages present in at least {thresholds['language_min_ratio']:.2%} "
                               "of labelled sample records; each language counts at most once per record.")
    elif selected_languages:
        languages, basis = selected_languages, "Hugging Face configuration/split"
        language_reason = "No column language labels; use the explicit selected HF partition label. No record ratios are inferred from it."
    else:
        languages, basis = [], "No dataset-column or Hugging Face selection evidence"
        language_reason = "No dataset-column or HF partition evidence; text language detection is disabled or no records were sampled."
    conflict = bool(evidence.observed and selected_languages
                    and set(evidence.observed) != set(selected_languages))
    coverage = coverage_label(languages)
    nepali_covered = ("ne" in languages) if languages else None
    nepali_scripts = evidence.nepali_scripts.copy()
    if "ne" in selected_languages:
        # Unlabelled rows may use HF evidence; explicitly non-Nepali rows may not.
        nepali_scripts.update(evidence.unlabelled_scripts)
    analysis_scripts = nepali_scripts.copy()
    if nepali_covered and "en" in languages:
        analysis_scripts.update(evidence.english_only_scripts)
    analysis_total = sum(analysis_scripts.values())
    other_ratio = analysis_scripts.get("Other", 0) / analysis_total if analysis_total else 0
    supported_total = analysis_scripts.get("Devanagari", 0) + analysis_scripts.get("Latin", 0)
    deva_ratio = analysis_scripts.get("Devanagari", 0) / supported_total if supported_total else 0
    latin_ratio = analysis_scripts.get("Latin", 0) / supported_total if supported_total else 0
    dominant = thresholds["script_dominance_ratio"]
    max_other = thresholds["script_max_other_ratio"]
    if nepali_covered is None:
        script, reason = "Unknown", "Nepali coverage is unknown; script classification requires Nepali language evidence."
    elif not nepali_covered:
        script, reason = "Not applicable", "The thresholded language coverage does not include Nepali."
    elif not nepali_scripts:
        script, reason = "Unknown", "Nepali is covered, but no letters/marks were found in its eligible sampled text."
    elif other_ratio > max_other or not supported_total:
        script = "Unknown" if coverage == "Nepali-only" else "Other"
        reason = (f"Needs review: other writing systems are {other_ratio:.2%} of eligible letters/marks "
                  f"(maximum {max_other:.2%}); no supported script category." if other_ratio > max_other else
                  "Needs review: no Devanagari/Latin letters or marks in eligible text.")
    elif deva_ratio >= dominant:
        script, reason = "Devanagari", f"Devanagari is {deva_ratio:.2%} of Devanagari/Latin letters/marks (threshold {dominant:.2%})."
    elif "en" in languages:
        script, reason = "Mixed (Nepali + English)", "Nepali and English meet language thresholds; Latin text exceeds the Devanagari dominance allowance."
    elif latin_ratio >= dominant:
        script, reason = "Romanized", f"Latin is {latin_ratio:.2%} of Devanagari/Latin letters/marks (threshold {dominant:.2%})."
    else:
        script, reason = "Mixed (Devanagari + romanized)", f"Neither Devanagari nor Latin reaches the {dominant:.2%} dominance threshold."
    if not nepali_covered:
        analysis_scripts.clear()
    analysis_total = sum(analysis_scripts.values())
    total = sum(evidence.scripts.values())
    language_category_counts = {name: evidence.language_category_records[name]
                                for name in LANGUAGE_COVERAGE_OPTIONS}
    script_category_counts = {name: evidence.script_category_records[name] for name in SCRIPT_OPTIONS}
    # Reporting only: expose the existing remainder counters without changing
    # evidence, denominators, thresholds, classifications, or votes.
    language_category_counts.update(Other=evidence.language_outside_categories_records,
                                    Unknown=evidence.language_no_evidence_records)
    script_category_counts.update(Other=evidence.script_outside_categories_records,
                                  Unknown=evidence.script_no_evidence_records)
    return {
        "language_coverage": coverage, "language_basis": basis,
        "languages": languages, "observed_language_counts": dict(evidence.observed),
        "language_evidence_policy": "metadata_then_text_detection_v3" if detector is not None else "dataset_columns_or_huggingface_selection_v2",
        **({"language_detection": detection} if detection is not None else {}),
        "classification_thresholds": thresholds, "language_reason": language_reason,
        "labelled_records": evidence.labelled_records, "labelled_record_ratio": labelled_ratio,
        "language_category_counts": language_category_counts,
        "language_category_percentages": {name: round(count * 100 / evidence.records, 2) if evidence.records else 0.0
                                           for name, count in language_category_counts.items()},
        "language_no_evidence_records": evidence.language_no_evidence_records,
        "language_outside_categories_records": evidence.language_outside_categories_records,
        "script_category_counts": script_category_counts,
        "script_category_percentages": {name: round(count * 100 / evidence.script_eligible_records, 2)
                                         if evidence.script_eligible_records else 0.0
                                         for name, count in script_category_counts.items()},
        "script_eligible_records": evidence.script_eligible_records,
        "script_no_evidence_records": evidence.script_no_evidence_records,
        "script_outside_categories_records": evidence.script_outside_categories_records,
        "language_record_counts": dict(evidence.language_records), "language_record_ratios": language_ratios,
        "language_evidence_sources": sources, "language_evidence_conflict": conflict,
        "language_evidence_rule": "Dataset columns take priority; the selected HF language partition supplies categorical fallback. When both are absent for a record, an enabled text detector can supply an accepted prediction. Apply the same record-ratio thresholds to column labels and accepted predictions; rejected predictions remain unlabelled. Column/HF conflicts are flagged.",
        "script": script, "script_basis": "Nepali language evidence first, then Unicode letters/marks in eligible sampled records",
        "script_policy": "nepali_required_v3", "nepali_covered": nepali_covered,
        "script_reason": reason,
        "script_decision_ratios": {"other_share_of_all": other_ratio,
                                  "devanagari_share_of_supported": deva_ratio,
                                  "latin_share_of_supported": latin_ratio},
        "script_scope": "Nepali-labelled or confidently predicted records (or unlabelled records in a Nepali HF partition), plus English-labelled or confidently predicted records when both languages are covered. Whole selected text fields are analyzed; mixed-language records are not separated into language spans.",
        "nepali_record_script_counts": dict(nepali_scripts) if nepali_covered else {},
        "script_analysis_counts": dict(analysis_scripts),
        "script_analysis_percentages": {name: round(count * 100 / analysis_total, 2) for name, count in analysis_scripts.items()},
        "script_counts": dict(evidence.scripts),
        "script_percentages": {name: round(count * 100 / total, 2) for name, count in evidence.scripts.items()},
        "sampled_records": evidence.records, "text_records": evidence.text_records,
        "sampled_characters": evidence.characters, "character_limit": character_limit,
        "note": "Language evidence comes from dataset columns, selected HF language partitions, or an explicitly enabled text detector when metadata is absent. Script is classified only when Nepali is covered. Romanized and mixed labels are evidence-based heuristics, not language identification of individual spans. Sample results do not guarantee the entire dataset uses one script.",
    }


def analyze_language_status(
    records: Sequence[Mapping[str, Any]], *, text_columns: Sequence[str] = (),
    declared_languages: Sequence[str] = (), language_hint: str | None = None,
    max_characters: int = 100_000, declaration_basis: str = "Repository/catalog declaration",
    hf_selection: Sequence[dict] = (), thresholds: Mapping[str, float] | None = None, detector=None,
) -> dict[str, Any]:
    """Compatibility path for the original small, character-limited sample."""
    if max_characters <= 0:
        raise ValueError("max_characters must be positive")
    evidence = LanguageEvidence()
    effective_thresholds = resolve_thresholds(thresholds)
    for record in records:
        evidence.add(record_language_evidence(
            record, text_columns, max_characters=max_characters - evidence.characters,
            detector=detector, hf_selection=hf_selection, thresholds=effective_thresholds,
        ))
    return status_from_evidence(
        evidence, declared_languages=declared_languages, language_hint=language_hint,
        declaration_basis=declaration_basis, character_limit=max_characters, hf_selection=hf_selection, thresholds=thresholds, detector=detector,
    )


def _selection_languages(value: str, *, split: bool = False) -> list[str]:
    """Recognize explicit ne/en labels and script tags, never incidental substrings."""
    label = str(value or "").strip().lower()
    if split:
        # Support ne_train/train_ne and en-ne_test, not plain train/test/default.
        label = re.sub(r"^(train|test|validation|valid|dev)[_/-]", "", label)
        label = re.sub(r"[_/-](train|test|validation|valid|dev)$", "", label)
    if not label:
        return []
    codes = []
    parts = re.split(r"[-_/+]", label)
    for index, part in enumerate(parts):
        if part in _ALIASES:
            codes.append(_ALIASES[part])
        elif index and part in {"dev", "deva", "devanagari", "latn", "latin"}:
            continue
        else:
            return []
    return sorted(set(codes))


def language_context(inventory: dict, declared_languages: Sequence[str] = ()) -> dict:
    selection = []
    if inventory.get("format") == "huggingface" or inventory.get("provider") == "huggingface":
        for key, name in (("dataset_config", "configuration"), ("dataset_split", "split")):
            value = inventory.get(key)
            codes = _selection_languages(value, split=name == "split")
            if codes:
                selection.append({"field": name, "value": value, "languages": codes})
    return {"hf_selection": selection}


def inventory_language_status(
    inventory: dict[str, Any], records: Sequence[Mapping[str, Any]], *,
    text_columns: Sequence[str] = (), declared_languages: Sequence[str] = (),
    thresholds: Mapping[str, float] | None = None, detector=None,
) -> dict[str, Any]:
    return analyze_language_status(
        records, text_columns=text_columns, thresholds=thresholds, detector=detector, **language_context(inventory, declared_languages),
    )
