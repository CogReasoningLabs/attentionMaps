"""Bounded, reproducible examples from the exact records already classified."""

import hashlib

EXAMPLE_LIMIT = 10
TEXT_LIMIT = 8000


class InspectionExamples:
    """Merge the lowest random priorities without consuming the sampling RNG."""

    def __init__(self, seed):
        self.seed = str(seed)
        self.groups = {"language": {}, "script": {}}

    def observe(self, evidence, text, position, *, hf_selection=(), prediction=None):
        language = next(iter(evidence.language_category_records),
                        "Other" if evidence.language_outside_categories_records else "Unknown")
        script = (next(iter(evidence.script_category_records),
                       "Other" if evidence.script_outside_categories_records else "Unknown")
                  if evidence.script_eligible_records else "Not applicable")
        priority = hashlib.sha256(f"inspection-examples:{self.seed}:{position}".encode()).hexdigest()
        targets = []
        for kind, category in (("language", language), ("script", script)):
            group = self.groups[kind].setdefault(category, [])
            if len(group) < EXAMPLE_LIMIT or (priority, position) < (group[-1]["priority"], group[-1]["population_row"]):
                targets.append(group)
        if not targets:
            return
        languages = sorted(evidence.observed or evidence.predicted or
                           {code for item in hf_selection for code in item["languages"]})
        origin = ("Dataset column" if evidence.observed else "fastText" if evidence.detection_reasons
                  else "Hugging Face partition" if hf_selection else "No language evidence")
        scripts = {name: evidence.scripts.get(name, 0) for name in ("Devanagari", "Latin", "Other")}
        total = sum(scripts.values())
        supported = scripts["Devanagari"] + scripts["Latin"]
        example = {
            "priority": priority, "population_row": position,
            "text": text[:TEXT_LIMIT], "text_characters": len(text), "text_truncated": len(text) > TEXT_LIMIT,
            "language_category": language, "script_category": script,
            "languages": languages, "language_origin": origin,
            "language_reason": next(iter(evidence.detection_reasons),
                                    "metadata" if languages else "no_metadata_or_detector"),
            "prediction": dict(prediction or {}),
            "script_counts": scripts,
            "script_percentages": {name: round(count * 100 / total, 2) if total else 0.0
                                   for name, count in scripts.items()},
            "devanagari_share_of_supported": scripts["Devanagari"] / supported if supported else None,
            "latin_share_of_supported": scripts["Latin"] / supported if supported else None,
        }
        for group in targets:
            group.append(example)
            group.sort(key=lambda item: (item["priority"], item["population_row"]))
            del group[EXAMPLE_LIMIT:]

    def merge(self, other):
        for kind, categories in other.groups.items():
            for category, examples in categories.items():
                group = self.groups[kind].setdefault(category, [])
                group.extend(examples)
                group.sort(key=lambda item: (item["priority"], item["population_row"]))
                del group[EXAMPLE_LIMIT:]

    def result(self):
        return {"version": 1, "seed": self.seed, "limit_per_category": EXAMPLE_LIMIT,
                "text_character_limit": TEXT_LIMIT,
                "row_reference": "1-based position within the selected sampling population, after any row/instance selection",
                "groups": {kind: dict(sorted(categories.items())) for kind, categories in self.groups.items()}}


def validate_inspection_examples(status):
    saved = status.get("record_examples")
    if saved is None:
        return
    if (not isinstance(saved, dict) or saved.get("version") != 1
            or saved.get("limit_per_category") != EXAMPLE_LIMIT
            or saved.get("text_character_limit") != TEXT_LIMIT
            or not isinstance(saved.get("groups"), dict) or set(saved["groups"]) != {"language", "script"}):
        raise ValueError("Inspection report has invalid record examples")
    population = status.get("sampling", {}).get("population_rows", status["sampled_records"])
    for kind, groups in saved["groups"].items():
        if not isinstance(groups, dict):
            raise ValueError("Inspection report has invalid example categories")
        expected = {name for name, count in status[f"{kind}_category_counts"].items() if count}
        if kind == "script" and status["sampled_records"] > status["script_eligible_records"]:
            expected.add("Not applicable")
        if set(groups) != expected:
            raise ValueError("Inspection report has incomplete example categories")
        for category, examples in groups.items():
            count = (status["sampled_records"] - status["script_eligible_records"]
                     if kind == "script" and category == "Not applicable"
                     else status[f"{kind}_category_counts"].get(category, 0))
            if not isinstance(examples, list) or len(examples) != min(EXAMPLE_LIMIT, count):
                raise ValueError("Inspection report has inconsistent example counts")
            positions = set()
            for item in examples:
                if (not isinstance(item, dict) or item.get(f"{kind}_category") != category
                        or type(item.get("population_row")) is not int
                        or not 1 <= item["population_row"] <= population
                        or item["population_row"] in positions
                        or not isinstance(item.get("text"), str) or len(item["text"]) > TEXT_LIMIT
                        or type(item.get("text_characters")) is not int or item["text_characters"] < len(item["text"])
                        or item.get("text_truncated") != (item["text_characters"] > len(item["text"]))
                        or not isinstance(item.get("script_counts"), dict)
                        or not isinstance(item.get("script_percentages"), dict)
                        or not isinstance(item.get("prediction"), dict)):
                    raise ValueError("Inspection report has invalid example evidence")
                positions.add(item["population_row"])
