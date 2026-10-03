"""Reporting of already-counted records; no classification or renormalization."""


def category_breakdown(status: dict, prefix: str) -> tuple[dict, dict]:
    """Include historical gap counters under their original names, without mutation."""
    counts = dict(status[f"{prefix}_category_counts"])
    percentages = dict(status[f"{prefix}_category_percentages"])
    denominator = status["sampled_records" if prefix == "language" else "script_eligible_records"]
    for bucket, label, suffix in (
        ("Other", "Outside listed categories", "outside_categories"),
        ("Unknown", f"No {prefix} evidence", "no_evidence"),
    ):
        # Intermediate reports already contain Other; its records must not be
        # counted twice. Original reports retain their original gap labels.
        if bucket not in counts:
            count = status[f"{prefix}_{suffix}_records"]
            counts[label] = count
            percentages[label] = round(count * 100 / denominator, 2) if denominator else 0.0
    return counts, percentages
