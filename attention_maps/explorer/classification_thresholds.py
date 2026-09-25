"""One set of classification/voting defaults for YAML, CLI, reports and tests."""

import math

DEFAULT_THRESHOLDS = {
    "language_min_ratio": 0.05,
    "language_dominance_ratio": 0.95,
    "language_min_labelled_ratio": 0.80,
    "script_dominance_ratio": 0.95,
    "script_max_other_ratio": 0.05,
    "vote_min_agreement": 0.60,
}


def resolve_thresholds(settings=None):
    settings = settings or {}
    values = {key: settings.get(key, default) for key, default in DEFAULT_THRESHOLDS.items()}
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number between 0 and 1")
        low = 0.5 if key in {"language_dominance_ratio", "script_dominance_ratio", "vote_min_agreement"} else 0
        inclusive = key in {"language_min_labelled_ratio", "script_max_other_ratio", "vote_min_agreement"}
        if not (low <= value <= 1 if inclusive else low < value <= 1):
            comparison = "at least" if inclusive else "greater than"
            raise ValueError(f"{key} must be {comparison} {low} and at most 1")
    return values
