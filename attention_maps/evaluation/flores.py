"""Memory-bounded English-to-Nepali evaluation on FLORES-200."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterable, Sequence

from attention_maps.inference.comparison import ComparisonResult


FLORES_DATASET_ID = "yash9439/flores200"
FLORES_DATASET_REVISION = "3c6628a4571f383d029d6e897a89ac953ae756d3"
FLORES_CONFIG = None
FLORES_SOURCE_COLUMN = "eng_Latn"
FLORES_REFERENCE_COLUMN = "npi_Deva"
FLORES_SPLIT_SIZES = {"dev": 997, "devtest": 1_012}
FLORES_TRANSLATION_PROMPT = (
    "Translate the following English sentence into natural Nepali. "
    "Return only the Nepali translation, without commentary.\n\n{text}"
)


class FloresEvaluationError(ValueError):
    """Raised when FLORES loading or scoring cannot be completed."""


@dataclass(frozen=True)
class FloresExample:
    """One aligned FLORES English/Nepali sentence pair."""

    position: int
    example_id: int | str
    source: str
    reference: str
    domain: str = ""
    topic: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "id": self.example_id,
            "English source": self.source,
            "Nepali reference": self.reference,
            "domain": self.domain,
            "topic": self.topic,
        }


def load_flores_examples(
    *,
    split: str = "devtest",
    offset: int = 0,
    limit: int = 10,
    dataset_id: str = FLORES_DATASET_ID,
    token: str | None = None,
) -> list[FloresExample]:
    """Stream only the requested contiguous FLORES slice from Hugging Face."""

    if split not in FLORES_SPLIT_SIZES:
        raise FloresEvaluationError("FLORES split must be dev or devtest")
    if offset < 0:
        raise FloresEvaluationError("FLORES offset must be non-negative")
    if limit <= 0:
        raise FloresEvaluationError("FLORES example count must be positive")
    if offset >= FLORES_SPLIT_SIZES[split]:
        raise FloresEvaluationError(
            f"FLORES offset must be below {FLORES_SPLIT_SIZES[split]:,} for {split}"
        )

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise FloresEvaluationError(
            "FLORES loading requires the `datasets` package"
        ) from error

    try:
        stream: Iterable[dict[str, Any]] = load_dataset(
            dataset_id,
            FLORES_CONFIG,
            split=split,
            streaming=True,
            token=token or None,
            revision=(
                FLORES_DATASET_REVISION
                if dataset_id == FLORES_DATASET_ID
                else None
            ),
        )
        rows = islice(stream, offset, min(offset + limit, FLORES_SPLIT_SIZES[split]))
        examples = []
        for position, row in enumerate(rows, start=offset):
            source = str(row.get(FLORES_SOURCE_COLUMN) or "").strip()
            reference = str(row.get(FLORES_REFERENCE_COLUMN) or "").strip()
            if not source or not reference:
                raise FloresEvaluationError(
                    f"FLORES row {position} is missing its aligned sentence pair"
                )
            examples.append(
                FloresExample(
                    position=position,
                    example_id=row.get("id", position),
                    source=source,
                    reference=reference,
                    domain=str(row.get("domain") or ""),
                    topic=str(row.get("topic") or ""),
                )
            )
    except FloresEvaluationError:
        raise
    except Exception as error:
        if "gated dataset" in str(error).lower():
            raise FloresEvaluationError(
                "FLORES-200 is gated on Hugging Face. Accept its dataset terms "
                "and set HF_TOKEN (or HF_token) in the Streamlit environment."
            ) from error
        raise FloresEvaluationError(f"Could not stream FLORES-200: {error}") from error

    if not examples:
        raise FloresEvaluationError("The requested FLORES slice returned no examples")
    return examples


def _sacrebleu() -> Any:
    try:
        import sacrebleu
    except ImportError as error:
        raise FloresEvaluationError(
            "FLORES scoring requires `sacrebleu`; reinstall requirements.txt"
        ) from error
    return sacrebleu


def score_flores_results(
    results: Sequence[ComparisonResult],
    examples: Sequence[FloresExample],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return per-example rows and corpus chrF++ summaries by model/config."""

    if not examples:
        raise FloresEvaluationError("Cannot score FLORES without examples")
    sacrebleu = _sacrebleu()
    details: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[tuple[str, str, float]]] = {}

    for result in results:
        if not 0 <= result.sample_index < len(examples):
            raise FloresEvaluationError(
                f"Result sample index {result.sample_index} has no FLORES reference"
            )
        example = examples[result.sample_index]
        score = None
        if not result.error and result.output.strip():
            score = round(
                float(
                    sacrebleu.sentence_chrf(
                        result.output, [example.reference], word_order=2
                    ).score
                ),
                3,
            )
            grouped.setdefault((result.model, result.decoding), []).append(
                (result.output, example.reference, result.latency_seconds)
            )
        details.append(
            {
                "model": result.model,
                "decoding": result.decoding,
                "position": example.position,
                "source": example.source,
                "reference": example.reference,
                "prediction": result.output,
                "chrF++": score,
                "latency_seconds": result.latency_seconds,
                "error": result.error,
            }
        )

    summaries = []
    all_groups = dict.fromkeys((result.model, result.decoding) for result in results)
    for model, decoding in all_groups:
        successful = grouped.get((model, decoding), [])
        predictions = [item[0] for item in successful]
        references = [item[1] for item in successful]
        latencies = [item[2] for item in successful]
        score = (
            round(
                float(
                    sacrebleu.corpus_chrf(
                        predictions, [references], word_order=2
                    ).score
                ),
                3,
            )
            if predictions
            else None
        )
        attempted = sum(
            result.model == model and result.decoding == decoding for result in results
        )
        summaries.append(
            {
                "model": model,
                "decoding": decoding,
                "chrF++": score,
                "successful": len(successful),
                "attempted": attempted,
                "coverage": round(len(successful) / attempted, 3) if attempted else 0.0,
                "mean_latency_seconds": (
                    round(sum(latencies) / len(latencies), 3) if latencies else None
                ),
            }
        )
    return details, summaries
