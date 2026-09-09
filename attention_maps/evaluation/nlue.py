"""Bounded text-generation evaluation for Nepali decoder-only models."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterable, Sequence

from attention_maps.inference.comparison import ComparisonResult


NLUE_COLLECTION_URL = (
    "https://huggingface.co/collections/IRIIS-RESEARCH/"
    "nepali-lanuguage-understanding-evaluation-benchmark"
)
NLUE_PAPER_URL = "https://aclanthology.org/2025.findings-ijcnlp.119/"
BELEBELE_URL = "https://huggingface.co/datasets/facebook/belebele"
GLOBAL_MMLU_URL = "https://huggingface.co/datasets/CohereLabs/Global-MMLU"
XLSUM_URL = "https://huggingface.co/datasets/GEM/xlsum"


class NLUEEvaluationError(ValueError):
    """Raised when NLUE loading, prediction parsing, or scoring fails."""


@dataclass(frozen=True)
class NLUETaskSpec:
    key: str
    label: str
    dataset_id: str
    revision: str
    split: str
    split_size: int
    category: str
    kind: str
    target_field: str | None
    class_labels: tuple[str, ...] = ()
    primary_metric: str | None = None
    config_name: str | None = None
    source_url: str = NLUE_COLLECTION_URL
    data_url: str | None = None


@dataclass(frozen=True)
class NLUEExample:
    position: int
    fields: dict[str, Any]
    target: int | float | str | None
    prompt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            **self.fields,
            "gold": self.target,
            "prompt": self.prompt,
        }


NLUE_TASKS = (
    NLUETaskSpec(
        "sentiment", "Sentiment Analysis (SA)",
        "IRIIS-RESEARCH/Sentiment-Analysis-Nepali",
        "823ea164365faabb77c38cc20096d7ccb668d68f", "test", 16_279,
        "Single sentence", "classification", "sentiment",
        ("negative", "positive"), "macro_f1",
    ),
    NLUETaskSpec(
        "cola", "Linguistic Acceptability (CoLA)",
        "IRIIS-RESEARCH/CoLA-Nepali",
        "fd598ceecdbb4e46708aa51ec06dc10c5bfcd5b8", "test", 1_950,
        "Single sentence", "classification", "label",
        ("unacceptable", "acceptable"), "macro_f1",
    ),
    NLUETaskSpec(
        "winogrande", "WinoGrande (WG)",
        "IRIIS-RESEARCH/WinoGrande-Nepali",
        "04272334ea3e21dc07915d4f83269dcc069a7230", "test", 8_135,
        "Single sentence", "classification", "answer",
        ("option 1", "option 2"), "macro_f1",
    ),
    NLUETaskSpec(
        "qqp", "Quora Question Pairs (QQP)",
        "IRIIS-RESEARCH/QQP-Nepali",
        "cb30b9ba030aef4a1e7cde9ac612349ebd6a1466", "test", 6_500,
        "Similarity / paraphrase", "classification", "label",
        ("not paraphrase", "paraphrase"), "macro_f1",
    ),
    NLUETaskSpec(
        "mrpc", "Microsoft Paraphrase Corpus (MRPC)",
        "IRIIS-RESEARCH/MRPC-Nepali",
        "f78d9b6c0395e53c842f77160d165bebce1b3904", "test", 1_047,
        "Similarity / paraphrase", "classification", "label",
        ("not paraphrase", "paraphrase"), "macro_f1",
    ),
    NLUETaskSpec(
        "stsb", "Semantic Textual Similarity (STS-B)",
        "IRIIS-RESEARCH/STS-B-Nepali",
        "7b2cfb176e66d958beb3df38e2ff7efa50505eef", "test", 1_363,
        "Similarity / paraphrase", "regression", "label", (), "spearman",
    ),
    NLUETaskSpec(
        "qadsm", "Query–Ad Matching (QADSM)",
        "IRIIS-RESEARCH/QADSM-Nepali",
        "d73cafacc82d72e9cc3fae397ed4867f58313dcd", "test", 14_859,
        "Similarity / paraphrase", "classification", "relevance_label",
        ("irrelevant", "relevant"), "macro_f1",
    ),
    NLUETaskSpec(
        "mnli", "Multi-Genre NLI (MNLI)",
        "IRIIS-RESEARCH/MNLI-Nepali",
        "f56eea7ba10727bf4ee78e0fa89b749fd7314d15", "test", 10_200,
        "Natural-language inference", "classification", "label",
        ("entailment", "neutral", "contradiction"), "macro_f1",
    ),
    NLUETaskSpec(
        "xnli", "Cross-lingual NLI (XNLI)",
        "IRIIS-RESEARCH/XNLI-Nepali",
        "1d9e057df981facd1774ef24ccbc977ef0c72ba7", "test", 12_794,
        "Natural-language inference", "classification", "label",
        ("entailment", "neutral", "contradiction"), "macro_f1",
    ),
    NLUETaskSpec(
        "qnli", "Question-answer NLI (QNLI)",
        "IRIIS-RESEARCH/QNLI-Nepali",
        "b4b904823bfc640f5095268c110bf5f13f99f208", "test", 6_995,
        "Natural-language inference", "classification", "label",
        ("entailment", "not entailment"), "macro_f1",
    ),
    NLUETaskSpec(
        "rte", "Recognizing Textual Entailment (RTE)",
        "IRIIS-RESEARCH/RTE-Nepali",
        "02a9c94cbe068fef2b7b10bfcbdc5189d14ef072", "test", 503,
        "Natural-language inference", "classification", "label",
        ("entailment", "not entailment"), "macro_f1",
    ),
    NLUETaskSpec(
        "coreference", "Co-reference Resolution (CR)",
        "IRIIS-RESEARCH/Co-Reference-Nepali",
        "60877cc227e1267cf0680f09fde0fc850966d1da", "test", 142,
        "Natural-language inference", "classification", "label",
        ("incorrect resolution", "correct resolution"), "macro_f1",
    ),
    NLUETaskSpec(
        "gmet", "General Masked Evaluation Task (GMET)",
        "IRIIS-RESEARCH/Filling-Masks-Nepali",
        "825174baac8699eedfd96c8f9b4ca5497f11bf37", "train", 1_500,
        "Masked evaluation", "manual", None, (), None,
    ),
)

# These tasks directly exercise next-token generation. Multiple-choice answers
# are emitted as short text and XL-Sum is scored as free-form generation.
DECODER_GENERATION_TASKS = (
    NLUETaskSpec(
        "belebele", "Belebele Nepali · Reading comprehension",
        "facebook/belebele",
        "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "test", 900,
        "Decoder generation", "classification", "correct_answer_num",
        ("1", "2", "3", "4"), "accuracy", "npi_Deva", BELEBELE_URL,
    ),
    NLUETaskSpec(
        "global_mmlu", "Global-MMLU Nepali · Knowledge and reasoning",
        "CohereLabs/Global-MMLU",
        "0e619dbeb34206cd48705a1a0ea7fb21cae09993", "test", 14_042,
        "Decoder generation", "classification", "answer",
        ("A", "B", "C", "D"), "accuracy", "ne", GLOBAL_MMLU_URL,
    ),
    NLUETaskSpec(
        "xlsum", "XL-Sum Nepali · Abstractive summarization",
        "GEM/xlsum",
        "00487191b2aa0461a3bb4783f5aba00fb432dbd2", "test", 725,
        "Free-form generation", "generation", "target", (), "rouge_l",
        "nepali", XLSUM_URL,
        (
            "https://huggingface.co/datasets/GEM/xlsum/resolve/"
            "00487191b2aa0461a3bb4783f5aba00fb432dbd2/"
            "nepali/xlsum-test.parquet"
        ),
    ),
)
DECODER_EVALUATION_TASKS = NLUE_TASKS + DECODER_GENERATION_TASKS
NLUE_TASK_BY_KEY = {task.key: task for task in DECODER_EVALUATION_TASKS}


# Best full-test scores reported in Tables 2–5 of the NLUE paper. Values are
# percentages. Each metric keeps its own winning model where winners differ.
NLUE_PUBLISHED_BEST: dict[str, dict[str, tuple[float, str]]] = {
    "sentiment": {"accuracy": (88.94, "m-DeBERTa-v3"), "macro_f1": (88.93, "m-DeBERTa-v3")},
    "cola": {"accuracy": (88.31, "m-DeBERTa-v3"), "macro_f1": (85.64, "m-DeBERTa-v3")},
    "winogrande": {"accuracy": (68.07, "RoBERTa Nepali"), "macro_f1": (68.07, "RoBERTa Nepali")},
    "qqp": {"accuracy": (84.34, "m-DeBERTa-v3"), "macro_f1": (83.82, "m-DeBERTa-v3")},
    "mrpc": {"accuracy": (83.48, "m-DeBERTa-v3"), "macro_f1": (81.93, "m-DeBERTa-v3")},
    "stsb": {
        "spearman": (90.22, "m-DeBERTa-v3"),
        "pearson": (89.57, "m-DeBERTa-v3"),
        "r2": (81.33, "m-DeBERTa-v3"),
    },
    "qadsm": {"accuracy": (66.42, "m-DeBERTa-v3"), "macro_f1": (66.42, "m-DeBERTa-v3")},
    "mnli": {"accuracy": (78.76, "m-DeBERTa-v3"), "macro_f1": (78.84, "m-DeBERTa-v3")},
    "qnli": {"accuracy": (86.65, "m-DeBERTa-v3"), "macro_f1": (86.65, "m-DeBERTa-v3")},
    "rte": {"accuracy": (68.19, "Multilingual BERT"), "macro_f1": (68.00, "Multilingual BERT")},
    "coreference": {"accuracy": (59.15, "BERT Nepali"), "macro_f1": (57.33, "NepBERTa")},
    "gmet": {"accuracy": (57.27, "RoBERTa Nepali"), "combined_score": (48.76, "RoBERTa Nepali")},
}


def nlue_task(task_key: str) -> NLUETaskSpec:
    try:
        return NLUE_TASK_BY_KEY[task_key]
    except KeyError as error:
        raise NLUEEvaluationError(f"Unknown NLUE task: {task_key}") from error


def _text(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise NLUEEvaluationError(f"NLUE row is missing {field!r}")
    return value


def build_nlue_prompt(task: NLUETaskSpec, row: dict[str, Any]) -> str:
    """Render a constrained zero-shot prompt for one official task row."""

    if task.key == "belebele":
        body = (
            f"अनुच्छेद:\n{_text(row, 'flores_passage')}\n\n"
            f"प्रश्न: {_text(row, 'question')}\n"
            f"1: {_text(row, 'mc_answer1')}\n"
            f"2: {_text(row, 'mc_answer2')}\n"
            f"3: {_text(row, 'mc_answer3')}\n"
            f"4: {_text(row, 'mc_answer4')}"
        )
        instruction = "अनुच्छेदका आधारमा सही उत्तर छान्नुहोस्: 1, 2, 3 वा 4."
    elif task.key == "global_mmlu":
        body = (
            f"प्रश्न: {_text(row, 'question')}\n"
            f"A: {_text(row, 'option_a')}\n"
            f"B: {_text(row, 'option_b')}\n"
            f"C: {_text(row, 'option_c')}\n"
            f"D: {_text(row, 'option_d')}"
        )
        instruction = "सही उत्तर छान्नुहोस्: A, B, C वा D."
    elif task.key == "xlsum":
        body = f"समाचार लेख:\n{_text(row, 'text')}"
        instruction = "यस समाचारको छोटो, तथ्यपरक नेपाली सारांश लेख्नुहोस्।"
    elif task.key == "sentiment":
        body = f"पाठ: {_text(row, 'sentences')}"
        instruction = "भावना वर्गीकरण गर्नुहोस्: 0 = negative, 1 = positive."
    elif task.key == "cola":
        body = f"वाक्य: {_text(row, 'sentence')}"
        instruction = "वाक्य व्याकरणिक रूपमा स्वीकार्य छ कि छैन: 0 = unacceptable, 1 = acceptable."
    elif task.key == "winogrande":
        body = (
            f"वाक्य: {_text(row, 'sentence')}\n"
            f"1: {_text(row, 'option1')}\n2: {_text(row, 'option2')}"
        )
        instruction = "रिक्त स्थानका लागि सही विकल्प छान्नुहोस्: 1 वा 2."
    elif task.key in {"qqp", "mrpc"}:
        first = "question1" if task.key == "qqp" else "sentence1"
        second = "question2" if task.key == "qqp" else "sentence2"
        body = f"पाठ 1: {_text(row, first)}\nपाठ 2: {_text(row, second)}"
        instruction = "दुवै पाठ paraphrase हुन् कि होइनन्: 0 = होइन, 1 = हुन्."
    elif task.key == "stsb":
        body = f"वाक्य 1: {_text(row, 'sentence1')}\nवाक्य 2: {_text(row, 'sentence2')}"
        instruction = "अर्थगत समानता 0 देखि 5 सम्म अंक दिनुहोस्."
    elif task.key == "qadsm":
        body = (
            f"खोज: {_text(row, 'query')}\nविज्ञापन शीर्षक: {_text(row, 'ad_title')}\n"
            f"विज्ञापन विवरण: {_text(row, 'ad_description')}"
        )
        instruction = "विज्ञापन खोजसँग सान्दर्भिक छ कि छैन: 0 = irrelevant, 1 = relevant."
    elif task.key in {"mnli", "xnli"}:
        body = f"Premise: {_text(row, 'premise')}\nHypothesis: {_text(row, 'hypothesis')}"
        instruction = "सम्बन्ध छान्नुहोस्: 0 = entailment, 1 = neutral, 2 = contradiction."
    elif task.key in {"qnli", "rte", "coreference"}:
        if task.key == "qnli":
            body = f"Question: {_text(row, 'question')}\nSentence: {_text(row, 'sentence')}"
            instruction = "Sentence ले उत्तर दिन्छ: 0 = entailment, 1 = not entailment."
        elif task.key == "rte":
            body = f"Text: {_text(row, 'sentence1')}\nHypothesis: {_text(row, 'sentence2')}"
            instruction = "सम्बन्ध छान्नुहोस्: 0 = entailment, 1 = not entailment."
        else:
            body = f"Context: {_text(row, 'sentence1')}\nResolution: {_text(row, 'sentence2')}"
            instruction = "Resolution सही छ कि छैन: 0 = incorrect, 1 = correct."
    elif task.key == "gmet":
        body = f"वाक्य: {_text(row, 'sentences')}"
        instruction = "[MASK] मा मिल्ने एउटा नेपाली शब्द मात्र लेख्नुहोस्."
    else:  # pragma: no cover - protected by the static task registry
        raise NLUEEvaluationError(f"No prompt renderer for {task.key}")
    if task.key == "global_mmlu":
        suffix = "केवल A, B, C वा D लेख्नुहोस्, व्याख्या नगर्नुहोस्।"
    elif task.kind == "classification" or task.kind == "regression":
        suffix = "केवल अंक मात्र लेख्नुहोस्, व्याख्या नगर्नुहोस्।"
    else:
        suffix = "व्याख्या वा थप टिप्पणी नलेख्नुहोस्।"
    return f"{instruction}\n{suffix}\n\n{body}"


def load_nlue_examples(
    task_key: str,
    *,
    offset: int = 0,
    limit: int = 10,
    token: str | None = None,
) -> list[NLUEExample]:
    """Stream a deterministic contiguous slice of one pinned NLUE dataset."""

    task = nlue_task(task_key)
    if offset < 0 or offset >= task.split_size:
        raise NLUEEvaluationError(
            f"NLUE offset must be between 0 and {task.split_size - 1:,}"
        )
    if limit <= 0:
        raise NLUEEvaluationError("NLUE example count must be positive")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise NLUEEvaluationError(
            "NLUE loading requires the `datasets` package"
        ) from error
    try:
        if task.data_url:
            stream = load_dataset(
                "parquet",
                data_files={task.split: task.data_url},
                split=task.split,
                streaming=True,
                token=token or None,
            )
        else:
            dataset_arguments: list[str] = [task.dataset_id]
            if task.config_name:
                dataset_arguments.append(task.config_name)
            stream = load_dataset(
                *dataset_arguments,
                split=task.split,
                streaming=True,
                token=token or None,
                revision=task.revision,
            )
        rows = islice(stream, offset, min(offset + limit, task.split_size))
        examples = []
        for position, row in enumerate(rows, start=offset):
            target = row.get(task.target_field) if task.target_field else None
            if task.kind == "classification":
                if target is None:
                    raise NLUEEvaluationError(f"NLUE row {position} has no gold label")
                if task.key == "global_mmlu":
                    answer = str(target).strip().upper()
                    if answer not in task.class_labels:
                        raise NLUEEvaluationError(
                            f"NLUE row {position} has invalid answer {target!r}"
                        )
                    target = task.class_labels.index(answer)
                else:
                    target = int(target)
                if task.key in {"winogrande", "belebele"}:
                    target -= 1
            elif task.kind == "regression":
                if target is None:
                    raise NLUEEvaluationError(f"NLUE row {position} has no gold score")
                target = float(target)
            elif task.kind == "generation":
                target = str(target or "").strip()
                if not target:
                    raise NLUEEvaluationError(
                        f"NLUE row {position} has no reference generation"
                    )
            fields = {key: value for key, value in row.items() if key != task.target_field}
            examples.append(
                NLUEExample(position, fields, target, build_nlue_prompt(task, row))
            )
    except NLUEEvaluationError:
        raise
    except Exception as error:
        raise NLUEEvaluationError(
            f"Could not stream {task.dataset_id}@{task.revision[:8]}: {error}"
        ) from error
    if not examples:
        raise NLUEEvaluationError("The requested NLUE slice returned no examples")
    return examples


def parse_nlue_prediction(task: NLUETaskSpec, output: str) -> int | float | str | None:
    """Parse a constrained generative answer into the task's prediction type."""

    cleaned = output.strip().strip("`").strip()
    if not cleaned:
        return None
    if task.kind == "generation":
        return cleaned
    if task.kind == "manual":
        return cleaned.splitlines()[0].strip()
    if task.key == "global_mmlu":
        answer = re.search(r"(?<![A-Z])[ABCD](?![A-Z])", cleaned.upper())
        return task.class_labels.index(answer.group()) if answer else None
    number = re.search(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])", cleaned)
    if not number:
        if task.kind == "classification":
            normalized = cleaned.casefold()
            for index, label in sorted(
                enumerate(task.class_labels),
                key=lambda item: len(item[1]),
                reverse=True,
            ):
                if re.search(rf"\b{re.escape(label.casefold())}\b", normalized):
                    return index
        return None
    if task.kind == "regression":
        value = float(number.group())
        return value if 0 <= value <= 5 else None
    value = int(float(number.group()))
    if task.key in {"winogrande", "belebele"}:
        value -= 1
    return value if 0 <= value < len(task.class_labels) else None


def _generation_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[\w\u0900-\u097f]+", normalized, flags=re.UNICODE)


def _ngram_f1(reference: Sequence[str], prediction: Sequence[str], size: int) -> float:
    if len(reference) < size or len(prediction) < size:
        return 0.0
    reference_counts: dict[tuple[str, ...], int] = {}
    prediction_counts: dict[tuple[str, ...], int] = {}
    for index in range(len(reference) - size + 1):
        ngram = tuple(reference[index:index + size])
        reference_counts[ngram] = reference_counts.get(ngram, 0) + 1
    for index in range(len(prediction) - size + 1):
        ngram = tuple(prediction[index:index + size])
        prediction_counts[ngram] = prediction_counts.get(ngram, 0) + 1
    overlap = sum(
        min(count, prediction_counts.get(ngram, 0))
        for ngram, count in reference_counts.items()
    )
    if not overlap:
        return 0.0
    precision = overlap / sum(prediction_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2 * precision * recall / (precision + recall)


def _rouge_l_f1(reference: Sequence[str], prediction: Sequence[str]) -> float:
    if not reference or not prediction:
        return 0.0
    previous = [0] * (len(prediction) + 1)
    for reference_token in reference:
        current = [0]
        for index, prediction_token in enumerate(prediction, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    length = previous[-1]
    precision = length / len(prediction)
    recall = length / len(reference)
    return 2 * precision * recall / (precision + recall) if length else 0.0


def _rouge_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    reference_tokens = _generation_tokens(reference)
    prediction_tokens = _generation_tokens(prediction)
    return (
        _ngram_f1(reference_tokens, prediction_tokens, 1) * 100,
        _ngram_f1(reference_tokens, prediction_tokens, 2) * 100,
        _rouge_l_f1(reference_tokens, prediction_tokens) * 100,
    )


def _classification_metrics(
    gold: Sequence[int], predictions: Sequence[int], class_count: int
) -> tuple[float, float]:
    accuracy = sum(left == right for left, right in zip(gold, predictions)) / len(gold)
    f1_scores = []
    for label in range(class_count):
        true_positive = sum(g == label and p == label for g, p in zip(gold, predictions))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predictions))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predictions))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(2 * true_positive / denominator if denominator else 0.0)
    return accuracy * 100, sum(f1_scores) / class_count * 100


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_sum * right_sum)
    return numerator / denominator if denominator else None


def published_baseline_rows(task_key: str) -> list[dict[str, object]]:
    return [
        {"metric": metric, "published_best": score, "reference_model": model}
        for metric, (score, model) in NLUE_PUBLISHED_BEST.get(task_key, {}).items()
    ]


def score_nlue_results(
    task_key: str,
    results: Sequence[ComparisonResult],
    examples: Sequence[NLUEExample],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score parsed generations and compare their primary metric to the paper."""

    task = nlue_task(task_key)
    if not examples:
        raise NLUEEvaluationError("Cannot score NLUE without examples")
    details = []
    grouped: dict[
        tuple[str, str],
        list[tuple[int | float | str, int | float | str, float]],
    ] = {}
    attempted: dict[tuple[str, str], int] = {}
    for result in results:
        if not 0 <= result.sample_index < len(examples):
            raise NLUEEvaluationError(
                f"Result sample index {result.sample_index} has no NLUE example"
            )
        example = examples[result.sample_index]
        key = (result.model, result.decoding)
        attempted[key] = attempted.get(key, 0) + 1
        prediction = None if result.error else parse_nlue_prediction(task, result.output)
        correct = None
        generation_scores = None
        if prediction is not None:
            grouped.setdefault(key, []).append(
                (example.target, prediction, result.latency_seconds)  # type: ignore[arg-type]
            )
            correct = (
                prediction == example.target
                if task.kind == "classification"
                else None
            )
            if task.kind == "generation":
                generation_scores = _rouge_scores(
                    str(example.target), str(prediction)
                )
        detail = {
            "model": result.model,
            "decoding": result.decoding,
            "position": example.position,
            "gold": example.target,
            "parsed_prediction": prediction,
            "correct": correct,
            "raw_output": result.output,
            "latency_seconds": round(result.latency_seconds, 3),
            "error": result.error,
        }
        for field in ("subject", "subject_category"):
            if field in example.fields:
                detail[field] = example.fields[field]
        if generation_scores:
            detail.update(
                rouge_1=round(generation_scores[0], 3),
                rouge_2=round(generation_scores[1], 3),
                rouge_l=round(generation_scores[2], 3),
            )
        details.append(detail)

    summaries = []
    for key, total in attempted.items():
        model, decoding = key
        parsed = grouped.get(key, [])
        summary: dict[str, object] = {
            "model": model,
            "decoding": decoding,
            "parsed": len(parsed),
            "attempted": total,
            "coverage": round(len(parsed) / total, 3),
            "mean_latency_seconds": (
                round(sum(item[2] for item in parsed) / len(parsed), 3)
                if parsed else None
            ),
        }
        if task.kind == "classification" and parsed:
            gold = [int(item[0]) for item in parsed]
            predictions = [int(item[1]) for item in parsed]
            accuracy, macro_f1 = _classification_metrics(
                gold, predictions, len(task.class_labels)
            )
            summary.update(accuracy=round(accuracy, 3), macro_f1=round(macro_f1, 3))
        elif task.kind == "regression" and parsed:
            gold = [float(item[0]) for item in parsed]
            predictions = [float(item[1]) for item in parsed]
            pearson = _pearson(gold, predictions)
            spearman = _pearson(_ranks(gold), _ranks(predictions))
            denominator = sum((value - sum(gold) / len(gold)) ** 2 for value in gold)
            r2 = (
                1 - sum((g - p) ** 2 for g, p in zip(gold, predictions)) / denominator
                if denominator else None
            )
            summary.update(
                spearman=round(spearman * 100, 3) if spearman is not None else None,
                pearson=round(pearson * 100, 3) if pearson is not None else None,
                r2=round(r2 * 100, 3) if r2 is not None else None,
            )
        elif task.kind == "generation" and parsed:
            rouge_scores = [
                _rouge_scores(str(item[0]), str(item[1])) for item in parsed
            ]
            summary.update(
                rouge_1=round(
                    sum(score[0] for score in rouge_scores) / len(rouge_scores), 3
                ),
                rouge_2=round(
                    sum(score[1] for score in rouge_scores) / len(rouge_scores), 3
                ),
                rouge_l=round(
                    sum(score[2] for score in rouge_scores) / len(rouge_scores), 3
                ),
            )
        if task.primary_metric and task.primary_metric in summary:
            published = NLUE_PUBLISHED_BEST.get(task.key, {}).get(task.primary_metric)
            if published and summary[task.primary_metric] is not None:
                summary["published_best"] = published[0]
                summary["delta_vs_published"] = round(
                    float(summary[task.primary_metric]) - published[0], 3
                )
        summaries.append(summary)
    return details, summaries
