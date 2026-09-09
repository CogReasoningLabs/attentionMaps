import unittest
from unittest.mock import patch

from attention_maps.evaluation.nlue import (
    DECODER_EVALUATION_TASKS,
    DECODER_GENERATION_TASKS,
    NLUE_PUBLISHED_BEST,
    NLUE_TASKS,
    build_nlue_prompt,
    load_nlue_examples,
    nlue_task,
    parse_nlue_prediction,
    published_baseline_rows,
    score_nlue_results,
)
from attention_maps.inference.comparison import ComparisonResult


class NLUETaskTests(unittest.TestCase):
    def test_registers_every_collection_dataset_with_pinned_revision(self):
        self.assertEqual(len(NLUE_TASKS), 13)
        self.assertTrue(all(len(task.revision) == 40 for task in NLUE_TASKS))
        self.assertEqual(nlue_task("gmet").split, "train")
        self.assertEqual(nlue_task("xnli").dataset_id, "IRIIS-RESEARCH/XNLI-Nepali")

    def test_registers_decoder_generation_benchmarks(self):
        self.assertEqual(len(DECODER_GENERATION_TASKS), 3)
        self.assertEqual(len(DECODER_EVALUATION_TASKS), 16)
        self.assertEqual(nlue_task("belebele").config_name, "npi_Deva")
        self.assertEqual(nlue_task("global_mmlu").config_name, "ne")
        self.assertEqual(nlue_task("xlsum").kind, "generation")
        self.assertTrue(all(len(task.revision) == 40 for task in DECODER_GENERATION_TASKS))

    def test_renders_constrained_task_prompts(self):
        prompt = build_nlue_prompt(
            nlue_task("mnli"),
            {"premise": "आधार", "hypothesis": "निष्कर्ष", "label": 0},
        )
        self.assertIn("0 = entailment", prompt)
        self.assertIn("Premise: आधार", prompt)
        self.assertIn("केवल अंक", prompt)

    def test_parses_classification_regression_and_manual_outputs(self):
        self.assertEqual(parse_nlue_prediction(nlue_task("mnli"), "0"), 0)
        self.assertEqual(
            parse_nlue_prediction(nlue_task("mnli"), "उत्तर: contradiction"),
            2,
        )
        self.assertEqual(
            parse_nlue_prediction(nlue_task("qnli"), "not entailment"),
            1,
        )
        self.assertEqual(parse_nlue_prediction(nlue_task("winogrande"), "2"), 1)
        self.assertEqual(parse_nlue_prediction(nlue_task("stsb"), "4.25"), 4.25)
        self.assertIsNone(parse_nlue_prediction(nlue_task("stsb"), "7"))
        self.assertEqual(parse_nlue_prediction(nlue_task("gmet"), "सम्मान"), "सम्मान")
        self.assertEqual(parse_nlue_prediction(nlue_task("belebele"), "उत्तर: 4"), 3)
        self.assertEqual(parse_nlue_prediction(nlue_task("global_mmlu"), "उत्तर: B"), 1)
        self.assertEqual(
            parse_nlue_prediction(
                nlue_task("xlsum"), "यो छोटो सारांश हो।\nदोस्रो वाक्य।"
            ),
            "यो छोटो सारांश हो।\nदोस्रो वाक्य।",
        )

    def test_renders_decoder_prompts(self):
        belebele = build_nlue_prompt(
            nlue_task("belebele"),
            {
                "flores_passage": "एउटा अनुच्छेद",
                "question": "कुन सही हो?",
                "mc_answer1": "एक",
                "mc_answer2": "दुई",
                "mc_answer3": "तीन",
                "mc_answer4": "चार",
            },
        )
        self.assertIn("अनुच्छेद", belebele)
        self.assertIn("4: चार", belebele)
        mmlu = build_nlue_prompt(
            nlue_task("global_mmlu"),
            {
                "question": "प्रश्न",
                "option_a": "एक",
                "option_b": "दुई",
                "option_c": "तीन",
                "option_d": "चार",
            },
        )
        self.assertIn("केवल A, B, C वा D", mmlu)
        summary = build_nlue_prompt(nlue_task("xlsum"), {"text": "समाचार"})
        self.assertIn("सारांश", summary)

    def test_streams_pinned_bounded_slice_and_normalizes_winogrande_target(self):
        rows = (
            {
                "sentence": f"वाक्य {index} _",
                "option1": "एक",
                "option2": "दुई",
                "answer": 2.0,
            }
            for index in range(10)
        )
        task = nlue_task("winogrande")
        with patch("datasets.load_dataset", return_value=rows) as load:
            examples = load_nlue_examples("winogrande", offset=3, limit=2)

        self.assertEqual([example.position for example in examples], [3, 4])
        self.assertEqual(examples[0].target, 1)
        load.assert_called_once_with(
            task.dataset_id,
            split="test",
            streaming=True,
            token=None,
            revision=task.revision,
        )

    def test_streams_configured_decoder_dataset_and_normalizes_answer(self):
        task = nlue_task("global_mmlu")
        row = {
            "question": "प्रश्न",
            "option_a": "एक",
            "option_b": "दुई",
            "option_c": "तीन",
            "option_d": "चार",
            "answer": "C",
        }
        with patch("datasets.load_dataset", return_value=iter([row])) as load:
            example = load_nlue_examples("global_mmlu", limit=1)[0]
        self.assertEqual(example.target, 2)
        load.assert_called_once_with(
            task.dataset_id,
            "ne",
            split="test",
            streaming=True,
            token=None,
            revision=task.revision,
        )

    def test_published_rows_use_paper_best_scores(self):
        rows = published_baseline_rows("mnli")
        self.assertEqual(NLUE_PUBLISHED_BEST["mnli"]["macro_f1"][0], 78.84)
        self.assertIn("m-DeBERTa-v3", {row["reference_model"] for row in rows})
        self.assertEqual(published_baseline_rows("xnli"), [])


class NLUEMetricTests(unittest.TestCase):
    @staticmethod
    def _result(index, output):
        return ComparisonResult(
            model="trained-model",
            decoding="greedy",
            sample_index=index,
            prompt="prompt",
            output=output,
            latency_seconds=0.25,
        )

    def test_scores_accuracy_macro_f1_and_paper_delta(self):
        rows = [
            {"sentences": "नराम्रो", "sentiment": 0},
            {"sentences": "राम्रो", "sentiment": 1},
        ]
        with patch("datasets.load_dataset", return_value=iter(rows)):
            examples = load_nlue_examples("sentiment", limit=2)
        details, summaries = score_nlue_results(
            "sentiment",
            [self._result(0, "0"), self._result(1, "1")],
            examples,
        )
        self.assertTrue(all(row["correct"] for row in details))
        self.assertEqual(summaries[0]["accuracy"], 100.0)
        self.assertEqual(summaries[0]["macro_f1"], 100.0)
        self.assertEqual(summaries[0]["delta_vs_published"], 11.07)

    def test_scores_stsb_correlations_and_r2(self):
        rows = [
            {"sentence1": "a", "sentence2": "b", "label": 0.0},
            {"sentence1": "c", "sentence2": "d", "label": 2.5},
            {"sentence1": "e", "sentence2": "f", "label": 5.0},
        ]
        with patch("datasets.load_dataset", return_value=iter(rows)):
            examples = load_nlue_examples("stsb", limit=3)
        _, summaries = score_nlue_results(
            "stsb",
            [self._result(0, "0"), self._result(1, "2.5"), self._result(2, "5")],
            examples,
        )
        self.assertEqual(summaries[0]["spearman"], 100.0)
        self.assertEqual(summaries[0]["pearson"], 100.0)
        self.assertEqual(summaries[0]["r2"], 100.0)

    def test_gmet_reports_output_coverage_without_fake_gold_score(self):
        with patch(
            "datasets.load_dataset",
            return_value=iter([{"sentences": "उहाँलाई [MASK] गर्नुहोस्।"}]),
        ):
            examples = load_nlue_examples("gmet", limit=1)
        details, summaries = score_nlue_results(
            "gmet", [self._result(0, "सम्मान")], examples
        )
        self.assertEqual(details[0]["parsed_prediction"], "सम्मान")
        self.assertEqual(summaries[0]["coverage"], 1.0)
        self.assertNotIn("accuracy", summaries[0])

    def test_scores_xlsum_free_form_generation_with_rouge(self):
        row = {"text": "लामो समाचार", "target": "नेपाल राम्रो देश हो"}
        with patch("datasets.load_dataset", return_value=iter([row])):
            examples = load_nlue_examples("xlsum", limit=1)
        details, summaries = score_nlue_results(
            "xlsum", [self._result(0, "नेपाल राम्रो देश हो")], examples
        )
        self.assertEqual(details[0]["rouge_l"], 100.0)
        self.assertEqual(summaries[0]["rouge_1"], 100.0)
        self.assertEqual(summaries[0]["rouge_2"], 100.0)
        self.assertEqual(summaries[0]["rouge_l"], 100.0)


if __name__ == "__main__":
    unittest.main()
