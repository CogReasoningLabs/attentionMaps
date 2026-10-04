"""All-subset execution retains the existing per-subset research calculations."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from attention_maps.explorer.inspection_history import load_history_report, read_history
from scripts.inspect_dataset import _arguments, main


class InspectionBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sheet = self.root / "history.csv"
        self.settings = self.root / "dataset.yaml"
        self.settings.write_text("""provider: huggingface
dataset: owner/conversations
config: "*"
split: npi_Deva
training_schema: instruction_finetuning
field_mapping: {messages: messages}
text_columns: null
row_filters: {}
sample_fraction: 0.2
sampling_runs: 5
seed: 42
output: history.csv
""")
        self.catalog = {"dataset_id": "owner/conversations", "revision": "fixed-revision",
                        "configs": ["beta", "no_nepali", "alpha"]}
        self.records = {
            name: [{"id": index, "messages": [
                {"role": "system", "content": "Answer the question in Nepali."},
                {"role": "user", "content": "नेपालको राजधानी के हो?"},
                {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"},
            ]} for index in range(count)] for name, count in (("alpha", 10), ("beta", 20))
        }
        self.configurations = {}
        for name in self.catalog["configs"]:
            split = "eng_Latn" if name == "no_nepali" else "npi_Deva"
            self.configurations[name] = {
                "dataset_id": self.catalog["dataset_id"], "revision": self.catalog["revision"],
                "config": name, "languages": ["ne", "en"],
                "schema": [{"column": "id", "type": "int64"}, {"column": "messages", "type": "list"}],
                "splits": {split: {"rows": len(self.records.get(name, [])), "memory_bytes": 100,
                                    "shards": [{"name": f"{name}.jsonl", "path": f"hf://{name}.jsonl", "bytes": 100}]}},
            }

    def run_cli(self, *flags):
        with contextlib.ExitStack() as stack:
            discovery = stack.enter_context(patch("scripts.inspect_dataset.discover_huggingface_dataset",
                                                 return_value=self.catalog))
            configuration = lambda catalog, name, **kwargs: copy.deepcopy(self.configurations[name])
            stack.enter_context(patch("scripts.inspect_dataset.inspect_huggingface_configuration", side_effect=configuration))
            batches = stack.enter_context(patch("attention_maps.explorer.inspection_batch.inspect_huggingface_configuration",
                                                side_effect=configuration))
            loader = stack.enter_context(patch("datasets.load_dataset",
                                                side_effect=lambda dataset, name, **kwargs: iter(self.records[name])))
            stdout = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = main(["--settings", str(self.settings), *flags])
        return code, json.loads(stdout.getvalue()) if stdout.getvalue() else None, stderr.getvalue(), loader, discovery, batches

    def test_all_available_subsets_keep_single_subset_evidence_and_old_csv_bytes(self):
        code, earlier, _, _, _, _ = self.run_cli("--config", "alpha")
        self.assertEqual(code, 0)
        before = self.sheet.read_bytes()
        code, result, errors, loader, discovery, batches = self.run_cli()
        self.assertEqual(code, 0, errors)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["selected_configurations"], ["alpha", "beta"])
        self.assertEqual([item["configuration"] for item in result["skipped_configurations"]], ["no_nepali"])
        self.assertEqual(result["failed_configurations"], [])
        self.assertEqual(result["reports"][0]["language_status"], earlier["language_status"])
        self.assertEqual([item["language_status"]["sampling"]["rows_per_run"] for item in result["reports"]], [2, 4])
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            name = call.args[1]
            self.assertEqual(call.kwargs["revision"], "fixed-revision")
            self.assertEqual(call.kwargs["split"], "npi_Deva")
            self.assertEqual(call.kwargs["data_files"], {"npi_Deva": [f"hf://{name}.jsonl"]})
        discovery.assert_called_once()
        self.assertEqual(batches.call_count, 3)
        self.assertTrue(self.sheet.read_bytes().startswith(before))
        rows = read_history(self.sheet)
        self.assertEqual([row["Configuration"] for row in rows], ["alpha", "alpha", "beta"])
        self.assertEqual(load_history_report(self.sheet, rows[0]), earlier)
        for row in rows[1:]:
            report = load_history_report(self.sheet, row)
            self.assertEqual(report["configuration_batch"]["batch_id"], result["batch_id"])
            self.assertEqual(report["settings"]["config"], row["Configuration"])
            self.assertEqual(report["settings"]["revision"], "fixed-revision")

    def test_listing_resolves_all_subsets_without_reading_records_or_saving(self):
        code, result, errors, loader, _, _ = self.run_cli("--list")
        self.assertEqual(code, 0, errors)
        self.assertEqual(result["selected_configurations"], ["alpha", "beta"])
        self.assertEqual(result["revision"], "fixed-revision")
        loader.assert_not_called()
        self.assertFalse(self.sheet.exists())

    def test_no_matching_split_fails_without_publishing(self):
        self.catalog["configs"] = ["no_nepali"]
        code, _, errors, loader, _, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("No configurations provide split", errors)
        loader.assert_not_called()
        self.assertFalse(self.sheet.exists())

    def test_bad_subset_is_reported_and_remaining_subsets_are_still_inspected(self):
        for record in self.records["alpha"]:
            record["messages"][0]["content"] = ""
        code, result, errors, _, _, _ = self.run_cli()
        self.assertEqual(code, 1)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["failed_configurations"][0]["configuration"], "alpha")
        self.assertIn("empty content", result["failed_configurations"][0]["error"])
        footer = errors.split("Subset inspections:", 1)[1]
        self.assertIn("1 completed, 1 without requested split, 1 failed", footer)
        self.assertIn("Failed subset details:", footer)
        self.assertIn(f"alpha: {result['failed_configurations'][0]['error']}", footer)
        self.assertEqual([row["Configuration"] for row in read_history(self.sheet)], ["beta"])

    def test_no_history_prints_complete_reports_without_writing_files(self):
        self.settings.write_text(self.settings.read_text().replace("output: history.csv", "output: null"))
        code, result, errors, _, _, _ = self.run_cli("--no-history")
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(result["reports"]), 2)
        self.assertFalse(self.sheet.exists())
        self.assertNotIn("history", result["reports"][0])

    def test_wildcard_rejects_ambiguous_splits_and_shared_export_paths(self):
        for extra in (["--split", "*"], ["--shard", "one.parquet"],
                      ["--min-devanagari-ratio", ".8", "--filtered-output", str(self.root / "kept.jsonl")]):
            with self.subTest(flags=extra), self.assertRaisesRegex(ValueError, "config: '\\*'"):
                _arguments(["--settings", str(self.settings), *extra])
        with self.assertRaisesRegex(ValueError, "requires one explicit split"):
            _arguments(["--dataset", "owner/data", "--config", "*"])


if __name__ == "__main__":
    unittest.main()
