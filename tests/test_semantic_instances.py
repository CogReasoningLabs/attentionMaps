import contextlib
import copy
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.explorer.semantic_instances import make_instance, resolve_instance_definition, iter_instance_records
from attention_maps.explorer.semantic_artifacts import pair_similarity, load_run, get_record
from scripts.cluster_dataset import main, parse_arguments
from tests.test_semantic_analysis import FixtureEncoder, make_run


def definition(record, schema="auto", fields=None, **options):
    inventory = {"columns": list(record), "format": "jsonl"}
    args = SimpleNamespace(training_schema=schema, **options)
    return resolve_instance_definition(inventory, fields or ["text"], args)


def adapt(record, contract):
    return make_instance(record, contract, source_row=7, source_identity={"dataset": "pinned/data"})


class SemanticInstanceTests(unittest.TestCase):
    def test_pretraining_preserves_entire_document_and_raw_metadata(self):
        text = "नेपाल\n" * 3000 + "END OF DOCUMENT"
        record = {"id": "doc-7", "text": text, "split": "train", "source": "publisher", "metadata": {"url": "example"}}
        original = copy.deepcopy(record)
        instance, embedded, generated = adapt(record, definition(record))
        self.assertEqual(instance, record)
        self.assertEqual(embedded, text)
        self.assertFalse(generated)
        self.assertEqual(record, original)
        self.assertNotIn("publisher", embedded)

    def test_missing_ids_are_stable_and_unknown_split_is_not_invented(self):
        record = {"text": "नेपाल"}
        first = adapt(record, definition(record))
        second = adapt(record, definition(record))
        self.assertEqual(first, second)
        self.assertTrue(first[2])
        self.assertIsNone(first[0]["split"])

    def test_conversation_keeps_all_turns_roles_and_final_response(self):
        record = {"id": "sft", "messages": [{"role": "system", "content": "नेपालीमा लेख्नुहोस्"},
            {"role": "user", "content": "प्रश्न एक"}, {"role": "assistant", "content": "उत्तर एक"},
            {"role": "user", "content": "प्रश्न दुई"}, {"role": "assistant", "content": "उत्तर दुई"}], "split": "train"}
        contract = definition(record)
        self.assertEqual(contract["schema"], "instruction_finetuning")
        instance, text, _ = adapt(record, contract)
        self.assertEqual(instance["messages"], record["messages"])
        self.assertTrue(text.endswith("[assistant]\nउत्तर दुई"))
        self.assertLess(text.index("प्रश्न एक"), text.index("उत्तर एक"))
        self.assertIn("[system]", text)
        record["messages"].pop()
        with self.assertRaisesRegex(ValueError, "assistant response"):
            adapt(record, contract)

    def test_instruction_input_output_becomes_one_complete_conversation(self):
        record = {"instruction": "Translate", "input": "Hello", "output": "नमस्ते", "system": "Be accurate"}
        instance, text, _ = adapt(record, definition(record))
        self.assertEqual(len(instance["messages"]), 3)
        for part in ("Translate", "Hello", "नमस्ते", "Be accurate"):
            self.assertIn(part, text)

    def test_supervised_includes_label_zero_task_and_all_input_fields(self):
        record = {"sentence1": "First sentence", "sentence2": "Second sentence", "label": 0}
        contract = definition(record, fields=["sentence1", "sentence2"], task_name="entailment")
        instance, text, _ = adapt(record, contract)
        self.assertEqual(instance["label"], 0)
        self.assertEqual(instance["text"], "First sentence\n\nSecond sentence")
        self.assertIn("[task]\nentailment", text)
        self.assertTrue(text.endswith("[label]\n0"))
        with self.assertRaisesRegex(ValueError, "label and task"):
            adapt(record, definition(record, fields=["sentence1"]))

    def test_preference_group_embeds_both_branches_and_swapping_changes_representation(self):
        record = {"prompt": "Question", "chosen": "Good answer", "rejected": "Bad answer"}
        contract = definition(record)
        instance, text, _ = adapt(record, contract)
        self.assertEqual(contract["schema"], "preference_tuning")
        for name, value in record.items():
            self.assertIn(f"[{name}]\n{value}", text)
        swapped = {**record, "chosen": record["rejected"], "rejected": record["chosen"]}
        self.assertNotEqual(text, adapt(swapped, contract)[1])
        with self.assertRaisesRegex(ValueError, "must differ"):
            adapt({**record, "rejected": record["chosen"]}, contract)

    def test_dotted_mapping_and_cli_override(self):
        record = {"payload": {"body": "नेपाल", "category": "news", "task": "classification"}}
        mapping = {"text": "payload.body", "label": "payload.category", "task": "payload.task"}
        instance, text, _ = adapt(record, definition(record, field_mapping=mapping))
        self.assertEqual(instance["text"], "नेपाल")
        self.assertEqual(instance["label"], "news")
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "embedding.yaml"
            config.write_text("local: source.jsonl\noutput_dir: result\nfield_mapping: {text: old}\n")
            args = parse_arguments(["run", "--settings", str(config), "--field-map", "text=payload.body", "--field-map", "label=payload.category"])
            self.assertEqual(args.field_mapping, {"text": "payload.body", "label": "payload.category"})

    def test_explicit_blank_line_boundaries_keep_whole_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.txt"
            source.write_text("Title one\nBody one\n\nTitle two\nBody two\n\n", encoding="utf-8")
            inventory = {"path": str(source), "format": "text", "columns": ["text"]}
            contract = resolve_instance_definition(inventory, ["text"], SimpleNamespace(training_schema="pretraining", text_record_unit="blank_line"))
            rows = list(iter_instance_records(inventory, contract))
            self.assertEqual(rows, [{"text": "Title one\nBody one"}, {"text": "Title two\nBody two"}])
            contract["text_record_unit"] = "line"
            self.assertEqual(len(list(iter_instance_records(inventory, contract))), 4)

    def test_blank_line_counts_are_instances_not_physical_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.txt"
            source.write_text("A title\nBody one\n\nB title\nBody two\n\n", encoding="utf-8")
            output = Path(directory) / "output"
            with patch("scripts.cluster_dataset.DocumentEncoder", return_value=FixtureEncoder()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--local", str(source), "--text-record-unit", "blank_line", "--clusters", "2", "--output-dir", str(output)]), 0)
            _, report = load_run(output)
            self.assertEqual(report["embedded_records"], 2)
            self.assertEqual(report["inventory"]["rows"], 4)
            self.assertEqual(pair_similarity(output, 0, 1)["first"]["instance"]["text"], "A title\nBody one")

    def test_legacy_records_remain_readable_without_relabelling_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root, report = make_run(Path(directory))
            original_score = pair_similarity(root, 0, 1)["cosine_similarity"]
            with sqlite3.connect(root / "records.sqlite") as db:
                db.execute("ALTER TABLE records RENAME TO modern_records")
                db.execute("CREATE TABLE records AS SELECT embedding_id, source_row, text, record_json, chunks FROM modern_records")
                db.execute("DROP TABLE modern_records")
            report.pop("instance_definition")
            (root / "report.json").write_text(json.dumps(report))
            result = pair_similarity(root, 0, 1)
            self.assertIsNone(result["instance_definition"])
            self.assertIsNone(result["first"]["instance"])
            self.assertEqual(result["first"]["record"]["extra"], 0)
            self.assertEqual(result["cosine_similarity"], original_score)

    def test_duplicate_canonical_ids_fail_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.jsonl"
            source.write_text(json.dumps({"id": "same", "text": "A"}) + "\n" + json.dumps({"id": "same", "text": "B"}) + "\n")
            with patch("scripts.cluster_dataset.DocumentEncoder", return_value=FixtureEncoder()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
                code = main(["run", "--local", str(source), "--clusters", "1", "--output-dir", str(root / "out")])
            self.assertEqual(code, 1)
            self.assertIn("Duplicate instance id", errors.getvalue())
            self.assertFalse((root / "out").exists())

    def test_preference_sampling_preserves_pair_and_ignores_added_english_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"prompt": "प्रश्न", "chosen": "उत्तर", "rejected": "गलत", "language": "ne"}
            source = root / "data.jsonl"
            source.write_text(json.dumps(record) + "\n")
            with patch("scripts.cluster_dataset.DocumentEncoder", return_value=FixtureEncoder()), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--local", str(source), "--clusters", "1", "--output-dir", str(root / "run")]), 0)
                self.assertEqual(main(["sample", "--run", str(root / "run"), "--method", "random", "--sample-fraction", "1", "--sampling-runs", "1", "--output-dir", str(root / "sample")]), 0)
            report = json.loads((root / "sample/report.json").read_text())
            self.assertEqual(report["language_status"]["script"], "Devanagari")
            self.assertEqual(json.loads((root / "sample/sample-1.jsonl").read_text()), record)

    def test_full_conversation_is_sent_to_encoder_and_shown_in_streamlit(self):
        from streamlit.testing.v1 import AppTest

        class CaptureEncoder(FixtureEncoder):
            def encode(self, texts):
                self.seen.extend(texts)
                return super().encode(texts)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [{"id": f"chat-{i}", "messages": [{"role": "user", "content": f"Question {i}"},
                       {"role": "assistant", "content": "COMPLETE FINAL ANSWER " + ("उत्तर " * 1000)}],
                       "split": "train", "metadata": {"source": "retained-original-metadata"}} for i in range(2)]
            source = root / "data.jsonl"
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            encoder = CaptureEncoder()
            encoder.seen = []
            output = root / "run"
            with patch("scripts.cluster_dataset.DocumentEncoder", return_value=encoder), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--local", str(source), "--training-schema", "instruction_finetuning", "--clusters", "1", "--output-dir", str(output)]), 0)
            self.assertTrue(all("COMPLETE FINAL ANSWER" in text and "[assistant]" in text for text in encoder.seen))
            pair = pair_similarity(output, 0, 1)
            self.assertEqual(pair["first"]["instance"]["messages"], records[0]["messages"])
            self.assertEqual(pair["first"]["record"], records[0])
            with patch.dict(os.environ, {"DATASET_EMBEDDING_RUNS": str(output)}), patch("scripts.cluster_dataset.DocumentEncoder", side_effect=AssertionError("No UI inference")):
                app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"), default_timeout=25).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(any(item.value.strip() == records[0]["messages"][-1]["content"].strip() for item in app.text))
                self.assertTrue(any(json.loads(item.value) == records[0] for item in app.json))
                self.assertEqual(sum(item.label == "Download complete instance" for item in app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
