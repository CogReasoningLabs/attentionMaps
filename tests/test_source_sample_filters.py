"""Source sampling combines row predicates and column projection without downloads."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_maps.explorer import VIEWER_PREFIX, file_signatures, inspect_dataset
from attention_maps.explorer.sample_filters import sample_filtered_rows


class SourceSampleFilterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.records = [
            {"inputs": f"input {i}", "targets": f"target {i}",
             "language": "English" if i < 80 else "Nepali", "domain": "news" if i % 2 else "books"}
            for i in range(100)
        ]

    def inventory(self, suffix="csv"):
        path = self.root / f"records.{suffix}"
        if suffix == "csv":
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.records[0]))
                writer.writeheader()
                writer.writerows(self.records)
        elif suffix == "jsonl":
            path.write_text("\n".join(json.dumps(record) for record in self.records))
        elif suffix == "json":
            path.write_text(json.dumps(self.records))
        elif suffix == "xlsx":
            from openpyxl import Workbook

            workbook = Workbook()
            sheet = workbook.active
            sheet.append(list(self.records[0]))
            for record in self.records:
                sheet.append(list(record.values()))
            workbook.save(path)
            workbook.close()
        return inspect_dataset(file_signatures([path]))

    def test_filters_before_sampling_and_projects_only_requested_fields(self):
        for suffix in ("csv", "jsonl", "json", "xlsx"):
            with self.subTest(format=suffix):
                inventory = self.inventory(suffix)
                original = copy.deepcopy(inventory)
                filters = {"language": ["Nepali", "npi"], "domain": ["news"]}
                result = sample_filtered_rows(inventory, 5, 42, ["inputs", "targets"], filters, 1000)
                self.assertEqual(result["rows_scanned"], 100)
                self.assertEqual(result["rows_matched"], 10)
                self.assertFalse(result["scan_limit_reached"])
                self.assertEqual(len(result["records"]), 5)
                for row in result["records"]:
                    index = row[f"{VIEWER_PREFIX}row_index"]
                    self.assertGreaterEqual(index, 80)
                    self.assertEqual(index % 2, 1)
                    self.assertEqual({key for key in row if not key.startswith(VIEWER_PREFIX)}, {"inputs", "targets"})
                    self.assertEqual(row["inputs"], f"input {index}")
                self.assertEqual(inventory, original)
                self.assertEqual(result, sample_filtered_rows(inventory, 5, 42, ["inputs", "targets"], filters, 1000))

    def test_scan_limit_and_case_sensitive_no_match_do_not_return_unfiltered_rows(self):
        inventory = self.inventory()
        limited = sample_filtered_rows(inventory, 5, 42, ["inputs"], {"language": ["Nepali"]}, 80)
        self.assertEqual(limited, {"records": [], "rows_scanned": 80, "rows_matched": 0, "scan_limit_reached": True})
        wrong_case = sample_filtered_rows(inventory, 5, 42, ["inputs"], {"language": ["nepali"]}, 1000)
        self.assertEqual(wrong_case["records"], [])
        self.assertEqual(wrong_case["rows_scanned"], 100)

    def test_invalid_filters_and_columns_fail_before_reading(self):
        inventory = self.inventory()
        for filters, columns, limit in (({"missing": ["Nepali"]}, ["inputs"], 100),
                                         ({"language": []}, ["inputs"], 100),
                                         ({"language": ["Nepali"]}, ["missing"], 100),
                                         ({"language": ["Nepali"]}, ["inputs"], 0)):
            with self.subTest(filters=filters, columns=columns, limit=limit), self.assertRaises(ValueError):
                sample_filtered_rows(inventory, 5, 42, columns, filters, limit)

    def test_filtered_parquet_preserves_file_and_global_row_identity(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        paths = [self.root / "first.parquet", self.root / "second.parquet"]
        for path, records in zip(paths, (self.records[:90], self.records[90:])):
            pq.write_table(pa.Table.from_pylist(records), path, row_group_size=10)
        inventory = inspect_dataset(file_signatures(paths))
        result = sample_filtered_rows(inventory, 30, 42, ["inputs"], {"language": ["Nepali"]}, 1000)
        self.assertEqual(len(result["records"]), 20)
        for row in result["records"]:
            index = row[f"{VIEWER_PREFIX}row_index"]
            self.assertEqual(row[f"{VIEWER_PREFIX}file"], str(paths[index >= 90]))
            self.assertEqual(row[f"{VIEWER_PREFIX}row_group"], index // 10 if index < 90 else 0)

    def test_huggingface_is_bounded_closes_stream_and_retains_base_filters(self):
        scanned, closed = [], []

        def records():
            try:
                for i in range(1000):
                    scanned.append(i)
                    yield {"id": str(i), "text": f"row {i}", "meta": {"languages": ["ne", "en"]},
                           "domain": "news" if i % 2 else "books", "split": "train" if i < 15 else "test"}
            finally:
                closed.append(True)

        inventory = {"format": "huggingface", "columns": ["id", "text", "meta", "domain", "split"],
                     "dataset_id": "fixture/data", "dataset_split": "train", "dataset_shards": ["second-shard"],
                     "row_filters": {"domain": ["news"]}, "filter_column": "split", "filter_value": "train"}
        with patch("attention_maps.explorer.sample_filters._load_inventory_huggingface_stream", side_effect=lambda *a, **k: records()) as loader:
            result = sample_filtered_rows(inventory, 20, 42, ["text"], {"meta.languages": ["ne"]}, 20)
        self.assertEqual(len(scanned), 20)
        self.assertEqual(closed, [True])
        self.assertEqual(result["rows_matched"], 7)
        self.assertTrue(result["scan_limit_reached"])
        self.assertEqual(loader.call_args.args[0]["dataset_shards"], ["second-shard"])
        self.assertEqual([row[f"{VIEWER_PREFIX}row_index"] for row in result["records"]], [str(i) for i in range(1, 15, 2)])
        self.assertTrue(all(row[f"{VIEWER_PREFIX}identity_stable"] for row in result["records"]))

    def test_ui_combines_filters_with_hidden_columns_and_resets_for_new_sources(self):
        from streamlit.testing.v1 import AppTest

        inventory = self.inventory()
        source = '''
from types import SimpleNamespace
import streamlit as st
from attention_maps.explorer import sample_dataset_rows
from attention_maps.explorer.sample_filters import sample_filtered_rows
from apps.explorer_tabs.overview import render_sample_tab
name = st.selectbox("Fixture source", ["first", "second"])
render_sample_tab(st=st, inventory=INVENTORY, spec=SimpleNamespace(key=name),
                  cached_sample=sample_dataset_rows, cached_filtered_sample=sample_filtered_rows)
'''.replace("INVENTORY", repr(inventory))
        app = AppTest.from_string(source, default_timeout=20).run()

        def widget(kind, label):
            return next(item for item in getattr(app, kind) if item.label == label)

        self.assertFalse(app.exception)
        widget("multiselect", "Columns to read").set_value(["inputs", "targets"]).run()
        widget("multiselect", "Filter rows by").set_value(["language", "domain"]).run()
        self.assertFalse(app.dataframe)  # Incomplete filters must not show an unfiltered sample.
        self.assertTrue(any("Enter at least one" in item.value for item in app.info))
        widget("text_area", "Allowed values for language").set_value("Nepali\nnpi")
        widget("text_area", "Allowed values for domain").set_value("news").run()
        self.assertFalse(app.exception)
        frame = app.dataframe[0].value
        self.assertEqual(list(frame.columns), ["inputs", "targets"])
        self.assertEqual(len(frame), 5)
        indices = [int(value.split()[-1]) for value in frame["inputs"]]
        self.assertTrue(all(index >= 80 and index % 2 for index in indices))
        self.assertTrue(any("100 source rows scanned · 10 matching rows" in item.value for item in app.caption))

        widget("number_input", "Maximum rows to scan").set_value(20).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.dataframe)
        self.assertTrue(any("No matching rows" in item.value for item in app.info))
        widget("number_input", "Maximum rows to scan").set_value(1000).run()
        self.assertEqual(len(app.dataframe[0].value), 5)
        widget("multiselect", "Filter rows by").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 5)
        self.assertFalse(any("matching rows" in item.value for item in app.caption))

        widget("multiselect", "Filter rows by").set_value(["language"]).run()
        widget("text_area", "Allowed values for language").set_value("Nepali").run()
        widget("selectbox", "Fixture source").set_value("second").run()
        self.assertFalse(app.exception)
        self.assertEqual(widget("multiselect", "Filter rows by").value, [])
        self.assertFalse(app.error)


if __name__ == "__main__":
    unittest.main()
