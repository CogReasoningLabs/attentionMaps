"""NepFakeV2 uses only its CSV, with bounded parsing and exact counts."""

import csv
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.datasets.nepfake import CSV_BATCH_ROWS, NEPFAKE_CSV, NEPFAKE_DATASET_ID
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec
from attention_maps.eda.pipeline import analyze_records, huggingface_records
from attention_maps.explorer.inspection import inspect_huggingface_dataset, sample_huggingface_rows


REVISION = "0123456789abcdef0123456789abcdef01234567"


@contextmanager
def local_nepfake():
    from datasets import load_dataset

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "nepfakev2.csv"
        rows = [{
            "example_id": f"NF2_{index}", "claim_text": f"नेपालको समाचार {index}",
            "verdict_label": index % 3, "verdict_label_text": "REAL",
            "evidence_text": 'प्रमाण, विवरण\nदोस्रो हरफ "उद्धरण"',
            "source_name": "example", "source_url": f"https://example.org/{index}",
            "date_published": "2026-09-01", "is_native_nepali": True,
            "topic_category": "society", "annotator_notes": "", "schema_version": "1.0",
        } for index in range(5)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        info = SimpleNamespace(sha=REVISION, card_data=None, siblings=[
            SimpleNamespace(rfilename=name, size=path.stat().st_size if name == NEPFAKE_CSV else 10)
            for name in (NEPFAKE_CSV, "data/nepfakev2.json", "data/stats.json")
        ])

        def local_load(*args, **kwargs):
            return load_dataset(*args, cache_dir=str(root / "cache"), **kwargs)

        with (
            patch("huggingface_hub.HfApi.dataset_info", return_value=info) as metadata,
            patch("huggingface_hub.hf_hub_url", return_value=str(path)) as urls,
            patch("datasets.load_dataset", side_effect=local_load) as loader,
            patch("attention_maps.explorer.inspection._huggingface_viewer_split_size",
                  side_effect=AssertionError("Viewer processing has failed")),
        ):
            yield path, metadata, urls, loader


class NepFakeTests(unittest.TestCase):
    def test_inventory_counts_only_csv_records_and_preserves_schema(self):
        with local_nepfake() as (path, metadata, urls, loader):
            inventory = inspect_huggingface_dataset(NEPFAKE_DATASET_ID, "train", revision="release")
            self.assertEqual(inventory["rows"], 5)
            self.assertEqual(inventory["files"], 1)
            self.assertEqual(len(inventory["columns"]), 12)
            self.assertNotIn("total_examples", inventory["columns"])
            self.assertEqual(inventory["hub_file_bytes"], path.stat().st_size)
            self.assertEqual(inventory["hub_file_size_basis"], "original files")
            self.assertEqual(inventory["dataset_config"], "default")
            self.assertEqual(inventory["dataset_revision"], REVISION)
            self.assertEqual(inventory["loading_strategy"], "memory_mapped_csv")
            self.assertGreater(inventory["memory_bytes"], 0)
            self.assertEqual(metadata.call_args.kwargs["revision"], "release")
            urls.assert_called_once_with(NEPFAKE_DATASET_ID, NEPFAKE_CSV, repo_type="dataset", revision=REVISION)
            self.assertEqual(loader.call_args.args, ("csv",))
            self.assertEqual(loader.call_args.kwargs["chunksize"], CSV_BATCH_ROWS)
            self.assertFalse(loader.call_args.kwargs["keep_in_memory"])

    def test_sampling_reuses_revision_and_preserves_multiline_text_and_types(self):
        with local_nepfake() as (_, metadata, _, _):
            inventory = inspect_huggingface_dataset(NEPFAKE_DATASET_ID, "train")
            columns = ("example_id", "claim_text", "evidence_text", "verdict_label", "is_native_nepali", "schema_version")
            first = sample_huggingface_rows(inventory, 5, 42, columns)
            second = sample_huggingface_rows(inventory, 5, 42, columns)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)
            self.assertTrue(all("\n" in row["evidence_text"] for row in first))
            self.assertTrue(all(isinstance(row["verdict_label"], int) for row in first))
            self.assertTrue(all(row["is_native_nepali"] is True for row in first))
            self.assertTrue(all(row["schema_version"] == "1.0" for row in first))
            self.assertEqual(metadata.call_args.kwargs["revision"], REVISION)

    def test_shared_eda_loader_uses_csv_records(self):
        with local_nepfake():
            spec = DatasetSpec("nepfake", NEPFAKE_DATASET_ID, text_columns=("claim_text", "evidence_text"))
            config = AnalysisConfig(sample_size=5)
            profile = analyze_records(spec, huggingface_records(spec, config), config)
            self.assertEqual(profile.summary.rows_seen, 5)
            self.assertEqual(profile.summary.usable_rows, 5)

    def test_filters_preserve_population_metadata(self):
        with local_nepfake():
            inventory = inspect_huggingface_dataset(
                NEPFAKE_DATASET_ID, "train", filter_column="example_id", filter_value="NF2_0"
            )
            self.assertEqual(inventory["rows"], 1)
            self.assertEqual(inventory["source_rows"], 5)
            rows = sample_huggingface_rows(inventory, 1, 42, ("example_id",))
            self.assertEqual(rows[0]["example_id"], "NF2_0")

    def test_invalid_selection_fails_before_network_access(self):
        for config, split in (("ne", "train"), (None, "test")):
            with self.subTest(config=config, split=split), patch("huggingface_hub.HfApi.dataset_info") as metadata:
                with self.assertRaisesRegex(ValueError, "configuration 'default'.*split 'train'"):
                    load_huggingface_stream(NEPFAKE_DATASET_ID, config, split=split)
                metadata.assert_not_called()

    def test_missing_csv_never_falls_back_to_stats_json(self):
        with (
            patch("huggingface_hub.HfApi.dataset_info", return_value=SimpleNamespace(
                sha=REVISION, siblings=[SimpleNamespace(rfilename="data/stats.json")]
            )),
            patch("datasets.load_dataset") as loader,
            self.assertRaisesRegex(ValueError, "no data/nepfakev2.csv"),
        ):
            load_huggingface_stream(NEPFAKE_DATASET_ID, split="train")
        loader.assert_not_called()

    def test_streamlit_loads_nepfake_and_displays_size_basis(self):
        from streamlit.testing.v1 import AppTest

        with local_nepfake():
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py").run(timeout=30)
            next(item for item in app.selectbox if item.label == "Data source").set_value("Hugging Face").run(timeout=30)
            next(item for item in app.text_input if item.label == "Hugging Face dataset ID").set_value(NEPFAKE_DATASET_ID)
            next(item for item in app.button if item.label == "Load Hugging Face source").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(item.value for item in app.metric if item.label == "Rows"), "5")
            self.assertNotEqual(next(item.value for item in app.metric if item.label == "Selected file size"), "Unknown")
            self.assertTrue(any("original files for the selected split" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
