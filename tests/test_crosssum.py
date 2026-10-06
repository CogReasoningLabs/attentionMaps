"""CrossSum archive ingestion preserves language direction, splits, and scope."""

import io
import json
import tarfile
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.datasets.crosssum import CROSSSUM_BATCH_ROWS, CROSSSUM_COLUMNS, CROSSSUM_DATASET_ID, _crosssum_rows
from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec
from attention_maps.eda.pipeline import analyze_records, huggingface_records
from attention_maps.explorer.inspection import inspect_huggingface_dataset, sample_huggingface_rows


REVISION = "0123456789abcdef0123456789abcdef01234567"


def add_member(archive, name, body):
    payload = body.encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


@contextmanager
def local_crosssum(train_count=4, empty_test=False):
    from datasets import Dataset

    from_generator = Dataset.from_generator
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {}
        for pair in ("nepali-nepali", "english-nepali", "nepali-english"):
            name = f"data/{pair}_CrossSum.tar.bz2"
            path = root / name
            path.parent.mkdir(exist_ok=True)
            with tarfile.open(path, "w:bz2") as archive:
                for suffix, count in (("train", train_count), ("val", 2), ("test", 0 if empty_test else 1)):
                    rows = [{
                        "source_url": f"https://example.org/{pair}/{suffix}/{index}",
                        "target_url": f"https://example.org/summary/{index}",
                        "text": f"{pair} {suffix} समाचार विवरण {index}",
                        "summary": f"समाचार सारांश {index}",
                    } for index in range(count)]
                    add_member(archive, f"./{pair}_{suffix}.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                add_member(archive, "../../unrelated.txt", "This must never be extracted.")
            paths[name] = path
        info = SimpleNamespace(sha=REVISION, card_data=None, siblings=[
            SimpleNamespace(rfilename=name, size=path.stat().st_size) for name, path in paths.items()
        ])

        def local_download(repo_id, filename, **kwargs):
            return str(paths[filename])

        def local_generator(*args, **kwargs):
            return from_generator(*args, cache_dir=str(root / "arrow"), **kwargs)

        with (
            patch("huggingface_hub.HfApi.dataset_info", return_value=info) as metadata,
            patch("huggingface_hub.hf_hub_download", side_effect=local_download) as download,
            patch("datasets.Dataset.from_generator", side_effect=local_generator) as indexer,
            patch("attention_maps.explorer.inspection._huggingface_viewer_split_size",
                  side_effect=AssertionError("CrossSum Viewer is disabled")),
            patch("datasets.load_dataset", side_effect=AssertionError("Do not load the legacy script")),
        ):
            yield root, metadata, download, indexer


class CrossSumTests(unittest.TestCase):
    def test_missing_configuration_explains_language_direction_before_network(self):
        for config in (None, "ne", "../nepali-nepali"):
            with self.subTest(config=config), patch("huggingface_hub.HfApi.dataset_info") as metadata:
                with self.assertRaisesRegex(ValueError, "requires a source-target language pair.*english-nepali"):
                    load_huggingface_stream(CROSSSUM_DATASET_ID, config, split="train")
                metadata.assert_not_called()

    def test_invalid_split_is_not_silently_replaced(self):
        with patch("huggingface_hub.HfApi.dataset_info") as metadata:
            with self.assertRaisesRegex(ValueError, "split must be 'train', 'validation', or 'test'"):
                load_huggingface_stream(CROSSSUM_DATASET_ID, "nepali-nepali", split="val")
            metadata.assert_not_called()

    def test_inventory_has_selected_split_counts_and_whole_pair_archive_size(self):
        with local_crosssum() as (root, metadata, download, indexer):
            inventory = inspect_huggingface_dataset(CROSSSUM_DATASET_ID, "train", config="nepali-nepali", revision="release", token="test-token")
            self.assertEqual(inventory["rows"], 4)
            self.assertEqual(inventory["files"], 1)
            self.assertEqual(inventory["columns"], list(CROSSSUM_COLUMNS))
            self.assertGreater(inventory["memory_bytes"], 0)
            self.assertEqual(inventory["dataset_revision"], REVISION)
            self.assertEqual(inventory["hub_file_bytes"], (root / "data/nepali-nepali_CrossSum.tar.bz2").stat().st_size)
            self.assertEqual(inventory["hub_file_size_basis"], "compressed archive")
            self.assertEqual(inventory["hub_file_size_scope"], "selected language pair (all splits)")
            self.assertEqual(inventory["loading_strategy"], "memory_mapped_archive")
            self.assertEqual(metadata.call_args.kwargs["revision"], "release")
            download.assert_called_once_with(CROSSSUM_DATASET_ID, "data/nepali-nepali_CrossSum.tar.bz2", repo_type="dataset", revision=REVISION, token="test-token")
            self.assertEqual(indexer.call_args.kwargs["writer_batch_size"], CROSSSUM_BATCH_ROWS)
            self.assertFalse(indexer.call_args.kwargs["keep_in_memory"])
            self.assertFalse((root / "unrelated.txt").exists())

    def test_validation_sampling_preserves_direction_and_reuses_revision(self):
        with local_crosssum() as (_, metadata, download, _):
            inventory = inspect_huggingface_dataset(CROSSSUM_DATASET_ID, "validation", config="english-nepali")
            self.assertEqual(inventory["rows"], 2)
            first = sample_huggingface_rows(inventory, 2, 42, ("text", "summary"))
            second = sample_huggingface_rows(inventory, 2, 42, ("text", "summary"))
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            self.assertTrue(all(row["text"].startswith("english-nepali val") for row in first))
            self.assertEqual(metadata.call_args.kwargs["revision"], REVISION)
            self.assertTrue(all(call.args[1] == "data/english-nepali_CrossSum.tar.bz2" for call in download.call_args_list))

    def test_eda_script_uses_the_same_reader(self):
        with local_crosssum():
            spec = DatasetSpec("crosssum-ne", CROSSSUM_DATASET_ID, config_name="nepali-nepali", text_columns=("text", "summary"))
            config = AnalysisConfig(sample_size=4)
            profile = analyze_records(spec, huggingface_records(spec, config), config)
            self.assertEqual(profile.summary.rows_seen, 4)
            self.assertEqual(profile.summary.usable_rows, 4)

    def test_filters_keep_unfiltered_population_and_archive_bytes(self):
        with local_crosssum():
            inventory = inspect_huggingface_dataset(CROSSSUM_DATASET_ID, "train", config="nepali-nepali",
                filter_column="target_url", filter_value="https://example.org/summary/0")
            self.assertEqual(inventory["source_rows"], 4)
            self.assertEqual(inventory["rows"], 1)
            self.assertGreater(inventory["hub_file_bytes"], 0)
            rows = sample_huggingface_rows(inventory, 1, 42, ("summary",))
            self.assertEqual(rows[0]["summary"], "समाचार सारांश 0")

    def test_empty_split_has_zero_rows_without_a_placeholder(self):
        with local_crosssum(empty_test=True):
            inventory = inspect_huggingface_dataset(CROSSSUM_DATASET_ID, "test", config="nepali-nepali")
            self.assertEqual(inventory["rows"], 0)
            self.assertEqual(inventory["memory_bytes"], 0)
            self.assertEqual(sample_huggingface_rows(inventory, 1, 42, ("text",)), [])

    def test_rows_survive_multiple_writer_batches(self):
        with local_crosssum(train_count=CROSSSUM_BATCH_ROWS + 3):
            inventory = inspect_huggingface_dataset(CROSSSUM_DATASET_ID, "train", config="nepali-nepali")
            self.assertEqual(inventory["rows"], CROSSSUM_BATCH_ROWS + 3)
            rows = sample_huggingface_rows(inventory, CROSSSUM_BATCH_ROWS + 3, 42, ("source_url",))
            self.assertEqual(len({row["source_url"] for row in rows}), CROSSSUM_BATCH_ROWS + 3)

    def test_unknown_pair_never_downloads_another_archive(self):
        with local_crosssum() as (_, _, download, _):
            with self.assertRaisesRegex(ValueError, "no archive for language pair"):
                load_huggingface_stream(CROSSSUM_DATASET_ID, "unknown-nepali", split="train")
            download.assert_not_called()

    def test_archive_members_must_be_exact_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar.bz2"
            member_name = "nepali-nepali_train.jsonl"
            for scenario in ("traversal", "duplicate", "symlink"):
                with self.subTest(scenario=scenario):
                    with tarfile.open(path, "w:bz2") as archive:
                        if scenario == "traversal":
                            add_member(archive, "../" + member_name, "{}\n")
                        elif scenario == "duplicate":
                            add_member(archive, member_name, "{}\n")
                            add_member(archive, "./" + member_name, "{}\n")
                        else:
                            member = tarfile.TarInfo(member_name)
                            member.type = tarfile.SYMTYPE
                            member.linkname = "elsewhere"
                            archive.addfile(member)
                    with self.assertRaisesRegex(ValueError, "exactly one regular file"):
                        list(_crosssum_rows(str(path), member_name))

    def test_bad_record_schema_fails_with_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.tar.bz2"
            member_name = "nepali-nepali_train.jsonl"
            with tarfile.open(path, "w:bz2") as archive:
                add_member(archive, member_name, '{"text": "missing summary"}\n')
            with self.assertRaisesRegex(ValueError, "record schema.*train.jsonl:1"):
                list(_crosssum_rows(str(path), member_name))

    def test_streamlit_loads_pair_and_explains_archive_size_scope(self):
        from streamlit.testing.v1 import AppTest

        with local_crosssum():
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py").run(timeout=30)
            next(item for item in app.selectbox if item.label == "Data source").set_value("Hugging Face").run(timeout=30)
            next(item for item in app.text_input if item.label == "Hugging Face dataset ID").set_value(CROSSSUM_DATASET_ID)
            next(item for item in app.button if item.label == "Load Hugging Face source").click().run(timeout=30)
            next(item for item in app.selectbox if item.label == "Dataset configuration").set_value("nepali-nepali").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(item.value for item in app.metric if item.label == "Rows"), "4")
            self.assertTrue(any("compressed archive for the selected language pair (all splits)" in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
