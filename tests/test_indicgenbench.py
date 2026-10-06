"""Nested-JSON benchmark loading must preserve language and evaluation splits."""

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.datasets.indicgenbench import FLORES_IN_DATASET_ID
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec
from attention_maps.eda.pipeline import analyze_records, huggingface_records
from attention_maps.explorer.inspection import inspect_huggingface_dataset, sample_huggingface_rows
from attention_maps.explorer.source_imports import huggingface_source_spec


REVISION = "0123456789abcdef0123456789abcdef01234567"


@contextmanager
def local_benchmark():
    from datasets import load_dataset

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        names = []
        for language in ("ne", "hi"):
            for direction in ("enxx", "xxen"):
                pair = f"en_{language}" if direction == "enxx" else f"{language}_en"
                for suffix, count in (("dev", 2), ("test", 1)):
                    filename = f"flores_{pair}_{suffix}.json"
                    names.append(filename)
                    rows = [{
                        "source": f"Source sentence {index}",
                        "target": f"अनुवाद गरिएको वाक्य {index}",
                        "lang": language,
                        "translation_direction": direction,
                    } for index in range(count)]
                    (root / filename).write_text(json.dumps({
                        "canary": "Evaluation metadata, not a corpus record.",
                        "examples": rows,
                    }, ensure_ascii=False), encoding="utf-8")
        info = SimpleNamespace(
            sha=REVISION, card_data=None, siblings=[
                SimpleNamespace(rfilename=name, size=(root / name).stat().st_size) for name in names
            ]
        )

        def local_load(*args, **kwargs):
            return load_dataset(*args, cache_dir=str(root / "cache"), **kwargs)

        def local_url(repo_id, filename, **kwargs):
            return str(root / filename)

        with (
            patch("huggingface_hub.HfApi.dataset_info", return_value=info) as metadata,
            patch("huggingface_hub.hf_hub_url", side_effect=local_url) as urls,
            patch("datasets.load_dataset", side_effect=local_load) as loader,
            patch("attention_maps.explorer.inspection._huggingface_viewer_split_size",
                  side_effect=AssertionError("The dataset viewer is disabled")),
        ):
            yield root, metadata, urls, loader


class IndicGenBenchTests(unittest.TestCase):
    def test_train_is_rejected_with_available_splits_before_network_access(self):
        with patch("huggingface_hub.HfApi.dataset_info") as metadata:
            with self.assertRaisesRegex(ValueError, "has no train split.*validation"):
                load_huggingface_stream(FLORES_IN_DATASET_ID, split="train")
        metadata.assert_not_called()

    def test_requires_explicit_language_for_bounded_downloads(self):
        with patch("huggingface_hub.HfApi.dataset_info") as metadata:
            with self.assertRaisesRegex(ValueError, "use 'ne' for Nepali"):
                load_huggingface_stream(FLORES_IN_DATASET_ID, split="validation")
        metadata.assert_not_called()

    def test_inventory_uses_exact_cached_counts_without_dataset_viewer(self):
        with local_benchmark() as (root, metadata, urls, loader):
            inventory = inspect_huggingface_dataset(
                FLORES_IN_DATASET_ID, "validation", config="ne", revision="release-tag"
            )
            self.assertEqual(inventory["rows"], 4)
            self.assertEqual(inventory["files"], 2)
            self.assertEqual(inventory["columns"], ["source", "target", "lang", "translation_direction"])
            self.assertEqual(inventory["dataset_revision"], REVISION)
            self.assertEqual(inventory["loading_strategy"], "memory_mapped_json")
            self.assertEqual(inventory["hub_file_bytes"], sum(
                (root / name).stat().st_size
                for name in ("flores_en_ne_dev.json", "flores_ne_en_dev.json")
            ))
            self.assertGreater(inventory["memory_bytes"], 0)
            self.assertEqual(metadata.call_args.kwargs["revision"], "release-tag")
            self.assertTrue(all(call.kwargs["revision"] == REVISION for call in urls.call_args_list))
            self.assertEqual(loader.call_args.kwargs["field"], "examples")
            self.assertFalse(loader.call_args.kwargs["keep_in_memory"])
            self.assertNotIn("canary", inventory["columns"])

    def test_sampler_reuses_resolved_revision_and_nepali_validation_rows(self):
        with local_benchmark() as (_, metadata, _, _):
            inventory = inspect_huggingface_dataset(FLORES_IN_DATASET_ID, "validation", config="ne")
            columns = ("source", "target", "lang", "translation_direction")
            first = sample_huggingface_rows(inventory, 4, 42, columns)
            second = sample_huggingface_rows(inventory, 4, 42, columns)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual({row["lang"] for row in first}, {"ne"})
            self.assertEqual({row["translation_direction"] for row in first}, {"enxx", "xxen"})
            self.assertEqual(metadata.call_args.kwargs["revision"], REVISION)

    def test_test_split_never_reads_dev_files(self):
        with local_benchmark() as (_, _, urls, _):
            inventory = inspect_huggingface_dataset(FLORES_IN_DATASET_ID, "test", config="ne")
            self.assertEqual(inventory["rows"], 2)
            self.assertEqual({call.args[1] for call in urls.call_args_list}, {
                "flores_en_ne_test.json", "flores_ne_en_test.json"
            })

    def test_unknown_language_does_not_download_other_languages(self):
        with local_benchmark() as (_, _, urls, loader):
            with self.assertRaisesRegex(ValueError, "Available languages: hi, ne"):
                inspect_huggingface_dataset(FLORES_IN_DATASET_ID, "validation", config="zz")
            urls.assert_not_called()
            loader.assert_not_called()

    def test_direction_filter_keeps_original_population_counts(self):
        with local_benchmark():
            inventory = inspect_huggingface_dataset(
                FLORES_IN_DATASET_ID, "validation", config="ne",
                filter_column="translation_direction", filter_value="enxx",
            )
            self.assertEqual(inventory["source_rows"], 4)
            self.assertEqual(inventory["rows"], 2)
            rows = sample_huggingface_rows(inventory, 2, 42, ("translation_direction",))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["translation_direction"] == "enxx" for row in rows))

    def test_eda_cli_uses_nested_examples_through_shared_loader(self):
        with local_benchmark():
            spec = DatasetSpec(
                "flores-in-ne", FLORES_IN_DATASET_ID, config_name="ne",
                split="validation", text_columns=("source", "target"),
            )
            config = AnalysisConfig(sample_size=4)
            profile = analyze_records(spec, huggingface_records(spec, config), config)
            self.assertEqual(profile.summary.rows_seen, 4)
            self.assertEqual(profile.summary.usable_rows, 4)

    def test_registered_source_retains_benchmark_purpose(self):
        spec = huggingface_source_spec(
            FLORES_IN_DATASET_ID, schema="pretraining", config="ne", split="validation"
        )
        self.assertEqual(spec.primary_purpose, "Evaluation / benchmark")
        self.assertEqual(spec.dataset_split, "validation")

    def test_streamlit_can_inspect_nested_json_benchmark(self):
        from streamlit.testing.v1 import AppTest

        with local_benchmark():
            app = AppTest.from_file(
                Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
            ).run(timeout=30)
            next(item for item in app.selectbox if item.label == "Data source").set_value("Hugging Face").run(timeout=30)
            next(item for item in app.text_input if item.label == "Hugging Face dataset ID").set_value(FLORES_IN_DATASET_ID)
            next(item for item in app.button if item.label == "Load Hugging Face source").click().run(timeout=30)
            next(item for item in app.selectbox if item.label == "Dataset configuration").set_value("ne").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(item.value for item in app.metric if item.label == "Rows"), "4")
            self.assertTrue(any("Evaluation / benchmark" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
