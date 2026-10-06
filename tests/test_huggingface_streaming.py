"""Regression coverage for script-free legacy Hub dataset loading."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec
from attention_maps.eda.pipeline import huggingface_records
from attention_maps.explorer.inspection import (
    inspect_huggingface_dataset,
    sample_huggingface_rows,
)


DATASET = "MBZUAI/Bactrian-X"
LEGACY_ERROR = "Dataset scripts are no longer supported, but found Bactrian-X.py"
VIEWER = "attention_maps.datasets.huggingface.urllib.request.urlopen"
RESOLVER = "attention_maps.datasets.huggingface._converted_parquet_files"


def parquet_file(config="ne", split="train"):
    return {
        "dataset": DATASET,
        "config": config,
        "split": split,
        "url": (
            f"https://huggingface.co/datasets/{DATASET}/resolve/"
            f"refs%2Fconvert%2Fparquet/{config}/{split}/0000.parquet"
        ),
        "size": 100,
    }


def viewer_response(files=None, **extra):
    return io.BytesIO(json.dumps({
        "parquet_files": files if files is not None else [
            parquet_file("en"), parquet_file("ne"), parquet_file("ne", "test")
        ],
        "partial": False,
        **extra,
    }).encode())


class FakeStream:
    num_shards = 1
    features = {key: "string" for key in ("id", "instruction", "input", "output")}
    info = SimpleNamespace(splits=None, download_size=None, download_checksums=None)

    def shuffle(self, **kwargs):
        self.shuffle_arguments = kwargs
        return self

    def take(self, count):
        return [{"id": "ne-1", "instruction": "प्रश्न", "input": "", "output": "उत्तर"}][:count]


class HuggingFaceStreamingTests(unittest.TestCase):
    def test_native_loader_keeps_revision_and_filters_without_fallback(self):
        with patch("datasets.load_dataset", return_value=FakeStream()) as loader, patch(RESOLVER) as resolver:
            loaded = load_huggingface_stream(
                DATASET, "ne", split="train", revision="commit123", token="test-token",
                filters=[("language", "==", "ne")],
            )
        self.assertFalse(loaded.parquet_files)
        self.assertEqual(loader.call_args.kwargs["revision"], "commit123")
        self.assertEqual(loader.call_args.kwargs["filters"], [("language", "==", "ne")])
        resolver.assert_not_called()

    def test_fallback_selects_only_requested_language_split_and_forwards_auth(self):
        with (
            patch("datasets.load_dataset", side_effect=[RuntimeError(LEGACY_ERROR), FakeStream()]) as loader,
            patch(VIEWER, return_value=viewer_response()) as viewer,
        ):
            loaded = load_huggingface_stream(
                DATASET, "ne", split="train", token="test-token",
                filters=[("id", "==", "ne-1")],
            )
        self.assertEqual(loaded.config, "ne")
        self.assertEqual(loaded.parquet_files, (parquet_file()["url"],))
        self.assertEqual(loader.call_args.args, ("parquet",))
        self.assertEqual(loader.call_args.kwargs["data_files"], {"train": [parquet_file()["url"]]})
        self.assertTrue(loader.call_args.kwargs["streaming"])
        self.assertEqual(loader.call_args.kwargs["token"], "test-token")
        self.assertEqual(loader.call_args.kwargs["filters"], [("id", "==", "ne-1")])
        self.assertNotIn("trust_remote_code", loader.call_args.kwargs)
        self.assertEqual(viewer.call_args.args[0].get_header("Authorization"), "Bearer test-token")

    def test_inspection_and_sampling_reuse_resolved_files_and_stable_ids(self):
        stream = FakeStream()
        with (
            patch("datasets.load_dataset", side_effect=[RuntimeError(LEGACY_ERROR), stream, stream]) as loader,
            patch(VIEWER, return_value=viewer_response()) as viewer,
            patch("attention_maps.explorer.inspection._huggingface_viewer_split_size", return_value={
                "num_rows": 67_017, "num_bytes_memory": 109_698_333,
                "num_bytes_parquet_files": 43_613_675,
            }) as size,
        ):
            inventory = inspect_huggingface_dataset(DATASET, "train", config="ne")
            rows = sample_huggingface_rows(inventory, 1, 42, ("instruction", "output"))
        self.assertEqual(inventory["rows"], 67_017)
        self.assertEqual(inventory["dataset_config"], "ne")
        self.assertEqual(inventory["dataset_parquet_files"], [parquet_file()["url"]])
        self.assertEqual(inventory["hub_file_bytes"], 43_613_675)
        self.assertEqual(rows[0]["output"], "उत्तर")
        self.assertEqual(rows[0]["__viewer_row_index"], "ne-1")
        self.assertTrue(rows[0]["__viewer_identity_stable"])
        self.assertEqual(loader.call_args.args, ("parquet",))
        self.assertEqual(size.call_args.kwargs["config"], "ne")
        viewer.assert_called_once()

    def test_eda_stream_uses_the_same_fallback(self):
        stream = FakeStream()
        with (
            patch("datasets.load_dataset", side_effect=[RuntimeError(LEGACY_ERROR), stream]) as loader,
            patch(VIEWER, return_value=viewer_response()),
        ):
            records = huggingface_records(
                DatasetSpec("bactrian-ne", DATASET, config_name="ne"),
                AnalysisConfig(shuffle_buffer_size=25),
            )
        self.assertIs(records, stream)
        self.assertEqual(loader.call_args.kwargs["data_files"], {"train": [parquet_file()["url"]]})
        self.assertEqual(stream.shuffle_arguments["buffer_size"], 25)

    def test_ambiguous_configuration_explains_nepali_selection(self):
        with (
            patch("datasets.load_dataset", side_effect=RuntimeError(LEGACY_ERROR)),
            patch(VIEWER, return_value=viewer_response()),
            self.assertRaisesRegex(ValueError, "For Nepali, use configuration 'ne'"),
        ):
            load_huggingface_stream(DATASET, split="train")

    def test_single_configuration_is_resolved_instead_of_mixing_subsets(self):
        with (
            patch("datasets.load_dataset", side_effect=[RuntimeError(LEGACY_ERROR), FakeStream()]),
            patch(VIEWER, return_value=viewer_response([parquet_file()])),
        ):
            loaded = load_huggingface_stream(DATASET, split="train")
        self.assertEqual(loaded.config, "ne")

    def test_missing_configuration_or_split_never_loads_another_subset(self):
        for config, split in (("missing", "train"), ("ne", "validation")):
            with (
                self.subTest(config=config, split=split),
                patch("datasets.load_dataset", side_effect=RuntimeError(LEGACY_ERROR)) as loader,
                patch(VIEWER, return_value=viewer_response()),
                self.assertRaisesRegex(ValueError, "No converted Parquet files"),
            ):
                load_huggingface_stream(DATASET, config, split=split)
            loader.assert_called_once()

    def test_pinned_source_revision_is_never_replaced_by_default_conversion(self):
        with (
            patch("datasets.load_dataset", side_effect=RuntimeError(LEGACY_ERROR)),
            patch(RESOLVER) as resolver,
            self.assertRaisesRegex(ValueError, "cannot guarantee the requested source revision"),
        ):
            load_huggingface_stream(DATASET, "ne", split="train", revision="commit123")
        resolver.assert_not_called()

    def test_unrelated_loading_errors_are_preserved(self):
        for error in (RuntimeError("network unavailable"), ValueError("unknown split")):
            with (
                self.subTest(error=str(error)),
                patch("datasets.load_dataset", side_effect=error),
                patch(RESOLVER) as resolver,
                self.assertRaises(type(error)) as raised,
            ):
                load_huggingface_stream(DATASET, "ne", split="train")
            self.assertIs(raised.exception, error)
            resolver.assert_not_called()

    def test_partial_conversion_is_rejected(self):
        with (
            patch("datasets.load_dataset", side_effect=RuntimeError(LEGACY_ERROR)),
            patch(VIEWER, return_value=viewer_response(partial=True)),
            self.assertRaisesRegex(ValueError, "incomplete Parquet conversion"),
        ):
            load_huggingface_stream(DATASET, "ne", split="train")

    def test_unavailable_conversion_has_actionable_error(self):
        with (
            patch("datasets.load_dataset", side_effect=RuntimeError(LEGACY_ERROR)),
            patch(VIEWER, side_effect=OSError("service unavailable")),
            self.assertRaisesRegex(ValueError, "Use a JSON/Parquet export"),
        ):
            load_huggingface_stream(DATASET, "ne", split="train")


if __name__ == "__main__":
    unittest.main()
