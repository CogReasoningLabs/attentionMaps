"""Official FLORES access, split selection, and revision-aware size metadata."""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from attention_maps.datasets.flores import FACEBOOK_FLORES_DATASET_ID
from attention_maps.datasets.huggingface import load_huggingface_stream
from attention_maps.eda.contracts import AnalysisConfig, DatasetSpec
from attention_maps.eda.pipeline import analyze_records, huggingface_records
from attention_maps.explorer.inspection import inspect_huggingface_dataset, sample_huggingface_rows
from attention_maps.explorer.source_imports import huggingface_source_spec


REVISION = "0123456789abcdef0123456789abcdef01234567"


def denied(code):
    from huggingface_hub.errors import HfHubHTTPError

    return HfHubHTTPError("Access denied", response=httpx.Response(
        code, request=httpx.Request("GET", "https://huggingface.co/api/datasets/facebook/flores/auth-check")
    ))


@contextmanager
def local_flores():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import SplitDict, SplitInfo, load_dataset

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = []
        for config in ("npi_Deva", "eng_Latn-npi_Deva"):
            folder = f"pair/{config}" if "-" in config else f"language/{config}"
            for split in ("dev", "devtest"):
                name = f"data/{folder}/{split}-00000-of-00001.parquet"
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = [{"id": index, "URL": "https://example.org", "domain": "news", "topic": "society",
                         "has_image": 0, "has_hyperlink": 0} for index in range(3 if split == "dev" else 2)]
                for row in rows:
                    if "-" in config:
                        row.update(sentence_eng_Latn="A sentence", sentence_npi_Deva="नेपाली वाक्य")
                    else:
                        row["sentence"] = "नेपाली वाक्य"
                pq.write_table(pa.Table.from_pylist(rows), path)
                files.append(SimpleNamespace(rfilename=name, size=path.stat().st_size))
        info = SimpleNamespace(sha=REVISION, card_data=None, siblings=files)

        def local_load(repo, config, *, split, **kwargs):
            folder = f"pair/{config}" if "-" in config else f"language/{config}"
            selected = [str(root / item.rfilename) for item in files if item.rfilename.startswith(f"data/{folder}/{split}-")]
            data = load_dataset("parquet", data_files={split: selected}, split=split, cache_dir=str(root / "cache"))
            stream = data.to_iterable_dataset()
            stream.info.config_name = config
            # Model native card metadata: download_size spans BOTH splits.
            stream.info.download_size = sum(item.size for item in files if item.rfilename.startswith(f"data/{folder}/"))
            stream.info.splits = SplitDict({
                split: SplitInfo(name=split, num_examples=len(data), num_bytes=data.data.nbytes),
                "unused": SplitInfo(name="unused", num_examples=7, num_bytes=500),
            })
            if kwargs.get("filters"):
                stream = stream.filter(lambda row: all(row.get(column) == value for column, _, value in kwargs["filters"]))
            return stream

        def local_builder(repo, *, name, **kwargs):
            folder = f"pair/{name}" if "-" in name else f"language/{name}"
            data_files = {
                split: [f"hf://datasets/{repo}@{REVISION}/{item.rfilename}" for item in files
                        if item.rfilename.startswith(f"data/{folder}/{split}-")]
                for split in ("dev", "devtest")
            }
            return SimpleNamespace(config=SimpleNamespace(data_files=data_files),
                                   info=SimpleNamespace(splits=None, features=None))

        def local_footer(url, **kwargs):
            return footer(str(root / url.split(REVISION + "/", 1)[1]), token=kwargs.get("token"))

        from attention_maps.explorer.huggingface import _parquet_metadata as footer
        with (
            patch("datasets.get_dataset_config_names", return_value=["npi_Deva", "eng_Latn-npi_Deva"]),
            patch("datasets.load_dataset_builder", side_effect=local_builder),
            patch("attention_maps.explorer.huggingface._parquet_metadata", side_effect=local_footer),
            patch("huggingface_hub.HfApi.auth_check") as access,
            patch("huggingface_hub.HfApi.dataset_info", return_value=info) as metadata,
            patch("datasets.load_dataset", side_effect=local_load) as loader,
            patch("attention_maps.explorer.inspection._huggingface_viewer_split_size",
                  side_effect=AssertionError("No gated Dataset Viewer access")),
            patch("attention_maps.datasets.huggingface._converted_parquet_files",
                  side_effect=AssertionError("Do not use a legacy conversion or mirror")),
        ):
            yield root, info, access, metadata, loader


class FacebookFloresTests(unittest.TestCase):
    def test_train_explains_split_and_nepali_configuration_before_network(self):
        with patch("huggingface_hub.HfApi.auth_check") as access:
            with self.assertRaisesRegex(ValueError, "no train split.*devtest.*npi_Deva"):
                load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, split="train")
        access.assert_not_called()

    def test_missing_or_wrong_configuration_explains_language_code(self):
        for config in (None, "ne", "../npi_Deva"):
            with self.subTest(config=config), patch("huggingface_hub.HfApi.auth_check") as access:
                with self.assertRaisesRegex(ValueError, "Select a FLORES configuration.*npi_Deva"):
                    load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, config, split="dev")
                access.assert_not_called()

    def test_access_denial_is_actionable_and_never_attempts_data_or_conversion(self):
        for status in (401, 403):
            with (
                self.subTest(status=status),
                patch("huggingface_hub.HfApi.auth_check", side_effect=denied(status)),
                patch("datasets.load_dataset") as loader,
                patch("attention_maps.datasets.huggingface._converted_parquet_files") as conversion,
                self.assertRaisesRegex(ValueError, "requires Hugging Face dataset access.*HF_TOKEN"),
            ):
                load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, "npi_Deva", split="dev", token="test-token")
            loader.assert_not_called()
            conversion.assert_not_called()

    def test_inventory_uses_selected_split_bytes_and_pins_current_commit(self):
        with local_flores() as (root, _, access, metadata, loader):
            inventory = inspect_huggingface_dataset(FACEBOOK_FLORES_DATASET_ID, "dev", config="npi_Deva", token="test-token")
            self.assertEqual(inventory["rows"], 3)
            self.assertEqual(inventory["files"], 1)
            self.assertEqual(inventory["dataset_revision"], REVISION)
            self.assertEqual(inventory["hub_file_bytes"], (root / "data/language/npi_Deva/dev-00000-of-00001.parquet").stat().st_size)
            self.assertEqual(inventory["hub_file_size_basis"], "original files")
            self.assertEqual(inventory["hub_file_size_scope"], "selected split")
            access.assert_called_once_with(FACEBOOK_FLORES_DATASET_ID, repo_type="dataset", token="test-token")
            self.assertEqual(metadata.call_args.kwargs["revision"], "main")
            self.assertEqual(loader.call_args.kwargs["revision"], REVISION)
            self.assertTrue(loader.call_args.kwargs["streaming"])
            self.assertEqual(loader.call_args.kwargs["token"], "test-token")
            self.assertNotIn("trust_remote_code", loader.call_args.kwargs)

    def test_devtest_sampling_reuses_the_inspected_revision(self):
        with local_flores() as (_, _, _, metadata, _):
            inventory = inspect_huggingface_dataset(FACEBOOK_FLORES_DATASET_ID, "devtest", config="npi_Deva", revision="release")
            self.assertEqual(metadata.call_args.kwargs["revision"], "release")
            self.assertEqual(inventory["rows"], 2)
            rows = sample_huggingface_rows(inventory, 2, 42, ("id", "sentence"))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["sentence"] == "नेपाली वाक्य" for row in rows))
            self.assertEqual(metadata.call_args.kwargs["revision"], REVISION)

    def test_english_nepali_pair_uses_shared_eda_loader(self):
        with local_flores():
            spec = DatasetSpec("flores-ne", FACEBOOK_FLORES_DATASET_ID, config_name="eng_Latn-npi_Deva",
                               split="dev", text_columns=("sentence_eng_Latn", "sentence_npi_Deva"))
            config = AnalysisConfig(sample_size=3)
            profile = analyze_records(spec, huggingface_records(spec, config), config)
            self.assertEqual(profile.summary.rows_seen, 3)
            self.assertEqual(profile.summary.usable_rows, 3)

    def test_explicit_old_revision_is_not_silently_replaced(self):
        with local_flores() as (_, info, _, metadata, loader):
            info.siblings = [SimpleNamespace(rfilename="flores.py", size=100)]
            with self.assertRaisesRegex(ValueError, "No Parquet export.*script-only revision"):
                load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, "npi_Deva", split="dev", revision="old-commit")
            self.assertEqual(metadata.call_args.kwargs["revision"], "old-commit")
            loader.assert_not_called()

    def test_unknown_language_does_not_load_another_configuration(self):
        with local_flores() as (_, _, _, _, loader):
            with self.assertRaisesRegex(ValueError, "No Parquet export"):
                load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, "zzz_Deva", split="dev")
            loader.assert_not_called()

    def test_gated_error_during_native_load_has_access_instructions(self):
        with local_flores() as (_, _, _, _, loader):
            loader.side_effect = denied(403)
            with self.assertRaisesRegex(ValueError, "requires Hugging Face dataset access"):
                load_huggingface_stream(FACEBOOK_FLORES_DATASET_ID, "npi_Deva", split="dev")

    def test_source_remains_an_evaluation_benchmark(self):
        spec = huggingface_source_spec(FACEBOOK_FLORES_DATASET_ID, schema="pretraining", config="npi_Deva", split="dev")
        self.assertEqual(spec.primary_purpose, "Evaluation / benchmark")

    def test_streamlit_displays_access_requirement_and_loads_authorized_source(self):
        from streamlit.testing.v1 import AppTest

        with local_flores():
            app = AppTest.from_file(Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py").run(timeout=30)
            next(item for item in app.selectbox if item.label == "Data source").set_value("Hugging Face").run(timeout=30)
            next(item for item in app.text_input if item.label == "Hugging Face dataset ID").set_value(FACEBOOK_FLORES_DATASET_ID)
            next(item for item in app.button if item.label == "Load Hugging Face source").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(next(item.value for item in app.metric if item.label == "Rows"), "3")
            self.assertTrue(any("Accept access conditions" in item.value for item in app.caption))
            self.assertTrue(any("Evaluation / benchmark" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
