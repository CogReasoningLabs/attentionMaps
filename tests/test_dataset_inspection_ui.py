"""Exercise dependent source selectors and language status in the actual app."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class DatasetInspectionUITests(unittest.TestCase):
    def test_config_split_and_shard_controls_follow_the_selection(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            first, second = Path(directory) / "first.parquet", Path(directory) / "second.parquet"
            pq.write_table(pa.table({"text": ["first shard"]}), first)
            pq.write_table(pa.table({"text": ["नेपाल सुन्दर छ।", "नेपाली भाषा"]}), second)
            catalog = {
                "dataset_id": "fixture/data", "revision": "fixture-revision",
                "configs": ["default", "ne"], "languages": ["ne"], "file_sizes": {},
            }

            def configuration(_catalog, config, **_kwargs):
                return {
                    "dataset_id": "fixture/data", "revision": "fixture-revision",
                    "config": config, "languages": ["ne"],
                    "schema": [{"column": "text", "type": "string", "nullable": "False"}],
                    "splits": {
                        "train": {"rows": 3, "memory_bytes": 150, "shards": [
                            {"path": str(path), "name": path.name, "bytes": path.stat().st_size}
                            for path in (first, second)
                        ]},
                        "test": {"rows": 2, "memory_bytes": 100, "shards": [
                            {"path": str(second), "name": second.name, "bytes": second.stat().st_size},
                        ]},
                    },
                }

            calls = []

            def sample(inventory, count, seed, columns):
                calls.append(tuple(inventory["dataset_shards"]))
                rows = []
                for path in inventory["dataset_shards"]:
                    rows.extend(pq.read_table(path, columns=list(columns)).to_pylist())
                return rows[:count]

            for name in ("discover_datasets", "himalaya_ai_dataset_specs", "aya_nepali_dataset_specs",
                         "iriis_nepali_text_corpus_specs", "kaggle_dataset_specs"):
                stack.enter_context(patch(f"attention_maps.explorer.{name}", return_value=[]))
            stack.enter_context(patch("attention_maps.explorer.huggingface.discover_huggingface_dataset", return_value=catalog))
            stack.enter_context(patch("attention_maps.explorer.huggingface.inspect_huggingface_configuration", side_effect=configuration))
            stack.enter_context(patch("attention_maps.explorer.sample_dataset_rows", side_effect=sample))
            # Model inference is independent of dataset selection and may scan model caches.
            stack.enter_context(patch("apps.explorer_tabs.render_inference_hub"))
            st.cache_data.clear()
            app_path = Path(__file__).resolve().parents[1] / "apps/dataset_explorer.py"
            app = AppTest.from_file(str(app_path), default_timeout=20).run()

            def widget(kind, label):
                return next(item for item in getattr(app, kind) if item.label == label)

            def metric(label):
                return widget("metric", label).value

            widget("selectbox", "Data source").select("Hugging Face").run()
            app.text_input(key="source-hf-id").input("fixture/data")
            widget("button", "Load Hugging Face source").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(widget("selectbox", "Dataset configuration").options, ["default", "ne"])
            self.assertEqual(metric("Rows"), "3")
            widget("radio", "Dataset shards").set_value("Choose shards").run()
            self.assertFalse(app.metric)  # No implicit fallback to shard zero.
            widget("multiselect", "Shards to use").set_value([str(second)]).run()
            self.assertFalse(app.exception)
            self.assertEqual(metric("Rows"), "2")
            self.assertEqual(metric("Remote shards"), "1")
            self.assertEqual(calls[-1], (str(second),))
            self.assertFalse(any(item.label == "Inspect language and script" for item in app.button))
            widget("selectbox", "Dataset split").select("test").run()
            self.assertFalse(app.exception)
            self.assertEqual(metric("Rows"), "2")
            widget("selectbox", "Dataset configuration").select("ne").run()
            self.assertFalse(app.exception)
            self.assertEqual(widget("selectbox", "Dataset split").value, "train")
            self.assertEqual(widget("radio", "Dataset shards").value, "All shards")
            self.assertEqual(metric("Rows"), "3")
            st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
