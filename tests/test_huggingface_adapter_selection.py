"""Local adapters remain available in the new discovery and inspection flow."""

import unittest
from unittest.mock import patch

from attention_maps.datasets.crosssum import CROSSSUM_DATASET_ID
from attention_maps.datasets.indicgenbench import FLORES_IN_DATASET_ID
from attention_maps.datasets.nepfake import NEPFAKE_DATASET_ID
from attention_maps.explorer.huggingface import (
    discover_huggingface_dataset, inspect_huggingface_configuration,
    inspect_huggingface_selection, load_selected_huggingface_stream,
)
from attention_maps.explorer.inspection import sample_huggingface_rows
from attention_maps.explorer.inspection_runs import iter_inspection_records
from tests.test_crosssum import local_crosssum
from tests.test_indicgenbench import local_benchmark
from tests.test_nepfake import local_nepfake


class AdapterSelectionTests(unittest.TestCase):
    def selection(self, dataset, config, split, **kwargs):
        catalog = discover_huggingface_dataset(dataset, revision="release")
        configuration = inspect_huggingface_configuration(catalog, config)
        return configuration, inspect_huggingface_selection(configuration, split, **kwargs)

    def test_discovery_and_formats_do_not_download_adapter_records(self):
        cases = (
            (local_crosssum, CROSSSUM_DATASET_ID, "nepali-nepali", "train"),
            (local_benchmark, FLORES_IN_DATASET_ID, "ne", "validation"),
            (local_nepfake, NEPFAKE_DATASET_ID, "default", "train"),
        )
        for fixture, dataset, config, split in cases:
            with self.subTest(dataset=dataset), fixture(), \
                 patch("datasets.get_dataset_config_names", side_effect=AssertionError("legacy builder")), \
                 patch("attention_maps.datasets.huggingface.load_huggingface_stream") as loader:
                configuration, inventory = self.selection(dataset, config, split, metadata_only=True)
                self.assertEqual(inventory["dataset_config"], config)
                self.assertTrue(inventory["metadata_only"])
                self.assertIsNone(inventory["rows"])
                self.assertGreater(inventory["hub_file_bytes"], 0)
                self.assertTrue(configuration["requires_complete_split"])
                loader.assert_not_called()

    def test_crosssum_new_inspection_and_sampling_keep_only_selected_split(self):
        with local_crosssum() as (_, metadata, _, _):
            _, inventory = self.selection(CROSSSUM_DATASET_ID, "english-nepali", "validation")
            self.assertEqual(inventory["rows"], 2)
            self.assertEqual(inventory["hub_file_size_scope"], "selected language pair (all splits)")
            rows = list(iter_inspection_records(inventory, batch_size=1))
            sampled = sample_huggingface_rows(inventory, 2, 42, ("text",))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all("english-nepali val" in row["text"] for row in rows + sampled))
            self.assertEqual(metadata.call_args.kwargs["revision"], inventory["dataset_revision"])

    def test_flores_in_keeps_both_directions_and_rejects_partial_pair(self):
        with local_benchmark():
            configuration, inventory = self.selection(FLORES_IN_DATASET_ID, "ne", "validation")
            rows = list(iter_inspection_records(inventory))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["translation_direction"] for row in rows}, {"enxx", "xxen"})
            with self.assertRaisesRegex(ValueError, "requires all files"):
                inspect_huggingface_selection(configuration, "validation", shards=inventory["dataset_shards"][:1])

    def test_nepfake_new_inspection_excludes_stats_and_json_exports(self):
        with local_nepfake():
            _, inventory = self.selection(NEPFAKE_DATASET_ID, "default", "train")
            self.assertEqual(len(inventory["dataset_shards"]), 1)
            self.assertTrue(inventory["dataset_shards"][0].endswith("data/nepfakev2.csv"))
            self.assertEqual(len(list(iter_inspection_records(inventory))), 5)
            self.assertEqual(len(sample_huggingface_rows(inventory, 5, 42, ("example_id",))), 5)

    def test_adapter_filters_survive_inventory_sampling_and_batch_iteration(self):
        with local_benchmark():
            _, inventory = self.selection(
                FLORES_IN_DATASET_ID, "ne", "validation",
                filter_column="translation_direction", filter_value="enxx",
            )
            self.assertEqual(inventory["source_rows"], 4)
            self.assertEqual(inventory["rows"], 2)
            rows = list(iter_inspection_records(inventory))
            sampled = sample_huggingface_rows(inventory, 4, 42, ("translation_direction",))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(sampled), 2)
            self.assertTrue(all(row["translation_direction"] == "enxx" for row in rows + sampled))

    def test_selected_adapter_rejects_wrong_revision_before_loading(self):
        with local_nepfake():
            _, inventory = self.selection(NEPFAKE_DATASET_ID, "default", "train", metadata_only=True)
            with patch("attention_maps.datasets.huggingface.load_huggingface_stream") as loader:
                with self.assertRaisesRegex(ValueError, "pinned revision"):
                    load_selected_huggingface_stream(
                        NEPFAKE_DATASET_ID, "default", "train", "another-commit",
                        inventory["dataset_shards"], loader="source_adapter",
                    )
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
