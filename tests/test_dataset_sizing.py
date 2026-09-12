import unittest

from attention_maps.explorer import bucket_dataset_size, dataset_size_bucket


class DatasetSizeBucketTests(unittest.TestCase):
    def test_classifies_binary_size_boundaries(self):
        mib = 1024**2
        gib = 1024**3

        self.assertEqual(bucket_dataset_size(None), "Unknown")
        self.assertEqual(bucket_dataset_size(100 * mib - 1), "Tiny (<100 MiB)")
        self.assertEqual(bucket_dataset_size(100 * mib), "Small (100 MiB–1 GiB)")
        self.assertEqual(bucket_dataset_size(gib), "Medium (1–10 GiB)")
        self.assertEqual(bucket_dataset_size(10 * gib), "Large (10–100 GiB)")
        self.assertEqual(bucket_dataset_size(100 * gib), "Very large (≥100 GiB)")

    def test_prefers_decoded_size_for_remote_datasets(self):
        bucket = dataset_size_bucket(
            {"bytes": 20, "memory_bytes": 2 * 1024**3, "bytes_estimated": False}
        )

        self.assertEqual(bucket.label, "Medium (1–10 GiB)")
        self.assertEqual(bucket.size_bytes, 2 * 1024**3)
        self.assertEqual(bucket.size_basis, "decoded dataset size")


if __name__ == "__main__":
    unittest.main()
