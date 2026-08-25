from __future__ import annotations

import unittest

from scripts.download_nepali_pretrain_corpus import (
    DEFAULT_DATASET_ID,
    build_patterns,
    human_size,
    matches_patterns,
    parse_args,
    parse_dataset_reference,
    resolve_download_request,
)


class DatasetDownloaderTests(unittest.TestCase):
    def test_defaults_target_nepali_pretrain_corpus(self) -> None:
        args = parse_args([])
        request = resolve_download_request(args)
        self.assertEqual(request.dataset_id, DEFAULT_DATASET_ID)
        self.assertEqual(str(request.output_dir), "data/raw/nepali_pretrain_corpus")

    def test_accepts_pdf_corpus_tree_url(self) -> None:
        url = (
            "https://huggingface.co/datasets/himalaya-ai/"
            "nepali_pdf_corpus/tree/main/data"
        )
        args = parse_args([url])
        request = resolve_download_request(args)
        self.assertEqual(request.dataset_id, "himalaya-ai/nepali_pdf_corpus")
        self.assertEqual(request.revision, "main")
        self.assertEqual(request.path_prefix, "data")
        self.assertEqual(str(request.output_dir), "data/raw/nepali_pdf_corpus")

    def test_accepts_dataset_id_and_custom_output(self) -> None:
        args = parse_args(
            ["himalaya-ai/nepali_pdf_corpus", "--output-dir", "custom/data"]
        )
        request = resolve_download_request(args)
        self.assertEqual(request.dataset_id, "himalaya-ai/nepali_pdf_corpus")
        self.assertEqual(str(request.output_dir), "custom/data")

    def test_rejects_non_huggingface_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only huggingface.co"):
            parse_dataset_reference("https://example.com/datasets/org/name")

    def test_parquet_patterns_include_nested_shards(self) -> None:
        patterns = ("*.parquet", "**/*.parquet")
        self.assertTrue(matches_patterns("data/train-00001.parquet", patterns))
        self.assertTrue(matches_patterns("train.parquet", patterns))
        self.assertFalse(matches_patterns("data/train.jsonl", patterns))

    def test_path_prefix_restricts_patterns(self) -> None:
        patterns = build_patterns("data", all_files=False)
        self.assertIsNotNone(patterns)
        self.assertTrue(matches_patterns("data/train-00001.parquet", patterns or ()))
        self.assertFalse(matches_patterns("other/train-00001.parquet", patterns or ()))

    def test_human_size(self) -> None:
        self.assertEqual(human_size(1024), "1.00 KiB")
        self.assertEqual(human_size(None), "unknown")


if __name__ == "__main__":
    unittest.main()
