from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_pretraining_dataset import (
    assign_split,
    include_sample,
    iter_lyrics_candidates,
    parse_args,
    validate_args,
)


class PretrainingDatasetBuilderTests(unittest.TestCase):
    def test_sampling_is_deterministic(self) -> None:
        first = include_sample("nepali_news", "article-1", 0.5, 42)
        second = include_sample("nepali_news", "article-1", 0.5, 42)
        self.assertEqual(first, second)
        self.assertTrue(include_sample("nepali_news", "article-1", 1.0, 42))
        self.assertFalse(include_sample("nepali_news", "article-1", 0.0, 42))

    def test_split_assignment_is_deterministic(self) -> None:
        first = assign_split(
            "document-1", train_ratio=0.8, validation_ratio=0.1, seed=7
        )
        second = assign_split(
            "document-1", train_ratio=0.8, validation_ratio=0.1, seed=7
        )
        self.assertEqual(first, second)
        self.assertIn(first, {"train", "validation", "test"})

    def test_split_ratios_must_sum_to_one(self) -> None:
        args = parse_args(["--train-ratio", "0.8"])
        with self.assertRaisesRegex(ValueError, "must sum to 1"):
            validate_args(args)

    def test_lyrics_are_grouped_by_song_from_artist_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artist = root / "Artist"
            artist.mkdir()
            (artist / "metadata.csv").write_text(
                "SegmentFilePath,SegmentLength,Lyrics\n"
                "output/Song/segment_2.wav,2.5,दोस्रो हरफ\n"
                "output/Song/segment_1.wav,3.0,पहिलो हरफ\n",
                encoding="utf-8",
            )
            # This duplicate per-song table must not be read.
            song_dir = artist / "output" / "Song"
            song_dir.mkdir(parents=True)
            (song_dir / "Song_segments.csv").write_text(
                "SegmentFilePath,SegmentLength,Lyrics\n"
                "segment_1.wav,3.0,पहिलो हरफ\n",
                encoding="utf-8",
            )

            candidates = list(iter_lyrics_candidates(root))
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].source_id, "Artist/Song")
            self.assertEqual(candidates[0].text, "पहिलो हरफ\nदोस्रो हरफ")
            self.assertEqual(candidates[0].metadata["segment_count"], 2)


if __name__ == "__main__":
    unittest.main()
