import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import Workbook

from attention_maps.datasets.kaggle import (
    KaggleDatasetError,
    inspect_kaggle_text,
    inspect_kaggle_workbook,
    sample_kaggle_text_rows,
    sample_kaggle_workbook_rows,
)


class KaggleWorkbookTests(unittest.TestCase):
    def _workbook(self, directory: str) -> Path:
        path = Path(directory) / "records.xlsx"
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Reviews"
        worksheet.append(["Reviews", "Emotion"])
        worksheet.append(["राम्रो चलचित्र", 1])
        worksheet.append(["नराम्रो चलचित्र", 0])
        worksheet.append(["ठीकै छ", None])
        workbook.save(path)
        workbook.close()
        return path

    def test_inspects_workbook_without_loading_a_dataframe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._workbook(directory)
            with patch(
                "attention_maps.datasets.kaggle.download_kaggle_file",
                return_value=path,
            ):
                inventory = inspect_kaggle_workbook("owner/dataset", "records.xlsx")

        self.assertEqual(inventory["rows"], 3)
        self.assertEqual(inventory["columns"], ["Reviews", "Emotion"])
        self.assertEqual(inventory["sheet_name"], "Reviews")
        self.assertEqual(inventory["schema"][1]["type"], "integer")
        self.assertEqual(inventory["schema"][1]["nullable"], "True")

    def test_reservoir_sample_is_bounded_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._workbook(directory)
            with patch(
                "attention_maps.datasets.kaggle.download_kaggle_file",
                return_value=path,
            ):
                inventory = inspect_kaggle_workbook("owner/dataset", "records.xlsx")
            first = sample_kaggle_workbook_rows(inventory, 2, 42, ("Reviews",))
            second = sample_kaggle_workbook_rows(inventory, 2, 42, ("Reviews",))

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(
            all(
                set(row) >= {"Reviews", "__viewer_row_index"}
                for row in first
            )
        )

    def test_rejects_unknown_columns(self):
        with self.assertRaises(KaggleDatasetError):
            sample_kaggle_workbook_rows(
                {"columns": [], "path": "unused.xlsx"}, 1, 0, ("missing",)
            )

    def test_inspects_and_randomly_samples_line_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ne_dedup.txt"
            path.write_text(
                "पहिलो नेपाली पङ्क्ति\n\nदोस्रो नेपाली पङ्क्ति\nतेस्रो नेपाली पङ्क्ति\n",
                encoding="utf-8",
            )
            with patch(
                "attention_maps.datasets.kaggle.download_kaggle_file",
                return_value=path,
            ):
                inventory = inspect_kaggle_text(
                    "hsebarp/oscar-corpus-nepali", "ne_dedup.txt"
                )
            first = sample_kaggle_text_rows(inventory, 2, 42, ("text",))
            second = sample_kaggle_text_rows(inventory, 2, 42, ("text",))

        self.assertEqual(inventory["rows"], 3)
        self.assertEqual(inventory["columns"], ["text"])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(record["text"] for record in first))
        self.assertTrue(
            all(
                str(record["__viewer_row_index"]).startswith("byte:")
                for record in first
            )
        )


if __name__ == "__main__":
    unittest.main()
