import ast
import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "Llama2_7B_Nepali_MultiDataset_QLoRA.ipynb"
)


class Llama2MultiDatasetNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_notebook_code_cells_are_valid_python(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith("%"):
                continue
            try:
                ast.parse(source)
            except SyntaxError as error:
                self.fail(f"Notebook code cell {index} is invalid: {error}")

    def test_exposes_both_dataset_choices_and_flexible_mapping(self):
        self.assertIn("saillab/alpaca-nepali-cleaned", self.source)
        self.assertIn("data_generation_pipeline", self.source)
        self.assertIn("lima_translations.json", self.source)
        self.assertIn("FIELD_MAP", self.source)
        self.assertIn("normalize_sft_example", self.source)
        self.assertIn("apply_pipeline_cleaning", self.source)

    def test_uses_smallest_llama2_chat_checkpoint_and_qlora(self):
        self.assertIn("meta-llama/Llama-2-7b-chat-hf", self.source)
        self.assertIn("load_in_4bit=True", self.source)
        self.assertIn("prepare_model_for_kbit_training", self.source)
        self.assertIn("completion_only_loss=True", self.source)


if __name__ == "__main__":
    unittest.main()
