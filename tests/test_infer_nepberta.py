from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from attention_maps.inference.nepberta import (
    load_model,
    parse_args,
    prepare_masked_text,
)


class NepBERTaInferenceTests(unittest.TestCase):
    def test_portable_mask_placeholder(self) -> None:
        text = prepare_masked_text("नेपाल एक {mask} देश हो।", "[MASK]")
        self.assertEqual(text, "नेपाल एक [MASK] देश हो।")

    def test_requires_exactly_one_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_masked_text("नेपाल एक देश हो।", "[MASK]")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_masked_text("{mask} नेपाल {mask}", "[MASK]")

    def test_cli_defaults_do_not_import_heavy_dependencies(self) -> None:
        args = parse_args([])
        self.assertEqual(args.model_id, "NepBERTa/NepBERTa")
        self.assertEqual(args.top_k, 5)
        self.assertEqual(args.device, "cpu")
        self.assertIn("{mask}", args.text)

    def test_tensorflow_loader_disables_other_transformer_backends(self) -> None:
        args = parse_args(["--local-files-only", "--device", "cpu"])
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "dependencies are missing"):
                # Blocking imports lets this test inspect environment setup
                # without importing the installed native ML runtimes.
                with patch.dict("sys.modules", {"tensorflow": None}):
                    load_model(args)
            self.assertEqual(os.environ["USE_TORCH"], "0")
            self.assertEqual(os.environ["USE_TF"], "1")
            self.assertEqual(os.environ["USE_FLAX"], "0")
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")


if __name__ == "__main__":
    unittest.main()
