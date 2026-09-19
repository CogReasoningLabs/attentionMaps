from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apps.explorer_tabs.manifest import render_manifest_tab
from attention_maps.explorer import DatasetSpec


class ExplorerManifestTests(unittest.TestCase):
    def test_staged_kaggle_xlsx_is_treated_as_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cached_file = Path(temporary_directory) / "records.xlsx"
            cached_file.touch()
            spec = DatasetSpec(
                "dynamic:kaggle:records",
                "Kaggle records",
                "dataset",
                (cached_file,),
                format="xlsx",
                source_uri="kaggle://datasets/owner/dataset/records.xlsx",
            )
            streamlit = Mock()

            with patch("apps.explorer_tabs.manifest.find_manifest") as find:
                render_manifest_tab(
                    st=streamlit,
                    spec=spec,
                    data_root=Path(temporary_directory),
                )

        find.assert_not_called()
        streamlit.info.assert_called_once_with(
            "Remote dataset metadata is shown above in this tab."
        )


if __name__ == "__main__":
    unittest.main()
