#!/usr/bin/env python3
"""Compatibility launcher for ``python -m attention_maps.tokenization``."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.tokenization.cli import main  # noqa: E402


if __name__ == "__main__":
    main()

