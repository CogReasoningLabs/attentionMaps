#!/usr/bin/env python3
"""Compatibility launcher for packaged Himalaya Gemma inference."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_maps.inference.himalaya_gemma import *  # noqa: E402,F403


if __name__ == "__main__":
    main()
