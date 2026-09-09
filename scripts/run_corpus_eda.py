"""Compatibility wrapper for ``python -m attention_maps.eda``."""

from attention_maps.eda.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
