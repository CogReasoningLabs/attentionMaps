"""Terminal logging for long-running interactive pipeline work."""

from __future__ import annotations

import logging
import sys


def pipeline_logger(component: str) -> logging.Logger:
    """Return an INFO logger that remains visible beside Streamlit startup logs."""

    logger = logging.getLogger(f"attention_maps.pipeline.{component}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        getattr(handler, "_attention_maps_pipeline_handler", False)
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler._attention_maps_pipeline_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger
