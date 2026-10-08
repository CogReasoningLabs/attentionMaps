"""Render word clouds with verified glyph coverage for every displayed word."""

from __future__ import annotations

import math
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable


@lru_cache(maxsize=1)
def _font_paths() -> tuple[Path, ...]:
    from matplotlib import font_manager
    from attention_maps.explorer.catalog import (
        DEVANAGARI_FONT_CANDIDATES, LATIN_FONT_CANDIDATES,
    )

    return tuple(dict.fromkeys([
        *DEVANAGARI_FONT_CANDIDATES,
        *LATIN_FONT_CANDIDATES,
        *(Path(path) for path in sorted(font_manager.findSystemFonts())),
    ]))


@lru_cache(maxsize=1024)
def _font_characters(path: Path) -> frozenset[int]:
    from matplotlib import ft2font

    try:
        return frozenset(
            code for code, glyph in ft2font.FT2Font(str(path)).get_charmap().items()
            if glyph
        )
    except (OSError, RuntimeError, ValueError):
        return frozenset()


def group_words_by_font(
    frequencies: dict[str, int], *, font_paths: Iterable[Path] | None = None,
) -> tuple[list[tuple[Path, dict[str, int]]], dict[str, int]]:
    """Prefer one font; otherwise partition words into fully supported groups.

    Pillow/WordCloud use a single font per cloud, without font fallback. A
    Devanagari font can lack even Latin letters, so check actual codepoints.
    """

    if not frequencies:
        return [], {}
    required = {
        word: {ord(char) for char in word
               if not char.isspace() and unicodedata.category(char) != "Cf"}
        for word in frequencies
    }
    coverage: list[tuple[Path, set[str]]] = []
    for path in _font_paths() if font_paths is None else font_paths:
        characters = _font_characters(path)
        supported = {word for word, codes in required.items() if codes <= characters}
        if len(supported) == len(frequencies):
            return [(path, dict(frequencies))], {}
        if supported:
            coverage.append((path, supported))
    remaining = dict(frequencies)
    groups = []
    while remaining and coverage:
        path, supported = max(coverage, key=lambda item: len(item[1] & remaining.keys()))
        words = {word: count for word, count in remaining.items() if word in supported}
        if not words:
            break
        groups.append((path, words))
        remaining = {word: count for word, count in remaining.items() if word not in words}
    return groups, remaining


def render_wordcloud(
    groups: list[tuple[Path, dict[str, int]]], *, seed: int, missing_words: int = 0,
):
    """Compose font-specific clouds without ever drawing unsupported glyphs."""

    from PIL import Image, ImageDraw
    from attention_maps.explorer.text import create_wordcloud

    columns = 2 if len(groups) > 3 else 1
    width, height = 1400 // columns, 700 if len(groups) == 1 else 400
    rows = math.ceil(len(groups) / columns)
    notes = []
    if len(groups) > 1:
        notes.append("Words grouped by font coverage; word sizes are relative within each panel.")
    if missing_words:
        notes.append(f"{missing_words} words lack an installed font; their counts remain in top_tokens.csv.")
    canvas = Image.new("RGB", (1400, rows * height + 30 * len(notes)), "white")
    for index, (font, frequencies) in enumerate(groups):
        cloud = create_wordcloud(
            frequencies, font_path=font, max_words=len(frequencies),
            width=width, height=height, seed=seed, prefer_horizontal=1.0,
        )
        canvas.paste(cloud.to_image(), ((index % columns) * width, (index // columns) * height))
    draw = ImageDraw.Draw(canvas)
    for index, note in enumerate(notes):
        draw.text((15, rows * height + index * 30 + 8), note, fill="black")
    return canvas
