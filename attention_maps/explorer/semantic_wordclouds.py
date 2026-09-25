"""Script-owned cluster term counts and word clouds, including Devanagari marks."""

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import unicodedata

import numpy as np


def configured_stopwords():
    from .catalog import ENGLISH_WORDCLOUD_STOPWORDS, configured_nepali_stopwords
    return frozenset(unicodedata.normalize("NFC", term).casefold()
                     for term in (*ENGLISH_WORDCLOUD_STOPWORDS, *configured_nepali_stopwords()))


def words(text, stopwords=None):
    """Keep letters and attached combining marks; digits/punctuation split tokens."""
    stopwords = configured_stopwords() if stopwords is None else stopwords
    token = []
    for character in unicodedata.normalize("NFC", text).casefold():
        category = unicodedata.category(character)[0]
        if category == "L" or (token and (category == "M" or character in "\u200c\u200d")):
            token.append(character)
        elif token:
            value = "".join(token)
            if value not in stopwords:
                yield value
            token = []
    if token and "".join(token) not in stopwords:
        yield "".join(token)


def content_text(instance, schema, fallback):
    if not instance or not schema:
        return fallback

    def content(value):
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(content(item) for item in value)
        if isinstance(value, dict):
            return content(value.get("content", ""))
        return ""

    fields = {"pretraining": ("text",), "task_specific_supervised": ("text",),
              "instruction_finetuning": ("messages",),
              "preference_tuning": ("prompt", "chosen", "rejected")}[schema]
    return "\n".join(content(instance.get(field)) for field in fields)


def find_font(requested=None):
    if requested:
        path = Path(requested).expanduser()
        if not path.is_file():
            raise ValueError(f"Word-cloud font does not exist: {path}")
        return str(path)
    from .text import find_devanagari_font
    path = find_devanagari_font()
    return str(path) if path else None



def save_wordclouds(records_root, output, labels, *, definition=None, seed=42, font_path=None):
    """Scan every saved instance once; aggregate vocabulary on disk, not in RAM."""
    from .semantic_artifacts import open_records
    from wordcloud import WordCloud

    output = Path(output)
    target = output / "wordclouds"
    target.mkdir()
    font = find_font(font_path)
    stopwords = configured_stopwords()
    database = target / "counts.sqlite"
    counts = sqlite3.connect(database)
    sizes = np.bincount(labels)
    summary = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "scope": "all embedded records in each cluster", "seed": seed,
               "tokenization": "NFC, lowercase, Unicode letters and combining marks; punctuation and numbers excluded; no stemming",
               "stopwords": sorted(stopwords), "font": font, "max_displayed_words": 80,
               "content": "Canonical content fields; metadata, supervised labels/tasks and role markers excluded. Legacy runs use saved embedding text.",
               "clusters": {}}
    try:
        counts.execute("CREATE TABLE counts (cluster INTEGER, word TEXT, count INTEGER, PRIMARY KEY(cluster, word)) WITHOUT ROWID")
        with open_records(records_root) as records:
            has_instance = "instance_json" in {row[1] for row in records.execute("PRAGMA table_info(records)")}
            query = "SELECT embedding_id, text" + (", instance_json" if has_instance else "") + " FROM records ORDER BY embedding_id"
            seen = 0
            for row in records.execute(query):
                if row[0] != seen or seen >= len(labels):
                    raise ValueError("Saved records do not match cluster labels")
                instance = json.loads(row[2]) if has_instance else None
                frequencies = Counter(words(content_text(instance, (definition or {}).get("schema"), row[1]), stopwords))
                counts.executemany("INSERT INTO counts VALUES (?, ?, ?) ON CONFLICT(cluster, word) DO UPDATE SET count=count+excluded.count",
                                   ((int(labels[seen]), word, count) for word, count in frequencies.items()))
                seen += 1
                if seen % 1000 == 0:
                    counts.commit()
            if seen != len(labels):
                raise ValueError("Saved records do not match cluster labels")
        counts.commit()
        for cluster, size in enumerate(sizes):
            top = counts.execute("SELECT word, count FROM counts WHERE cluster=? ORDER BY count DESC, word LIMIT 200", (cluster,)).fetchall()
            total = counts.execute("SELECT COALESCE(SUM(count),0) FROM counts WHERE cluster=?", (cluster,)).fetchone()[0]
            item = {"records": int(size), "counted_tokens": total, "top_words": top, "image": None}
            if top:
                # Do not silently render Nepali as missing-glyph squares on machines without a suitable font.
                has_devanagari = any("DEVANAGARI" in unicodedata.name(ch, "") for word, _ in top for ch in word)
                if has_devanagari and not font:
                    item["note"] = "Install a Noto Devanagari font or pass --wordcloud-font; saved word counts are available."
                else:
                    filename = f"cluster-{cluster}.png"
                    cloud = WordCloud(width=1200, height=600, background_color="white", font_path=font,
                                      max_words=80, prefer_horizontal=1, random_state=seed, collocations=False)
                    cloud.generate_from_frequencies(dict(top)).to_file(str(target / filename))
                    item["image"] = filename
            summary["clusters"][str(cluster)] = item
        (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        counts.close()
        database.unlink(missing_ok=True)
    return {"summary": "wordclouds/summary.json", "scope": summary["scope"], "max_displayed_words": 80}
