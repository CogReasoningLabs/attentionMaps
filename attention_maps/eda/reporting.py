"""Machine-readable artifacts and compact publication-oriented figures."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from attention_maps.eda.contracts import DatasetProfile, SurveyRun


def write_survey_report(
    run: SurveyRun,
    output_dir: Path,
    *,
    plots: bool = True,
    plot_names: Iterable[str] | None = None,
) -> list[Path]:
    """Write stable JSON/CSV products and optional derived-only figures."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    selected_plots = None if plot_names is None else frozenset(plot_names)
    plotters = (
        ("profile", plot_dataset_profile),
        ("document_size", plot_document_size_distribution),
        ("text_structure", plot_segment_length_kde),
        ("ngrams", plot_top_ngrams),
        ("cooccurrence", plot_cooccurrence_network),
        ("deduplication", plot_deduplication_stages),
        ("wordcloud", plot_wordcloud),
    )
    known_plots = {name for name, _ in plotters} | {"cross_dataset"}
    if selected_plots is not None and not selected_plots <= known_plots:
        unknown = ", ".join(sorted(selected_plots - known_plots))
        raise ValueError(f"Unknown EDA plot name(s): {unknown}")
    summary_rows = [profile.summary.as_dict() for profile in run.profiles]
    for profile in run.profiles:
        dataset_dir = output_dir / profile.summary.dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        written.extend(_write_profile(profile, dataset_dir))
        if plots:
            written.extend(
                plotter(profile, dataset_dir)
                for name, plotter in plotters
                if selected_plots is None or name in selected_plots
            )

    written.append(_write_json(output_dir / "survey_summary.json", summary_rows))
    written.append(_write_csv(output_dir / "survey_summary.csv", summary_rows))
    manifest = {
        "survey": run.plan.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": asdict(run.plan.analysis),
        "datasets_requested": [asdict(dataset) for dataset in run.plan.datasets],
        "datasets_completed": [profile.summary.dataset_key for profile in run.profiles],
        "sampling_results": [
            {
                "dataset_key": profile.summary.dataset_key,
                "population_rows": profile.summary.population_rows,
                "rows_seen": profile.summary.rows_seen,
                "population_coverage_pct": profile.summary.population_coverage_pct,
                "seed": profile.summary.seed,
            }
            for profile in run.profiles
        ],
        "failures": dict(run.failures),
        "raw_text_persisted": False,
        "runtime": _runtime_metadata(),
    }
    written.append(_write_json(output_dir / "survey_manifest.json", manifest))
    if (
        plots
        and run.profiles
        and (selected_plots is None or "cross_dataset" in selected_plots)
    ):
        written.append(plot_cross_dataset(run.profiles, output_dir))
    return written


def _write_profile(profile: DatasetProfile, output_dir: Path) -> list[Path]:
    written = [
        _write_json(output_dir / "eda_summary.json", profile.summary.as_dict()),
        _write_csv(
            output_dir / "metric_sample.csv",
            [asdict(sample) for sample in profile.samples],
        ),
        _write_csv(
            output_dir / "top_tokens.csv",
            [{"token": token, "count": count} for token, count in profile.top_tokens],
        ),
        _write_csv(
            output_dir / "top_sources.csv",
            [
                {"source": source, "count": count}
                for source, count in profile.top_sources
            ],
        ),
        _write_csv(
            output_dir / "sentence_length_sample.csv",
            [{"sentence_tokens": length} for length in profile.sentence_length_samples],
        ),
        _write_csv(
            output_dir / "line_length_sample.csv",
            [{"line_characters": length} for length in profile.line_length_samples],
        ),
        _write_csv(
            output_dir / "cooccurrence_edges.csv",
            [
                {"source": source, "target": target, "document_count": count}
                for source, target, count in profile.cooccurrence_edges
            ],
        ),
        _write_csv(
            output_dir / "deduplication_stages.csv",
            [
                {
                    "stage": "1_exact_document",
                    "method": profile.summary.dedup_hash_algorithm,
                    "flagged": profile.summary.exact_duplicate_rows,
                    "normalization": profile.summary.dedup_normalization,
                },
                {
                    "stage": "2_near_document",
                    "method": "MinHash-LSH",
                    "flagged": profile.summary.near_duplicate_rows,
                    "normalization": profile.summary.dedup_normalization,
                },
                {
                    "stage": "3_repeated_paragraph",
                    "method": "normalized paragraph SHA-256 frequency",
                    "flagged": profile.summary.boilerplate_unique_paragraphs,
                    "normalization": profile.summary.dedup_normalization,
                },
            ],
        ),
    ]
    for order, values in sorted(profile.top_ngrams.items()):
        written.append(
            _write_csv(
                output_dir / f"top_{order}grams.csv",
                [{"ngram": ngram, "count": count} for ngram, count in values],
            )
        )
    return written


def plot_dataset_profile(profile: DatasetProfile, output_dir: Path) -> Path:
    """Create one consistent four-panel profile without retaining source text."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    samples = profile.samples
    axes[0, 0].hist([sample.tokens for sample in samples], bins=40, color="#2a6f97")
    axes[0, 0].set(title="Document length", xlabel="Tokens", ylabel="Sampled records")
    axes[0, 1].hist(
        [sample.devanagari_ratio for sample in samples],
        bins=30,
        range=(0, 1),
        color="#4c956c",
    )
    axes[0, 1].set(title="Devanagari composition", xlabel="Letter/mark ratio")
    tokens = list(reversed(profile.top_tokens[:20]))
    axes[1, 0].barh(
        [item[0] for item in tokens], [item[1] for item in tokens], color="#d99a2b"
    )
    axes[1, 0].set(title="Frequent tokens", xlabel="Occurrences")
    sources = list(reversed(profile.top_sources[:15]))
    if sources:
        axes[1, 1].barh(
            [item[0] for item in sources],
            [item[1] for item in sources],
            color="#9c6fb6",
        )
    else:
        axes[1, 1].text(0.5, 0.5, "No source metadata", ha="center", va="center")
    axes[1, 1].set(title="Source provenance", xlabel="Records")
    figure.suptitle(profile.summary.dataset_key, fontsize=15, fontweight="bold")
    figure.tight_layout()
    path = output_dir / "dataset_profile.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_document_size_distribution(profile: DatasetProfile, output_dir: Path) -> Path:
    """Plot log-scaled character and word-count histograms."""

    import matplotlib.pyplot as plt
    import numpy as np

    _configure_plotting(plt)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    series = (
        ([sample.characters for sample in profile.samples], "Characters"),
        ([sample.tokens for sample in profile.samples], "Words / regex tokens"),
    )
    for axis, (values, label) in zip(axes, series):
        positive = np.asarray([value for value in values if value > 0], dtype=float)
        if positive.size:
            lower = max(1.0, float(positive.min()))
            upper = max(lower * 1.01, float(positive.max()))
            bins = np.geomspace(lower, upper, num=min(50, max(10, positive.size + 1)))
            axis.hist(positive, bins=bins, color="#2a6f97", alpha=0.88)
            axis.set_xscale("log")
        else:
            axis.text(0.5, 0.5, "No usable documents", ha="center", va="center")
        axis.set(title=f"Document {label.lower()}", xlabel=f"{label} (log scale)")
        axis.set_ylabel("Sampled documents")
    figure.suptitle(
        f"{profile.summary.dataset_key}: document size distributions",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    path = output_dir / "document_size_distribution.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_segment_length_kde(profile: DatasetProfile, output_dir: Path) -> Path:
    """Plot bounded Gaussian KDE estimates for sentence and line lengths."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    series = (
        (profile.sentence_length_samples, "Sentence length", "Regex tokens"),
        (profile.line_length_samples, "Line length", "Normalized characters"),
    )
    for axis, (values, title, xlabel) in zip(axes, series):
        grid, density = _gaussian_kde(values)
        if grid:
            axis.plot(grid, density, color="#4c956c", linewidth=2.2)
            axis.fill_between(grid, density, color="#4c956c", alpha=0.22)
        else:
            axis.text(0.5, 0.5, "No segments observed", ha="center", va="center")
        axis.set(title=title, xlabel=xlabel, ylabel="Estimated density")
    figure.suptitle(
        f"{profile.summary.dataset_key}: text structure",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    path = output_dir / "segment_length_kde.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_top_ngrams(profile: DatasetProfile, output_dir: Path) -> Path:
    """Plot top phrases for every configured n-gram order."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    orders = sorted(profile.top_ngrams)
    figure, axes = plt.subplots(
        1, len(orders), figsize=(7 * len(orders), 8), squeeze=False
    )
    for axis, order in zip(axes[0], orders):
        values = list(reversed(profile.top_ngrams[order][:20]))
        if values:
            axis.barh(
                [ngram for ngram, _ in values],
                [count for _, count in values],
                color="#d99a2b",
            )
        else:
            axis.text(0.5, 0.5, "No phrases observed", ha="center", va="center")
        axis.set(title=f"Top {order}-grams", xlabel="Occurrences")
    figure.suptitle(
        f"{profile.summary.dataset_key}: recurring phrases",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    path = output_dir / "top_ngrams.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_cooccurrence_network(profile: DatasetProfile, output_dir: Path) -> Path:
    """Draw a dependency-free bipartite network of seed terms and neighbors."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    edges = profile.cooccurrence_edges[:30]
    figure, axis = plt.subplots(figsize=(14, max(7, len(edges) * 0.24)))
    if edges:
        seeds = list(dict.fromkeys(source for source, _, _ in edges))
        neighbors = list(dict.fromkeys(target for _, target, _ in edges))
        seed_y = _even_positions(seeds)
        neighbor_y = _even_positions(neighbors)
        maximum = max(count for _, _, count in edges)
        for source, target, count in edges:
            axis.plot(
                (0, 1),
                (seed_y[source], neighbor_y[target]),
                color="#829399",
                alpha=0.25 + 0.65 * count / maximum,
                linewidth=0.5 + 4 * count / maximum,
                zorder=1,
            )
        axis.scatter(
            [0] * len(seeds),
            [seed_y[value] for value in seeds],
            s=180,
            color="#c44536",
            zorder=2,
        )
        axis.scatter(
            [1] * len(neighbors),
            [neighbor_y[value] for value in neighbors],
            s=120,
            color="#2a6f97",
            zorder=2,
        )
        for value in seeds:
            axis.text(-0.03, seed_y[value], value, ha="right", va="center")
        for value in neighbors:
            axis.text(1.03, neighbor_y[value], value, ha="left", va="center")
        axis.set_xlim(-0.35, 1.35)
        axis.set_ylim(-0.05, 1.05)
    else:
        axis.text(
            0.5,
            0.5,
            "No configured seed-term co-occurrences observed",
            ha="center",
            va="center",
        )
    axis.set_axis_off()
    axis.set_title(
        f"{profile.summary.dataset_key}: seed-term co-occurrence network\n"
        "edge weight = documents containing the local token pairing",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    path = output_dir / "term_cooccurrence_network.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_deduplication_stages(profile: DatasetProfile, output_dir: Path) -> Path:
    """Plot sequential document flags and paragraph-affected documents."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    labels = ("Exact SHA-256", "Near MinHash-LSH", "Boilerplate affected")
    values = (
        profile.summary.exact_duplicate_rows,
        profile.summary.near_duplicate_rows,
        profile.summary.boilerplate_affected_rows,
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color=("#2a6f97", "#d99a2b", "#9c6fb6"))
    axis.bar_label(bars, fmt="%d")
    axis.set(title="Multi-stage duplicate screening", ylabel="Sampled documents")
    axis.text(
        0.01,
        -0.2,
        (
            f"{profile.summary.dedup_normalization} → SHA-256; "
            f"MinHash threshold={profile.summary.near_duplicate_threshold:.2f}; "
            "boilerplate is flagged, not removed"
        ),
        transform=axis.transAxes,
    )
    figure.tight_layout()
    path = output_dir / "deduplication_stages.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_wordcloud(profile: DatasetProfile, output_dir: Path) -> Path:
    """Render a fixed, stopword-filtered cloud from derived token counts."""

    from attention_maps.explorer.catalog import (
        ENGLISH_WORDCLOUD_STOPWORDS,
        configured_nepali_stopwords,
    )
    from attention_maps.explorer.text import (
        create_wordcloud,
        find_devanagari_font,
        find_latin_font,
    )

    frequencies = _eligible_wordcloud_frequencies(
        profile.top_tokens,
        (
            *configured_nepali_stopwords(),
            *ENGLISH_WORDCLOUD_STOPWORDS,
        ),
    )
    if not frequencies:
        return _plot_wordcloud_unavailable(
            output_dir,
            "No eligible words remain after configs/eda/stopwords.txt filtering.",
        )
    contains_devanagari = any(
        "\u0900" <= character <= "\u097f"
        for token in frequencies
        for character in token
    )
    font_path = find_devanagari_font() if contains_devanagari else find_latin_font()
    if contains_devanagari and font_path is None:
        return _plot_wordcloud_unavailable(
            output_dir,
            "A Devanagari-capable font is required to render this WordCloud.",
        )
    cloud = create_wordcloud(
        frequencies,
        font_path=font_path,
        max_words=100,
        width=1_400,
        height=700,
        background_color="white",
        colormap="viridis",
        seed=profile.summary.seed,
    )
    path = output_dir / "wordcloud.png"
    cloud.to_file(str(path))
    return path


def _eligible_wordcloud_frequencies(
    top_tokens: tuple[tuple[str, int], ...],
    stopwords: tuple[str, ...],
) -> dict[str, int]:
    """Filter WordCloud candidates without ever restoring excluded words."""

    import unicodedata

    from attention_maps.eda.text import strip_nepali_suffix

    excluded: set[str] = set()
    for token in stopwords:
        normalized = unicodedata.normalize("NFC", token).casefold()
        excluded.add(normalized)
        excluded.add(strip_nepali_suffix(normalized, minimum_stem_length=2))

    frequencies: dict[str, int] = {}
    for token, count in top_tokens:
        normalized = unicodedata.normalize("NFC", token).casefold()
        if (
            len(normalized) >= 2
            and normalized not in excluded
            and strip_nepali_suffix(normalized, minimum_stem_length=2) not in excluded
            and any(character.isalpha() for character in normalized)
        ):
            frequencies[normalized] = count
    return frequencies


def _plot_wordcloud_unavailable(output_dir: Path, message: str) -> Path:
    """Write a non-fatal explanatory figure when a cloud cannot be rendered."""

    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=14)
    axis.set_axis_off()
    axis.set_title("WordCloud unavailable", fontsize=16, fontweight="bold")
    path = output_dir / "wordcloud.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_cross_dataset(profiles: Iterable[DatasetProfile], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    _configure_plotting(plt)
    profiles = tuple(profiles)
    labels = [profile.summary.dataset_key for profile in profiles]
    positions = list(range(len(profiles)))
    figure, axes = plt.subplots(1, 3, figsize=(20, max(6, len(profiles) * 0.45)))
    axes[0].barh(
        positions,
        [profile.summary.average_tokens for profile in profiles],
        color="#2a6f97",
    )
    axes[0].set(
        title="Average document length",
        xlabel="Tokens",
        yticks=positions,
        yticklabels=labels,
    )
    axes[1].barh(
        positions,
        [profile.summary.quality_pass_ratio_pct for profile in profiles],
        color="#4c956c",
    )
    axes[1].set(
        title="Basic quality pass", xlabel="Percent", yticks=positions, yticklabels=[]
    )
    axes[2].barh(
        positions,
        [profile.summary.duplicate_ratio_pct for profile in profiles],
        color="#c44536",
    )
    axes[2].set(
        title="Duplicate screening", xlabel="Percent", yticks=positions, yticklabels=[]
    )
    figure.suptitle("Cross-dataset corpus survey", fontsize=16, fontweight="bold")
    figure.tight_layout()
    path = output_dir / "cross_dataset_comparison.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def _configure_plotting(plt: Any) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    font_family = _mixed_script_font()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "font.family": font_family,
        }
    )


def _gaussian_kde(values: Iterable[int]) -> tuple[list[float], list[float]]:
    """Compute a small Gaussian KDE without adding scipy/seaborn dependencies."""

    import numpy as np

    data = np.asarray([value for value in values if value >= 0], dtype=float)
    if not data.size:
        return [], []
    data.sort()
    if data.size > 3_000:
        indexes = np.linspace(0, data.size - 1, 3_000, dtype=int)
        data = data[indexes]
    upper = float(np.quantile(data, 0.99))
    clipped = data[data <= upper]
    if not clipped.size:
        clipped = data
    deviation = float(clipped.std(ddof=1)) if clipped.size > 1 else 0.0
    bandwidth = 1.06 * deviation * clipped.size ** (-0.2) if deviation else 0.0
    bandwidth = max(bandwidth, max(1.0, float(clipped.mean()) * 0.05))
    grid_upper = max(float(clipped.max()) + 2 * bandwidth, bandwidth)
    grid = np.linspace(0.0, grid_upper, 256)
    differences = (grid[:, None] - clipped[None, :]) / bandwidth
    density = np.exp(-0.5 * differences**2).mean(axis=1) / (
        bandwidth * np.sqrt(2 * np.pi)
    )
    return grid.tolist(), density.tolist()


def _even_positions(values: list[str]) -> dict[str, float]:
    if len(values) <= 1:
        return {value: 0.5 for value in values}
    return {value: index / (len(values) - 1) for index, value in enumerate(values)}


def _mixed_script_font() -> str:
    """Choose one installed font containing both Latin and Devanagari glyphs."""

    from matplotlib import font_manager, ft2font

    for family in ("Nirmala UI", "FreeSerif", "Arial Unicode MS"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=family),
                fallback_to_default=False,
            )
            characters = ft2font.FT2Font(path).get_charmap()
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if ord("A") in characters and ord("क") in characters:
            return family
    return "DejaVu Sans"


def _runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("datasets", "matplotlib", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
    }


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
