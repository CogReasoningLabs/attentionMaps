"""Streaming analysis orchestration with bounded-memory accumulators."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping

from attention_maps.eda.contracts import (
    AnalysisConfig,
    DatasetProfile,
    DatasetSpec,
    DatasetSummary,
    ProfileSample,
    SurveyPlan,
    SurveyRun,
)
from attention_maps.eda.deduplication import (
    DeduplicationConfig,
    MultiStageDeduplicator,
)
from attention_maps.eda.script import (
    ScriptEvidence,
    classify_script_evidence,
    script_evidence,
)
from attention_maps.eda.text import (
    devanagari_ratio,
    extract_source,
    extract_text,
    line_character_lengths,
    normalize_text,
    sentence_token_lengths,
    strip_nepali_suffix,
    token_ngrams,
    tokenize,
)


RecordSource = Callable[[DatasetSpec, AnalysisConfig], Iterable[Mapping[str, Any]]]
ProgressCallback = Callable[[str, str, int], None]


class _Reservoir:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.random = random.Random(seed)
        self.seen = 0
        self.values: list[Any] = []

    def add(self, value: Any) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        position = self.random.randrange(self.seen)
        if position < self.capacity:
            self.values[position] = value


class _BoundedCounter:
    """Counter that periodically prunes low-frequency vocabulary entries."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.counts: Counter[Any] = Counter()
        self.truncated = False

    def update(self, values: Iterable[Any]) -> None:
        self.counts.update(values)
        if len(self.counts) > self.capacity * 2:
            self.counts = Counter(dict(self.counts.most_common(self.capacity)))
            self.truncated = True

    def finish(self) -> None:
        if len(self.counts) > self.capacity:
            self.counts = Counter(dict(self.counts.most_common(self.capacity)))
            self.truncated = True


def analyze_records(
    spec: DatasetSpec,
    records: Iterable[Mapping[str, Any]],
    config: AnalysisConfig,
    *,
    progress: ProgressCallback | None = None,
) -> DatasetProfile:
    """Analyze at most the configured number of records without retaining text."""

    sample_limit = spec.sample_size or config.sample_size
    reservoir = _Reservoir(config.reservoir_size, config.seed)
    sentence_reservoir = _Reservoir(config.reservoir_size, config.seed + 1)
    line_reservoir = _Reservoir(config.reservoir_size, config.seed + 2)
    vocabulary = _BoundedCounter(config.max_vocabulary)
    ngram_counters = {
        order: _BoundedCounter(config.max_ngrams) for order in config.ngram_orders
    }
    cooccurrences = _BoundedCounter(config.max_cooccurrence_edges)
    cooccurrence_terms = {
        strip_nepali_suffix(term_tokens[0])
        for term in config.cooccurrence_terms
        if len(term_tokens := tokenize(term)) == 1
    }
    cooccurrence_stopwords = {
        strip_nepali_suffix(token)
        for term in config.cooccurrence_stopwords
        for token in tokenize(term)
    }
    sources: Counter[str] = Counter()
    aggregate_script_evidence = ScriptEvidence()
    deduplication = MultiStageDeduplicator(
        DeduplicationConfig(
            normalization=config.dedup_normalization,
            lowercase=config.dedup_lowercase,
            collapse_whitespace=config.dedup_collapse_whitespace,
            shingle_size=config.minhash_shingle_size,
            minhash_permutations=config.minhash_permutations,
            minhash_bands=config.minhash_bands,
            near_duplicate_threshold=config.near_duplicate_threshold,
            boilerplate_min_documents=config.boilerplate_min_documents,
            boilerplate_min_characters=config.boilerplate_min_characters,
            seed=config.seed,
        )
    )

    rows_seen = usable_rows = missing_text_rows = 0
    total_characters = total_tokens = maximum_characters = maximum_tokens = 0
    sentences_observed = total_sentence_tokens = maximum_sentence_tokens = 0
    lines_observed = total_line_characters = maximum_line_characters = 0
    devanagari_total = devanagari_clean_rows = quality_pass_rows = 0
    exact_duplicate_rows = near_duplicate_rows = source_metadata_rows = 0
    pattern_truncated_rows = 0

    for record in records:
        if rows_seen >= sample_limit:
            break
        rows_seen += 1
        text = extract_text(record, spec.text_columns)
        if not text:
            missing_text_rows += 1
            continue
        normalized = normalize_text(text)
        if normalized:
            aggregate_script_evidence += script_evidence(normalized)
        tokens = tokenize(normalized)
        if not normalized or not tokens:
            missing_text_rows += 1
            continue

        usable_rows += 1
        characters = len(normalized)
        token_count = len(tokens)
        ratio = devanagari_ratio(normalized)
        total_characters += characters
        total_tokens += token_count
        maximum_characters = max(maximum_characters, characters)
        maximum_tokens = max(maximum_tokens, token_count)
        devanagari_total += ratio
        devanagari_clean_rows += ratio >= config.min_devanagari_ratio
        quality_pass_rows += (
            token_count >= config.min_tokens and ratio >= config.min_devanagari_ratio
        )
        reservoir.add(ProfileSample(characters, token_count, ratio))
        vocabulary.update(tokens)

        sentence_lengths = sentence_token_lengths(text)
        line_lengths = line_character_lengths(text)
        sentences_observed += len(sentence_lengths)
        total_sentence_tokens += sum(sentence_lengths)
        maximum_sentence_tokens = max(
            maximum_sentence_tokens, max(sentence_lengths, default=0)
        )
        lines_observed += len(line_lengths)
        total_line_characters += sum(line_lengths)
        maximum_line_characters = max(
            maximum_line_characters, max(line_lengths, default=0)
        )
        for length in sentence_lengths:
            sentence_reservoir.add(length)
        for length in line_lengths:
            line_reservoir.add(length)

        pattern_tokens = tokens[: config.max_pattern_tokens_per_document]
        pattern_truncated_rows += len(tokens) > len(pattern_tokens)
        for order, counter in ngram_counters.items():
            counter.update(token_ngrams(pattern_tokens, order))
        document_edges: set[tuple[str, str]] = set()
        cooccurrence_tokens = [strip_nepali_suffix(token) for token in pattern_tokens]
        for index, token in enumerate(cooccurrence_tokens):
            if token not in cooccurrence_terms:
                continue
            lower = max(0, index - config.cooccurrence_window)
            upper = min(
                len(cooccurrence_tokens), index + config.cooccurrence_window + 1
            )
            for neighbor in cooccurrence_tokens[lower:upper]:
                if neighbor != token and neighbor not in cooccurrence_stopwords:
                    document_edges.add((token, neighbor))
        cooccurrences.update(document_edges)

        source = extract_source(record, spec.source_columns)
        if source:
            source_metadata_rows += 1
            sources[source] += 1

        duplicate = deduplication.observe(text)
        if duplicate.exact_duplicate:
            exact_duplicate_rows += 1
        elif duplicate.near_duplicate:
            near_duplicate_rows += 1

        if progress and rows_seen % 1_000 == 0:
            progress(spec.key, "analyzing", rows_seen)

    vocabulary.finish()
    for counter in ngram_counters.values():
        counter.finish()
    cooccurrences.finish()
    samples = tuple(reservoir.values)
    sentence_samples = tuple(sentence_reservoir.values)
    line_samples = tuple(line_reservoir.values)
    character_values = [sample.characters for sample in samples]
    token_values = [sample.tokens for sample in samples]
    duplicate_rows = exact_duplicate_rows + near_duplicate_rows
    boilerplate = deduplication.boilerplate_summary()
    dominant_source, dominant_count = (
        sources.most_common(1)[0] if sources else (None, 0)
    )
    warnings: list[str] = []
    if rows_seen == sample_limit and (
        spec.population_rows is None or rows_seen < spec.population_rows
    ):
        warnings.append("Metrics describe a bounded sample, not the complete dataset.")
    if vocabulary.truncated:
        warnings.append(
            "Vocabulary tracking reached its configured cap; type-token ratio is a lower bound."
        )
    if any(counter.truncated for counter in ngram_counters.values()):
        warnings.append(
            "N-gram tracking reached its configured cap; top phrases are approximate."
        )
    if cooccurrences.truncated:
        warnings.append(
            "Co-occurrence tracking reached its configured cap; top edges are approximate."
        )
    if pattern_truncated_rows:
        warnings.append(
            "Pattern analysis used only the configured leading-token cap for some documents."
        )
    if not source_metadata_rows:
        warnings.append(
            "No source/provenance field was detected in the sampled records."
        )

    summary = DatasetSummary(
        dataset_key=spec.key,
        dataset_id=spec.dataset_id,
        config_name=spec.config_name,
        split=spec.split,
        requested_revision=spec.revision,
        population_rows=spec.population_rows,
        population_coverage_pct=(
            _percent(rows_seen, spec.population_rows)
            if spec.population_rows is not None
            else None
        ),
        sample_limit=sample_limit,
        seed=config.seed,
        rows_seen=rows_seen,
        usable_rows=usable_rows,
        missing_text_rows=missing_text_rows,
        average_characters=_safe_ratio(total_characters, usable_rows),
        median_characters=_quantile(character_values, 0.50),
        p95_characters=_quantile(character_values, 0.95),
        maximum_characters=maximum_characters,
        average_tokens=_safe_ratio(total_tokens, usable_rows),
        median_tokens=_quantile(token_values, 0.50),
        p95_tokens=_quantile(token_values, 0.95),
        maximum_tokens=maximum_tokens,
        sentences_observed=sentences_observed,
        average_sentence_tokens=_safe_ratio(total_sentence_tokens, sentences_observed),
        median_sentence_tokens=_quantile(list(sentence_samples), 0.50),
        p95_sentence_tokens=_quantile(list(sentence_samples), 0.95),
        maximum_sentence_tokens=maximum_sentence_tokens,
        lines_observed=lines_observed,
        average_line_characters=_safe_ratio(total_line_characters, lines_observed),
        median_line_characters=_quantile(list(line_samples), 0.50),
        p95_line_characters=_quantile(list(line_samples), 0.95),
        maximum_line_characters=maximum_line_characters,
        average_devanagari_ratio=_safe_ratio(devanagari_total, usable_rows),
        script_category=classify_script_evidence(aggregate_script_evidence),
        devanagari_letter_share_pct=round(
            100
            * aggregate_script_evidence.letter_share(
                aggregate_script_evidence.devanagari_letters
            ),
            4,
        ),
        latin_letter_share_pct=round(
            100
            * aggregate_script_evidence.letter_share(
                aggregate_script_evidence.latin_letters
            ),
            4,
        ),
        other_letter_share_pct=round(
            100
            * aggregate_script_evidence.letter_share(
                aggregate_script_evidence.other_letters
            ),
            4,
        ),
        romanized_latin_token_share_pct=round(
            100 * aggregate_script_evidence.romanized_token_share,
            4,
        ),
        devanagari_clean_rows=devanagari_clean_rows,
        devanagari_clean_ratio_pct=_percent(devanagari_clean_rows, usable_rows),
        quality_pass_rows=quality_pass_rows,
        quality_pass_ratio_pct=_percent(quality_pass_rows, usable_rows),
        exact_duplicate_rows=exact_duplicate_rows,
        near_duplicate_rows=near_duplicate_rows,
        duplicate_ratio_pct=_percent(duplicate_rows, usable_rows),
        dedup_hash_algorithm="SHA-256",
        dedup_normalization=config.dedup_normalization,
        dedup_lowercase=config.dedup_lowercase,
        dedup_collapse_whitespace=config.dedup_collapse_whitespace,
        minhash_shingle_size=config.minhash_shingle_size,
        minhash_permutations=config.minhash_permutations,
        minhash_bands=config.minhash_bands,
        near_duplicate_threshold=config.near_duplicate_threshold,
        boilerplate_unique_paragraphs=boilerplate.unique_repeated_paragraphs,
        boilerplate_paragraph_occurrences=boilerplate.repeated_paragraph_occurrences,
        boilerplate_affected_rows=boilerplate.affected_documents,
        boilerplate_affected_ratio_pct=_percent(
            boilerplate.affected_documents, usable_rows
        ),
        total_tokens=total_tokens,
        tracked_vocabulary=len(vocabulary.counts),
        type_token_ratio=_safe_ratio(len(vocabulary.counts), total_tokens),
        vocabulary_truncated=vocabulary.truncated,
        ngram_tracking_truncated=any(
            counter.truncated for counter in ngram_counters.values()
        ),
        cooccurrence_tracking_truncated=cooccurrences.truncated,
        pattern_truncated_rows=pattern_truncated_rows,
        source_metadata_rows=source_metadata_rows,
        source_categories_observed=len(sources),
        dominant_source=dominant_source,
        dominant_source_share_pct=_percent(dominant_count, source_metadata_rows),
        warnings=tuple(warnings),
    )
    return DatasetProfile(
        summary=summary,
        samples=samples,
        top_tokens=tuple(vocabulary.counts.most_common(config.top_tokens)),
        top_sources=tuple(sources.most_common(config.top_tokens)),
        sentence_length_samples=sentence_samples,
        line_length_samples=line_samples,
        top_ngrams={
            order: tuple(counter.counts.most_common(config.top_ngrams))
            for order, counter in ngram_counters.items()
        },
        cooccurrence_edges=tuple(
            (source, target, count)
            for (source, target), count in cooccurrences.counts.most_common(
                config.top_cooccurrence_edges
            )
        ),
    )


def huggingface_records(
    spec: DatasetSpec,
    config: AnalysisConfig,
    *,
    token: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Open a Hugging Face iterable dataset without downloading all rows."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face EDA requires the 'datasets' package"
        ) from error
    arguments: dict[str, Any] = {
        "split": spec.split,
        "streaming": True,
        "token": token or None,
    }
    if spec.revision:
        arguments["revision"] = spec.revision
    dataset = load_dataset(spec.dataset_id, spec.config_name, **arguments)
    if config.shuffle_buffer_size:
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer_size,
        )
    return dataset


def run_survey(
    plan: SurveyPlan,
    *,
    record_source: RecordSource | None = None,
    token: str | None = None,
    progress: ProgressCallback | None = None,
) -> SurveyRun:
    """Run every dataset independently so one schema/network failure is isolated."""

    profiles: list[DatasetProfile] = []
    failures: dict[str, str] = {}
    source = record_source or (
        lambda spec, config: huggingface_records(spec, config, token=token)
    )
    for index, spec in enumerate(plan.datasets, start=1):
        if progress:
            progress(spec.key, "starting", index)
        try:
            records = source(spec, plan.analysis)
            profiles.append(
                analyze_records(spec, records, plan.analysis, progress=progress)
            )
        except Exception as error:
            failures[spec.key] = f"{type(error).__name__}: {error}"
        if progress:
            progress(
                spec.key, "complete" if spec.key not in failures else "failed", index
            )
    return SurveyRun(plan, tuple(profiles), failures)


def select_datasets(plan: SurveyPlan, keys: set[str]) -> SurveyPlan:
    """Return a validated plan containing only selected catalog keys."""

    unknown = keys.difference(dataset.key for dataset in plan.datasets)
    if unknown:
        raise ValueError(f"unknown dataset key(s): {', '.join(sorted(unknown))}")
    return replace(
        plan,
        datasets=tuple(dataset for dataset in plan.datasets if dataset.key in keys),
    )


def _safe_ratio(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _percent(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 4) if denominator else 0.0


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 4)
