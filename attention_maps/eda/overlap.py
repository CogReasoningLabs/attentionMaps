"""Report-only internal deduplication and cross-dataset containment analysis."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Mapping

from .deduplication import (
    DeduplicationConfig,
    MultiStageDeduplicator,
    _DocumentFingerprint,
    _MinHashLSHIndex,
    _document_fingerprint,
    _edit_similarity_at_least,
    _sorted_jaccard,
    normalize_for_deduplication,
)
from .text import tokenize


@dataclass(frozen=True)
class DatasetDeduplicationSummary:
    dataset_id: str
    sampled_documents: int
    usable_documents: int
    skipped_documents: int
    exact_duplicates: int
    near_duplicates: int
    unique_documents: int
    internal_duplicate_ratio: float
    retained_token_count: int


@dataclass(frozen=True)
class DirectionalContainment:
    source_dataset_id: str
    covering_dataset_id: str
    source_unique_documents: int
    exact_matched_documents: int
    near_matched_documents: int
    matched_documents: int
    containment_ratio: float
    source_retained_tokens: int
    matched_document_tokens: int
    matched_document_token_ratio: float


@dataclass(frozen=True)
class DatasetOverlapAnalysis:
    datasets: tuple[DatasetDeduplicationSummary, ...]
    directional_containment: tuple[DirectionalContainment, ...]


@dataclass(frozen=True)
class _CanonicalDocument:
    dataset_id: str
    document_index: int
    digest: bytes
    fingerprint: _DocumentFingerprint

    @property
    def tokens(self) -> int:
        return len(self.fingerprint.token_hashes)


def analyze_dataset_overlap(
    datasets: Mapping[str, Iterable[str]],
    config: DeduplicationConfig | None = None,
) -> DatasetOverlapAnalysis:
    """Measure internal duplicates and every cross-dataset direction.

    Inputs are never modified. Cross-dataset denominators use the canonical
    documents retained by an independent internal deduplication pass.
    """

    if len(datasets) < 2:
        raise ValueError("dataset overlap analysis requires at least two datasets")
    if any(not str(dataset_id).strip() for dataset_id in datasets):
        raise ValueError("dataset IDs cannot be empty")
    dedup_config = config or DeduplicationConfig()

    summaries: list[DatasetDeduplicationSummary] = []
    canonical_by_dataset: dict[str, list[_CanonicalDocument]] = {}
    for dataset_id in sorted(map(str, datasets)):
        source_texts = list(datasets[dataset_id])
        deduplicator = MultiStageDeduplicator(dedup_config)
        normalized_texts = [
            normalized
            for text in source_texts
            if (
                normalized := normalize_for_deduplication(
                    str(text),
                    normalization=dedup_config.normalization,
                    lowercase=dedup_config.lowercase,
                    collapse_whitespace=dedup_config.collapse_whitespace,
                )
            )
            and tokenize(normalized)
        ]
        exact_duplicates = near_duplicates = 0
        unique_texts: list[str] = []
        for text in normalized_texts:
            decision = deduplicator.observe(text)
            if decision.exact_duplicate:
                exact_duplicates += 1
            elif decision.near_duplicate:
                near_duplicates += 1
            else:
                unique_texts.append(text)

        documents = [
            _CanonicalDocument(
                dataset_id=dataset_id,
                document_index=index,
                digest=hashlib.sha256(text.encode("utf-8")).digest(),
                fingerprint=_document_fingerprint(text, dedup_config.shingle_size),
            )
            for index, text in enumerate(unique_texts)
        ]
        canonical_by_dataset[dataset_id] = documents
        usable = len(normalized_texts)
        summaries.append(
            DatasetDeduplicationSummary(
                dataset_id=dataset_id,
                sampled_documents=len(source_texts),
                usable_documents=usable,
                skipped_documents=len(source_texts) - usable,
                exact_duplicates=exact_duplicates,
                near_duplicates=near_duplicates,
                unique_documents=len(documents),
                internal_duplicate_ratio=(
                    (exact_duplicates + near_duplicates) / usable if usable else 0.0
                ),
                retained_token_count=sum(document.tokens for document in documents),
            )
        )

    match_sets: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    all_documents = [
        document
        for dataset_id in sorted(canonical_by_dataset)
        for document in canonical_by_dataset[dataset_id]
    ]
    exact_groups: dict[bytes, list[_CanonicalDocument]] = defaultdict(list)
    for document in all_documents:
        exact_groups[document.digest].append(document)
    for group in exact_groups.values():
        for source in group:
            for covering in group:
                if source.dataset_id != covering.dataset_id:
                    match_sets[
                        (source.dataset_id, covering.dataset_id, "exact")
                    ].add(source.document_index)

    near_index = _MinHashLSHIndex(dedup_config)
    indexed_documents: list[_CanonicalDocument] = []
    for document in all_documents:
        shingles = document.fingerprint.shingle_hashes
        if not shingles:
            indexed_documents.append(document)
            continue
        signature = near_index.signature(shingles)
        for candidate_index in sorted(near_index.candidates(signature)):
            candidate = indexed_documents[candidate_index]
            if (
                candidate.dataset_id == document.dataset_id
                or candidate.digest == document.digest
            ):
                continue
            if (
                _sorted_jaccard(
                    document.fingerprint.shingle_hashes,
                    candidate.fingerprint.shingle_hashes,
                )
                < dedup_config.near_duplicate_threshold
                or not _edit_similarity_at_least(
                    document.fingerprint.token_hashes,
                    candidate.fingerprint.token_hashes,
                    dedup_config.edit_similarity_threshold,
                )
            ):
                continue
            match_sets[
                (document.dataset_id, candidate.dataset_id, "near")
            ].add(document.document_index)
            match_sets[
                (candidate.dataset_id, document.dataset_id, "near")
            ].add(candidate.document_index)
        near_index.add(signature, len(indexed_documents))
        indexed_documents.append(document)

    containment: list[DirectionalContainment] = []
    for source_id, covering_id in permutations(sorted(canonical_by_dataset), 2):
        source_documents = canonical_by_dataset[source_id]
        exact = match_sets[(source_id, covering_id, "exact")]
        near = match_sets[(source_id, covering_id, "near")].difference(exact)
        matched = exact.union(near)
        source_tokens = sum(document.tokens for document in source_documents)
        matched_tokens = sum(
            document.tokens
            for document in source_documents
            if document.document_index in matched
        )
        containment.append(
            DirectionalContainment(
                source_dataset_id=source_id,
                covering_dataset_id=covering_id,
                source_unique_documents=len(source_documents),
                exact_matched_documents=len(exact),
                near_matched_documents=len(near),
                matched_documents=len(matched),
                containment_ratio=(
                    len(matched) / len(source_documents) if source_documents else 0.0
                ),
                source_retained_tokens=source_tokens,
                matched_document_tokens=matched_tokens,
                matched_document_token_ratio=(
                    matched_tokens / source_tokens if source_tokens else 0.0
                ),
            )
        )

    return DatasetOverlapAnalysis(tuple(summaries), tuple(containment))
