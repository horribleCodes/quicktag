"""Tag selection from SigLIP sigmoid scores."""

from __future__ import annotations

from dataclasses import dataclass

from quicktag.config import ScoringConfig


@dataclass(frozen=True)
class ScoredTag:
    label: str
    score: float


def select_tags(
    scored: list[ScoredTag],
    config: ScoringConfig,
) -> list[ScoredTag]:
    """
    Select applicable tags using min_score, top_p, and top_k.

    1. Filter by min_score
    2. Sort descending by score
    3. Apply top_p nucleus cutoff on normalized scores (if set)
    4. Cap at top_k (if set)
    """
    candidates = [item for item in scored if item.score >= config.min_score]
    candidates.sort(key=lambda item: item.score, reverse=True)

    if not candidates:
        return []

    if config.top_p is not None:
        total = sum(item.score for item in candidates)
        if total > 0:
            cumulative = 0.0
            selected: list[ScoredTag] = []
            for item in candidates:
                selected.append(item)
                cumulative += item.score / total
                if cumulative >= config.top_p:
                    break
            candidates = selected

    if config.top_k is not None:
        candidates = candidates[: config.top_k]

    return candidates
