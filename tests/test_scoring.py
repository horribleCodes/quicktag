"""Tests for tag scoring selection."""

from quicktag.config import ScoringConfig
from quicktag.scoring import ScoredTag, select_tags


def test_select_tags_min_score():
    scored = [
        ScoredTag("cat", 0.9),
        ScoredTag("dog", 0.04),
        ScoredTag("bird", 0.2),
    ]
    selected = select_tags(
        scored, ScoringConfig(min_score=0.05, top_k=None, top_p=None)
    )
    labels = [t.label for t in selected]
    assert labels == ["cat", "bird"]


def test_select_tags_top_k():
    scored = [
        ScoredTag("a", 0.9),
        ScoredTag("b", 0.8),
        ScoredTag("c", 0.7),
    ]
    selected = select_tags(scored, ScoringConfig(min_score=0.0, top_k=2, top_p=None))
    assert [t.label for t in selected] == ["a", "b"]


def test_select_tags_top_p():
    scored = [
        ScoredTag("a", 0.5),
        ScoredTag("b", 0.3),
        ScoredTag("c", 0.2),
    ]
    selected = select_tags(scored, ScoringConfig(min_score=0.0, top_k=None, top_p=0.6))
    assert [t.label for t in selected] == ["a", "b"]
