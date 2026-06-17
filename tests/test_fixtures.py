"""Tests for shared test fixtures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_sample_fixture_is_large_enough_for_transformers():
    """Integration tests need a non-degenerate image size for HF processors."""
    sample = FIXTURES_DIR / "sample.png"
    assert sample.is_file()
    with Image.open(sample) as image:
        width, height = image.size
        assert width >= 64 and height >= 64
        assert image.mode in {"RGB", "RGBA"}


def test_tiny_fixture_stays_small_for_fast_unit_tests():
    tiny = FIXTURES_DIR / "tiny.png"
    assert tiny.is_file()
    with Image.open(tiny) as image:
        assert max(image.size) <= 8
