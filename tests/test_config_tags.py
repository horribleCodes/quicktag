"""Tests for config and tags loading."""

from pathlib import Path

import pytest

from quicktag.config import load_config
from quicktag.tags import load_tags


def test_load_config(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
paths:
  input: photos/in
  output: photos/out
scoring:
  min_score: 0.1
  top_k: 5
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.paths.input == "photos/in"
    assert config.scoring.min_score == 0.1
    assert config.scoring.top_k == 5


def test_load_config_invalid_on_error(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("processing:\n  on_error: maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="on_error"):
        load_config(config_file)


def test_load_tags_simple(tmp_path: Path):
    tags_file = tmp_path / "tags.yaml"
    tags_file.write_text("tags:\n  - cat\n  - dog\n", encoding="utf-8")
    tags = load_tags(tags_file)
    assert [t.label for t in tags] == ["cat", "dog"]
    assert tags[0].prompt == "cat"


def test_load_tags_with_prompts(tmp_path: Path):
    tags_file = tmp_path / "tags.yaml"
    tags_file.write_text(
        "tags:\n  - label: cat\n    prompt: a photo of a cat\n",
        encoding="utf-8",
    )
    tags = load_tags(tags_file)
    assert tags[0].label == "cat"
    assert tags[0].prompt == "a photo of a cat"
