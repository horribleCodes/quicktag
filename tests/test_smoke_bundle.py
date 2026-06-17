"""Tests for the committed CI smoke ONNX bundle."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quicktag.model_onnx import SMOKE_ONNX_MODEL_REPO
from quicktag.onnx_tagger import OnnxSigLIPTagger
from quicktag.paths import onnx_export_dir
from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SMOKE_BUNDLE_DIR = FIXTURES_DIR / "onnx-smoke-bundle"


def _stage_smoke_bundle(cache_home: Path) -> Path:
    export_dir = onnx_export_dir(cache_home, SMOKE_ONNX_MODEL_REPO)
    export_dir.mkdir(parents=True, exist_ok=True)
    for artifact in SMOKE_BUNDLE_DIR.iterdir():
        shutil.copy2(artifact, export_dir / artifact.name)
    return export_dir


def test_smoke_bundle_artifacts_exist():
    assert SMOKE_BUNDLE_DIR.is_dir()
    for name in (
        "model.onnx",
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        assert (SMOKE_BUNDLE_DIR / name).is_file()


def test_smoke_bundle_scores_without_download(tmp_path: Path, isolate_hf_cache_env: None):
    cache_home = tmp_path / "cache"
    _stage_smoke_bundle(cache_home)

    tagger = OnnxSigLIPTagger(
        SMOKE_ONNX_MODEL_REPO,
        cache_home,
        local_files_only=True,
    )
    tags = [
        TagDefinition(label="cat", prompt="cat"),
        TagDefinition(label="dog", prompt="dog"),
    ]
    scored = tagger.score(FIXTURES_DIR / "sample.png", tags)

    assert len(scored) == 2
    assert all(isinstance(item, ScoredTag) for item in scored)
    assert all(0.0 <= item.score <= 1.0 for item in scored)
    assert scored[0].score >= scored[1].score
