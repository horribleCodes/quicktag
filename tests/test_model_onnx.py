"""Tests for hosted SigLIP2 ONNX bundle download helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_download_constants():
    from quicktag.model_onnx import DEFAULT_ONNX_MODEL_REPO

    assert DEFAULT_ONNX_MODEL_REPO == "horrible/siglip2-base-patch16-224"


def test_download_onnx_bundle_writes_model(tmp_path: Path):
    from quicktag.model_onnx import download_onnx_bundle

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()

    def fake_snapshot_download(*, repo_id, local_dir, local_files_only, allow_patterns):
        bundle = Path(local_dir)
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "model.onnx").write_bytes(b"onnx")
        (bundle / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        return bundle

    with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
        model_path = download_onnx_bundle(output_dir, repo_id="example/onnx-bundle")

    assert model_path == output_dir / "model.onnx"
