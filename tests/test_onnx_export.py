"""Tests for SigLIP2 ONNX bundle tooling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_onnx_model.py"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_onnx_model.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_onnx_model", EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_constants():
    pytest.importorskip("optimum")
    module = _load_export_module()
    assert "pixel_values" in module.EXPECTED_INPUTS
    assert "input_ids" in module.EXPECTED_INPUTS
    assert "logits_per_image" in module.EXPECTED_OUTPUTS


def test_download_constants():
    from quicktag.model_onnx import DEFAULT_ONNX_MODEL_REPO, DEFAULT_SIGLIP_MODEL

    assert DEFAULT_SIGLIP_MODEL == "google/siglip2-base-patch16-224"
    assert DEFAULT_ONNX_MODEL_REPO


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


@pytest.mark.integration
def test_export_onnx_model_creates_valid_session(tmp_path: Path):
    pytest.importorskip("optimum")
    module = _load_export_module()
    import onnxruntime as ort

    model_path = module.export_onnx_model(tmp_path)
    assert model_path.is_file()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    output_names = {item.name for item in session.get_outputs()}

    assert set(module.EXPECTED_INPUTS).issubset(input_names)
    assert set(module.EXPECTED_OUTPUTS).issubset(output_names)

    pixel_values = np.zeros((1, 3, 224, 224), dtype=np.float32)
    input_ids = np.zeros((1, 64), dtype=np.int64)
    outputs = session.run(
        None,
        {"pixel_values": pixel_values, "input_ids": input_ids},
    )
    assert len(outputs) >= 1
