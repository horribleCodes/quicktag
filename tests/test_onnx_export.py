"""Tests for SigLIP2 ONNX export tooling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("optimum")
pytest.importorskip("onnxruntime")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "export_onnx_model.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_onnx_model", EXPORT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_constants():
    module = _load_export_module()
    assert "pixel_values" in module.EXPECTED_INPUTS
    assert "input_ids" in module.EXPECTED_INPUTS
    assert "logits_per_image" in module.EXPECTED_OUTPUTS


@pytest.mark.integration
def test_export_onnx_model_creates_valid_session(tmp_path: Path):
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
