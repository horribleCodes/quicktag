"""Tests for PyInstaller packaging hooks."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PyInstaller")


def _load_onnx_hook():
    hook_path = Path(__file__).resolve().parents[1] / "hooks" / "hook-onnxruntime.py"
    spec = importlib.util.spec_from_file_location("hook_onnxruntime", hook_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_includes_onnxruntime():
    hook = _load_onnx_hook()
    assert "onnxruntime" in hook.hiddenimports
    assert "onnxruntime.capi._pybind_state" in hook.hiddenimports


def test_hook_includes_tokenizers():
    hook = _load_onnx_hook()
    assert "tokenizers" in hook.hiddenimports


def test_runtime_metadata_packages_cover_onnxruntime():
    hooks_dir = Path(__file__).resolve().parents[1] / "hooks"
    spec = importlib.util.spec_from_file_location(
        "metadata_packages",
        hooks_dir / "metadata_packages.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "onnxruntime" in module.RUNTIME_METADATA
    assert "tokenizers" in module.RUNTIME_METADATA
