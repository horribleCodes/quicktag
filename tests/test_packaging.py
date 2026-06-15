"""Tests for PyInstaller packaging hooks."""

import importlib.util
import pkgutil
from pathlib import Path

import pytest

pytest.importorskip("PyInstaller")


def _load_transformers_hook():
    hook_path = Path(__file__).resolve().parents[1] / "hooks" / "hook-transformers.py"
    spec = importlib.util.spec_from_file_location("hook_transformers", hook_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_includes_diffusion_gemma():
    hook = _load_transformers_hook()
    assert "transformers.models.diffusion_gemma" in hook.hiddenimports


def test_hook_includes_regex():
    hook = _load_transformers_hook()
    assert "regex" in hook.hiddenimports


def test_runtime_metadata_packages_cover_transformers_checks():
    hooks_dir = Path(__file__).resolve().parents[1] / "hooks"
    spec = importlib.util.spec_from_file_location(
        "metadata_packages",
        hooks_dir / "metadata_packages.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "regex" in module.TRANSFORMERS_RUNTIME_METADATA
    assert "packaging" in module.TRANSFORMERS_RUNTIME_METADATA
    assert len(module.TRANSFORMERS_RUNTIME_METADATA) >= 10


def test_hook_includes_all_transformers_model_packages():
    import transformers.models

    hook = _load_transformers_hook()
    expected = {
        f"transformers.models.{name}"
        for _, name, is_pkg in pkgutil.iter_modules(transformers.models.__path__)
        if is_pkg
    }
    assert expected <= set(hook.hiddenimports)
