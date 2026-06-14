"""Tests for path resolution."""

import os
from pathlib import Path

from quicktag.paths import configure_huggingface_cache, get_install_dir, resolve_path


def test_resolve_relative_path():
    install = Path("/app/quicktag")
    assert resolve_path(install, "input") == Path("/app/quicktag/input").resolve()


def test_resolve_absolute_path():
    install = Path("/app/quicktag")
    assert resolve_path(install, "/tmp/out") == Path("/tmp/out").resolve()


def test_configure_huggingface_cache(tmp_path: Path):
    cache = configure_huggingface_cache(tmp_path, ".cache/huggingface")
    assert cache.is_dir()
    assert os.environ["HF_HOME"] == str(cache)
