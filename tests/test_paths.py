"""Tests for path resolution."""

import os
from pathlib import Path

import pytest

from quicktag.paths import (
    configure_huggingface_cache,
    get_global_hf_home,
    is_huggingface_cli_installed,
    model_is_cached,
    resolve_hf_cache,
    resolve_path,
    setup_huggingface_cache,
)


def test_resolve_relative_path():
    install = Path("/app/quicktag")
    assert resolve_path(install, "input") == Path("/app/quicktag/input").resolve()


def test_resolve_absolute_path():
    install = Path("/app/quicktag")
    assert resolve_path(install, "/tmp/out") == Path("/tmp/out").resolve()


def test_configure_huggingface_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    cache = configure_huggingface_cache(tmp_path / "hf-cache")
    assert cache.is_dir()
    assert os.environ["HF_HOME"] == str(cache)
    assert os.environ["TRANSFORMERS_CACHE"] == str(cache)
    assert os.environ["HF_HUB_CACHE"] == str(cache / "hub")


@pytest.mark.parametrize("command", ["hf", "huggingface-cli"])
def test_is_huggingface_cli_installed_true(monkeypatch: pytest.MonkeyPatch, command: str):
    monkeypatch.setattr(
        "quicktag.paths.shutil.which",
        lambda name: "/usr/bin/hf" if name == command else None,
    )
    assert is_huggingface_cli_installed() is True


def test_is_huggingface_cli_installed_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("quicktag.paths.shutil.which", lambda _name: None)
    assert is_huggingface_cli_installed() is False


def test_resolve_hf_cache_primary_global_when_cli_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"
    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)
    monkeypatch.setattr("quicktag.paths.model_is_cached", lambda *_args: False)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert layout.load_home == global_home
    assert layout.source == "primary"


def test_resolve_hf_cache_primary_local_when_cli_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"
    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: False)
    monkeypatch.setattr("quicktag.paths.model_is_cached", lambda *_args: False)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == local_home
    assert layout.load_home == local_home
    assert layout.source == "primary"


def test_resolve_hf_cache_loads_from_global_when_only_cached_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"

    def fake_model_is_cached(_repo_id: str, hf_home: Path) -> bool:
        return hf_home == global_home

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: False)
    monkeypatch.setattr("quicktag.paths.model_is_cached", fake_model_is_cached)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == local_home
    assert layout.load_home == global_home
    assert layout.source == "global"


def test_resolve_hf_cache_loads_from_local_when_only_cached_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"

    def fake_model_is_cached(_repo_id: str, hf_home: Path) -> bool:
        return hf_home == local_home

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)
    monkeypatch.setattr("quicktag.paths.model_is_cached", fake_model_is_cached)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert layout.load_home == local_home
    assert layout.source == "local"


def test_setup_huggingface_cache_configures_primary_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home",
        lambda install, cache: resolve_path(install, cache),
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)
    monkeypatch.setattr("quicktag.paths.model_is_cached", lambda *_args: False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    layout = setup_huggingface_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert global_home.is_dir()
    assert os.environ["HF_HOME"] == str(global_home)


def test_get_global_hf_home_uses_huggingface_hub_default():
    from huggingface_hub.constants import HF_HUB_CACHE

    assert get_global_hf_home() == Path(HF_HUB_CACHE).parent


def _write_cached_model(cache_root: Path, repo_id: str, revision: str) -> None:
    from huggingface_hub.file_download import repo_folder_name

    snapshot_dir = (
        cache_root
        / repo_folder_name(repo_id=repo_id, repo_type="model")
        / "snapshots"
        / revision
    )
    snapshot_dir.mkdir(parents=True)
    _ = (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")


def test_model_is_cached_detects_commit_hash_snapshot_in_hub(tmp_path: Path):
    hf_home = tmp_path / "hf-home"
    _write_cached_model(
        hf_home / "hub",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    )

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is True


def test_model_is_cached_detects_commit_hash_snapshot_in_hf_home(tmp_path: Path):
    hf_home = tmp_path / "hf-home"
    _write_cached_model(
        hf_home,
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    )

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is True


def test_model_is_cached_false_when_repo_missing(tmp_path: Path):
    assert model_is_cached("google/siglip2-base-patch16-224", tmp_path / "hf-home") is False


def test_model_is_cached_false_when_snapshot_has_no_config(tmp_path: Path):
    hf_home = tmp_path / "hf-home"
    snapshot_dir = (
        hf_home
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is False
