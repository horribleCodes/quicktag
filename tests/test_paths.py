"""Tests for path resolution."""

import os
from pathlib import Path

import pytest

from quicktag.paths import (
    _snapshot_has_onnx,
    _snapshot_has_weights,
    configure_huggingface_cache,
    find_model_in_cache,
    get_global_hf_home,
    is_huggingface_cli_installed,
    model_is_cached,
    resolve_hf_cache,
    resolve_onnx_model_dir,
    resolve_path,
    setup_huggingface_cache,
)


@pytest.fixture
def isolate_hf_cache_env(monkeypatch: pytest.MonkeyPatch):
    """Prevent host HF cache env vars from affecting cache lookup tests."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)


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
    monkeypatch.setattr("quicktag.paths.find_model_in_cache", lambda *_args: None)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert layout.load_home == global_home
    assert layout.hub_dir == global_home / "hub"
    assert layout.source == "primary"
    assert layout.local_files_only is False


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
    monkeypatch.setattr("quicktag.paths.find_model_in_cache", lambda *_args: None)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == local_home
    assert layout.load_home == local_home
    assert layout.hub_dir == local_home / "hub"
    assert layout.source == "primary"
    assert layout.local_files_only is False


def test_resolve_hf_cache_loads_from_global_when_only_cached_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"

    def fake_find_model_in_cache(_repo_id: str, hf_home: Path) -> Path | None:
        return hf_home if hf_home == global_home else None

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: False)
    monkeypatch.setattr("quicktag.paths.find_model_in_cache", fake_find_model_in_cache)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == local_home
    assert layout.load_home == global_home
    assert layout.hub_dir == global_home
    assert layout.source == "global"
    assert layout.local_files_only is True


def test_resolve_hf_cache_loads_from_local_when_only_cached_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    global_home = tmp_path / "global-hf"
    local_home = tmp_path / "local-hf"

    def fake_find_model_in_cache(_repo_id: str, hf_home: Path) -> Path | None:
        return hf_home if hf_home == local_home else None

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr(
        "quicktag.paths.get_local_hf_home", lambda _install, _cache: local_home
    )
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)
    monkeypatch.setattr("quicktag.paths.find_model_in_cache", fake_find_model_in_cache)

    layout = resolve_hf_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert layout.load_home == local_home
    assert layout.hub_dir == local_home
    assert layout.source == "local"
    assert layout.local_files_only is True


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
    monkeypatch.setattr("quicktag.paths.find_model_in_cache", lambda *_args: None)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    layout = setup_huggingface_cache(tmp_path, ".cache/huggingface", "google/siglip2")

    assert layout.primary_home == global_home
    assert global_home.is_dir()
    assert os.environ["HF_HOME"] == str(global_home)
    assert os.environ["HF_HUB_CACHE"] == str(global_home / "hub")


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
    _ = (snapshot_dir / "model.safetensors").write_bytes(b"weights")


def test_model_is_cached_detects_commit_hash_snapshot_in_hub(
    tmp_path: Path, isolate_hf_cache_env: None
):
    hf_home = tmp_path / "hf-home"
    _write_cached_model(
        hf_home / "hub",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    )

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is True
    assert find_model_in_cache("google/siglip2-base-patch16-224", hf_home) == (
        hf_home / "hub"
    ).resolve()


def test_model_is_cached_detects_commit_hash_snapshot_in_hf_home(
    tmp_path: Path, isolate_hf_cache_env: None
):
    hf_home = tmp_path / "hf-home"
    _write_cached_model(
        hf_home,
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    )

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is True
    assert find_model_in_cache("google/siglip2-base-patch16-224", hf_home) == hf_home.resolve()


def test_model_is_cached_false_when_repo_missing(tmp_path: Path, isolate_hf_cache_env: None):
    assert model_is_cached("google/siglip2-base-patch16-224", tmp_path / "hf-home") is False


def test_model_is_cached_false_when_snapshot_has_no_config(
    tmp_path: Path, isolate_hf_cache_env: None
):
    hf_home = tmp_path / "hf-home"
    snapshot_dir = (
        hf_home
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is False


def test_model_is_cached_false_when_snapshot_has_config_but_no_weights(
    tmp_path: Path, isolate_hf_cache_env: None
):
    hf_home = tmp_path / "hf-home"
    snapshot_dir = (
        hf_home
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is False


def test_find_model_in_cache_prefers_hub_subdir(tmp_path: Path, isolate_hf_cache_env: None):
    model_name = "google/siglip2-base-patch16-224"
    hf_home = tmp_path / "hf"

    hub_snapshot = hf_home / "hub" / "models--google--siglip2-base-patch16-224" / "snapshots" / "hub"
    hub_snapshot.mkdir(parents=True)
    (hub_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (hub_snapshot / "model.safetensors").write_bytes(b"hub")

    root_snapshot = hf_home / "models--google--siglip2-base-patch16-224" / "snapshots" / "root"
    root_snapshot.mkdir(parents=True)
    (root_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (root_snapshot / "model.safetensors").write_bytes(b"root")

    assert find_model_in_cache(model_name, hf_home) == (hf_home / "hub").resolve()


def test_setup_huggingface_cache_uses_hf_home_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    global_home = tmp_path / "global-hf"
    snapshot = global_home / "models--google--siglip2-base-patch16-224" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)

    layout = setup_huggingface_cache(tmp_path, ".cache/huggingface", "google/siglip2-base-patch16-224")

    assert layout.load_home == global_home.resolve()
    assert layout.hub_dir == global_home.resolve()
    assert layout.local_files_only is True
    assert os.environ["HF_HUB_CACHE"] == str(global_home.resolve())


def test_setup_huggingface_cache_prefers_global_cached_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolate_hf_cache_env: None
):
    global_home = tmp_path / "global-hf"
    global_hub = global_home / "hub"
    snapshot = global_hub / "models--google--siglip2-base-patch16-224" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr("quicktag.paths.get_global_hf_home", lambda: global_home)
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: True)

    layout = setup_huggingface_cache(tmp_path, ".cache/huggingface", "google/siglip2-base-patch16-224")

    assert layout.load_home == global_home.resolve()
    assert layout.hub_dir == global_hub.resolve()
    assert layout.local_files_only is True
    assert os.environ["HF_HUB_CACHE"] == str(global_hub.resolve())


def test_snapshot_has_weights_accepts_sharded_files(tmp_path: Path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a")

    assert _snapshot_has_weights(snapshot) is True


def test_snapshot_has_onnx_detects_model_file(tmp_path: Path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.onnx").write_bytes(b"onnx")

    assert _snapshot_has_onnx(snapshot) is True


def test_model_is_cached_true_for_onnx_snapshot(tmp_path: Path, isolate_hf_cache_env: None):
    hf_home = tmp_path / "hf-home"
    snapshot_dir = (
        hf_home
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "onnx-rev"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model.onnx").write_bytes(b"onnx")

    assert model_is_cached("google/siglip2-base-patch16-224", hf_home) is True


def test_resolve_onnx_model_dir_returns_snapshot(tmp_path: Path, isolate_hf_cache_env: None):
    hf_home = tmp_path / "hf-home"
    snapshot_dir = (
        hf_home
        / "models--google--siglip2-base-patch16-224"
        / "snapshots"
        / "onnx-rev"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
    (snapshot_dir / "model.onnx").write_bytes(b"onnx")

    assert resolve_onnx_model_dir("google/siglip2-base-patch16-224", hf_home) == snapshot_dir.resolve()
