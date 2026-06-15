"""Resolve install directory and config-relative paths."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
_ONNX_FILENAMES = ("model.onnx",)


@dataclass(frozen=True)
class HuggingFaceCacheLayout:
    """Resolved Hugging Face cache locations for env setup and model loading."""

    primary_home: Path
    load_home: Path
    hub_dir: Path
    source: Literal["global", "local", "primary"]
    local_files_only: bool


def get_install_dir(root: Path | None = None) -> Path:
    """Return the directory containing the application (exe or project root)."""
    if root is not None:
        return root.resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    # Dev mode: src/quicktag/paths.py -> project root
    return Path(__file__).resolve().parents[2]


def resolve_path(install_dir: Path, path: str | Path) -> Path:
    """Resolve a path relative to install_dir unless already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (install_dir / p).resolve()


def is_huggingface_cli_installed() -> bool:
    """Return True when the Hugging Face CLI is available on PATH."""
    return shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None


def get_global_hf_home() -> Path:
    """Return the default global Hugging Face cache home directory."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE).parent


def get_local_hf_home(install_dir: Path, cache_dir: str | Path) -> Path:
    """Return the install-local Hugging Face cache home directory."""
    return resolve_path(install_dir, cache_dir)


def _snapshot_has_weights(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False
    if not (snapshot / "config.json").is_file():
        return False
    if any((snapshot / name).is_file() for name in _WEIGHT_FILENAMES):
        return True
    return any(snapshot.glob("model-*.safetensors")) or any(snapshot.glob("pytorch_model-*.bin"))


def _snapshot_has_onnx(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False
    if not (snapshot / "config.json").is_file():
        return False
    if any((snapshot / name).is_file() for name in _ONNX_FILENAMES):
        return True
    onnx_dir = snapshot / "onnx" / "model.onnx"
    return onnx_dir.is_file()


def _snapshot_is_cached(snapshot: Path) -> bool:
    return _snapshot_has_weights(snapshot) or _snapshot_has_onnx(snapshot)


def _iter_cache_roots(hf_home: Path) -> list[Path]:
    """Return candidate cache roots, preferring HF_HUB_CACHE then HF_HOME/hub then HF_HOME."""
    roots: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        add(Path(env_hub))
    add(hf_home / "hub")
    add(hf_home)
    return roots


def find_model_in_cache(repo_id: str, hf_home: Path) -> Path | None:
    """Return the cache root containing a complete model snapshot, if any."""
    from huggingface_hub.file_download import repo_folder_name

    repo_folder = repo_folder_name(repo_id=repo_id, repo_type="model")

    for cache_root in _iter_cache_roots(hf_home):
        snapshots_dir = cache_root / repo_folder / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        if any(_snapshot_is_cached(child) for child in snapshots_dir.iterdir()):
            return cache_root

    return None


def onnx_export_dir(hf_home: Path, repo_id: str) -> Path:
    """Return the install-local ONNX export directory for a model repo."""
    return hf_home / "onnx-export" / repo_id.replace("/", "--")


def find_onnx_in_cache(repo_id: str, hf_home: Path) -> Path | None:
    """Return the cache root containing ONNX weights or an export, if any."""
    from huggingface_hub.file_download import repo_folder_name

    repo_folder = repo_folder_name(repo_id=repo_id, repo_type="model")

    for cache_root in _iter_cache_roots(hf_home):
        snapshots_dir = cache_root / repo_folder / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        if any(_snapshot_has_onnx(child) for child in snapshots_dir.iterdir()):
            return cache_root

    export_dir = onnx_export_dir(hf_home, repo_id)
    if export_dir.is_dir() and _snapshot_has_onnx(export_dir):
        return hf_home

    return None


def resolve_onnx_model_dir(repo_id: str, hf_home: Path) -> Path | None:
    """Return the snapshot directory containing ONNX weights, if cached."""
    from huggingface_hub.file_download import repo_folder_name

    repo_folder = repo_folder_name(repo_id=repo_id, repo_type="model")

    for cache_root in _iter_cache_roots(hf_home):
        snapshots_dir = cache_root / repo_folder / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for child in snapshots_dir.iterdir():
            if _snapshot_has_onnx(child):
                return child

    return None


def model_is_cached(repo_id: str, hf_home: Path) -> bool:
    """Return True when a model snapshot is present under an HF cache home directory."""
    return find_model_in_cache(repo_id, hf_home) is not None


def resolve_hf_cache(
    install_dir: Path,
    config_cache_dir: str | Path,
    model_name: str,
) -> HuggingFaceCacheLayout:
    """Choose primary and load cache dirs, probing global and local caches first."""
    global_home = get_global_hf_home()
    local_home = get_local_hf_home(install_dir, config_cache_dir)
    primary_home = global_home if is_huggingface_cli_installed() else local_home

    for source, home in (("global", global_home), ("local", local_home)):
        hub_dir = find_onnx_in_cache(model_name, home)
        if hub_dir is not None:
            return HuggingFaceCacheLayout(
                primary_home=primary_home,
                load_home=home,
                hub_dir=hub_dir,
                source=source,
                local_files_only=True,
            )

    return HuggingFaceCacheLayout(
        primary_home=primary_home,
        load_home=primary_home,
        hub_dir=primary_home / "hub",
        source="primary",
        local_files_only=False,
    )


def _default_model_cache_dir(hf_home: Path) -> Path:
    """Default download location for new models (standard HF_HOME/hub layout)."""
    cache_dir = hf_home / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _apply_hf_cache_env(hf_home: Path, hub_dir: Path) -> None:
    """Point Hugging Face libraries at hf_home and hub_dir."""
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_dir)


def configure_huggingface_cache(primary_home: Path) -> Path:
    """Pin Hugging Face env vars to the primary cache directory."""
    primary_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(primary_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(primary_home))
    os.environ.setdefault("HF_HUB_CACHE", str(primary_home / "hub"))
    return primary_home


def setup_huggingface_cache(
    install_dir: Path,
    config_cache_dir: str | Path,
    model_name: str,
) -> HuggingFaceCacheLayout:
    """Resolve cache layout and configure Hugging Face env vars."""
    layout = resolve_hf_cache(install_dir, config_cache_dir, model_name)
    if layout.source == "primary":
        _apply_hf_cache_env(layout.primary_home, _default_model_cache_dir(layout.primary_home))
    else:
        _apply_hf_cache_env(layout.load_home, layout.hub_dir)
    return layout


def get_exiftool_path(install_dir: Path) -> Path:
    """Locate bundled ExifTool executable."""
    exe_name = "exiftool.exe" if sys.platform == "win32" else "exiftool"

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled = Path(meipass) / "exiftool" / exe_name
            if bundled.is_file():
                return bundled

    candidates = [
        install_dir / "exiftool" / exe_name,
        install_dir / "assets" / "exiftool" / exe_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    system_exiftool = shutil.which("exiftool")
    if system_exiftool:
        return Path(system_exiftool)

    raise FileNotFoundError(
        f"ExifTool not found. Expected {exe_name} under {install_dir / 'exiftool'}"
    )
