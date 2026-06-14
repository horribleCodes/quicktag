"""Resolve install directory and config-relative paths."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class HuggingFaceCacheLayout:
    """Resolved Hugging Face cache locations for env setup and model loading."""

    primary_home: Path
    load_home: Path
    source: Literal["global", "local", "primary"]


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


def model_is_cached(repo_id: str, hub_cache_dir: Path) -> bool:
    """Return True when a model snapshot is present in the given hub cache."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        repo_id=repo_id,
        filename="config.json",
        cache_dir=str(hub_cache_dir),
    )
    return isinstance(cached, str)


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
        if model_is_cached(model_name, home / "hub"):
            return HuggingFaceCacheLayout(
                primary_home=primary_home,
                load_home=home,
                source=source,
            )

    return HuggingFaceCacheLayout(
        primary_home=primary_home,
        load_home=primary_home,
        source="primary",
    )


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
    configure_huggingface_cache(layout.primary_home)
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
