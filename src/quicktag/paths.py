"""Resolve install directory and config-relative paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


def configure_huggingface_cache(install_dir: Path, cache_dir: str | Path) -> Path:
    """Pin Hugging Face cache to a directory beside the install folder."""
    cache_path = resolve_path(install_dir, cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_path / "hub"))
    return cache_path


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

    import shutil

    system_exiftool = shutil.which("exiftool")
    if system_exiftool:
        return Path(system_exiftool)

    raise FileNotFoundError(
        f"ExifTool not found. Expected {exe_name} under {install_dir / 'exiftool'}"
    )
