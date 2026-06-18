"""Locate and optionally install ExifTool."""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

EXIFTOOL_VERSION = "13.59"
EXIFTOOL_WINDOWS_ZIP_URL = f"https://exiftool.org/exiftool-{EXIFTOOL_VERSION}_64.zip"


class ExifToolSetupError(Exception):
    """ExifTool is unavailable and could not be installed automatically."""

    def __init__(self, message: str, install_hint: str) -> None:
        super().__init__(message)
        self.install_hint = install_hint


def _exe_name() -> str:
    return "exiftool.exe" if sys.platform == "win32" else "exiftool"


def find_exiftool(install_dir: Path) -> Path | None:
    """Return the ExifTool executable path if found, otherwise None."""
    exe_name = _exe_name()

    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "exiftool" / exe_name)
        candidates.append(Path(sys.executable).resolve().parent / "exiftool" / exe_name)

    candidates.extend(
        [
            install_dir / "exiftool" / exe_name,
            install_dir / "assets" / "exiftool" / exe_name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    system_exiftool = shutil.which("exiftool")
    if system_exiftool:
        return Path(system_exiftool)

    return None


def get_exiftool_path(install_dir: Path) -> Path:
    """Locate ExifTool executable or raise FileNotFoundError."""
    found = find_exiftool(install_dir)
    if found is not None:
        return found

    exe_name = _exe_name()
    raise FileNotFoundError(
        f"ExifTool not found. Expected {exe_name} under {install_dir / 'exiftool'}"
    )


def _windows_install_hint(install_dir: Path) -> str:
    exiftool_dir = install_dir / "exiftool"
    return (
        "Install ExifTool manually:\n"
        "  1. Download from https://exiftool.org/install.html\n"
        f"  2. Place exiftool.exe and exiftool_files/ in {exiftool_dir}\n"
        "  3. Or install ExifTool system-wide so the exiftool command is on PATH"
    )


def _linux_install_hint(install_dir: Path) -> str:
    return (
        "Install ExifTool via your package manager:\n"
        "  Arch Linux:        sudo pacman -S perl-image-exiftool\n"
        "  Debian / Ubuntu:   sudo apt install libimage-exiftool-perl\n"
        "  Fedora:            sudo dnf install perl-Image-ExifTool\n"
        "  macOS:             brew install exiftool\n"
        "Or follow https://exiftool.org/install.html"
    )


def _install_hint(install_dir: Path) -> str:
    if sys.platform == "win32":
        return _windows_install_hint(install_dir)
    return _linux_install_hint(install_dir)


def _extract_windows_exiftool(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir / "_extract")

    extract_root = dest_dir / "_extract"
    win_dirs = sorted(extract_root.glob("exiftool-*"))
    if not win_dirs:
        raise FileNotFoundError("No exiftool-* directory found in downloaded zip")

    win_dir = win_dirs[0]
    source_exe = win_dir / "exiftool(-k).exe"
    if not source_exe.is_file():
        raise FileNotFoundError(f"Expected {source_exe.name} in downloaded zip")

    shutil.copy2(source_exe, dest_dir / "exiftool.exe")

    source_files = win_dir / "exiftool_files"
    if not source_files.is_dir():
        raise FileNotFoundError("Expected exiftool_files/ in downloaded zip")

    target_files = dest_dir / "exiftool_files"
    if target_files.exists():
        shutil.rmtree(target_files)
    shutil.copytree(source_files, target_files)

    shutil.rmtree(extract_root)


def _download_windows_exiftool(dest_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "exiftool.zip"
        logger.info("Downloading ExifTool %s for Windows...", EXIFTOOL_VERSION)
        urllib.request.urlretrieve(EXIFTOOL_WINDOWS_ZIP_URL, zip_path)
        _extract_windows_exiftool(zip_path, dest_dir)


def ensure_exiftool(install_dir: Path) -> Path:
    """Return ExifTool path, downloading on Windows when missing."""
    found = find_exiftool(install_dir)
    if found is not None:
        return found

    if sys.platform != "win32":
        raise ExifToolSetupError(
            "ExifTool not found.",
            _install_hint(install_dir),
        )

    dest_dir = install_dir / "exiftool"
    try:
        _download_windows_exiftool(dest_dir)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, FileNotFoundError) as exc:
        raise ExifToolSetupError(
            f"Failed to download ExifTool: {exc}",
            _install_hint(install_dir),
        ) from exc

    found = find_exiftool(install_dir)
    if found is not None:
        logger.info("ExifTool installed at %s", found)
        return found

    raise ExifToolSetupError(
        "ExifTool download completed but executable was not found.",
        _install_hint(install_dir),
    )
