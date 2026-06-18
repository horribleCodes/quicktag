"""Tests for ExifTool lookup and Windows runtime installation."""

from __future__ import annotations

import sys
import types
import zipfile
from pathlib import Path

import pytest

from quicktag.exiftool_setup import get_exiftool_path
from quicktag.exiftool_setup import (
    ExifToolSetupError,
    ensure_exiftool,
    find_exiftool,
)


def _write_fake_exiftool_zip(zip_path: Path) -> None:
    root = zip_path.parent / "bundle"
    win_dir = root / "exiftool-13.59"
    files_dir = win_dir / "exiftool_files"
    files_dir.mkdir(parents=True)
    (win_dir / "exiftool(-k).exe").write_bytes(b"fake-exe")
    (files_dir / "support.txt").write_text("support", encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))


def test_find_exiftool_from_install_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install = tmp_path / "install"
    exe_name = "exiftool.exe" if sys.platform == "win32" else "exiftool"
    exiftool = install / "exiftool" / exe_name
    exiftool.parent.mkdir(parents=True)
    exiftool.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "quicktag.exiftool_setup.sys",
        types.SimpleNamespace(frozen=False, platform=sys.platform, executable=""),
    )
    assert find_exiftool(install) == exiftool


def test_find_exiftool_uses_exe_dir_when_frozen_with_root_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = tmp_path / "dist" / "quicktag"
    smoke = tmp_path / "smoke-root"
    bundle.mkdir(parents=True)
    smoke.mkdir()

    exe_name = "exiftool.exe" if sys.platform == "win32" else "exiftool"
    exiftool = bundle / "exiftool" / exe_name
    exiftool.parent.mkdir(parents=True)
    exiftool.write_text("", encoding="utf-8")

    fake_exe = bundle / "quicktag"
    fake_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "quicktag.exiftool_setup.sys",
        types.SimpleNamespace(
            frozen=True,
            platform=sys.platform,
            executable=str(fake_exe),
            _MEIPASS=None,
        ),
    )

    assert find_exiftool(smoke) == exiftool


def test_find_exiftool_falls_back_to_which(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "quicktag.exiftool_setup.shutil.which",
        lambda name: "/usr/bin/exiftool" if name == "exiftool" else None,
    )
    assert find_exiftool(tmp_path) == Path("/usr/bin/exiftool")


def test_get_exiftool_path_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("quicktag.exiftool_setup.find_exiftool", lambda _install: None)
    with pytest.raises(FileNotFoundError, match="ExifTool not found"):
        get_exiftool_path(tmp_path)


def test_ensure_exiftool_returns_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    expected = Path("/usr/bin/exiftool")
    monkeypatch.setattr(
        "quicktag.exiftool_setup.find_exiftool", lambda _install: expected
    )
    called = {"download": False}

    def fake_download(_dest: Path) -> None:
        called["download"] = True

    monkeypatch.setattr(
        "quicktag.exiftool_setup._download_windows_exiftool", fake_download
    )
    assert ensure_exiftool(tmp_path) == expected
    assert called["download"] is False


def test_ensure_exiftool_non_windows_raises_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("quicktag.exiftool_setup.sys.platform", "linux")
    monkeypatch.setattr("quicktag.exiftool_setup.find_exiftool", lambda _install: None)
    called = {"download": False}
    monkeypatch.setattr(
        "quicktag.exiftool_setup._download_windows_exiftool",
        lambda _dest: called.update({"download": True}),
    )

    with pytest.raises(ExifToolSetupError) as exc_info:
        ensure_exiftool(tmp_path)

    assert "ExifTool not found" in str(exc_info.value)
    assert exc_info.value.install_hint
    assert called["download"] is False


def test_ensure_exiftool_windows_downloads_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    install = tmp_path / "install"
    expected = install / "exiftool" / "exiftool.exe"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"fake")

    calls = {"download": 0}

    def fake_find(install_dir: Path) -> Path | None:
        if calls["download"] == 0:
            return None
        return expected

    def fake_download(dest: Path) -> None:
        calls["download"] += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "exiftool.exe").write_bytes(b"fake")

    monkeypatch.setattr("quicktag.exiftool_setup.sys.platform", "win32")
    monkeypatch.setattr("quicktag.exiftool_setup.find_exiftool", fake_find)
    monkeypatch.setattr(
        "quicktag.exiftool_setup._download_windows_exiftool", fake_download
    )

    assert ensure_exiftool(install) == expected
    assert calls["download"] == 1


def test_ensure_exiftool_windows_download_failure_raises_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("quicktag.exiftool_setup.sys.platform", "win32")
    monkeypatch.setattr("quicktag.exiftool_setup.find_exiftool", lambda _install: None)

    def fail_download(_dest: Path) -> None:
        raise OSError("network down")

    monkeypatch.setattr(
        "quicktag.exiftool_setup._download_windows_exiftool", fail_download
    )

    with pytest.raises(ExifToolSetupError) as exc_info:
        ensure_exiftool(tmp_path)

    assert "Failed to download ExifTool" in str(exc_info.value)
    assert "exiftool.org" in exc_info.value.install_hint


def test_extract_windows_exiftool_from_zip(tmp_path: Path):
    from quicktag.exiftool_setup import _extract_windows_exiftool

    zip_path = tmp_path / "exiftool.zip"
    dest_dir = tmp_path / "exiftool"
    _write_fake_exiftool_zip(zip_path)

    _extract_windows_exiftool(zip_path, dest_dir)

    assert (dest_dir / "exiftool.exe").read_bytes() == b"fake-exe"
    assert (dest_dir / "exiftool_files" / "support.txt").read_text(
        encoding="utf-8"
    ) == "support"
