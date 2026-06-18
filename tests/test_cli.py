"""Tests for CLI argument parsing and multiprocessing spawn guard."""

from pathlib import Path

import pytest

from quicktag.cli import build_parser, is_multiprocessing_bootstrap_argv, main
from quicktag.exiftool_setup import ExifToolSetupError


def test_build_parser_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


def test_build_parser_quiet_levels():
    parser = build_parser()
    assert parser.parse_args([]).quiet == 0
    assert parser.parse_args(["-q"]).quiet == 1
    assert parser.parse_args(["-qq"]).quiet == 2
    assert parser.parse_args(["-q", "-q"]).quiet == 2
    assert parser.parse_args(["--quiet", "--quiet"]).quiet == 2


def test_main_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_invalid_args_exits_two():
    with pytest.raises(SystemExit) as exc_info:
        main(["--not-a-real-flag"])
    assert exc_info.value.code == 2


def test_is_multiprocessing_bootstrap_argv_resource_tracker():
    argv = [
        "-B",
        "-S",
        "-I",
        "-c",
        "from multiprocessing.resource_tracker import main;main(5)",
    ]
    assert is_multiprocessing_bootstrap_argv(argv) is True


def test_is_multiprocessing_bootstrap_argv_normal_cli():
    assert is_multiprocessing_bootstrap_argv(["--root", "/tmp/quicktag"]) is False


def test_main_ignores_multiprocessing_resource_tracker_argv():
    argv = [
        "-B",
        "-S",
        "-I",
        "-c",
        "from multiprocessing.resource_tracker import main;main(5)",
    ]
    assert main(argv) == 0


def test_main_empty_input_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "config.yaml").write_text(
        """
paths:
  input: input
  output: output
""",
        encoding="utf-8",
    )
    (tmp_path / "tags.yaml").write_text("tags:\n  - cat\n", encoding="utf-8")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: False)
    monkeypatch.setattr(
        "quicktag.cli.ensure_exiftool",
        lambda _install: Path("/usr/bin/exiftool"),
    )

    assert main(["--root", str(tmp_path)]) == 0


def test_main_exiftool_setup_error_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "config.yaml").write_text(
        """
paths:
  input: input
  output: output
""",
        encoding="utf-8",
    )
    (tmp_path / "tags.yaml").write_text("tags:\n  - cat\n", encoding="utf-8")

    def raise_setup_error(_install: Path) -> Path:
        raise ExifToolSetupError("ExifTool not found.", "Install exiftool manually.")

    monkeypatch.setattr("quicktag.cli.ensure_exiftool", raise_setup_error)

    assert main(["--root", str(tmp_path)]) == 1


def test_main_qq_disables_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    (tmp_path / "config.yaml").write_text(
        """
paths:
  input: input
  output: output
""",
        encoding="utf-8",
    )
    (tmp_path / "tags.yaml").write_text("tags:\n  - cat\n", encoding="utf-8")
    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    monkeypatch.setattr("quicktag.paths.is_huggingface_cli_installed", lambda: False)
    monkeypatch.setattr(
        "quicktag.cli.ensure_exiftool",
        lambda _install: Path("/usr/bin/exiftool"),
    )

    captured: list[bool] = []

    def fake_run_pipeline(*args, show_progress: bool = True, **kwargs):
        captured.append(show_progress)
        from quicktag.pipeline import PipelineSummary

        return PipelineSummary()

    monkeypatch.setattr("quicktag.cli.run_pipeline", fake_run_pipeline)

    assert main(["--root", str(tmp_path), "-qq"]) == 0
    assert captured == [False]
    assert "Done: 0 processed" in capsys.readouterr().err
