"""Tests for CLI argument parsing and multiprocessing spawn guard."""

from pathlib import Path

import pytest

from quicktag.cli import build_parser, is_multiprocessing_bootstrap_argv, main


def test_build_parser_help():
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0


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

    assert main(["--root", str(tmp_path)]) == 0
