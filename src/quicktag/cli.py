"""Command-line interface for QuickTag."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

from quicktag.config import load_config
from quicktag.paths import (
    get_install_dir,
    is_huggingface_cli_installed,
    resolve_path,
    setup_huggingface_cache,
)
from quicktag.pipeline import run_pipeline
from quicktag.tags import apply_prompt_template, load_tags


def is_multiprocessing_bootstrap_argv(argv: list[str]) -> bool:
    """Return True when re-executed as a multiprocessing spawn child."""
    if not argv:
        return False
    if argv[0] in {"-c", "--multiprocessing-fork"}:
        return True
    if "-c" in argv:
        idx = argv.index("-c")
        if idx + 1 < len(argv) and "multiprocessing" in argv[idx + 1]:
            return True
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quicktag",
        description="Tag images using SigLIP2 and write metadata to copies in the output folder.",
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config.yaml beside the executable)",
    )
    _ = parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Install root directory for resolving relative paths (dev override)",
    )
    _ = parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if is_multiprocessing_bootstrap_argv(argv):
        multiprocessing.freeze_support()
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    log = logging.getLogger("quicktag")

    install_dir = get_install_dir(args.root)
    config_path = args.config or (install_dir / "config.yaml")

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    tags_path = resolve_path(install_dir, config.tags_file)
    try:
        tags = load_tags(tags_path)
        if config.scoring.prompt_template:
            tags = apply_prompt_template(
                tags,
                config.scoring.prompt_template,
                prompt_overrides_template=config.scoring.prompt_overrides_template,
            )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    hf_cache = setup_huggingface_cache(
        install_dir, config.model.cache_dir, config.model.name
    )

    if is_huggingface_cli_installed():
        log.info("Hugging Face CLI detected; using global cache at %s", hf_cache.primary_home)
    else:
        log.info(
            "Hugging Face CLI not found; using local cache at %s",
            hf_cache.primary_home,
        )

    if hf_cache.source == "primary":
        log.info("Model not cached yet; will download to %s", hf_cache.load_home)
    elif hf_cache.source == "global":
        log.info("Model found in global cache at %s", hf_cache.load_home)
    else:
        log.info("Model found in local cache at %s", hf_cache.load_home)

    log.info("Install directory: %s", install_dir)
    log.info("Input: %s", resolve_path(install_dir, config.paths.input))
    log.info("Output: %s", resolve_path(install_dir, config.paths.output))
    log.info("Tags: %d candidates from %s", len(tags), tags_path.name)

    try:
        summary = run_pipeline(config, install_dir, hf_cache, tags)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        if args.verbose:
            log.exception("Details:")
        return 1

    log.info(
        "Done: %d processed, %d succeeded, %d failed",
        summary.processed,
        summary.succeeded,
        summary.failed,
    )

    if summary.failed > 0 and config.processing.on_error == "skip":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
