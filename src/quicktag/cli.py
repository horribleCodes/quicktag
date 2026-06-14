"""Command-line interface for QuickTag."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from quicktag.config import load_config
from quicktag.paths import configure_huggingface_cache, get_install_dir, resolve_path
from quicktag.pipeline import run_pipeline
from quicktag.tags import load_tags


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quicktag",
        description="Tag images using SigLIP2 and write metadata to copies in the output folder.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config.yaml beside the executable)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Install root directory for resolving relative paths (dev override)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
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
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    cache_dir = configure_huggingface_cache(install_dir, config.model.cache_dir)

    log.info("Install directory: %s", install_dir)
    log.info("Input: %s", resolve_path(install_dir, config.paths.input))
    log.info("Output: %s", resolve_path(install_dir, config.paths.output))
    log.info("Tags: %d candidates from %s", len(tags), tags_path.name)

    try:
        summary = run_pipeline(config, install_dir, cache_dir, tags)
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
