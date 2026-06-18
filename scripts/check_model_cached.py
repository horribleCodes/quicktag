#!/usr/bin/env python3
"""Check whether a Hugging Face model is cached globally or locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from quicktag.paths import (
    get_global_hf_home,
    get_install_dir,
    model_is_cached,
    resolve_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="horrible/siglip2-base-patch16-224",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Install root for resolving the local cache (default: project/exe root)",
    )
    parser.add_argument(
        "--local-cache-dir",
        default=".cache/huggingface",
        help="Local HF_HOME path relative to --root",
    )
    args = parser.parse_args()

    install_dir = get_install_dir(args.root)
    local_home = resolve_path(install_dir, args.local_cache_dir)
    global_home = get_global_hf_home()

    for label, home in (("global", global_home), ("local", local_home)):
        cached = model_is_cached(args.model, home)
        print(f"{label}: {home} -> {cached}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
