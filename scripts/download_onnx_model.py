#!/usr/bin/env python3
"""Download the hosted QuickTag SigLIP2 ONNX bundle from Hugging Face.

The fused zero-shot model (``pixel_values`` + ``input_ids`` -> ``logits_per_image``)
is published separately from the PyTorch checkpoint. Runtime and CI use this script;
maintainers republish with ``scripts/export_onnx_model.py``.

Example::

    python scripts/download_onnx_model.py \\
        --output .cache/huggingface/onnx-export/google--siglip2-base-patch16-224
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quicktag.model_onnx import DEFAULT_ONNX_MODEL_REPO, download_onnx_bundle

DEFAULT_MODEL = "google/siglip2-base-patch16-224"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write ONNX bundle artifacts",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_ONNX_MODEL_REPO,
        help="Hugging Face repo id for the ONNX bundle",
    )
    args = parser.parse_args()

    model_path = download_onnx_bundle(args.output, repo_id=args.repo)
    print(f"Downloaded ONNX model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
