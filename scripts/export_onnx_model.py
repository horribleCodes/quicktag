#!/usr/bin/env python3
"""Export SigLIP2 to ONNX for maintainer republishing.

Uses Hugging Face Optimum to export ``google/siglip2-base-patch16-224`` for
zero-shot image classification. Output includes ``model.onnx`` plus tokenizer
and preprocessor files needed at runtime.

Normal dev/CI use ``scripts/download_onnx_model.py`` instead. Republish flow:

1. ``python scripts/export_onnx_model.py --output ./onnx-bundle``
2. Upload the bundle to ``horribleCodes/quicktag-siglip2-onnx`` on Hugging Face.

Example::

    python scripts/export_onnx_model.py --output ./onnx-bundle
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_TASK = "zero-shot-image-classification"
DEFAULT_OPSET = 18

EXPECTED_INPUTS = ("pixel_values", "input_ids")
EXPECTED_OUTPUTS = ("logits_per_image",)


def export_onnx_model(
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    task: str = DEFAULT_TASK,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export *model_name* to ONNX under *output_dir*; return path to model.onnx."""
    from optimum.exporters.onnx import main_export

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=model_name,
        output=str(output_dir),
        task=task,
        opset=opset,
    )

    model_path = output_dir / "model.onnx"
    if not model_path.is_file():
        onnx_files = sorted(output_dir.glob("*.onnx"))
        if len(onnx_files) == 1:
            model_path = onnx_files[0]
        else:
            raise FileNotFoundError(
                f"Expected model.onnx under {output_dir}, found: {onnx_files}"
            )

    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write ONNX export artifacts",
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Optimum export task",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    model_path = export_onnx_model(
        args.output,
        model_name=args.model,
        task=args.task,
        opset=args.opset,
    )
    print(f"Exported ONNX model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
