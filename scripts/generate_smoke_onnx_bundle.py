#!/usr/bin/env python3
"""Generate the committed CI smoke ONNX bundle under tests/fixtures/onnx-smoke-bundle/.

The graph matches OnnxSigLIPTagger I/O: pixel_values + input_ids -> logits_per_image.
Regenerate when the ONNX contract in quicktag.onnx_tagger changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from onnx import TensorProto, helper, numpy_helper
import numpy as np

from quicktag.model_onnx import SMOKE_ONNX_MODEL_REPO

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "onnx-smoke-bundle"

# Prompts from tags.example.yaml (lowercased by load_tags).
SMOKE_VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
    "cat": 2,
    "dog": 3,
    "landscape": 4,
    "portrait": 5,
}


def _build_smoke_onnx() -> bytes:
    pixel_values = helper.make_tensor_value_info(
        "pixel_values",
        TensorProto.FLOAT,
        [1, 3, 224, 224],
    )
    input_ids = helper.make_tensor_value_info(
        "input_ids",
        TensorProto.INT64,
        ["num_prompts", 64],
    )
    logits_per_image = helper.make_tensor_value_info(
        "logits_per_image",
        TensorProto.FLOAT,
        [1, "num_prompts"],
    )

    reduce_axes = numpy_helper.from_array(
        np.array([1], dtype=np.int64), name="reduce_axes"
    )
    cast_node = helper.make_node(
        "Cast",
        inputs=["input_ids"],
        outputs=["input_ids_float"],
        to=TensorProto.FLOAT,
    )
    reduce_node = helper.make_node(
        "ReduceSum",
        inputs=["input_ids_float", "reduce_axes"],
        outputs=["prompt_sums"],
        keepdims=0,
    )
    unsqueeze_axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name="unsqueeze_axes"
    )
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["prompt_sums", "unsqueeze_axes"],
        outputs=["logits_per_image"],
    )

    graph = helper.make_graph(
        nodes=[cast_node, reduce_node, unsqueeze_node],
        name="quicktag_smoke",
        inputs=[pixel_values, input_ids],
        outputs=[logits_per_image],
        initializer=[reduce_axes, unsqueeze_axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    return model.SerializeToString()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_tokenizer(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": SMOKE_VOCAB,
                    "unk_token": "<unk>",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_bundle(output_dir: Path = OUTPUT_DIR) -> Path:
    """Write smoke ONNX bundle files to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "model.onnx").write_bytes(_build_smoke_onnx())

    _write_json(
        output_dir / "preprocessor_config.json",
        {
            "size": {"height": 224, "width": 224},
            "do_rescale": True,
            "rescale_factor": 1.0 / 255.0,
            "do_normalize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    )
    _write_json(output_dir / "config.json", {"model_type": "siglip", "smoke": True})
    _write_json(
        output_dir / "tokenizer_config.json",
        {"model_max_length": 64, "tokenizer_class": "PreTrainedTokenizerFast"},
    )
    _write_json(
        output_dir / "special_tokens_map.json",
        {"unk_token": "<unk>", "pad_token": "<pad>"},
    )
    _write_tokenizer(output_dir / "tokenizer.json")

    return output_dir


def main() -> int:
    out = generate_bundle()
    print(f"Generated smoke ONNX bundle for {SMOKE_ONNX_MODEL_REPO!r} at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
