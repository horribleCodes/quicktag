"""Hosted SigLIP2 ONNX bundle constants and download helpers."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ONNX_MODEL_REPO = "horrible/siglip2-base-patch16-224"

_ONNX_BUNDLE_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.onnx",
]


def download_onnx_bundle(
    output_dir: Path,
    *,
    repo_id: str = DEFAULT_ONNX_MODEL_REPO,
    local_files_only: bool = False,
) -> Path:
    """Download ONNX bundle files into *output_dir*; return path to model.onnx."""
    from huggingface_hub import snapshot_download

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        local_files_only=local_files_only,
        allow_patterns=_ONNX_BUNDLE_FILES,
    )

    model_path = output_dir / "model.onnx"
    if not model_path.is_file():
        nested = output_dir / "onnx" / "model.onnx"
        if nested.is_file():
            model_path = nested
        else:
            raise FileNotFoundError(
                f"Expected model.onnx under {output_dir} after downloading {repo_id!r}"
            )

    return model_path
