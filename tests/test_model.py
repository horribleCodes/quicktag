"""Tests for image loading and ONNX tag scoring."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from quicktag.image_io import load_rgb_image
from quicktag.onnx_tagger import OnnxSigLIPTagger
from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition


def _write_p_mode_byte_transparency_png(path: Path) -> None:
    """Create a P-mode PNG whose transparency is stored as a byte string."""
    img = Image.new("P", (2, 2))
    img.putpalette(
        [255, 0, 0, 0, 255, 0, 0, 0, 255] + [0, 0, 0] * (256 - 3)
    )
    img.putpixel((0, 0), 0)
    img.putpixel((1, 0), 1)
    img.putpixel((0, 1), 2)
    img.putpixel((1, 1), 0)
    img.info["transparency"] = bytes([255, 128, 64])
    img.save(path)


def test_load_rgb_image_handles_p_mode_byte_transparency(tmp_path: Path):
    png_path = tmp_path / "palette.png"
    _write_p_mode_byte_transparency_png(png_path)

    with Image.open(png_path) as opened:
        assert opened.mode == "P"
        assert isinstance(opened.info.get("transparency"), bytes)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rgb = load_rgb_image(png_path)

    assert rgb.mode == "RGB"
    assert rgb.size == (2, 2)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []


def test_load_rgb_image_naive_convert_warns(tmp_path: Path):
    png_path = tmp_path / "palette.png"
    _write_p_mode_byte_transparency_png(png_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(png_path) as img:
            img.convert("RGB")

    assert any(
        "Palette images with Transparency" in str(w.message) for w in caught
    )


def test_score_passes_pil_image_without_user_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    png_path = tmp_path / "palette.png"
    _write_p_mode_byte_transparency_png(png_path)
    tags = [TagDefinition(label="cat", prompt="a photo of a cat")]

    mock_session = MagicMock()
    mock_output = MagicMock()
    mock_output.name = "logits_per_image"
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[0.0]], dtype=np.float32)]

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text(
        '{"size": {"height": 224, "width": 224}}',
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,'
        '"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,'
        '"model":{"type":"WordLevel","vocab":{"a":0,"photo":1,"cat":2,"[UNK]":3},"unk_token":"[UNK]"}}',
        encoding="utf-8",
    )
    (model_dir / "model.onnx").write_bytes(b"onnx")

    monkeypatch.setattr(
        "quicktag.onnx_tagger.resolve_onnx_model_dir",
        lambda *_args, **_kwargs: model_dir,
    )
    monkeypatch.setattr(
        "quicktag.onnx_tagger.ort.InferenceSession",
        lambda *_args, **_kwargs: mock_session,
    )

    tagger = OnnxSigLIPTagger("horrible/siglip2-base-patch16-224", tmp_path, local_files_only=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scored = tagger.score(png_path, tags)

    assert scored == [ScoredTag(label="cat", score=0.5)]
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []
