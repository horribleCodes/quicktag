"""Tests for SigLIP image loading and scoring."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from quicktag.model import SigLIPTagger, _load_rgb_image
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
        rgb = _load_rgb_image(png_path)

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

    mock_pipe = MagicMock(
        return_value=[{"label": "a photo of a cat", "score": 0.5}]
    )
    monkeypatch.setattr(
        SigLIPTagger,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "_pipe", mock_pipe),
    )

    tagger = SigLIPTagger("google/siglip2-base-patch16-224", tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scored = tagger.score(png_path, tags)

    assert scored == [ScoredTag(label="cat", score=0.5)]
    assert mock_pipe.call_count == 1
    passed_image = mock_pipe.call_args.args[0]
    assert isinstance(passed_image, Image.Image)
    assert passed_image.mode == "RGB"
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []
