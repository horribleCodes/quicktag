"""Tests for ONNX SigLIP tagger."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from quicktag.onnx_tagger import OnnxSigLIPTagger, _preprocess_image, _sigmoid
from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition


def test_sigmoid():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(10.0) > 0.99


def test_preprocess_image_shape():
    from PIL import Image

    image = Image.new("RGB", (100, 50), color=(128, 64, 32))
    config = {
        "size": {"height": 224, "width": 224},
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    tensor = _preprocess_image(image, config)
    assert tensor.shape == (1, 3, 224, 224)


def test_score_with_mock_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    mock_output = MagicMock()
    mock_output.name = "logits_per_image"
    mock_session = MagicMock()
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.array([[1.0, -1.0]], dtype=np.float32)]

    monkeypatch.setattr(
        "quicktag.onnx_tagger.resolve_onnx_model_dir",
        lambda *_args, **_kwargs: model_dir,
    )
    monkeypatch.setattr(
        "quicktag.onnx_tagger.ort.InferenceSession",
        lambda *_args, **_kwargs: mock_session,
    )

    tagger = OnnxSigLIPTagger("horrible/siglip2-base-patch16-224", tmp_path, local_files_only=True)
    tags = [
        TagDefinition(label="cat", prompt="a photo of a cat"),
        TagDefinition(label="dog", prompt="a photo of a dog"),
    ]
    image_path = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(image_path)

    scored = tagger.score(image_path, tags)
    assert len(scored) == 2
    assert all(isinstance(item, ScoredTag) for item in scored)
    assert scored[0].score >= scored[1].score
