"""SigLIP2 zero-shot image classification via ONNX Runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

from quicktag.image_io import load_rgb_image
from quicktag.paths import resolve_onnx_model_dir
from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition

DEFAULT_MAX_LENGTH = 64
_ONNX_ALLOW_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.onnx",
    "onnx/*",
]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _load_preprocessor_config(model_dir: Path) -> dict:
    config_path = model_dir / "preprocessor_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing preprocessor config: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _preprocess_image(image: Image.Image, config: dict) -> np.ndarray:
    size = config.get("size", {})
    height = int(size.get("height", 224))
    width = int(size.get("width", 224))
    resample = Image.Resampling.BILINEAR
    resized = image.resize((width, height), resample=resample)
    pixels = np.asarray(resized, dtype=np.float32)

    if config.get("do_rescale", True):
        pixels *= float(config.get("rescale_factor", 1.0 / 255.0))

    if config.get("do_normalize", True):
        mean = np.array(config.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        std = np.array(config.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        pixels = (pixels - mean) / std

    pixels = np.transpose(pixels, (2, 0, 1))
    return np.expand_dims(pixels, axis=0)


def _encode_prompts(tokenizer: Tokenizer, prompts: list[str], max_length: int) -> np.ndarray:
    encoded = tokenizer.encode_batch(prompts)
    batch = np.zeros((len(prompts), max_length), dtype=np.int64)
    for row, item in enumerate(encoded):
        ids = item.ids[:max_length]
        batch[row, : len(ids)] = ids
    return batch


def _find_model_onnx(model_dir: Path) -> Path:
    direct = model_dir / "model.onnx"
    if direct.is_file():
        return direct
    nested = model_dir / "onnx" / "model.onnx"
    if nested.is_file():
        return nested
    onnx_files = sorted(model_dir.glob("**/*.onnx"))
    if len(onnx_files) == 1:
        return onnx_files[0]
    raise FileNotFoundError(f"No ONNX model found under {model_dir}")


def _ensure_onnx_snapshot(
    model_name: str,
    hub_cache_dir: str | Path,
    *,
    local_files_only: bool,
) -> Path:
    cache_home = Path(hub_cache_dir)
    cached = resolve_onnx_model_dir(model_name, cache_home)
    if cached is not None:
        return cached

    export_dir = cache_home / "onnx-export" / model_name.replace("/", "--")
    if export_dir.is_dir():
        try:
            _find_model_onnx(export_dir)
            return export_dir
        except FileNotFoundError:
            pass

    if local_files_only:
        raise FileNotFoundError(
            f"ONNX model {model_name!r} not found in cache: {hub_cache_dir}"
        )

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_name,
        cache_dir=str(cache_home),
        local_files_only=False,
        allow_patterns=_ONNX_ALLOW_PATTERNS,
    )

    cached = resolve_onnx_model_dir(model_name, cache_home)
    if cached is not None:
        return cached

    exported = _try_export_onnx_model(model_name, cache_home)
    if exported is not None:
        return exported

    raise FileNotFoundError(
        f"Could not locate ONNX weights for {model_name!r} under {hub_cache_dir}. "
        "Run scripts/export_onnx_model.py with dev dependencies installed."
    )


def _try_export_onnx_model(model_name: str, cache_home: Path) -> Path | None:
    try:
        import importlib.util
        import sys

        script = Path(__file__).resolve().parents[2] / "scripts" / "export_onnx_model.py"
        spec = importlib.util.spec_from_file_location("export_onnx_model", script)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        export_dir = cache_home / "onnx-export" / model_name.replace("/", "--")
        module.export_onnx_model(export_dir, model_name=model_name)
        return export_dir
    except Exception:
        return None


class OnnxSigLIPTagger:
    """CPU-only SigLIP2 tag scorer using ONNX Runtime."""

    def __init__(
        self,
        model_name: str,
        hub_cache_dir: str | Path,
        *,
        local_files_only: bool = False,
    ) -> None:
        model_dir = _ensure_onnx_snapshot(
            model_name,
            hub_cache_dir,
            local_files_only=local_files_only,
        )
        model_onnx = _find_model_onnx(model_dir)
        self._session = ort.InferenceSession(
            str(model_onnx),
            providers=["CPUExecutionProvider"],
        )
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Missing tokenizer.json under {model_dir}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._preprocessor = _load_preprocessor_config(model_dir)
        self._max_length = DEFAULT_MAX_LENGTH

    def score(self, image_path: Path, tags: list[TagDefinition]) -> list[ScoredTag]:
        """Score an image against all candidate tag prompts."""
        prompts = [tag.prompt for tag in tags]
        prompt_to_label = {tag.prompt: tag.label for tag in tags}

        pixel_values = _preprocess_image(load_rgb_image(image_path), self._preprocessor)
        input_ids = _encode_prompts(self._tokenizer, prompts, self._max_length)

        outputs = self._session.run(None, {"pixel_values": pixel_values, "input_ids": input_ids})
        output_meta = {item.name: index for index, item in enumerate(self._session.get_outputs())}
        logits = outputs[output_meta["logits_per_image"]]
        scores = [_sigmoid(float(logits[0, index])) for index in range(len(prompts))]

        scored = [
            ScoredTag(label=prompt_to_label[prompt], score=score)
            for prompt, score in zip(prompts, scores, strict=True)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored
