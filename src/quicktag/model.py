"""SigLIP2 zero-shot image classification wrapper."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition


def _load_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB, handling palette transparency correctly."""
    with Image.open(path) as img:
        if img.mode == "P" and isinstance(img.info.get("transparency"), bytes):
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")


class SigLIPTagger:
    """CPU-only SigLIP2 tag scorer."""

    def __init__(
        self,
        model_name: str,
        hub_cache_dir: str | Path,
        *,
        local_files_only: bool = False,
    ) -> None:
        from transformers import pipeline

        self._prompt_to_label = {}
        pipe_kwargs: dict[str, object] = {
            "task": "zero-shot-image-classification",
            "model": model_name,
            "device": -1,
            "model_kwargs": {"cache_dir": str(hub_cache_dir)},
        }
        if local_files_only:
            pipe_kwargs["local_files_only"] = True
        self._pipe = pipeline(**pipe_kwargs)

    def score(self, image_path: Path, tags: list[TagDefinition]) -> list[ScoredTag]:
        """Score an image against all candidate tag prompts."""
        prompts = [tag.prompt for tag in tags]
        self._prompt_to_label = {tag.prompt: tag.label for tag in tags}

        results = self._pipe(_load_rgb_image(image_path), candidate_labels=prompts)

        scored: list[ScoredTag] = []
        for item in results:
            prompt = item["label"]
            label = self._prompt_to_label.get(prompt, prompt)
            scored.append(ScoredTag(label=label, score=float(item["score"])))

        return scored
