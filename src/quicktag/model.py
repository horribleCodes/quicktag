"""SigLIP2 zero-shot image classification wrapper."""

from __future__ import annotations

from pathlib import Path

from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition


class SigLIPTagger:
    """CPU-only SigLIP2 tag scorer."""

    def __init__(self, model_name: str, cache_dir: str | Path) -> None:
        from transformers import pipeline

        self._prompt_to_label = {}
        self._pipe = pipeline(
            task="zero-shot-image-classification",
            model=model_name,
            device=-1,
            model_kwargs={"cache_dir": str(cache_dir)},
        )

    def score(self, image_path: Path, tags: list[TagDefinition]) -> list[ScoredTag]:
        """Score an image against all candidate tag prompts."""
        prompts = [tag.prompt for tag in tags]
        self._prompt_to_label = {tag.prompt: tag.label for tag in tags}

        results = self._pipe(str(image_path), candidate_labels=prompts)

        scored: list[ScoredTag] = []
        for item in results:
            prompt = item["label"]
            label = self._prompt_to_label.get(prompt, prompt)
            scored.append(ScoredTag(label=label, score=float(item["score"])))

        return scored
