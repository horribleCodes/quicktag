"""SigLIP2 zero-shot image classification wrapper."""

from __future__ import annotations

from pathlib import Path

from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition


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
        model_kwargs: dict[str, object] = {"cache_dir": str(hub_cache_dir)}
        if local_files_only:
            model_kwargs["local_files_only"] = True
        self._pipe = pipeline(
            task="zero-shot-image-classification",
            model=model_name,
            device=-1,
            model_kwargs=model_kwargs,
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
