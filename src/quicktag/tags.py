"""Load tags.yaml and build SigLIP prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TagDefinition:
    label: str
    prompt: str


def load_tags(path: Path) -> list[TagDefinition]:
    """Load tag definitions from YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Tags file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    entries = raw.get("tags")
    if not entries:
        raise ValueError(f"No tags defined in {path}")

    tags: list[TagDefinition] = []
    for entry in entries:
        if isinstance(entry, str):
            label = entry.strip()
            if not label:
                continue
            tags.append(TagDefinition(label=label, prompt=label.lower()))
        elif isinstance(entry, dict):
            label = str(entry.get("label", "")).strip()
            if not label:
                raise ValueError(f"Tag entry missing label: {entry!r}")
            prompt = str(entry.get("prompt", label)).strip().lower()
            tags.append(TagDefinition(label=label, prompt=prompt))
        else:
            raise ValueError(f"Invalid tag entry: {entry!r}")

    if not tags:
        raise ValueError(f"No valid tags found in {path}")

    return tags
