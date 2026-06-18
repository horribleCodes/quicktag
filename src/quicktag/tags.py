"""Load tags.yaml and build SigLIP prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TagDefinition:
    label: str
    prompt: str
    custom_prompt: bool = False
    override: bool = False


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
            tags.append(TagDefinition(
                label=label,
                prompt=label.lower(),
                custom_prompt=False,
                override=False
            ))
        elif isinstance(entry, dict):
            label = str(entry.get("label", "")).strip()
            if not label:
                raise ValueError(f"Tag entry missing label: {entry!r}")
            custom_prompt = "prompt" in entry
            override = entry.get("override", False)
            prompt = str(entry.get("prompt", label)).strip().lower()
            tags.append(TagDefinition(
                label=label,
                prompt=prompt,
                custom_prompt=custom_prompt,
                override=override
            ))
        else:
            raise ValueError(f"Invalid tag entry: {entry!r}")

    if not tags:
        raise ValueError(f"No valid tags found in {path}")

    return tags


def apply_prompt_template(
    tags: list[TagDefinition],
    template: str,
    *,
    prompt_overrides_template: bool = False,
) -> list[TagDefinition]:
    """Format each tag's classification prompt using a config template."""
    result: list[TagDefinition] = []
    for tag in tags:
        if (prompt_overrides_template or tag.override) and tag.custom_prompt:
            result.append(tag)
            continue
        try:
            prompt = template.format(tag=tag.label, label=tag.label, prompt=tag.prompt)
        except KeyError as exc:
            raise ValueError(
                f"Invalid prompt_template placeholder: {exc}. "
                "Use {{tag}}, {{label}}, or {{prompt}}."
            ) from exc
        result.append(TagDefinition(label=tag.label, prompt=prompt.strip().lower()))
    return result
