"""Load and validate config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    input: str = "input"
    output: str = "output"


@dataclass
class ModelConfig:
    name: str = "horrible/siglip2-base-patch16-224"
    cache_dir: str = ".cache/huggingface"


@dataclass
class ScoringConfig:
    min_score: float = 0.05
    top_k: int | None = 10
    top_p: float | None = 0.9
    prompt_template: str | None = None
    prompt_overrides_template: bool = False


@dataclass
class MetadataConfig:
    fields: list[str] = field(default_factory=lambda: ["Keywords", "XMP:Subject"])
    merge_existing: bool = False


@dataclass
class ProcessingConfig:
    extensions: list[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp", "tiff", "tif"]
    )
    preserve_timestamps: bool = True
    on_error: str = "skip"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    tags_file: str = "tags.yaml"
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    paths_raw = raw.get("paths", {})
    model_raw = raw.get("model", {})
    scoring_raw = raw.get("scoring", {})
    metadata_raw = raw.get("metadata", {})
    processing_raw = raw.get("processing", {})

    on_error = processing_raw.get("on_error", "skip")
    if on_error not in {"skip", "fail"}:
        raise ValueError(
            f"Invalid processing.on_error: {on_error!r} (expected 'skip' or 'fail')"
        )

    return AppConfig(
        paths=PathsConfig(
            input=paths_raw.get("input", "input"),
            output=paths_raw.get("output", "output"),
        ),
        model=ModelConfig(
            name=model_raw.get("name", "horrible/siglip2-base-patch16-224"),
            cache_dir=model_raw.get("cache_dir", ".cache/huggingface"),
        ),
        scoring=ScoringConfig(
            min_score=float(scoring_raw.get("min_score", 0.05)),
            top_k=scoring_raw.get("top_k", 10),
            top_p=scoring_raw.get("top_p", 0.9),
            prompt_template=scoring_raw.get("prompt_template"),
            prompt_overrides_template=bool(
                scoring_raw.get("prompt_overrides_template", False)
            ),
        ),
        metadata=MetadataConfig(
            fields=list(metadata_raw.get("fields", ["Keywords", "XMP:Subject"])),
            merge_existing=bool(metadata_raw.get("merge_existing", False)),
        ),
        tags_file=str(raw.get("tags_file", "tags.yaml")),
        processing=ProcessingConfig(
            extensions=[
                ext.lower().lstrip(".") for ext in processing_raw.get("extensions", [])
            ]
            or ["jpg", "jpeg", "png", "webp", "tiff", "tif"],
            preserve_timestamps=bool(processing_raw.get("preserve_timestamps", True)),
            on_error=on_error,
        ),
    )
