"""Batch image tagging pipeline."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from quicktag.config import AppConfig
from quicktag.metadata import MetadataWriter
from quicktag.model import SigLIPTagger
from quicktag.paths import HuggingFaceCacheLayout, get_exiftool_path, resolve_path
from quicktag.scoring import ScoredTag, select_tags
from quicktag.tags import TagDefinition

logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    source: Path
    destination: Path
    tags: list[ScoredTag] = field(default_factory=list)
    error: str | None = None


@dataclass
class PipelineSummary:
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[FileResult] = field(default_factory=list)


def discover_images(input_dir: Path, extensions: list[str]) -> list[Path]:
    """List image files in input_dir (non-recursive)."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ext_set = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    files = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in ext_set
    ]
    return files


def run_pipeline(
    config: AppConfig,
    install_dir: Path,
    hf_cache: HuggingFaceCacheLayout,
    tags: list[TagDefinition],
    tagger: SigLIPTagger | None = None,
) -> PipelineSummary:
    """Process all images from input to output with tagging."""
    input_dir = resolve_path(install_dir, config.paths.input)
    output_dir = resolve_path(install_dir, config.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(input_dir, config.processing.extensions)
    summary = PipelineSummary()

    if not images:
        logger.warning("No images found in %s", input_dir)
        return summary

    owns_tagger = tagger is None
    if owns_tagger:
        if hf_cache.local_files_only:
            logger.info("Loading model %s from cache...", config.model.name)
        else:
            logger.info("Loading model %s (first run may download weights)...", config.model.name)
        tagger = SigLIPTagger(
            config.model.name,
            hf_cache.hub_dir,
            local_files_only=hf_cache.local_files_only,
        )

    exiftool_path = get_exiftool_path(install_dir)

    try:
        with MetadataWriter(exiftool_path, config.metadata) as writer:
            for image_path in images:
                summary.processed += 1
                dest_path = output_dir / image_path.name
                result = FileResult(source=image_path, destination=dest_path)

                try:
                    _copy_image(image_path, dest_path, config.processing.preserve_timestamps)
                    scored = tagger.score(image_path, tags)
                    selected = select_tags(scored, config.scoring)
                    result.tags = selected

                    writer.write_tags(dest_path, [item.label for item in selected])
                    summary.succeeded += 1
                    logger.info(
                        "%s -> %s",
                        image_path.name,
                        ", ".join(f"{t.label} ({t.score:.3f})" for t in selected) or "(no tags)",
                    )
                except Exception as exc:
                    result.error = str(exc)
                    summary.failed += 1
                    logger.error("Failed to process %s: %s", image_path.name, exc)
                    if config.processing.on_error == "fail":
                        raise

                summary.results.append(result)
    finally:
        if owns_tagger:
            del tagger

    return summary


def _copy_image(source: Path, dest: Path, preserve_timestamps: bool) -> None:
    if preserve_timestamps:
        shutil.copy2(source, dest)
    else:
        shutil.copy(source, dest)
