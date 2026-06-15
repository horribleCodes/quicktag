"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB, handling palette transparency correctly."""
    with Image.open(path) as img:
        if img.mode == "P" and isinstance(img.info.get("transparency"), bytes):
            return img.convert("RGBA").convert("RGB")
        return img.convert("RGB")
