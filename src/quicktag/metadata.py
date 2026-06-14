"""Write image metadata via bundled ExifTool."""

from __future__ import annotations

from pathlib import Path

from exiftool import ExifToolHelper

from quicktag.config import MetadataConfig


class MetadataWriter:
    """Apply keyword tags to image files using ExifTool."""

    def __init__(self, exiftool_path: Path, config: MetadataConfig) -> None:
        self._config = config
        self._et = ExifToolHelper(executable=str(exiftool_path))
        self._et.__enter__()

    def close(self) -> None:
        self._et.__exit__(None, None, None)

    def __enter__(self) -> MetadataWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write_tags(self, image_path: Path, tags: list[str]) -> None:
        """Write tags to configured metadata fields."""
        if not tags and not self._config.merge_existing:
            tag_values: list[str] = []
        elif self._config.merge_existing:
            tag_values = self._merge_existing(image_path, tags)
        else:
            tag_values = tags

        write_tags = {field: tag_values for field in self._config.fields}
        self._et.set_tags([str(image_path)], tags=write_tags)

    def _merge_existing(self, image_path: Path, new_tags: list[str]) -> list[str]:
        existing: set[str] = set()
        for record in self._et.get_tags([str(image_path)], tags=self._config.fields):
            for field in self._config.fields:
                value = record.get(field)
                if value is None:
                    continue
                if isinstance(value, list):
                    existing.update(str(v) for v in value)
                else:
                    existing.add(str(value))

        merged = list(existing)
        for tag in new_tags:
            if tag not in existing:
                merged.append(tag)
        return merged
