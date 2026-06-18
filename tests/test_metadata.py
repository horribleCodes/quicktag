"""Tests for Unicode-safe metadata writing via ExifTool."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quicktag.config import MetadataConfig
from quicktag.metadata import MetadataWriter, _CHARSET_PARAMS

FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNICODE_TAGS = ["Müller", "Straße", "café"]
KEYWORD_FIELDS = ["Keywords", "XMP:Subject"]


class FakeExifToolHelper:
    """Records ExifToolHelper construction and method calls."""

    instances: list[FakeExifToolHelper] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.set_tags_calls: list[tuple[object, object, object]] = []
        self.get_tags_calls: list[tuple[object, object, object]] = []
        FakeExifToolHelper.instances.append(self)

    def __enter__(self) -> FakeExifToolHelper:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def set_tags(self, files: object, tags: object, params: object = None) -> None:
        self.set_tags_calls.append((files, tags, params))

    def get_tags(
        self, files: object, tags: object, params: object = None
    ) -> list[dict]:
        self.get_tags_calls.append((files, tags, params))
        return [{}]


def _get_metadata(exiftool_path: Path, config: MetadataConfig, image_path: Path):
    with MetadataWriter(exiftool_path, config) as reader:
        records = reader._et.get_tags(
            [str(image_path)],
            tags=config.fields,
            params=_CHARSET_PARAMS,
        )
    return records


def _compare_keywords(
    records,
    expected,
):
    record = records[0]
    keywords = record.get("Keywords") or record.get("IPTC:Keywords")
    expected_set = set(expected)
    assert set(keywords or []) == expected_set
    assert set(record.get("XMP:Subject") or []) == expected_set


def test_metadata_writer_passes_utf8_charset_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    FakeExifToolHelper.instances.clear()
    monkeypatch.setattr("quicktag.metadata.ExifToolHelper", FakeExifToolHelper)

    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"fake")
    exiftool_path = tmp_path / "exiftool"
    exiftool_path.write_text("", encoding="utf-8")
    config = MetadataConfig(fields=KEYWORD_FIELDS, merge_existing=True)

    with MetadataWriter(exiftool_path, config) as writer:
        writer.write_tags(image_path, UNICODE_TAGS)

    assert len(FakeExifToolHelper.instances) == 1
    assert FakeExifToolHelper.instances[0].kwargs == {
        "executable": str(exiftool_path),
        "encoding": "utf-8",
    }

    helper = FakeExifToolHelper.instances[0]
    assert len(helper.get_tags_calls) == 1
    assert helper.get_tags_calls[0][2] == _CHARSET_PARAMS

    assert len(helper.set_tags_calls) == 1
    files, tags, params = helper.set_tags_calls[0]
    assert files == [str(image_path)]
    assert tags == {"Keywords": UNICODE_TAGS, "XMP:Subject": UNICODE_TAGS}
    assert params == ["-overwrite_original", *_CHARSET_PARAMS]


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not on PATH")
def test_metadata_unicode_round_trip(tmp_path: Path):
    image_path = tmp_path / "unicode.png"
    image_path.write_bytes((FIXTURES / "tiny.png").read_bytes())
    exiftool_path = Path(shutil.which("exiftool"))  # type: ignore[arg-type]
    config = MetadataConfig(fields=KEYWORD_FIELDS, merge_existing=False)

    with MetadataWriter(exiftool_path, config) as writer:
        writer.write_tags(image_path, UNICODE_TAGS)

    records = _get_metadata(
        exiftool_path=exiftool_path, config=config, image_path=image_path
    )

    expected = UNICODE_TAGS
    _compare_keywords(records, expected)


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not on PATH")
def test_metadata_tool_overwrites_prior_tags(tmp_path: Path):
    image_path = tmp_path / "merge.png"
    image_path.write_bytes((FIXTURES / "tiny.png").read_bytes())
    exiftool_path = Path(shutil.which("exiftool"))  # type: ignore[arg-type]
    config = MetadataConfig(fields=KEYWORD_FIELDS, merge_existing=False)
    initial_tags = ["alpha", "beta"]
    new_tags = ["gamma", "delta"]

    with MetadataWriter(exiftool_path, config) as writer:
        writer.write_tags(image_path, initial_tags)

    with MetadataWriter(exiftool_path, config) as writer:
        writer.write_tags(image_path, new_tags)

    records = _get_metadata(
        exiftool_path=exiftool_path,
        config=MetadataConfig(fields=KEYWORD_FIELDS),
        image_path=image_path,
    )

    expected = {"gamma", "delta"}
    _compare_keywords(records, expected)


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not on PATH")
def test_metadata_merge_existing_preserves_prior_tags(tmp_path: Path):
    image_path = tmp_path / "merge.png"
    image_path.write_bytes((FIXTURES / "tiny.png").read_bytes())
    exiftool_path = Path(shutil.which("exiftool"))  # type: ignore[arg-type]
    initial_tags = ["alpha", "beta"]
    new_tags = ["beta", "gamma"]

    with MetadataWriter(
        exiftool_path, MetadataConfig(fields=KEYWORD_FIELDS, merge_existing=False)
    ) as writer:
        writer.write_tags(image_path, initial_tags)

    with MetadataWriter(
        exiftool_path, MetadataConfig(fields=KEYWORD_FIELDS, merge_existing=True)
    ) as writer:
        writer.write_tags(image_path, new_tags)

    records = _get_metadata(
        exiftool_path=exiftool_path,
        config=MetadataConfig(fields=KEYWORD_FIELDS),
        image_path=image_path,
    )

    expected = {"alpha", "beta", "gamma"}
    _compare_keywords(records, expected)
