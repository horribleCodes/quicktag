"""End-to-end pipeline tests using stub tagger (no model download)."""

from pathlib import Path

import pytest

from quicktag.config import load_config
from quicktag.paths import HuggingFaceCacheLayout
from quicktag.pipeline import run_pipeline
from quicktag.scoring import ScoredTag
from quicktag.tags import TagDefinition, load_tags

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StubTagger:
    """Minimal tagger stand-in for pipeline integration tests."""

    def score(self, image_path: Path, tags: list[TagDefinition]) -> list[ScoredTag]:
        return [ScoredTag(label=tags[0].label, score=0.9)]


class FakeMetadataWriter:
    """MetadataWriter stand-in that records written tags."""

    instances: list["FakeMetadataWriter"] = []

    def __init__(self, exiftool_path: Path, config: object) -> None:
        self.exiftool_path = exiftool_path
        self.config = config
        self.written: list[tuple[Path, list[str]]] = []
        FakeMetadataWriter.instances.append(self)

    def __enter__(self) -> FakeMetadataWriter:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def write_tags(self, image_path: Path, tags: list[str]) -> None:
        self.written.append((image_path, tags))


@pytest.fixture
def install_dir(tmp_path: Path) -> Path:
    (tmp_path / "config.yaml").write_text(
        """
paths:
  input: input
  output: output
scoring:
  min_score: 0.05
  top_k: 10
""",
        encoding="utf-8",
    )
    (tmp_path / "tags.yaml").write_text(
        "tags:\n  - cat\n  - dog\n",
        encoding="utf-8",
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "tiny.png").write_bytes((FIXTURES / "tiny.png").read_bytes())
    return tmp_path


@pytest.fixture
def hf_cache(install_dir: Path) -> HuggingFaceCacheLayout:
    cache_home = install_dir / ".cache" / "huggingface"
    hub_dir = cache_home / "hub"
    hub_dir.mkdir(parents=True)
    return HuggingFaceCacheLayout(
        primary_home=cache_home,
        load_home=cache_home,
        hub_dir=hub_dir,
        source="primary",
        local_files_only=False,
    )


def test_pipeline_processes_image_with_stub_tagger(
    install_dir: Path,
    hf_cache: HuggingFaceCacheLayout,
    monkeypatch: pytest.MonkeyPatch,
):
    FakeMetadataWriter.instances.clear()
    monkeypatch.setattr("quicktag.pipeline.MetadataWriter", FakeMetadataWriter)
    monkeypatch.setattr(
        "quicktag.pipeline.get_exiftool_path",
        lambda _install: Path("/usr/bin/exiftool"),
    )

    config = load_config(install_dir / "config.yaml")
    tags = load_tags(install_dir / "tags.yaml")
    summary = run_pipeline(
        config,
        install_dir,
        hf_cache,
        tags,
        tagger=StubTagger(),
    )

    output_file = install_dir / "output" / "tiny.png"
    assert output_file.is_file()
    assert summary.processed == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(FakeMetadataWriter.instances) == 1
    assert FakeMetadataWriter.instances[0].written == [(output_file, ["cat"])]
