"""Shared pytest hooks and fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolate_hf_cache_env(monkeypatch: pytest.MonkeyPatch):
    """Prevent host HF cache env vars from affecting cache lookup tests."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)


def _markexpr_passed_on_cli(config: pytest.Config) -> bool:
    args = config.invocation_params.args
    for index, arg in enumerate(args):
        if arg in ("-m", "--markers"):
            return True
        if arg.startswith("-m") and arg != "-m":
            return True
    return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _markexpr_passed_on_cli(config):
        return
    skip = pytest.mark.skip(
        reason="integration tests opt-in; use: pytest -m integration"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
