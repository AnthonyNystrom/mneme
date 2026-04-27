"""Shared pytest fixtures and CLI flags."""

from __future__ import annotations


def pytest_addoption(parser):  # type: ignore[no-untyped-def]
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="Run performance benchmarks (tests marked @pytest.mark.perf).",
    )


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    if config.getoption("--run-perf"):
        return
    import pytest

    skip_perf = pytest.mark.skip(reason="performance benchmark; pass --run-perf to run")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
