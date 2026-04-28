"""Shared pytest fixtures and CLI flags."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest


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
    skip_perf = pytest.mark.skip(reason="performance benchmark; pass --run-perf to run")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)


# --- Redis / Postgres connection fixtures ---
#
# Resolution order:
# 1. If MNEME_REDIS_URL / MNEME_PG_URL is set, use it directly (treats local
#    install or any reachable service as integration target).
# 2. Else if RUN_REDIS_INTEGRATION=1 / RUN_POSTGRES_INTEGRATION=1, spin up a
#    testcontainers-managed Docker container for the session.
# 3. Else skip cleanly: each test that depends on the URL fixture is skipped.


def _redis_url_from_env() -> str | None:
    return os.environ.get("MNEME_REDIS_URL")


def _pg_url_from_env() -> str | None:
    return os.environ.get("MNEME_PG_URL")


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """Return a Redis connection URL or skip cleanly."""
    direct = _redis_url_from_env()
    if direct is not None:
        yield direct
        return
    if os.environ.get("RUN_REDIS_INTEGRATION") != "1":
        pytest.skip("Redis integration disabled (set MNEME_REDIS_URL or RUN_REDIS_INTEGRATION=1)")
    try:
        from testcontainers.redis import RedisContainer  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("testcontainers not installed")
    with RedisContainer("redis:7-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """Return a Postgres DSN or skip cleanly."""
    direct = _pg_url_from_env()
    if direct is not None:
        yield direct
        return
    if os.environ.get("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip(
            "Postgres integration disabled (set MNEME_PG_URL or RUN_POSTGRES_INTEGRATION=1)"
        )
    try:
        from testcontainers.postgres import PostgresContainer  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("testcontainers not installed")
    with PostgresContainer("postgres:16") as container:
        yield container.get_connection_url()


# Per-test isolation helpers: each test should construct stores with a unique
# key_prefix (Redis) or schema (Postgres) so tests don't collide.


@pytest.fixture
def redis_prefix() -> str:
    return f"mneme_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def pg_schema() -> str:
    return f"mneme_test_{uuid.uuid4().hex[:12]}"
