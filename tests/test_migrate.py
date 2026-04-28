"""Phase-11 migration tests: reembed and areembed."""

from __future__ import annotations

from pathlib import Path

import pytest

from mneme import SemanticCache
from mneme.tools.migrate import areembed, reembed

from .fakes import FakeAsyncEmbedder, FakeEmbedder, HighDimEmbedder

# --- reembed (sync) ---


def test_reembed_returns_count_migrated(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        for i in range(5):
            cache.put(f"q{i}", f"r{i}")
    n = reembed(src, dst, e2)
    assert n == 5


def test_reembed_preserves_responses_and_metadata(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q1", "r1", metadata={"k": "v"})
        cache.put("q2", "r2", namespace="tenant_a")
    reembed(src, dst, e2)
    with SemanticCache(path=dst, embedder=e2) as restored:
        h1 = restored.get("q1")
        assert h1 is not None
        assert h1.response == "r1"
        assert h1.metadata == {"k": "v"}
        h2 = restored.get("q2", namespace="tenant_a")
        assert h2 is not None
        assert h2.response == "r2"


def test_reembed_changes_dim(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = HighDimEmbedder(dim=8, fingerprint="fp:8d")
    e2 = HighDimEmbedder(dim=16, fingerprint="fp:16d")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q", "r")
    reembed(src, dst, e2)
    with SemanticCache(path=dst, embedder=e2) as restored:
        assert restored.health().entries == 1
        # Dest cache uses the new embedder dim.
        assert restored.health().embedder_fingerprint_match is True


def test_reembed_preserves_namespaces(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q", "r", namespace="alpha")
        cache.put("q", "r", namespace="beta")
        cache.put("q", "r", namespace="gamma")
    reembed(src, dst, e2)
    with SemanticCache(path=dst, embedder=e2) as restored:
        assert restored.list_namespaces() == ["alpha", "beta", "gamma"]


def test_reembed_namespace_filter(tmp_path: Path):
    """Only-listed namespaces survive the migration."""
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q", "r", namespace="keep")
        cache.put("q", "r", namespace="drop")
    n = reembed(src, dst, e2, namespaces=["keep"])
    assert n == 1
    with SemanticCache(path=dst, embedder=e2) as restored:
        assert restored.list_namespaces() == ["keep"]


def test_reembed_does_not_modify_source(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q1", "r1")
    reembed(src, dst, e2)
    # Source still readable with the original embedder.
    with SemanticCache(path=src, embedder=e1) as restored_src:
        assert restored_src.get("q1") is not None


def test_reembed_missing_source_raises(tmp_path: Path):
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with pytest.raises(FileNotFoundError):
        reembed(tmp_path / "nope.db", tmp_path / "dst.db", e2)


def test_reembed_same_source_and_dest_raises(tmp_path: Path):
    src = tmp_path / "same.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q", "r")
    with pytest.raises(ValueError, match="differ"):
        reembed(src, src, e2)


def test_reembed_overwrites_existing_dest(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    # Prime dst with junk
    dst.write_bytes(b"junk")
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        cache.put("q", "r")
    n = reembed(src, dst, e2)
    assert n == 1


def test_reembed_progress_writes_to_stderr(tmp_path: Path, capsys):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e1 = FakeEmbedder(dim=8, fingerprint="fp:v1")
    e2 = FakeEmbedder(dim=8, fingerprint="fp:v2")
    with SemanticCache(path=src, embedder=e1) as cache:
        for i in range(3):
            cache.put(f"q{i}", "r")
    reembed(src, dst, e2, progress=True, batch_size=1)
    err = capsys.readouterr().err
    assert "reembed:" in err


def test_reembed_to_different_fingerprint(tmp_path: Path):
    """Critical: source's old fingerprint != dest's new fingerprint."""
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    e_old = FakeEmbedder(dim=8, fingerprint="openai:old:v1")
    e_new = FakeEmbedder(dim=8, fingerprint="openai:new:v2")
    with SemanticCache(path=src, embedder=e_old) as cache:
        cache.put("q", "r")
    reembed(src, dst, e_new)
    # Opening dest with OLD embedder should raise EmbedderMismatchError.
    from mneme import EmbedderMismatchError

    with pytest.raises(EmbedderMismatchError):
        SemanticCache(path=dst, embedder=e_old)
    # Opening dest with NEW embedder works.
    with SemanticCache(path=dst, embedder=e_new) as restored:
        assert restored.get("q") is not None


# --- areembed (async) ---


async def test_areembed_returns_count_migrated(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    sync_e = FakeEmbedder(dim=8, fingerprint="fp:v1")
    async_e = FakeAsyncEmbedder(dim=8, fingerprint="fp:async:v2")
    with SemanticCache(path=src, embedder=sync_e) as cache:
        for i in range(5):
            cache.put(f"q{i}", f"r{i}")
    n = await areembed(src, dst, async_e)
    assert n == 5


async def test_areembed_preserves_entries(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    sync_e = FakeEmbedder(dim=8, fingerprint="fp:v1")
    async_e = FakeAsyncEmbedder(dim=8, fingerprint="fp:async:v2")
    with SemanticCache(path=src, embedder=sync_e) as cache:
        cache.put("hi", "there", metadata={"k": "v"})
    await areembed(src, dst, async_e, concurrency=4)
    # Open dest synchronously by adapting the async embedder via to_sync_embedder.
    from mneme import to_sync_embedder

    sync_view = to_sync_embedder(async_e)
    with SemanticCache(path=dst, embedder=sync_view) as restored:
        h = restored.get("hi")
        assert h is not None
        assert h.response == "there"
        assert h.metadata == {"k": "v"}


async def test_areembed_namespace_filter(tmp_path: Path):
    src = tmp_path / "src.db"
    dst = tmp_path / "dst.db"
    sync_e = FakeEmbedder(dim=8, fingerprint="fp:v1")
    async_e = FakeAsyncEmbedder(dim=8, fingerprint="fp:async:v2")
    with SemanticCache(path=src, embedder=sync_e) as cache:
        cache.put("q", "r", namespace="keep")
        cache.put("q", "r", namespace="drop")
    n = await areembed(src, dst, async_e, namespaces=["keep"])
    assert n == 1


async def test_areembed_missing_source_raises(tmp_path: Path):
    async_e = FakeAsyncEmbedder(dim=8, fingerprint="fp:async")
    with pytest.raises(FileNotFoundError):
        await areembed(tmp_path / "nope.db", tmp_path / "dst.db", async_e)


async def test_areembed_same_source_and_dest_raises(tmp_path: Path):
    src = tmp_path / "same.db"
    sync_e = FakeEmbedder(dim=8, fingerprint="fp:v1")
    async_e = FakeAsyncEmbedder(dim=8, fingerprint="fp:async:v2")
    with SemanticCache(path=src, embedder=sync_e) as cache:
        cache.put("q", "r")
    with pytest.raises(ValueError, match="differ"):
        await areembed(src, src, async_e)
