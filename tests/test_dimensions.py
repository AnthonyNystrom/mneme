"""Phase-7 dimension tests: end-to-end across 384/768/1024/1536/3072 dims.

 acceptance criterion #18: dimension tests pass across all the listed
embedder dimensions in production use (sentence-transformers, BGE, OpenAI,
Cohere, Voyage, Bedrock).
"""

from __future__ import annotations

import numpy as np
import pytest

from mneme import EmbedderDimensionError, MemoryStore, SemanticCache

from .fakes import HighDimEmbedder

# Dimensions covering the vendors enumerated in .
_DIMS = [384, 768, 1024, 1536, 3072]


@pytest.mark.parametrize("dim", _DIMS)
def test_open_close_at_dim(dim: int):
    cache = SemanticCache(store=MemoryStore(), embedder=HighDimEmbedder(dim=dim))
    try:
        assert cache.health().entries == 0
    finally:
        cache.close()


@pytest.mark.parametrize("dim", _DIMS)
def test_put_get_round_trip_at_dim(dim: int):
    with SemanticCache(store=MemoryStore(), embedder=HighDimEmbedder(dim=dim)) as cache:
        cache.put("hello world", "the answer")
        hit = cache.get("hello world")
        assert hit is not None
        assert hit.layer == "exact"
        assert hit.response == "the answer"


@pytest.mark.parametrize("dim", _DIMS)
def test_memory_estimate_matches_dim(dim: int):
    """``memory_bytes_estimate`` should reflect entries * dim * dtype_size."""
    with SemanticCache(store=MemoryStore(), embedder=HighDimEmbedder(dim=dim)) as cache:
        cache.put("a", "r")
        cache.put("b", "r")
        s = cache.stats()
        # float32 = 4 bytes; 2 entries * dim * 4.
        assert s.memory_bytes_estimate == 2 * dim * 4


def test_dim_mismatch_raises_clearly():
    """Pass a wrong-shape embedding to put -> EmbedderDimensionError."""
    with SemanticCache(store=MemoryStore(), embedder=HighDimEmbedder(dim=768)) as cache:
        bad = np.zeros(384, dtype=np.float32)
        with pytest.raises(EmbedderDimensionError, match="dim"):
            cache.put("q", "r", embedding=bad)


def test_reopen_with_different_dim_raises(tmp_path):
    """A cache built at dim=N must reject reopening with a different-dim embedder.

    Use the SAME fingerprint so the fingerprint check passes and the dim
    check is the one that fires.
    """
    fp = "stable:fp"
    e1 = HighDimEmbedder(dim=768, fingerprint=fp)
    e2 = HighDimEmbedder(dim=1024, fingerprint=fp)
    path = tmp_path / "c.db"
    cache = SemanticCache(path=path, embedder=e1)
    cache.put("q", "r")
    cache.close()
    with pytest.raises(EmbedderDimensionError):
        SemanticCache(path=path, embedder=e2)
