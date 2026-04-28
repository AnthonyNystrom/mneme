"""Test embedders.

Per PRD §17.2:
- ``FakeEmbedder``: deterministic vectors keyed by query hash. Sync.
- ``FakeAsyncEmbedder``: async equivalent.
- ``ParaphraseEmbedder``: shared keywords -> high cosine similarity. Used for
  semantic-match tests without a real model.
- ``FlakyEmbedder``: raises after the Nth call (for embedder-failure paths).
- ``SlowEmbedder``: configurable per-call latency (for async cancellation).
- ``HighDimEmbedder``: parameterizable dim (for ``test_dimensions``).
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np
import numpy.typing as npt


def _seeded_unit_vector(seed: int, dim: int) -> npt.NDArray[np.float32]:
    """Deterministic L2-normalized vector for a given (seed, dim)."""
    rng = np.random.default_rng(seed % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def _query_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class FakeEmbedder:
    """Deterministic, fast, sync. Identical text -> identical vector."""

    def __init__(self, dim: int = 8, fingerprint: str = "fake:v1") -> None:
        self._dim = dim
        self._fingerprint = fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return _seeded_unit_vector(_query_seed(text), self._dim)


class FakeAsyncEmbedder:
    """Async counterpart of FakeEmbedder."""

    def __init__(self, dim: int = 8, fingerprint: str = "fake:async:v1") -> None:
        self._dim = dim
        self._fingerprint = fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        return _seeded_unit_vector(_query_seed(text), self._dim)


class ParaphraseEmbedder:
    """Vectors close in cosine when queries share words.

    Each lowercase word contributes a seeded basis vector; the sum is
    L2-normalized. Sentences with overlapping vocabulary score high.
    """

    def __init__(self, dim: int = 16, fingerprint: str = "paraphrase:v1") -> None:
        self._dim = dim
        self._fingerprint = fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        words = [w.lower() for w in text.split() if w.strip()]
        if not words:
            return np.zeros(self._dim, dtype=np.float32)
        v = np.zeros(self._dim, dtype=np.float32)
        for w in words:
            v += _seeded_unit_vector(_query_seed(w), self._dim)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v.astype(np.float32, copy=False)


class FlakyEmbedder:
    """Raises ``RuntimeError`` on the Nth call (1-indexed)."""

    def __init__(self, dim: int = 8, fail_on: int = 1, fingerprint: str = "flaky:v1") -> None:
        self._dim = dim
        self._fail_on = fail_on
        self._fingerprint = fingerprint
        self._calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError(f"FlakyEmbedder: forced failure on call {self._calls}")
        return _seeded_unit_vector(_query_seed(text), self._dim)


class SlowEmbedder:
    """Sync embedder with a configurable per-call delay."""

    def __init__(
        self, dim: int = 8, delay_seconds: float = 0.01, fingerprint: str = "slow:v1"
    ) -> None:
        self._dim = dim
        self._delay = delay_seconds
        self._fingerprint = fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        import time

        time.sleep(self._delay)
        return _seeded_unit_vector(_query_seed(text), self._dim)


class SlowAsyncEmbedder:
    """Async embedder with a configurable per-call delay (for cancellation tests)."""

    def __init__(
        self,
        dim: int = 8,
        delay_seconds: float = 0.05,
        fingerprint: str = "slow:async:v1",
    ) -> None:
        self._dim = dim
        self._delay = delay_seconds
        self._fingerprint = fingerprint

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def embed(self, text: str) -> npt.NDArray[np.float32]:
        await asyncio.sleep(self._delay)
        return _seeded_unit_vector(_query_seed(text), self._dim)


class HighDimEmbedder:
    """Deterministic embedder parameterizable to any dim (for dimension tests)."""

    def __init__(self, dim: int, fingerprint: str | None = None) -> None:
        self._dim = dim
        self._fingerprint = fingerprint or f"highdim:v1:{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return _seeded_unit_vector(_query_seed(text), self._dim)
