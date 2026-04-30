"""Cache-aware classifier wrapping mneme + Nemotron.

This is the place where the cache pays off. ``classify()`` calls the cache
first; on a hit it returns immediately. On a miss it invokes Nemotron, then
writes the result back so the next paraphrase is a hit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from nemotron_client import NemotronClient

from mneme import SemanticCache

Layer = Literal["exact", "semantic", "miss"]


@dataclass(frozen=True)
class ClassifyResult:
    query: str
    intent: str
    layer: Layer
    similarity: float | None    # None for exact / miss
    confidence: float | None    # None for miss
    age_seconds: int | None     # None for miss
    namespace: str
    latency_ms: float
    llm_seconds: float | None   # None on cache hit


class CachedClassifier:
    """Wires together a ``SemanticCache`` and a ``NemotronClient``.

    Thread-safe: ``SemanticCache`` is internally ``RLock``-guarded.
    """

    def __init__(self, cache: SemanticCache, llm: NemotronClient) -> None:
        self._cache = cache
        self._llm = llm

    def classify(
        self, query: str, *, namespace: str = "support", bypass: bool = False
    ) -> ClassifyResult:
        t0 = time.monotonic()
        hit = self._cache.get(query, namespace=namespace, bypass=bypass)
        if hit is not None:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            return ClassifyResult(
                query=query,
                intent=hit.response,
                layer=hit.layer,
                similarity=float(hit.similarity) if hit.layer == "semantic" else None,
                confidence=float(hit.confidence),
                age_seconds=int(hit.age_seconds),
                namespace=namespace,
                latency_ms=elapsed_ms,
                llm_seconds=None,
            )

        # Cache miss → real LLM call.
        llm_resp = self._llm.classify(query)
        self._cache.put(query, llm_resp.intent, namespace=namespace)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return ClassifyResult(
            query=query,
            intent=llm_resp.intent,
            layer="miss",
            similarity=None,
            confidence=None,
            age_seconds=None,
            namespace=namespace,
            latency_ms=elapsed_ms,
            llm_seconds=llm_resp.duration_sec,
        )

    def classify_uncached(self, query: str) -> ClassifyResult:
        """Bypass the cache entirely. Used by the side-by-side comparison
        on the Try-it page."""
        t0 = time.monotonic()
        llm_resp = self._llm.classify(query)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return ClassifyResult(
            query=query,
            intent=llm_resp.intent,
            layer="miss",
            similarity=None,
            confidence=None,
            age_seconds=None,
            namespace="(bypassed)",
            latency_ms=elapsed_ms,
            llm_seconds=llm_resp.duration_sec,
        )
