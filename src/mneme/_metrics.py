"""Per-namespace counters + metrics-hook dispatch.

Counters tracked per PRD §15.1:

- ``hits_exact`` — layer-1 (normalized hash) hits
- ``hits_semantic`` — layer-2 (cosine similarity) hits
- ``misses`` — both layers missed
- ``evictions`` — entries removed by LRU
- ``expirations`` — entries removed by TTL

Hook events per PRD §15.2:

- ``hit`` ``{layer, similarity, confidence, age_seconds, namespace}``
- ``miss`` ``{reason, namespace}``
- ``eviction`` ``{count, namespace}``
- ``expired`` ``{count, namespace}``

Hook exceptions are caught and logged at WARNING (PRD §15.2). Writes from
inside a hook are forbidden by §30 invariant #12 — that is the user's
responsibility, not the library's.
"""

from __future__ import annotations

import logging
from typing import Any

from ._types import HitLayer, MetricsHook

logger = logging.getLogger("mneme.metrics")


_COUNTER_NAMES = (
    "hits_exact",
    "hits_semantic",
    "misses",
    "evictions",
    "expirations",
)


class Counters:
    """Per-namespace counter store.

    Internally a flat ``dict[(namespace, name) -> int]``; ``get_namespace``
    and ``aggregate`` provide the views the cache layer uses.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], int] = {}

    def increment(self, namespace: str, name: str, delta: int = 1) -> None:
        if delta <= 0:
            return
        key = (namespace, name)
        self._counts[key] = self._counts.get(key, 0) + delta

    def get(self, namespace: str, name: str) -> int:
        return self._counts.get((namespace, name), 0)

    def get_namespace(self, namespace: str) -> dict[str, int]:
        """All counters for a single namespace (zero-filled for the
        documented set)."""
        out = dict.fromkeys(_COUNTER_NAMES, 0)
        for (ns, name), count in self._counts.items():
            if ns == namespace:
                out[name] = count
        return out

    def aggregate(self) -> dict[str, int]:
        """Sum each counter across all namespaces (zero-filled)."""
        out = dict.fromkeys(_COUNTER_NAMES, 0)
        for (_ns, name), count in self._counts.items():
            out[name] = out.get(name, 0) + count
        return out

    def clear_namespace(self, namespace: str) -> None:
        for key in list(self._counts):
            if key[0] == namespace:
                del self._counts[key]

    def serialize(self) -> dict[str, int]:
        """Flat ``"ns:name" -> int`` mapping for persistence via store meta."""
        return {f"{ns}:{name}": v for (ns, name), v in self._counts.items()}

    def restore(self, data: dict[str, int]) -> None:
        """Inverse of ``serialize``. Idempotent: clears existing first."""
        self._counts.clear()
        for key, value in data.items():
            ns, _, name = key.partition(":")
            if name:
                self._counts[(ns, name)] = int(value)


class MetricsDispatcher:
    """Bundles counters with optional hook fan-out.

    The hook signature matches PRD §8.7: ``Callable[[str, dict], None]``.
    ``None`` means counters are still tracked but no external dispatch
    happens.
    """

    def __init__(self, hook: MetricsHook | None = None) -> None:
        self._hook = hook
        self.counters = Counters()

    @property
    def hook(self) -> MetricsHook | None:
        return self._hook

    def set_hook(self, hook: MetricsHook | None) -> None:
        self._hook = hook

    # --- Public emission API ---

    def emit_hit(
        self,
        namespace: str,
        layer: HitLayer,
        similarity: float,
        confidence: float,
        age_seconds: int,
    ) -> None:
        if layer == "exact":
            self.counters.increment(namespace, "hits_exact")
        else:
            self.counters.increment(namespace, "hits_semantic")
        self._fire(
            "hit",
            {
                "namespace": namespace,
                "layer": layer,
                "similarity": float(similarity),
                "confidence": float(confidence),
                "age_seconds": int(age_seconds),
            },
        )

    def emit_miss(self, namespace: str, reason: str) -> None:
        self.counters.increment(namespace, "misses")
        self._fire("miss", {"namespace": namespace, "reason": reason})

    def emit_eviction(self, namespace: str, count: int) -> None:
        if count <= 0:
            return
        self.counters.increment(namespace, "evictions", count)
        self._fire("eviction", {"namespace": namespace, "count": int(count)})

    def emit_expired(self, namespace: str, count: int) -> None:
        if count <= 0:
            return
        self.counters.increment(namespace, "expirations", count)
        self._fire("expired", {"namespace": namespace, "count": int(count)})

    # --- Internal ---

    def _fire(self, event: str, attrs: dict[str, Any]) -> None:
        if self._hook is None:
            return
        try:
            self._hook(event, attrs)
        except Exception:
            logger.warning("metrics hook raised on event %r", event, exc_info=True)


__all__ = ["Counters", "MetricsDispatcher"]
