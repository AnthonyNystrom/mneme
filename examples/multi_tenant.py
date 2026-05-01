"""Multi-tenant ``SemanticCache`` with per-namespace LRU quotas.

Each tenant gets its own namespace and its own entry cap. When tenant_a's
cap is hit, only tenant_a's oldest entries are evicted; tenant_b is
unaffected. Global caps are enforced separately.

Run:
    python examples/multi_tenant.py
"""

from __future__ import annotations

import hashlib

import numpy as np

from mneme import MemoryStore, SemanticCache


class ToyEmbedder:
    dim = 32
    fingerprint = "toy:hash:v1"

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def main() -> None:
    cache = SemanticCache(
        store=MemoryStore(),
        embedder=ToyEmbedder(),
        namespace_quotas={
            "tenant_a": 5,  # tenant_a is on the small plan
            "tenant_b": 50,  # tenant_b is on a larger plan
        },
    )
    try:
        # Fill tenant_a past its quota.
        for i in range(10):
            cache.put(f"a-query-{i}", f"a-response-{i}", namespace="tenant_a")

        # Tenant_b is independent.
        for i in range(20):
            cache.put(f"b-query-{i}", f"b-response-{i}", namespace="tenant_b")

        a_count = cache.stats(namespace="tenant_a").entries
        b_count = cache.stats(namespace="tenant_b").entries
        print(f"tenant_a entries: {a_count} (capped at 5)")
        print(f"tenant_b entries: {b_count} (cap is 50)")

        # The most recent tenant_a writes survive.
        hit = cache.get("a-query-9", namespace="tenant_a")
        assert hit is not None
        assert hit.response == "a-response-9"

        # The earliest tenant_a writes were evicted to make room.
        miss = cache.get("a-query-0", namespace="tenant_a")
        assert miss is None
        print("oldest tenant_a entries were evicted; newest survived")

        # Cross-tenant isolation: tenant_a's hash collisions can't leak
        # into tenant_b.
        cache.put("shared-query", "FROM-A", namespace="tenant_a")
        cache.put("shared-query", "FROM-B", namespace="tenant_b")
        assert cache.get("shared-query", namespace="tenant_a").response == "FROM-A"  # type: ignore[union-attr]
        assert cache.get("shared-query", namespace="tenant_b").response == "FROM-B"  # type: ignore[union-attr]
        print("namespaces are isolated")
    finally:
        cache.close()


if __name__ == "__main__":
    main()
