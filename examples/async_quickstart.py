"""Quickstart for ``AsyncSemanticCache``.

Same shape as ``examples/quickstart.py`` but with the async API. The
embedder here is async; for sync embedders, use ``to_async_embedder``
from ``mneme``.

Run:
    python examples/async_quickstart.py
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np

from mneme import AsyncSemanticCache, MemoryStore


class ToyAsyncEmbedder:
    dim = 32
    fingerprint = "toy:async-hash:v1"

    async def embed(self, text: str) -> np.ndarray:
        # Real async embedders await an HTTP call here. The toy version
        # just hashes synchronously inside an async function for shape parity.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


async def main() -> None:
    embedder = ToyAsyncEmbedder()
    async with AsyncSemanticCache(store=MemoryStore(), embedder=embedder) as cache:
        await cache.put("How do I reset my password?", "Forgot password on login.")
        await cache.put("Where is the billing dashboard?", "Settings -> Billing.")

        # 100 concurrent gets are fine — the cache holds an RLock around the
        # core for each operation but releases it across embedder awaits.
        async def lookup(query: str) -> str | None:
            hit = await cache.get(query)
            return None if hit is None else hit.response

        results = await asyncio.gather(
            *(lookup("How do I reset my password?") for _ in range(100))
        )
        assert all(r == "Forgot password on login." for r in results)
        print("100 concurrent hits, all returned the cached response.")

        # stats() and health() are sync on AsyncSemanticCache (cheap counters).
        s = cache.stats()
        print(f"hits_exact={s.hits_exact}  misses={s.misses}")


if __name__ == "__main__":
    asyncio.run(main())
