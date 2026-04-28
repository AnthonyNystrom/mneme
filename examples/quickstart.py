"""Quickstart for the sync ``SemanticCache``.

A toy embedder is used so this script runs without external services or
GPU dependencies. Replace ``ToyEmbedder`` with a real one (sentence-
transformers, OpenAI, Bedrock, etc. — see examples/reference_embedders/)
in your own code.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mneme import MemoryStore, SemanticCache


class ToyEmbedder:
    """Deterministic 32-dim hash embedder. Not for production use."""

    dim = 32
    fingerprint = "toy:hash:v1"

    def embed(self, text: str) -> np.ndarray:
        # Hash to bytes, unpack as int8, normalize. This is *not* a real
        # embedding — it's just enough that identical strings produce
        # identical vectors and similar strings often produce similar ones.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat to fill 32 dims.
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def main() -> None:
    embedder = ToyEmbedder()

    # In-memory store: state is lost on process exit.
    with SemanticCache(store=MemoryStore(), embedder=embedder) as cache:
        cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
        cache.put("Where is the billing dashboard?", "Settings -> Billing.")

        # Exact-match hit (Layer 1).
        hit = cache.get("How do I reset my password?")
        assert hit is not None
        assert hit.layer == "exact"
        print(f"L1 hit:  similarity={hit.similarity:.3f}  response={hit.response!r}")

        # Miss for an unrelated query.
        miss = cache.get("What is the weather today?")
        print(f"miss:    {miss}")

        # Stats are namespace-aware.
        s = cache.stats()
        print(
            f"stats:   entries={s.entries}  hits_exact={s.hits_exact}  "
            f"hits_semantic={s.hits_semantic}  misses={s.misses}"
        )

    # SQLite-backed cache: durable across process restarts. Same API.
    db = Path("cache.db")
    if db.exists():
        db.unlink()
    with SemanticCache(path=db, embedder=embedder) as cache:
        cache.put("How do I cancel my subscription?", "Settings -> Subscription.")
    with SemanticCache(path=db, embedder=embedder) as cache:
        hit = cache.get("How do I cancel my subscription?")
        assert hit is not None
        print(f"durable: response={hit.response!r}")
    db.unlink()


if __name__ == "__main__":
    main()
