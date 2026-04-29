"""Use case: cache RAG retrieval results.

A cross-encoder reranker is 10-100x the cost of a cache lookup. Wrap your
retrieval pipeline so paraphrased questions reuse the same top-k chunks.

This script simulates the pattern with a fake retrieval function. The
'cache' is a real ``SemanticCache``; replace ``simulate_retrieval`` with
your real vector-db query + reranker call.

Run:
    python examples/use_cases/rag_retrieval.py
"""

from __future__ import annotations

import json
import time

from _embedder import TokenBagEmbedder

from mneme import MemoryStore, SemanticCache


def simulate_retrieval(question: str) -> list[dict]:
    """Pretend this is your vector DB + cross-encoder reranker.

    Costs ~200ms of wall time so the cache savings are visible.
    """
    time.sleep(0.2)
    # Returns the top-3 chunks for the question. In real life this is
    # a real similarity search + rerank.
    return [
        {"id": f"doc-{i}", "score": 0.9 - i * 0.1, "text": f"snippet about {question[:30]}..."}
        for i in range(3)
    ]


def cached_retrieve(cache: SemanticCache, question: str) -> tuple[list[dict], str]:
    """Cache-aware retrieval. Returns (chunks, layer)."""
    hit = cache.get(question, namespace="rag")
    if hit is not None:
        return json.loads(hit.response), hit.layer
    chunks = simulate_retrieval(question)
    cache.put(question, json.dumps(chunks), namespace="rag")
    return chunks, "miss"


def main() -> None:
    embedder = TokenBagEmbedder()
    with SemanticCache(store=MemoryStore(), embedder=embedder, similarity_threshold=0.4) as cache:
        questions = [
            "How do I configure rate limiting?",
            "How do I configure rate limiting?",            # exact dup
            "How can I configure rate limiting?",           # near-paraphrase, shares words
            "How do I set up rate limiting?",               # near-paraphrase
            "How do I deploy to production?",               # different intent
        ]
        for q in questions:
            t0 = time.monotonic()
            chunks, layer = cached_retrieve(cache, q)
            elapsed_ms = (time.monotonic() - t0) * 1000
            top_id = chunks[0]["id"]
            print(f"  {layer:8s}  {elapsed_ms:7.1f} ms  top={top_id}  q={q!r}")

        s = cache.stats()
        print(
            f"\n{s.entries} unique retrievals stored; "
            f"{s.hits_exact + s.hits_semantic}/{s.hits_exact + s.hits_semantic + s.misses} cache hits"
        )


if __name__ == "__main__":
    main()
