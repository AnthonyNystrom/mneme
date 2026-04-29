"""Use case: cache classification labels (any classifier, not just LLM).

Wraps a tiny rule-based classifier in mneme. Production code would
swap the classifier for sklearn / fastText / a trained transformer —
the cache layer is indifferent to what produces the label.

The point: paraphrased inputs reuse the cached label without re-running
the model. Useful for any expensive classifier under high paraphrase
load.

Run:
    python examples/use_cases/classification.py
"""

from __future__ import annotations

import time

from _embedder import TokenBagEmbedder

from mneme import MemoryStore, SemanticCache


def slow_classifier(text: str) -> str:
    """Pretend this is a 100ms sklearn pipeline / transformer / etc."""
    time.sleep(0.1)
    text_lower = text.lower()
    if any(w in text_lower for w in ("password", "login")):
        return "account"
    if "refund" in text_lower:
        return "billing"
    if "cancel" in text_lower or "subscription" in text_lower:
        return "billing"
    if "joke" in text_lower or "hello" in text_lower or "hi " in text_lower:
        return "smalltalk"
    if "crash" in text_lower or "error" in text_lower or "broken" in text_lower:
        return "technical"
    return "other"


def classify(cache: SemanticCache, text: str) -> tuple[str, str, float]:
    """Cache-aware classification. Returns (label, layer, latency_ms)."""
    t0 = time.monotonic()
    hit = cache.get(text, namespace="moderation")
    if hit is not None:
        return hit.response, hit.layer, (time.monotonic() - t0) * 1000

    label = slow_classifier(text)
    cache.put(text, label, namespace="moderation")
    return label, "miss", (time.monotonic() - t0) * 1000


def main() -> None:
    inputs = [
        "How do I reset my password?",
        "How can I reset my password?",              # paraphrase, shares words
        "How do I reset my password again?",         # paraphrase
        "I want a refund for my order",
        "I want a refund for my purchase",           # paraphrase
        "I want to cancel my subscription",          # different intent
        "I want to cancel my subscription now",      # paraphrase of above
        "Tell me a joke",                             # smalltalk
        "Tell me a funny joke",                       # paraphrase
    ]

    with SemanticCache(store=MemoryStore(), embedder=TokenBagEmbedder(), similarity_threshold=0.4) as cache:
        for text in inputs:
            label, layer, ms = classify(cache, text)
            print(f"  {layer:8s}  {ms:7.1f} ms  {label:10s}  {text!r}")

        s = cache.stats()
        hits = s.hits_exact + s.hits_semantic
        total = hits + s.misses
        print(f"\n{hits}/{total} cache hits ({hits/total:.0%}); ~{0.1 * hits:.1f}s of classifier work avoided")


if __name__ == "__main__":
    main()
