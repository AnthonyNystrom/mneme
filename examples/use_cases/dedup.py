"""Use case: semantic deduplication.

Different shape than the LLM cache: read ``Hit.similarity``, ignore the
response. The cache becomes a 'have I seen something this close before?'
detector. The threshold here is stricter than for LLM caching because a
false-positive in dedup means dropping real content.

Run:
    python examples/use_cases/dedup.py
"""

from __future__ import annotations

from _embedder import TokenBagEmbedder

from mneme import MemoryStore, SemanticCache

SIMILAR_ENOUGH = 0.85  # stricter than the LLM-caching default


def is_duplicate(cache: SemanticCache, content: str) -> tuple[bool, float | None]:
    """Returns (is_duplicate, similarity) for a piece of content."""
    hit = cache.get(content, namespace="dedup")
    if hit is not None and hit.layer == "exact":
        return True, 1.0
    if hit is not None and hit.layer == "semantic" and hit.similarity >= SIMILAR_ENOUGH:
        return True, float(hit.similarity)
    # Marker response: any non-empty string. The default Validator rejects
    # empty strings on Layer-2 candidates, so we store a sentinel instead.
    cache.put(content, "seen", namespace="dedup")
    return False, None if hit is None else float(hit.similarity)


def main() -> None:
    # Imagine these are ingested news articles or feedback messages.
    incoming = [
        "Apple announces new MacBook Pro with M5 chip",
        "Apple announces new MacBook Pro with M5 chip",  # exact dup
        "Apple announces new MacBook Pro with M5 chip processor",  # near dup
        "Google launches new Pixel phone",  # different
        "Google launches a new Pixel phone today",  # near dup of above
        "Microsoft releases Windows 12",  # different
        "Apple announces new iPad with M5 chip",  # related but different topic
    ]

    with SemanticCache(
        store=MemoryStore(), embedder=TokenBagEmbedder(), similarity_threshold=0.5
    ) as cache:
        kept, dropped = 0, 0
        for content in incoming:
            dup, sim = is_duplicate(cache, content)
            if dup:
                dropped += 1
                print(f"  DROP (sim={sim:.3f})  {content!r}")
            else:
                kept += 1
                sim_str = f"sim={sim:.3f}" if sim is not None else "novel"
                print(f"  KEEP ({sim_str:12s})  {content!r}")

        print(f"\n{kept} kept, {dropped} dropped as near-duplicates")


if __name__ == "__main__":
    main()
