"""Use case: cache translations.

Translation APIs charge per character. Multiple phrasings of the same
idea reuse one paid translation. Each (source_lang, target_lang) pair
gets its own namespace so a French cache hit can't leak into a German
request.

Run:
    python examples/use_cases/translation.py
"""

from __future__ import annotations

import time

from _embedder import TokenBagEmbedder

from mneme import MemoryStore, SemanticCache

# Pretend translation API. Real one would be DeepL / Google Translate / etc.
_FAKE_TRANSLATIONS = {
    ("en", "fr"): {
        "How do I reset my password?": "Comment réinitialiser mon mot de passe ?",
        "I forgot my password": "J'ai oublié mon mot de passe",
        "I want a refund": "Je veux un remboursement",
    },
    ("en", "de"): {
        "How do I reset my password?": "Wie setze ich mein Passwort zurück?",
        "I want a refund": "Ich möchte eine Rückerstattung",
    },
}


def call_translation_api(text: str, source: str, target: str) -> str:
    """Pretend network call to a paid translation API."""
    time.sleep(0.15)            # simulate network + billing
    pair = _FAKE_TRANSLATIONS.get((source, target), {})
    return pair.get(text, f"[{target}] " + text)


def translate(cache: SemanticCache, text: str, source: str, target: str) -> tuple[str, str]:
    """Cache-aware translation. Returns (translation, layer)."""
    namespace = f"translate:{source}-{target}"
    hit = cache.get(text, namespace=namespace)
    if hit is not None:
        return hit.response, hit.layer
    translated = call_translation_api(text, source, target)
    cache.put(text, translated, namespace=namespace)
    return translated, "miss"


def main() -> None:
    with SemanticCache(store=MemoryStore(), embedder=TokenBagEmbedder(), similarity_threshold=0.55) as cache:
        sources = [
            ("How do I reset my password?", "en", "fr"),
            ("How do I reset my password?", "en", "fr"),    # exact
            ("How can I reset my password?", "en", "fr"),   # paraphrase, semantic hit
            ("How do I reset my password?", "en", "de"),    # different target → miss
            ("How do I reset my password?", "en", "de"),    # exact in de
            ("I want a refund", "en", "fr"),
            ("I want a refund", "en", "de"),
        ]
        for text, src, tgt in sources:
            t0 = time.monotonic()
            translated, layer = translate(cache, text, src, tgt)
            elapsed_ms = (time.monotonic() - t0) * 1000
            print(f"  {layer:8s}  {elapsed_ms:7.1f} ms  [{src}→{tgt}]  {text!r:50s} → {translated!r}")

        # Per-pair breakdown
        for ns in cache.list_namespaces():
            print(f"  namespace {ns}: {cache.stats(namespace=ns).entries} cached translations")


if __name__ == "__main__":
    main()
