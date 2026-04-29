# Your first cached LLM

The killer use case: an LLM call wrapped in `mneme` so paraphrased queries hit a microsecond cache instead of a multi-second model.

## The pattern

```python
def classify(query: str) -> str:
    """Customer-support intent classifier with caching."""
    hit = cache.get(query, namespace="support")
    if hit is not None:
        return hit.response                    # <1 ms — exact or semantic

    intent = call_llm(query)                   # ~1-3 s — the expensive part
    cache.put(query, intent, namespace="support")
    return intent
```

That's it. Three lines wrap any deterministic LLM call. The only thing you change is **what `call_llm` does** — classify, summarize, route, score, etc.

## A concrete example

A real intent classifier against a local LLM (Ollama):

```python
import json

import numpy as np
import requests

from mneme import SemanticCache


class SentenceTransformersEmbedder:
    """See the 'Bring your own embedder' page for the full version."""
    # ... (omitted for brevity)


def call_llm(query: str) -> str:
    """Classify an inbound customer-support message into one of seven intents."""
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "nemotron-3-nano:latest",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. Reply with JSON "
                        '{"intent": "<one of: billing, technical, refund, account, '
                        'how_to, complaint, other>"}. No other text.'
                    ),
                },
                {"role": "user", "content": query},
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 40},
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    return json.loads(body["message"]["content"])["intent"]


with SemanticCache(
    path="cache.db",
    embedder=SentenceTransformersEmbedder(),
    similarity_threshold=0.65,            # tune for your embedder + corpus
) as cache:

    def classify(query: str) -> str:
        hit = cache.get(query, namespace="support")
        if hit is not None:
            return hit.response
        intent = call_llm(query)
        cache.put(query, intent, namespace="support")
        return intent

    print(classify("How do I reset my password?"))   # 1st call — ~500 ms (LLM)
    print(classify("How do I reset my password?"))   # 2nd call — <1 ms (exact)
    print(classify("I forgot my password"))          # paraphrase — ~25 ms (semantic)
```

For a fully runnable version with a 73-message corpus, a Flask UI, and live metrics, see the [showcase](../showcase.md).

## What you get

| Hit type | Typical latency | What's running |
| --- | --- | --- |
| **Exact** (Layer 1) | ~50 µs to 1 ms (depends on store) | One normalized SHA-256 + one store lookup |
| **Semantic** (Layer 2) | 1–10 ms at 100k entries | One embedding + one matvec across the in-memory matrix |
| **Miss** | Whatever your LLM costs (1–3 s) | LLM call + insert |

A reasonable cache-warming workload — say a chatbot that classifies 1000 customer-support messages with high paraphrase overlap — hits Layer 2 on 60–80% of requests after the first ~50. That's 600–800 LLM calls *avoided*, in time and tokens.

## Failure modes

- **Embedder fails on `get`.** Treated as a miss + WARNING. No exception bubbles up; the call falls through to `call_llm`. This is the right default — observability cares, but the user doesn't.
- **Embedder fails on `put`.** The exception propagates. You wanted to cache but couldn't; that's a real failure.
- **LLM fails.** Your problem, not `mneme`'s. The cache is uninvolved.
- **`call_llm` returns garbage** (e.g. `"[ERROR]"`, empty string). The default `Validator` rejects empty / `[ERROR]` / `[LLM Error]` strings on `put`. Plug in your own `validator=` for stricter rules. See [Confidence and validators](../concepts/confidence-and-validators.md).

## Calibrate before you ship

The default `similarity_threshold=0.85` is a starting point, not a recommendation. **Calibrate against your own embedder + paraphrase corpus** before going to production:

```python
from mneme.tools.calibrate import find_threshold

result = find_threshold(
    paraphrase_pairs=labeled_paraphrases,    # list[(query_a, query_b)]
    distractor_pairs=labeled_distractors,    # list[(query_a, query_b)] — should NOT match
    embedder=embedder,
    target_metric="f1",
    min_precision=0.95,                      # never accept below 95% precision
    vector_dtype="float16",                  # calibrate against the same dtype prod uses
)
print(result.threshold, result.precision, result.recall, result.f1)
```

Wire the result into your `SemanticCache(similarity_threshold=...)` call. See [Calibration](../guides/calibration.md) for the full walkthrough.

## Where to go next

- **[Layered cache](../concepts/layered-cache.md)** — the two-layer story in depth.
- **[Calibration](../guides/calibration.md)** — find the right threshold.
- **[Showcase](../showcase.md)** — the same pattern, packaged with a Flask UI and 73-message corpus.
- **[Multi-tenant](../concepts/multi-tenant.md)** — namespaces and per-tenant LRU quotas.
- **[Metrics](../guides/metrics.md)** — track hit rate and `LLM seconds saved` over time.
