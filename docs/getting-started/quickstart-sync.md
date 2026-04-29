# Sync quickstart

The five-minute version: install `mneme`, write an embedder, wrap an LLM call.

## Install

```bash
pip install mneme-cache
```

That's the whole runtime dependency story for this quickstart. NumPy comes along.

## A toy embedder

`mneme` does not bundle an embedder. You provide one that returns a 1-D `float32` `numpy.ndarray` of length `dim` with a stable `fingerprint` string. For real models see [Bring your own embedder](bring-your-own-embedder.md).

For this quickstart, a deterministic toy embedder that produces stable vectors from a hash:

```python
import hashlib
import numpy as np


class ToyEmbedder:
    """Deterministic 32-dim hash embedder. Not for production use."""

    dim = 32
    fingerprint = "toy:hash:v1"

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
```

The full file lives at [`examples/quickstart.py`](https://github.com/anthonynystrom/mneme/blob/main/examples/quickstart.py).

## Open a cache

Two ways: pass `path=` for a SQLite-backed cache (durable), or pass an explicit `store=` (e.g. `MemoryStore` for tests). Exactly one of the two.

```python
from mneme import MemoryStore, SemanticCache

with SemanticCache(store=MemoryStore(), embedder=ToyEmbedder()) as cache:
    cache.put("How do I reset my password?", "Click 'Forgot password' on login.")

    hit = cache.get("How do I reset my password?")     # exact-match
    assert hit is not None
    assert hit.layer == "exact"
```

`SemanticCache` is a context manager - `__exit__` calls `close()` which flushes counters to the store and releases connections.

## Layered hits

Submit the same query twice and you get **`exact`** hits (Layer 1, normalized hash, O(1) `dict` lookup). Submit a paraphrase and you get a **`semantic`** hit (Layer 2, cosine similarity over the embedded vectors).

```python
hit = cache.get("Where do I reset my password?")        # paraphrase
assert hit is not None
assert hit.layer == "semantic"
print(hit.similarity)         # cosine score, e.g. 0.79
print(hit.confidence)         # confidence score (default: 24h half-life)
```

The `similarity_threshold` (default `0.85`) controls how close two queries must be to count as a semantic match. Calibrate it for your embedder + corpus - see [Calibration](../guides/calibration.md).

## Persistence

Switch from `MemoryStore` to a SQLite file and the cache survives a process restart:

```python
with SemanticCache(path="cache.db", embedder=ToyEmbedder()) as cache:
    cache.put("How do I cancel?", "Settings → Subscription → Cancel.")
# Process exits.

# New process:
with SemanticCache(path="cache.db", embedder=ToyEmbedder()) as cache:
    hit = cache.get("How do I cancel?")
    assert hit is not None  # still there
```

The cache validates the embedder's `fingerprint` on open and refuses to mix incompatible vectors with [`EmbedderMismatchError`](../reference/exceptions.md). Same for dimension changes.

## Stats and observability

```python
s = cache.stats()
print(f"entries={s.entries}  L1={s.hits_exact}  L2={s.hits_semantic}  miss={s.misses}")
```

Every query emits a `MetricsHook` event. Plug in your own callback or use the [shipped adapters](../reference/adapters.md) for Prometheus / OpenTelemetry.

## Where to go next

- **[Async quickstart](quickstart-async.md)** - same shape, async API.
- **[Bring your own embedder](bring-your-own-embedder.md)** - OpenAI, sentence-transformers, Bedrock, Ollama.
- **[Your first cached LLM](your-first-cached-llm.md)** - the killer use case end-to-end.
- **[Layered cache](../concepts/layered-cache.md)** - what's actually happening on each `get`.
