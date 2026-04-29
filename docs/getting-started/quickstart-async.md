# Async quickstart

Same shape as the [sync quickstart](quickstart-sync.md), but with `AsyncSemanticCache`. Use this when your embedder is async (network call to OpenAI, Bedrock, Ollama, etc.) or you're inside an `asyncio` event loop.

## Hello, async

```python
import asyncio
import hashlib

import numpy as np

from mneme import AsyncSemanticCache, MemoryStore


class ToyAsyncEmbedder:
    dim = 32
    fingerprint = "toy:async-hash:v1"

    async def embed(self, text: str) -> np.ndarray:
        # Real async embedders await an HTTP call here. The toy version is
        # synchronous-inside-an-async-function for shape parity.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


async def main() -> None:
    async with AsyncSemanticCache(store=MemoryStore(), embedder=ToyAsyncEmbedder()) as cache:
        await cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
        hit = await cache.get("How do I reset my password?")
        assert hit is not None and hit.layer == "exact"


asyncio.run(main())
```

The full file is at [`examples/async_quickstart.py`](https://github.com/anystrom/mneme/blob/main/examples/async_quickstart.py).

## Concurrency

The cache holds an internal `RLock` for each operation but releases it across embedder awaits. So 100 concurrent `get` calls against a populated cache complete in parallel:

```python
async with AsyncSemanticCache(store=MemoryStore(), embedder=embedder) as cache:
    for i in range(20):
        await cache.put(f"q{i}", f"r{i}")

    async def lookup(query):
        hit = await cache.get(query)
        return hit.response if hit else None

    results = await asyncio.gather(*(lookup(f"q{i % 20}") for i in range(100)))
```

## Sync ↔ async embedder adapters

Sometimes you have a sync embedder (sentence-transformers) but want an async cache, or vice versa:

```python
from mneme import to_async_embedder, to_sync_embedder

async_embedder = to_async_embedder(my_sync_embedder)   # wraps with asyncio.to_thread
sync_embedder = to_sync_embedder(my_async_embedder)    # runs an event loop per call
```

`to_sync_embedder` is for cases where you have an async embedder API but want to use the sync `SemanticCache`. It's slower per call (one event-loop spin per `embed`); prefer the async cache when the embedder is async.

## Sync vs async - what differs

| | `SemanticCache` | `AsyncSemanticCache` |
| --- | --- | --- |
| Embedder is awaited | n/a | yes (drops the cache lock) |
| Store work is awaited | no | yes (`asyncio.to_thread`) |
| `stats()`, `health()`, `list_namespaces()`, `clear_namespace()`, `clear()`, `set_similarity_threshold()` | sync | sync (cheap, no I/O) |
| `__enter__` / `__exit__` | sync | async (`__aenter__` / `__aexit__`) |
| Counter / locking semantics | identical | identical |

The two share the same conceptual surface, the same exceptions, the same metrics events. Pick whichever matches your call site.

## Where to go next

- **[Bring your own embedder](bring-your-own-embedder.md)** - production embedder patterns (sync + async).
- **[Your first cached LLM](your-first-cached-llm.md)** - the canonical async use case.
- **[Performance tuning](../guides/performance-tuning.md)** - perf knobs for async workloads.
