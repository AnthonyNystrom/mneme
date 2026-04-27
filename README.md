# mneme

A layered semantic cache for LLM applications.

`mneme` (Greek: μνήμη, "memory"; pronounced *NEE-mee*) is an embeddable, in-process Python library that caches LLM completions across paraphrased queries. It pairs an exact-match layer (normalized query hash) with a semantic-match layer (cosine similarity over L2-normalized embeddings) and persists durably to a single SQLite file by default.

```python
from mneme import SemanticCache
from my_embedders import OpenAIEmbedder

embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)

with SemanticCache(path="cache.db", embedder=embedder) as cache:
    hit = cache.get("How do I reset my password?")
    if hit is None:
        response = call_my_llm("How do I reset my password?")
        cache.put("How do I reset my password?", response)
    else:
        response = hit.response
```

## Features

- **One required runtime dependency: NumPy.** Optional extras for hnswlib, redis, postgres, prometheus, opentelemetry.
- **Layered cache:** O(1) exact match, then cosine similarity over an in-memory matrix.
- **Sync and async** APIs with parity.
- **Four `Store` backends shipped:** Memory, SQLite (default), Redis, Postgres. `Store` Protocol for custom backends.
- **Two `Index` backends:** NumPy (default, fast to ~500k entries) and hnswlib (opt-in, scales past 1M).
- **Three vector dtypes:** `float32` (default), `float16`, `int8` for memory-constrained deployments.
- **Three multi-process modes:** `single`, `stale-tolerant`, `mmap-shared`.
- **Multi-tenant** via namespaces with per-namespace quotas.
- **Calibration tooling** (Python API + CLI) for tuning thresholds.
- **Checkpoint export/import** for backup and environment promotion.
- **Re-embed migration tool** when the embedder changes.
- **Prometheus and OpenTelemetry** metrics adapters shipped.
- **Strict typing** with `py.typed`.

## Status

Pre-release.

## Install

```bash
pip install mneme                       # core (NumPy only)
pip install "mneme[hnsw]"               # add hnswlib for >500k entries
pip install "mneme[redis]"              # RedisStore
pip install "mneme[postgres]"           # PostgresStore
pip install "mneme[prometheus,otel]"    # metrics adapters
pip install "mneme[all]"                # everything
```

## License

Apache 2.0. See [LICENSE](LICENSE).
