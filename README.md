# mneme

A layered semantic cache for LLMs and any expensive function with embeddable input.

`mneme` (Greek: μνήμη, "memory"; pronounced *NEE-mee*) is an embeddable, in-process Python library for **semantic memoization**: cache an expensive function once, return the cached result whenever a later input means the same thing. LLM completions are the canonical use case; the same machinery covers RAG retrievals, translations, classifications, deduplication, and agent memory. It pairs an exact-match layer (normalized query hash) with a semantic-match layer (cosine similarity over L2-normalized embeddings) and persists durably to a single SQLite file by default.

**Full documentation: <https://anthonynystrom.github.io/mneme/>**

```python
from mneme import SemanticCache

with SemanticCache(path="cache.db", embedder=my_embedder) as cache:
    hit = cache.get("How do I reset my password?")
    if hit is None:
        response = call_my_llm("How do I reset my password?")
        cache.put("How do I reset my password?", response)
    else:
        response = hit.response
```

## Why

- **Cache before you call.** Turn redundant expensive operations - LLM calls, RAG rerankers, paid translation APIs, slow classifiers - into a microsecond `dict` lookup or a millisecond NumPy matvec. For chatbots, agent loops, classification pipelines, and batch-style scoring, this is the difference between a viable product and one that pays for every paraphrase.
- **One required dependency.** NumPy. Optional extras for `hnsw`, `redis`, `postgres`, `dynamodb`, `prometheus`, `otel`. Bring your own embedder, your own LLM client, your own server.
- **In-process, no daemon.** A library you `import`, not a service you operate. Persists to a single SQLite file by default; swap in Redis / Postgres / DynamoDB for cross-host shared state.
- **Strict typing, zero magic.** Public surface is a small set of frozen `@dataclass`es and `Protocol`s. `py.typed` shipped.

## Features

- Layered cache - O(1) exact match, then cosine similarity over an in-memory matrix
- Sync + async APIs (`SemanticCache`, `AsyncSemanticCache`)
- 5 `Store` backends: Memory, SQLite (default), Redis, Postgres, DynamoDB
- 2 `Index` backends: NumPy (default; bandwidth-bound exact search, comfortable at typical d=768 to ~500k and at d=384 well past 1M) and hnswlib (opt-in; sub-millisecond approximate search at 1M+)
- 3 vector dtypes: `float32`, `float16`, `int8` for memory-constrained deployments
- 3 multi-process modes: `single`, `stale-tolerant`, `mmap-shared`
- Multi-tenant via namespaces with per-namespace LRU quotas
- Calibration tooling (Python API + CLI) for tuning similarity thresholds
- Checkpoint export/import for backup and environment promotion
- Re-embed migration tool when the embedder changes
- Prometheus and OpenTelemetry metrics adapters

## Install

```bash
pip install mneme                       # core (NumPy only)
pip install "mneme[hnsw]"               # approximate-NN at 1M+ entries
pip install "mneme[redis]"              # RedisStore
pip install "mneme[postgres]"           # PostgresStore
pip install "mneme[dynamodb]"           # DynamoDBStore
pip install "mneme[prometheus,otel]"    # metrics adapters
pip install "mneme[all]"                # everything
```

Python 3.10+. See the full [install matrix](https://anthonynystrom.github.io/mneme/install/).

## Quickstart

```python
from mneme import SemanticCache, MemoryStore

with SemanticCache(store=MemoryStore(), embedder=my_embedder) as cache:
    cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
    hit = cache.get("Where do I reset my password?")  # paraphrase
    assert hit is not None
    print(hit.layer, hit.similarity, hit.response)
```

For the async API, see [Async quickstart](https://anthonynystrom.github.io/mneme/getting-started/quickstart-async/). For wrapping an actual LLM call, see [Your first cached LLM](https://anthonynystrom.github.io/mneme/getting-started/your-first-cached-llm/).

## Use cases

The same machinery covers more than LLM caching. Each pattern is the same three lines (`cache.get`, `cache.put`, your function); only what your function does changes.

| Pattern | What it caches |
| --- | --- |
| **LLM caching** | Wrap any LLM call so paraphrases hit a microsecond cache instead of a multi-second model |
| **RAG retrieval** | Top-k chunks behind paraphrased questions; skips the cross-encoder reranker on cache hits |
| **Translation** | "Source text → translated text" per language pair; cuts billed translation API calls |
| **Semantic deduplication** | Read `Hit.similarity` directly to detect near-duplicate content in ingestion pipelines |
| **Classification** | Cache labels from any classifier (sklearn, fastText, BERT, rules engines) |
| **Agent memory** | Per-agent task → plan lookup; consistency on similar tasks across runs |

[Full walkthrough with runnable scripts →](https://anthonynystrom.github.io/mneme/use-cases/)

## Performance

Apple M4 Max baseline at 100k entries (full table on the [docs site](https://anthonynystrom.github.io/mneme/performance/)):

| Operation | Latency |
| --- | --- |
| Exact-match `get` | ~2.3 ms p99 |
| Semantic `get` (fp32, d=768) | ~2.7 ms p99 |
| `put` (no eviction) | ~0.9 ms p99 |
| Single-thread throughput | ~5,700 ops/sec |

## Documentation

| | |
| --- | --- |
| [Getting started](https://anthonynystrom.github.io/mneme/getting-started/quickstart-sync/) | Sync + async quickstarts, bring your own embedder |
| [Use cases](https://anthonynystrom.github.io/mneme/use-cases/) | Five patterns: LLM, RAG retrieval, translation, dedup, classification, agent memory |
| [How mneme is different](https://anthonynystrom.github.io/mneme/comparison/) | Where mneme makes different choices than other semantic-cache libraries |
| [Concepts](https://anthonynystrom.github.io/mneme/concepts/layered-cache/) | Layered cache, embedders, quantization, multi-process, multi-tenant |
| [Stores](https://anthonynystrom.github.io/mneme/stores/memory/) | Memory · SQLite · Redis · Postgres · DynamoDB |
| [Guides](https://anthonynystrom.github.io/mneme/guides/calibration/) | Calibration, checkpoints, re-embed migration, metrics, custom stores, perf tuning |
| [API reference](https://anthonynystrom.github.io/mneme/reference/cache/) | Auto-generated from docstrings |
| [Performance](https://anthonynystrom.github.io/mneme/performance/) | Measured baseline against the original targets |
| [Showcase](https://anthonynystrom.github.io/mneme/showcase/) | Flask demo: customer-support intent classification w/ Nemotron on a DGX Spark |
| [Changelog](https://anthonynystrom.github.io/mneme/changelog/) | Release notes |

## Comparison

| | mneme | GPTCache |
| --- | --- | --- |
| Required runtime deps | NumPy | many (faiss, etc.) |
| Bundled embedder | no (BYOE) | yes |
| Bundled LLM client | no | yes |
| Sync + async parity | yes | partial |
| Strict typing (`py.typed`) | yes | no |
| Multi-process modes | 3 | n/a |
| Multi-tenant quotas | per-namespace LRU | n/a |
| Calibration tooling | yes (CLI + Python API) | no |

## Status

**v1.0.** Public surface locked; future minor versions are additive. See [Changelog](https://anthonynystrom.github.io/mneme/changelog/).

## License

Apache 2.0. See [LICENSE](LICENSE).
