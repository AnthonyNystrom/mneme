# mneme

A layered semantic cache for LLM applications.

`mneme` (Greek: μνήμη, "memory"; pronounced *NEE-mee*) is an embeddable, in-process Python library that caches LLM completions across paraphrased queries. It pairs an exact-match layer (normalized query hash) with a semantic-match layer (cosine similarity over L2-normalized embeddings) and persists durably to a single SQLite file by default.

📚 **Full documentation: <https://anystrom.github.io/mneme/>**

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

- **Cache before you call.** A semantic cache turns redundant LLM calls into a microsecond `dict` lookup or a millisecond NumPy matvec. For chatbots, agent loops, and batch-style scoring jobs, this is the difference between a viable product and one that burns tokens on every paraphrase.
- **One required dependency.** NumPy. Optional extras for `hnsw`, `redis`, `postgres`, `dynamodb`, `prometheus`, `otel`. Bring your own embedder, your own LLM client, your own server.
- **In-process, no daemon.** A library you `import`, not a service you operate. Persists to a single SQLite file by default; swap in Redis / Postgres / DynamoDB for cross-host shared state.
- **Strict typing, zero magic.** Public surface is a small set of frozen `@dataclass`es and `Protocol`s. `py.typed` shipped.

## Features

- Layered cache — O(1) exact match, then cosine similarity over an in-memory matrix
- Sync + async APIs (`SemanticCache`, `AsyncSemanticCache`)
- 5 `Store` backends: Memory, SQLite (default), Redis, Postgres, DynamoDB
- 2 `Index` backends: NumPy (default, ~500k entries) and hnswlib (opt-in, scales past 1M)
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
pip install "mneme[hnsw]"               # hnswlib for >500k entries
pip install "mneme[redis]"              # RedisStore
pip install "mneme[postgres]"           # PostgresStore
pip install "mneme[dynamodb]"           # DynamoDBStore
pip install "mneme[prometheus,otel]"    # metrics adapters
pip install "mneme[all]"                # everything
```

Python 3.10+. See the full [install matrix](https://anystrom.github.io/mneme/install/).

## Quickstart

```python
from mneme import SemanticCache, MemoryStore

with SemanticCache(store=MemoryStore(), embedder=my_embedder) as cache:
    cache.put("How do I reset my password?", "Click 'Forgot password' on login.")
    hit = cache.get("Where do I reset my password?")  # paraphrase
    assert hit is not None
    print(hit.layer, hit.similarity, hit.response)
```

For the async API, see [Async quickstart](https://anystrom.github.io/mneme/getting-started/quickstart-async/). For wrapping an actual LLM call, see [Your first cached LLM](https://anystrom.github.io/mneme/getting-started/your-first-cached-llm/).

## Documentation

| | |
| --- | --- |
| 🚀 [Getting started](https://anystrom.github.io/mneme/getting-started/quickstart-sync/) | Sync + async quickstarts, bring your own embedder |
| 💡 [Concepts](https://anystrom.github.io/mneme/concepts/layered-cache/) | Layered cache, embedders, quantization, multi-process, multi-tenant |
| 📦 [Stores](https://anystrom.github.io/mneme/stores/memory/) | Memory · SQLite · Redis · Postgres · DynamoDB |
| 🛠 [Guides](https://anystrom.github.io/mneme/guides/calibration/) | Calibration, checkpoints, re-embed migration, metrics, custom stores, perf tuning |
| 📖 [API reference](https://anystrom.github.io/mneme/reference/cache/) | Auto-generated from docstrings |
| 📊 [Performance](https://anystrom.github.io/mneme/performance/) | Measured baseline against PRD §16 targets |
| 🧪 [Showcase](https://anystrom.github.io/mneme/showcase/) | Flask demo: customer-support intent classification w/ Nemotron on a DGX Spark |
| 📝 [Changelog](https://anystrom.github.io/mneme/changelog/) | Release notes |

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

**v1.0.** Public surface locked; future minor versions are additive. See [Changelog](https://anystrom.github.io/mneme/changelog/).

## License

Apache 2.0. See [LICENSE](LICENSE).
