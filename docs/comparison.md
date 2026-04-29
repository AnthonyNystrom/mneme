# How mneme is different

There are other semantic-cache libraries in the Python ecosystem. The most established is [GPTCache](https://github.com/zilliztech/gptcache) - a mature, capable project with a strong feature set. This page is **not** a critique of GPTCache; it's a description of where `mneme` makes deliberately different choices, so you can pick the tool that matches your priorities.

If you're already happy with GPTCache, stay with it. The trade-offs below explain why mneme exists alongside it, not why you should switch.

## Different choices, different shape

| Decision | mneme | Other semantic caches |
| --- | --- | --- |
| Required runtime dependency | NumPy alone | Typically multiple (faiss / Pinecone clients / LangChain / etc.) |
| Embedder | Bring-your-own (`Embedder` Protocol) | Often bundled, with built-in adapters per model |
| LLM client | Bring-your-own (you wrap the call) | Often bundled adapters (OpenAI/Anthropic/etc.) |
| Public API surface | ~25 symbols, locked at v1.0 | Larger, evolves with adapters |
| Similarity metric | Cosine on L2-normalized vectors only | Often configurable (Euclidean, IP, Hamming, …) |
| Type system | Strict mypy + `py.typed` + Protocols | Varies |
| Stores shipped | 5 (Memory, SQLite, Redis, Postgres, DynamoDB) - same Protocol, same conformance battery | Varies; broader or narrower |
| Multi-process modes | 3 first-class (`single`, `stale-tolerant`, `mmap-shared`) | Varies |
| Multi-tenant | Per-namespace LRU quotas, namespace-scoped Layer 2 search | Varies |
| Sync + async parity | Both, sharing one core | Varies (often sync-primary) |
| Calibration tooling | `find_threshold` + `precision_recall_curve` (CLI + Python API) | Varies |

## Design philosophies

The bullets above come from a handful of underlying philosophies. These are the *why* behind the table.

### "Library, not framework"

`import mneme` should give you cache primitives, not a way to structure your application. You bring your own embedder, your own LLM client, your own server, your own logging. mneme owns the cache surface and nothing else. Frameworks are valuable; this is the trade-off in the other direction - fewer assumptions about your architecture.

The practical result: there is no `from mneme.adapters.openai import ...` step. There is no LangChain integration. The Embedder is whatever object you pass that satisfies the [Protocol](reference/types.md#mneme._types.Embedder); the LLM call site is wherever you put `cache.get` / `cache.put`.

### "One required dependency"

NumPy. Optional extras add `hnswlib`, `redis`, `psycopg`, `boto3`, `prometheus_client`, `opentelemetry-api` - each one independent, none transitively pulled in. `pip install mneme` is small and fast; deployment artifacts stay tight.

This matters when you're shipping mneme into a serverless function (cold-start cost), an edge device, or a constrained CI image. It also keeps the dependency-tree audit short for security-conscious teams.

### "Strict typing + Protocols"

`py.typed` ships. `mypy --strict` is clean across the source. The 24 methods of the `Store` Protocol are the contract for every backend, and the conformance battery (50+ tests) verifies every shipped store satisfies it identically. Custom stores get the same yardstick.

The result is that switching `MemoryStore()` for `RedisStore(...)` for `DynamoDBStore(...)` is mechanical - they really do behave the same, because the test suite enforces it. No surprise behaviors hiding behind backend-specific adapters.

### "Single similarity metric, calibrated"

Cosine similarity on L2-normalized vectors, full stop. No knob to switch to Euclidean, dot product, or Hamming. Two reasons:

1. **Cosine on normalized vectors is the right answer for embedding-based semantic match.** The literature converged here; making it configurable invites footguns.
2. **Calibration tooling is more useful than knob-turning.** `find_threshold` calibrates the *threshold* against your own corpus and dtype, which is the actual production knob you want - not the metric.

This is the "fewer knobs by design" choice. The cost is that exotic similarity needs (dot product on un-normalized vectors, etc.) aren't supported.

### "Multi-tenant from the start"

Namespaces are not bolted on. Every `get`/`put` takes one. Layer 2 search is namespace-scoped at the index level. Per-namespace LRU quotas evict within a namespace, never across. The conformance battery exercises namespace isolation against every store backend.

This makes mneme a comfortable fit for SaaS workloads where one cache file serves many tenants and you can't have tenant_a's hits leaking into tenant_b's request path.

### "Sync + async parity, one core"

`SemanticCache` and `AsyncSemanticCache` share the same store, the same index, the same locking. The async cache awaits the embedder directly (drops the lock around the await) and dispatches blocking store work via `asyncio.to_thread`. Counters, metrics, persistence - all identical between them.

The trade-off: there's no separate "async-native" implementation to optimize independently. Both classes are equally first-class because they share a core.

### "Calibration is a first-class concern"

`mneme.tools.calibrate` ships with `find_threshold` and `precision_recall_curve`. CLI: `python -m mneme.tools.calibrate`. The expectation is that you calibrate against your embedder + corpus before going to production - the default threshold is a starting point, not a recommendation.

This is documented as the [Calibration guide](guides/calibration.md), with explicit guidance on sourcing pairs, calibrating against the production dtype, and recognizing when low F1 means "your embedder doesn't fit this task."

### "Performance honesty"

The performance baseline ([Performance](performance.md)) records observed numbers against PRD §16 targets, **including the gaps**. int8 search at d=1536 is ~50–60 ms p99 - not the 6 ms that PRD §16 aspired to - and the docs explain why (no fused int8 GEMM in NumPy) and what to do about it (use hnsw, or accept the memory-footprint win without the latency win).

The test thresholds are regression bars above the observed baseline, not the aspirational targets. Honesty over marketing.

## When mneme is the wrong tool

For balance, here are workloads where you should pick something else:

- **You need a metric other than cosine.** Pick a tool that exposes the metric as a knob.
- **You want LangChain integration out of the box.** mneme has none; add a thin adapter or use a LangChain-native cache.
- **You need >10M cached vectors per process.** That's vector-DB territory - Pinecone, Weaviate, Qdrant, Milvus. mneme can run hnsw past 1M but it isn't designed for the multi-server-shard use case.
- **You can't bring your own embedder.** mneme deliberately doesn't bundle one. If you want a one-line install that handles the embedder for you, pick a tool with bundled adapters.

## When mneme is the right tool

- **You want a small dependency footprint** (one required: NumPy).
- **You already operate Redis / Postgres / DynamoDB** and want a cache that uses what you have.
- **You're a multi-tenant SaaS** and namespaces + per-tenant quotas matter.
- **You want strict typing all the way through** to your IDE and CI.
- **You want calibration tooling, not just a threshold parameter**.
- **You want sync and async to be equally first-class**, not one a thin wrapper over the other.
- **You want to swap stores without rewriting your code** and have automated tests prove the swap is safe.

## Where to go next

- **[Use cases](use-cases.md)** - five patterns the library was built for.
- **[Showcase](showcase.md)** - the design philosophies above, in motion.
- **[Custom stores](guides/custom-stores.md)** - the conformance battery as a portability contract.
