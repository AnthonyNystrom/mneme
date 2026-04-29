# Changelog

All notable changes to `mneme` are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.0 — 2026-04-29

The first stable release. The public surface in `mneme/__init__.py` is locked; future minor versions are additive.

### Added

#### Core cache

- **`SemanticCache`** — sync layered cache with exact-match (Layer 1) over normalized SHA-256 hashes and semantic-match (Layer 2) over cosine similarity on L2-normalized embeddings.
- **`AsyncSemanticCache`** — async wrapper that awaits the embedder directly and dispatches blocking store work via `asyncio.to_thread`. Sync ↔ async parity.
- **`Hit`, `Stats`, `Health`, `StoredEntry`** — frozen, slotted dataclasses for the public read surface.
- **`Embedder`, `AsyncEmbedder`, `Index`, `Store`** — runtime-checkable Protocols for bring-your-own backends.
- **`to_async_embedder` / `to_sync_embedder`** — adapter helpers between the two embedder Protocols.
- **`SemanticCache.clear()`** — backend-agnostic whole-cache wipe across all namespaces. Works against any `Store` implementation; bumps `version_counter` per cleared namespace; rebuilds the in-memory index empty.
- **`SemanticCache.set_similarity_threshold(value)` and `.similarity_threshold` property** — adjust the Layer-2 threshold at runtime. Validates `[-1.0, 1.0]`. Affects subsequent `get` calls only.

#### Stores (5 backends, one Protocol)

- **`MemoryStore`** — dict-backed; tests, scratch, ephemeral.
- **`SQLiteStore`** (default) — single-file, WAL mode, atomic writes, `mode 0o600`. Implements `snapshot_to` / `restore_from` via SQLite's online backup API.
- **`RedisStore`** — `[redis]` extra. `MULTI`/`EXEC` for atomicity. `version_counter` incremented in the same pipeline as every write.
- **`PostgresStore`** — `[postgres]` extra. Schema-scoped, `BIGSERIAL` ids, `BYTEA` embeddings, `JSONB` metadata. `version_counter` UPDATE in the same transaction as data writes.
- **`DynamoDBStore`** — `[dynamodb]` extra. Single table + 2 GSIs. `TransactWriteItems` pairs every data op with a counter `Update`. Auto-create-on-open opt-in. Snapshot stub directs operators to AWS native backup.

All five satisfy the same `Store` Protocol and pass the same conformance battery.

#### Index backends

- **`NumpyIndex`** — default; in-memory matrix, cosine via `M @ q`, L2-normalized, geometric capacity growth. Sub-5 ms p99 to ~500k entries at d=768.
- **`HnswIndex`** — `[hnsw]` extra. Approximate-NN past 1M entries. Per-namespace hnswlib indices, configurable `M` / `ef_construction` / `ef`.
- **`index_backend="auto"`** — picks NumPy below 500k, hnsw above; falls back to NumPy + WARNING when hnswlib is missing.

#### Vector dtypes

- **`float32`** (default) — fastest, no cast at search time.
- **`float16`** — 2× memory cut, negligible accuracy drift.
- **`int8`** — 4× memory cut. Symmetric quantization, scale 127, assumes L2-normalized input.
- **`requantize(dtype)`** — switch dtypes at runtime without rebuilding the store.

#### Multi-process modes

- **`single`** (default) — no coordination, fastest. One owner.
- **`stale-tolerant`** — periodic poll of `version_counter` + `iter_since` deltas. Eventually consistent. The right mode for multi-worker apps on one host.
- **`mmap-shared`** — single mmap matrix shared across processes under `fcntl.flock` (POSIX) / `msvcrt.locking` (Windows, best-effort). Strong consistency on the same host.

#### Multi-tenant

- Per-namespace `get`/`put`. Layer 2 search is namespace-scoped.
- **`namespace_quotas`** — per-namespace LRU caps. Eviction batches at 10% of the cap (min 1).
- **`clear_namespace(ns)`** — wipe one tenant.

#### Calibration and migration

- **`mneme.tools.calibrate.find_threshold`** — picks a `similarity_threshold` against your paraphrase + distractor pairs.
- **`mneme.tools.calibrate.precision_recall_curve`** — full sweep for hand-picking.
- **`mneme.tools.migrate.reembed` / `areembed`** — re-embed an existing cache through a new embedder when model or dim changes.
- CLI: `python -m mneme.tools.calibrate --help`.

#### Checkpoint

- **`SemanticCache.dumps(dest)` and `SemanticCache.loads(source, ...)`** — round-trip a full cache as a single `tar.gz` archive (manifest + store snapshot + optional vectors). Validates fingerprint + dim before unpacking.

#### Metrics

- **`metrics_hook`** — single `Callable[[event, fields], None]` for fan-out. Fires `hit_exact`, `hit_semantic`, `miss`, `put`, `put_rejected`, `evict`, `expire`, `embedder_failure`. Hook exceptions are caught and downgraded to WARNING.
- **`PrometheusMetricsHook`** — `[prometheus]` extra. Standard `mneme_*` counters and histograms.
- **`OTelMetricsHook`** — `[otel]` extra. Same metric shape, OTel meter API.

#### Examples

- 7 standalone runnable scripts: `quickstart.py`, `async_quickstart.py`, `multi_tenant.py`, `high_dim_quantized.py`, `custom_store.py`, `calibration.py`, `dynamodb_quickstart.py`.
- 4 reference embedder snippets: OpenAI, sentence-transformers, AWS Bedrock (Titan + Cohere), Ollama.
- Flask showcase under `examples/showcase/` — 5-page UI demonstrating the layered cache, namespace isolation, persistence, and a live similarity-threshold slider against `nemotron-3-nano` on a DGX Spark.

#### Documentation

- This site, built with MkDocs + Material + mkdocstrings. Auto-generated API reference plus hand-written concepts, guides, per-store deep-dives, and the showcase.
- GitHub Actions workflow auto-deploys to `gh-pages` on every push to `main`.

### Quality gates

- **689 tests** passing on Python 3.10–3.13, Linux + macOS.
- **`mypy --strict`** clean (27 source files).
- **`ruff`** clean across `src/`, `tests/`, and `examples/`.
- **Branch coverage 93%** with all optional services (Redis, Postgres, DynamoDB) running.
- **Conformance battery** parameterized across all 5 store backends.
- Performance regression bars enforced in `tests/test_perf.py`.

### Acknowledgments

The PRD and architecture decisions are documented in the project's design history; the v1.0 surface reflects 16 phases of staged implementation against that PRD.
