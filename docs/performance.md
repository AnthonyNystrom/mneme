# Performance baseline

This document records observed performance for `mneme` against PRD §16
targets, the hardware they were measured on, and notes on the engineering
trade-offs behind each gap.

The numbers below are produced by `tests/test_perf.py` (run with
`pytest --run-perf -s`). The `--run-perf` flag is opt-in; the suite is
skipped by default so that ordinary CI runs stay fast and stable. Larger
1M-entry hnsw benchmarks are gated behind `MNEME_PERF_HEAVY=1`.

## Baseline machine

| | |
| --- | --- |
| Model | Apple MacBook Pro (M4 Max) |
| OS | macOS 26.3 (Darwin 25.3) |
| Python | 3.12.13 |
| NumPy | 2.4.4 (Apple Accelerate BLAS) |
| Storage | Internal NVMe SSD |

## Measured baseline vs PRD §16

| Workload | PRD §16 target | Observed (p99) | Status |
| --- | --- | --- | --- |
| Exact-match `get` @ 100k | < 500 µs | ~2.3 ms | over target |
| Semantic `get` fp32 @ 100k/d=768 | < 5 ms | ~2.7 ms | meets |
| Semantic `get` fp32 @ 100k/d=1536 | < 8 ms | ~4.0 ms | meets |
| Semantic `get` int8 @ 100k/d=1536 | < 6 ms | ~50–60 ms | over target |
| `put` @ 100k (no eviction) | < 2 ms | ~0.9 ms | meets |
| `put` with 10% eviction (cap 10k) | < 20 ms | ~40–45 ms | over target |
| Open + rebuild fp32 @ 100k/d=768 | < 100 ms | ~300 ms | over target |
| Open + rebuild int8 @ 100k/d=768 | < 200 ms | ~400–450 ms | over target |
| Single-thread throughput | > 5000 ops/sec | ~5700 ops/sec | meets |
| Async throughput (100 concurrent) | > 2000 ops/sec | ~5100 ops/sec | meets |
| Direct `NumpyIndex.search` p99 @ 100k/d=768 | < 5 ms | ~2.8 ms | meets |

The test assertions in `tests/test_perf.py` use **regression bars**
above the observed baseline (typically 1.5–2× headroom) so that the
suite stays green on a typical contributor laptop while still flagging
gross regressions. The PRD §16 targets remain documented in the test
docstrings as aspirational goals.

## Notes on the gaps

### Exact-match `get` (~2.3 ms vs 500 µs)

Each `get()` issues a SQLite `UPDATE entries SET last_used_unix = …`
to keep the LRU ordering accurate across processes. Per-`get` UPDATE
through SQLite WAL is ~2 ms on this hardware. Removing the UPDATE
brings exact-match `get` under 100 µs but degrades cross-process LRU
accuracy. Possible future optimization: batch `last_used_unix` writes
in memory and flush every N gets / on close; keeps the on-disk LRU
approximately accurate while collapsing N writes into 1.

### Semantic `get` int8 @ d=1536 (~50–60 ms vs 6 ms)

The PRD target was written assuming a fused int8 GEMM (oneDNN, ARM
SDOT). Pure NumPy has **no int8 GEMM**: the only supported path is
`matrix.astype(float32) @ query`, which expands a 150 MB int8 matrix
into 600 MB of fp32 — the cast is the bottleneck, not the matmul.

`mneme` uses chunked dequant-and-matvec with a reused L2-resident
fp32 buffer (`src/mneme/_index.py:_chunked_matvec`) and pushes the
`1/127` int8 scale onto the query side to halve memory traffic. After
those optimizations the floor is dominated by the int8 → fp32
expansion (~750 MB of memory traffic per search at 100k × 1536).

The win for int8 in the current implementation is **memory footprint**
(4× smaller in-memory matrix at d=1536), not search latency on
pure-NumPy stacks. Workloads that need both small footprint and low
latency should use the hnsw backend (`index_backend="hnsw"`).

### `put` with eviction (~40 ms vs 20 ms)

A 10% batch eviction at cap=10k means 1000 row deletes through SQLite
WAL plus 1000 index tombstones. Each row delete is a small WAL write;
they aren't batched into a single transaction in v1.0. Possible future
optimization: a single `DELETE … WHERE id IN (…)` per eviction batch
plus bulk index tombstone.

### Open time (~300–450 ms vs 100–200 ms)

Open reads every row from the store and reconstructs the in-memory
index. SQLite reads and Python-level row iteration dominate. The fast
path `SQLiteStore.iter_index_rows()` already skips full `StoredEntry`
construction (no `json.loads` of metadata, no `last_used_unix` read
on the rebuild path), which cut open time roughly in half. Pushing
below the PRD target needs a binary blob format or memory-mapped
store; out of scope for v1.0.

## Reproducing

```bash
conda activate NuGenLLMCache    # or your equivalent env
pytest tests/test_perf.py --run-perf -s
# 1M-entry hnsw benchmarks (skipped by default):
MNEME_PERF_HEAVY=1 pytest tests/test_perf.py --run-perf -s
```

The suite prints each measurement so you can record your own baseline
in the table above.
