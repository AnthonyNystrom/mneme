# Examples

Runnable code that demonstrates one feature at a time. All live under [`examples/`](https://github.com/anystrom/mneme/tree/main/examples) in the repo. Run any of them after `pip install -e .` from a checkout.

## Standalone scripts

Each file is self-contained and uses a toy embedder so it runs without external services or GPU dependencies. Replace the toy with a real embedder (see [Bring your own embedder](getting-started/bring-your-own-embedder.md)) when you adapt to your own code.

| File | What it shows |
| --- | --- |
| [`quickstart.py`](https://github.com/anystrom/mneme/blob/main/examples/quickstart.py) | Sync `SemanticCache` with `MemoryStore`, then a SQLite round-trip across processes. The shortest possible introduction. |
| [`async_quickstart.py`](https://github.com/anystrom/mneme/blob/main/examples/async_quickstart.py) | `AsyncSemanticCache` with 100 concurrent gets to demonstrate the async cache's concurrency story. |
| [`multi_tenant.py`](https://github.com/anystrom/mneme/blob/main/examples/multi_tenant.py) | Per-namespace LRU quotas in action. Two tenants, different caps, eviction within each tenant only. |
| [`high_dim_quantized.py`](https://github.com/anystrom/mneme/blob/main/examples/high_dim_quantized.py) | float32 vs float16 vs int8 in-memory matrices at d=1536. Prints memory footprint per dtype. |
| [`custom_store.py`](https://github.com/anystrom/mneme/blob/main/examples/custom_store.py) | A complete `Store` Protocol implementation in ~150 lines. Pythonic dict-backed reference for writing your own backend. |
| [`calibration.py`](https://github.com/anystrom/mneme/blob/main/examples/calibration.py) | `find_threshold` and `precision_recall_curve` against a small labeled corpus. |
| [`dynamodb_quickstart.py`](https://github.com/anystrom/mneme/blob/main/examples/dynamodb_quickstart.py) | `DynamoDBStore` running against [moto](https://github.com/getmoto/moto) (in-process AWS mock). No real AWS account needed. |

Run any of them:

```bash
python examples/quickstart.py
python examples/async_quickstart.py
# ...
```

## Reference embedders

Production-quality embedder wrappers, not toys. Documentation-only — `mneme` never imports them. Copy into your own code.

| File | Service |
| --- | --- |
| [`reference_embedders/openai_embedder.py`](https://github.com/anystrom/mneme/blob/main/examples/reference_embedders/openai_embedder.py) | `text-embedding-3-small` / `-large` (sync + async pair) |
| [`reference_embedders/sentence_transformers_embedder.py`](https://github.com/anystrom/mneme/blob/main/examples/reference_embedders/sentence_transformers_embedder.py) | Local sentence-transformers model (any one) |
| [`reference_embedders/bedrock_embedder.py`](https://github.com/anystrom/mneme/blob/main/examples/reference_embedders/bedrock_embedder.py) | AWS Bedrock — Titan Text Embeddings v2 + Cohere Embed |
| [`reference_embedders/ollama_embedder.py`](https://github.com/anystrom/mneme/blob/main/examples/reference_embedders/ollama_embedder.py) | Self-hosted Ollama (sync + async via httpx) |

Walkthroughs of the embedder Protocol and the trade-offs between models live in [Bring your own embedder](getting-started/bring-your-own-embedder.md).

## Showcase

The Flask app at [`examples/showcase/`](https://github.com/anystrom/mneme/tree/main/examples/showcase) is a separate, fully wired demo. It's not a single-file example — it has its own README, requirements, and 5-page UI. Covered on its own page: [Showcase](showcase.md).

## Verifying examples on every release

The CI suite runs each standalone example as a smoke test before publishing a release. A passing example is the load-bearing guarantee that the documented patterns still work; if you find one that doesn't run, please open an issue on [GitHub](https://github.com/anystrom/mneme/issues).
