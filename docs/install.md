# Install

`mneme` requires Python 3.10 or newer.

The distribution is published to PyPI as **`mneme-cache`**; the import name is **`mneme`**. So `pip install mneme-cache`, then `from mneme import SemanticCache`.

## Core (NumPy only)

```bash
pip install mneme-cache
```

This is the minimal install. You get `MemoryStore`, `SQLiteStore`, the NumPy index backend, the calibration CLI, and the full sync + async cache surface. Every other backend is opt-in.

## Optional extras

Each extra installs the additional dependencies needed for one optional feature. They are independent - pick the ones you need.

| Extra | Pulls in | Enables |
| --- | --- | --- |
| `hnsw` | `hnswlib>=0.7` | The hnsw approximate-NN index backend (`index_backend="hnsw"`); useful when NumPy's exact-search latency stops fitting your budget - typically past ~500k entries at d=768, sooner at higher dim |
| `redis` | `redis>=5.0` | `RedisStore` for shared cache across hosts |
| `postgres` | `psycopg[binary]>=3.1`, `psycopg-pool>=3.2` | `PostgresStore` |
| `dynamodb` | `boto3>=1.34` | `DynamoDBStore` for serverless / multi-region |
| `prometheus` | `prometheus_client>=0.17` | `PrometheusMetricsHook` adapter |
| `otel` | `opentelemetry-api>=1.20` | `OTelMetricsHook` adapter |
| `all` | All of the above | Everything except dev tools |
| `dev` | pytest, mypy, ruff, testcontainers, moto, boto3, opentelemetry-sdk | Test + lint stack |
| `docs` | mkdocs, mkdocs-material, mkdocstrings, etc. | Build this site locally |

## Examples

=== "Just SQLite (default)"

    ```bash
    pip install mneme-cache
    ```

=== "SQLite + hnsw for scale"

    ```bash
    pip install "mneme-cache[hnsw]"
    ```

=== "Redis-backed cache"

    ```bash
    pip install "mneme-cache[redis]"
    ```

=== "Multi-host with metrics"

    ```bash
    pip install "mneme-cache[postgres,prometheus,otel]"
    ```

=== "Everything"

    ```bash
    pip install "mneme-cache[all]"
    ```

=== "Build docs locally"

    ```bash
    pip install -e ".[docs]"
    mkdocs serve
    ```

## Verifying install

```python
>>> import mneme
>>> mneme.__version__
'1.0.0'
>>> from mneme import SemanticCache, MemoryStore  # always available
>>> from mneme._store_redis import RedisStore     # only if `mneme[redis]` was installed
```

The optional-dep store classes live at `mneme._store_redis`, `mneme._store_postgres`, `mneme._store_dynamodb` so that `import mneme` stays lightweight regardless of which extras are installed.

## Python version matrix

| Python | Status |
| --- | --- |
| 3.10 | Supported |
| 3.11 | Supported |
| 3.12 | Supported (CI primary) |
| 3.13 | Supported |

CI runs the full test suite + conformance battery on Linux and macOS for each version.
