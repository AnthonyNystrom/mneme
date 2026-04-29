# Showcase

A self-contained Flask app that classifies customer-support messages into 7 intents using `nemotron-3-nano` running on a [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (or any Ollama-compatible host) — and shows what `mneme` does for an LLM workload that has paraphrases.

The pitch in one sentence: an intent classifier talking to a real LLM is slow and expensive on every call; this app makes that obvious in five pages, then makes it disappear by wrapping the call in `mneme.SemanticCache`.

The full project lives at [`examples/showcase/`](https://github.com/anystrom/mneme/tree/main/examples/showcase).

## What it shows

| Page | What it demonstrates |
| --- | --- |
| **Dashboard** | Live counters: queries served, exact-match hits, semantic-match hits, misses, cumulative LLM-seconds saved, cache memory footprint. Plus a similarity-threshold slider that adjusts the cache's runtime knob from the UI. Polls `/api/stats` once a second. |
| **Try it** | Single-query playground. Submit a message, see the cache layer hit, similarity, latency, and intent. A "Same query, no cache" button bypasses the cache for direct LLM-call timing in a side-by-side comparison. |
| **Stress test** | Streams the full 73-query seed corpus through the classifier in shuffled order via Server-Sent Events, plotting the cumulative cache hit rate climbing live in a Chart.js graph. |
| **Cache inspector** | Browse every entry currently in the cache: namespace, query, intent, age, hit count. Filter by namespace or substring. Confirms persistence — the entries survive a restart of the Flask app. |
| **Multi-tenant** | Run the same six queries against `tenant_a` and `tenant_b`. The first run for each tenant is mostly misses; namespaces are isolated, so warming `tenant_a` does not help `tenant_b` until `tenant_b` populates its own slice. |

## What's running under the hood

- **LLM**: `nemotron-3-nano` (31.6 B Nemotron-H-MoE, Q4_K_M) served by Ollama at `http://spark-245d.local:11434`. Cold call ~4 s (model load); warm ~0.5 s.
- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) running locally on CPU, ~80 MB memory.
- **Cache**: `SemanticCache` against SQLite at `examples/showcase/cache.db`. `vector_dtype="float16"`. Threshold calibrated to 0.65 against the seed corpus.
- **Web**: Flask 3 in `app.run(threaded=True)`. No external services; no auth.

## Running it

```bash
git clone https://github.com/anystrom/mneme.git
cd mneme/examples/showcase

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ../..                             # editable install of mneme

# Sanity-check the LLM host (defaults to spark-245d.local; override below)
curl -fsS http://spark-245d.local:11434/api/tags

python app.py
```

Open <http://127.0.0.1:5001>. The dashboard renders immediately; first classification takes a few hundred extra ms while the embedder warms.

## Configuration

Everything lives in [`config.py`](https://github.com/anystrom/mneme/blob/main/examples/showcase/config.py) and accepts `MNEME_SHOWCASE_*` env-var overrides:

| Variable | Default | Notes |
| --- | --- | --- |
| `MNEME_SHOWCASE_SPARK_URL` | `http://spark-245d.local:11434` | Ollama host |
| `MNEME_SHOWCASE_MODEL` | `nemotron-3-nano:latest` | Any Ollama model that follows JSON-format instructions |
| `MNEME_SHOWCASE_LLM_TIMEOUT` | `60` | Seconds |
| `MNEME_SHOWCASE_EMBEDDER` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedder |
| `MNEME_SHOWCASE_SIM_THRESHOLD` | `0.65` | Calibrated; tweak via the dashboard slider too |
| `MNEME_SHOWCASE_DTYPE` | `float16` | `float32`, `float16`, or `int8` |
| `MNEME_SHOWCASE_PORT` | `5001` | macOS uses 5000 for AirPlay Receiver |

## What this is not

- **Not a library.** The showcase is a demo, not part of the installed `mneme` wheel. It lives in `examples/showcase/` and ships its own `requirements.txt`.
- **Not production code.** No auth, no TLS, no WSGI server. It's `app.run()` for clarity. Don't expose it on the open internet.
- **Not the only way to use mneme.** It's *one* shape — Flask in front of an LLM. Most production users wrap the cache around the LLM call inside their own service. See [Your first cached LLM](getting-started/your-first-cached-llm.md).

## Code layout

```
examples/showcase/
  README.md                 # quickstart + troubleshooting
  requirements.txt          # Flask, sentence-transformers, requests, numpy
  config.py                 # central config + env var overrides
  app.py                    # Flask routes + AppState
  classifier.py             # the cache wrapping the LLM (the demo's whole point)
  nemotron_client.py        # Ollama HTTP client with think:false + format:json
  embedder.py               # SentenceTransformersEmbedder
  seed_data.py              # 73 labeled paraphrases across 7 intents
  calibrate.py              # threshold tuning script
  templates/                # 5 pages
  static/                   # style.css + app.js
```

`app.py` is ~270 lines and shows every public `mneme` API in use. If you want to copy a pattern into your own service, start there.

## Where to go next

- **[Your first cached LLM](getting-started/your-first-cached-llm.md)** — the same caching pattern, without the demo wrapping.
- **[Multi-tenant](concepts/multi-tenant.md)** — what the namespace switcher demonstrates.
- **[Performance baseline](performance.md)** — the numbers behind the "saved seconds" counter.
