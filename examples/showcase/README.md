# mneme showcase

A small Flask app that classifies customer-support messages into 7 intents using `nemotron-3-nano` running on a local [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) (or any Ollama host) — and shows what `mneme` does for an LLM workload that has paraphrases.

The pitch in one sentence: a customer-support intent classifier that talks to a real LLM is slow and expensive on every call; this app makes that obvious in five pages, then makes it disappear by wrapping the call in `mneme.SemanticCache`.

## What it shows

| Page | What it demonstrates |
| --- | --- |
| **Dashboard** | Live counters: queries served, exact-match hits, semantic-match hits, misses, cumulative LLM-seconds saved. Polls `/api/stats` once a second. |
| **Try it** | Single-query playground. Submit a message, see the cache layer hit, similarity, latency, and intent. Side-by-side button bypasses the cache so you can directly time `cached vs not`. |
| **Stress test** | Streams the full 73-query seed corpus through the classifier in shuffled order and plots the cumulative cache hit rate climbing live. |
| **Cache inspector** | Browse every entry in the cache with namespace, query, intent, age, hit count. Filter by namespace or substring. |
| **Multi-tenant** | Run the same six queries against `tenant_a` and `tenant_b`. The first run for each tenant is mostly misses; namespaces are isolated, so warming `tenant_a` does not help `tenant_b`. |

## Requirements

- Python 3.10+
- A reachable Ollama host with `nemotron-3-nano` (or any model — set `MNEME_SHOWCASE_MODEL`)
- ~1 GB of disk for the embedder model on first run (`sentence-transformers/all-MiniLM-L6-v2` weights cache)

The default points at `http://spark-245d.local:11434` running `nemotron-3-nano:latest`. Override with `MNEME_SHOWCASE_SPARK_URL` and `MNEME_SHOWCASE_MODEL` if your host is different.

## Run

```bash
cd examples/showcase

# 1. Isolated venv (recommended; the showcase pulls in torch via sentence-transformers).
python -m venv .venv
source .venv/bin/activate

# 2. Install the showcase deps + an editable install of mneme from the parent repo.
pip install -r requirements.txt
pip install -e ../..

# 3. Sanity check: is the Spark reachable?
curl -fsS "$MNEME_SHOWCASE_SPARK_URL"/api/tags 2>/dev/null \
  || curl -fsS http://spark-245d.local:11434/api/tags

# 4. Boot.
python app.py
```

Open <http://127.0.0.1:5001>. The dashboard renders immediately; the first classification request takes a few hundred extra ms while the embedder model warms.

## Tuning

Everything sits in [config.py](config.py) and accepts environment-variable overrides (`MNEME_SHOWCASE_*`):

| Variable | Default | Notes |
| --- | --- | --- |
| `MNEME_SHOWCASE_SPARK_URL` | `http://spark-245d.local:11434` | Ollama host |
| `MNEME_SHOWCASE_MODEL` | `nemotron-3-nano:latest` | Any Ollama model that follows JSON-format instructions |
| `MNEME_SHOWCASE_LLM_TIMEOUT` | `60` | seconds |
| `MNEME_SHOWCASE_EMBEDDER` | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~80 MB |
| `MNEME_SHOWCASE_SIM_THRESHOLD` | `0.78` | Run `calibrate.py` to retune |
| `MNEME_SHOWCASE_DTYPE` | `float16` | `float32`, `float16`, or `int8` |
| `MNEME_SHOWCASE_HOST` | `127.0.0.1` | bind address |
| `MNEME_SHOWCASE_PORT` | `5001` | macOS uses 5000 for AirPlay Receiver; we default to 5001 to dodge it |
| `MNEME_SHOWCASE_DEBUG` | `0` | `1` to enable Flask debug mode |

To re-tune the similarity threshold against the seed corpus and your embedder of choice:

```bash
python calibrate.py
```

## Layout

```
config.py              # central config + env overrides
seed_data.py           # 73 labeled paraphrases + click-to-fill presets
nemotron_client.py     # Ollama HTTP client with format=json + retry
embedder.py            # sentence-transformers wrapper for mneme's Embedder Protocol
classifier.py          # the cache wrapping the LLM (the demo's whole point)
app.py                 # Flask routes + AppState
calibrate.py           # threshold tuning script
templates/             # 5 pages
static/                # style.css + app.js
```

## Troubleshooting

**"Spark: unreachable" on the dashboard.** Confirm the host: `curl http://spark-245d.local:11434/api/tags`. Set `MNEME_SHOWCASE_SPARK_URL` to whatever works.

**First classification takes 4 seconds.** That's cold-start on Ollama's side: it loads `nemotron-3-nano` into VRAM. Subsequent calls are typically 0.4–0.6 s. The cache makes those go to <5 ms.

**Embedder model downloads on first boot.** sentence-transformers caches weights under `~/.cache/huggingface/`. Subsequent boots are instant.

**`Address already in use`.** Pick another with `MNEME_SHOWCASE_PORT=5002`.

**Want to nuke the cache.** Click "Clear cache" on the dashboard, or `rm cache.db*`.

## What this is not

- A library. The showcase is a demo, not part of the installed `mneme` wheel. Look at `src/mneme/` for the library and the top-level [README](../../README.md) for its public API.
- Production code. There's no auth, no TLS, no WSGI server. It's `app.run()` for clarity. Don't expose it on the open internet.
