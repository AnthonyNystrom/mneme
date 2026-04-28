"""Flask showcase for ``mneme`` against Nemotron on the local DGX Spark.

Run:
    cd examples/showcase
    pip install -r requirements.txt
    pip install -e ../..      # editable install of mneme
    python app.py
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import config
from classifier import CachedClassifier, ClassifyResult
from embedder import SentenceTransformersEmbedder
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from nemotron_client import NemotronClient
from seed_data import MESSAGES, TRY_IT_PRESETS, stress_run_order

from mneme import SemanticCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("showcase")


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


@dataclass
class Counters:
    queries: int = 0
    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    total_llm_seconds_observed: float = 0.0
    llm_calls: int = 0


class AppState:
    """Live counters + rolling avg LLM latency + recent-queries ring."""

    def __init__(self) -> None:
        self.counters = Counters()
        self.recent: deque[dict] = deque(maxlen=config.RECENT_QUERIES_KEEP)
        self.per_namespace: dict[str, Counters] = collections.defaultdict(Counters)
        self._lock = threading.RLock()

        logger.info("loading embedder %s …", config.EMBEDDER_MODEL)
        self.embedder = SentenceTransformersEmbedder()
        logger.info("embedder loaded (dim=%d)", self.embedder.dim)

        self.cache = SemanticCache(
            path=config.CACHE_DB,
            embedder=self.embedder,
            similarity_threshold=config.SIMILARITY_THRESHOLD,
            vector_dtype=config.VECTOR_DTYPE,  # type: ignore[arg-type]
            namespace_quotas=dict(config.NAMESPACE_QUOTAS),
        )
        self.llm = NemotronClient()
        self.classifier = CachedClassifier(cache=self.cache, llm=self.llm)
        logger.info(
            "cache opened at %s (entries=%d)",
            config.CACHE_DB,
            self.cache.stats().entries,
        )

    # --- counter updates -------------------------------------------------

    def record(self, result: ClassifyResult) -> None:
        with self._lock:
            self.counters.queries += 1
            ns = self.per_namespace[result.namespace]
            ns.queries += 1
            if result.layer == "exact":
                self.counters.hits_exact += 1
                ns.hits_exact += 1
            elif result.layer == "semantic":
                self.counters.hits_semantic += 1
                ns.hits_semantic += 1
            else:  # miss
                self.counters.misses += 1
                ns.misses += 1
                if result.llm_seconds is not None:
                    self.counters.total_llm_seconds_observed += result.llm_seconds
                    self.counters.llm_calls += 1
            self.recent.appendleft(self._summarize(result))

    @staticmethod
    def _summarize(result: ClassifyResult) -> dict:
        return {
            "query": result.query,
            "intent": result.intent,
            "layer": result.layer,
            "similarity": result.similarity,
            "namespace": result.namespace,
            "latency_ms": round(result.latency_ms, 2),
            "llm_seconds": result.llm_seconds,
            "ts": time.time(),
        }

    # --- derived ---------------------------------------------------------

    def avg_llm_seconds(self) -> float:
        with self._lock:
            if self.counters.llm_calls == 0:
                return config.APPROX_LLM_SECONDS_DEFAULT
            return self.counters.total_llm_seconds_observed / self.counters.llm_calls

    def llm_seconds_saved(self) -> float:
        with self._lock:
            return self.avg_llm_seconds() * (
                self.counters.hits_exact + self.counters.hits_semantic
            )

    def stats_payload(self) -> dict:
        with self._lock:
            cache_stats = self.cache.stats()
            spark_ok = self.llm.healthy()
            ns_breakdown = {
                ns: {
                    "queries": c.queries,
                    "hits_exact": c.hits_exact,
                    "hits_semantic": c.hits_semantic,
                    "misses": c.misses,
                }
                for ns, c in self.per_namespace.items()
            }
            return {
                "queries": self.counters.queries,
                "hits_exact": self.counters.hits_exact,
                "hits_semantic": self.counters.hits_semantic,
                "misses": self.counters.misses,
                "hit_rate": (
                    (self.counters.hits_exact + self.counters.hits_semantic)
                    / self.counters.queries
                    if self.counters.queries > 0
                    else 0.0
                ),
                "avg_llm_seconds": round(self.avg_llm_seconds(), 3),
                "llm_seconds_saved": round(self.llm_seconds_saved(), 2),
                "cache_entries": cache_stats.entries,
                "memory_bytes_estimate": cache_stats.memory_bytes_estimate,
                "vector_dtype": cache_stats.vector_dtype,
                "embedder_fingerprint": cache_stats.embedder_fingerprint,
                # Live value from the cache, not the startup constant —
                # this changes when /api/threshold is hit.
                "similarity_threshold": self.cache.similarity_threshold,
                "namespaces": ns_breakdown,
                "recent": list(self.recent),
                "spark_ok": spark_ok,
                "model": config.LLM_MODEL,
                "spark_url": config.SPARK_URL,
            }

    def reset_counters(self) -> None:
        with self._lock:
            self.counters = Counters()
            self.per_namespace.clear()
            self.recent.clear()

    def clear_cache(self) -> None:
        """Wipe every entry under every namespace."""
        with self._lock:
            self.cache.clear()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


app = Flask(__name__)
state = AppState()


# --- pages ------------------------------------------------------------------


@app.route("/")
def dashboard():  # type: ignore[no-untyped-def]
    return render_template("dashboard.html", config=config)


@app.route("/try")
def try_it():  # type: ignore[no-untyped-def]
    return render_template("try_it.html", presets=TRY_IT_PRESETS)


@app.route("/stress")
def stress():  # type: ignore[no-untyped-def]
    return render_template("stress.html", corpus_size=len(MESSAGES))


@app.route("/inspector")
def inspector():  # type: ignore[no-untyped-def]
    return render_template("inspector.html")


@app.route("/tenants")
def tenants():  # type: ignore[no-untyped-def]
    return render_template("tenants.html")


# --- API --------------------------------------------------------------------


@app.route("/api/stats")
def api_stats():  # type: ignore[no-untyped-def]
    return jsonify(state.stats_payload())


@app.route("/api/classify", methods=["POST"])
def api_classify():  # type: ignore[no-untyped-def]
    body = request.get_json(force=True)
    query = (body or {}).get("query", "").strip()
    namespace = (body or {}).get("namespace", "support")
    bypass_cache = bool((body or {}).get("bypass", False))
    if not query:
        return jsonify({"error": "missing 'query'"}), 400

    if bypass_cache:
        result = state.classifier.classify_uncached(query)
    else:
        result = state.classifier.classify(query, namespace=namespace)
        state.record(result)

    return jsonify(
        {
            "query": result.query,
            "intent": result.intent,
            "layer": result.layer,
            "similarity": result.similarity,
            "confidence": result.confidence,
            "age_seconds": result.age_seconds,
            "namespace": result.namespace,
            "latency_ms": round(result.latency_ms, 2),
            "llm_seconds": result.llm_seconds,
        }
    )


@app.route("/api/inspector")
def api_inspector():  # type: ignore[no-untyped-def]
    """Return all cache entries.

    Reaches into the cache's underlying ``Store`` via the private
    ``_store`` attr — the public ``SemanticCache`` API doesn't expose
    iteration, but for a demo this is the cleanest path. In production
    code you'd add an ``iter_entries()`` method to the cache or work
    against your own ``Store`` directly.
    """
    out: list[dict] = []
    now = int(time.time())
    for entry in state.cache._store.iter_all():
        out.append(
            {
                "id": entry.id,
                "namespace": entry.namespace,
                "query": entry.query[:200],
                "response": entry.response[:200],
                "age_seconds": now - entry.created_at,
                "access_count": entry.access_count,
            }
        )
    out.sort(key=lambda e: (e["namespace"], e["id"]))
    return jsonify({"entries": out})


@app.route("/api/clear", methods=["POST"])
def api_clear():  # type: ignore[no-untyped-def]
    state.clear_cache()
    state.reset_counters()
    return jsonify({"ok": True})


@app.route("/api/threshold", methods=["POST"])
def api_threshold():  # type: ignore[no-untyped-def]
    """Adjust the cache's similarity threshold at runtime."""
    body = request.get_json(force=True) or {}
    try:
        value = float(body.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "missing or invalid 'value'"}), 400
    try:
        state.cache.set_similarity_threshold(value)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"threshold": state.cache.similarity_threshold})


@app.route("/api/stress", methods=["POST"])
def api_stress():  # type: ignore[no-untyped-def]
    """Run the seed corpus and stream SSE per-query progress."""
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "support")
    seed = int(body.get("seed", 1))
    run = stress_run_order(seed=seed)

    @stream_with_context
    def gen():  # type: ignore[no-untyped-def]
        n = len(run)
        hits = 0
        for i, (query, _true_intent) in enumerate(run, start=1):
            result = state.classifier.classify(query, namespace=namespace)
            state.record(result)
            if result.layer in ("exact", "semantic"):
                hits += 1
            payload = {
                "seq": i,
                "total": n,
                "query": query,
                "intent": result.intent,
                "layer": result.layer,
                "similarity": result.similarity,
                "latency_ms": round(result.latency_ms, 2),
                "cumulative_hit_rate": round(hits / i, 4),
            }
            yield f"data: {json.dumps(payload)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/tenants/run", methods=["POST"])
def api_tenants_run():  # type: ignore[no-untyped-def]
    """Run a small set of queries against a chosen namespace.

    Powers the multi-tenant page: pressing the button for tenant_a runs the
    same 6 queries as tenant_b. The first run for either namespace is mostly
    misses; the second is mostly hits.
    """
    body = request.get_json(force=True) or {}
    namespace = body.get("namespace", "tenant_a")
    queries = body.get("queries") or [
        "How do I reset my password?",
        "I forgot my password.",
        "I want a refund.",
        "Please refund my order.",
        "How do I export to CSV?",
        "Where can I download my records?",
    ]
    out = []
    for q in queries:
        result = state.classifier.classify(q, namespace=namespace)
        state.record(result)
        out.append(
            {
                "query": q,
                "intent": result.intent,
                "layer": result.layer,
                "similarity": result.similarity,
                "latency_ms": round(result.latency_ms, 2),
            }
        )
    return jsonify({"namespace": namespace, "results": out})


@app.route("/healthz")
def healthz():  # type: ignore[no-untyped-def]
    spark = state.llm.healthy()
    return jsonify(
        {
            "ok": True,
            "spark": spark,
            "spark_url": config.SPARK_URL,
            "model": config.LLM_MODEL,
            "cache_entries": state.cache.stats().entries,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logger.info(
        "showcase starting on http://%s:%d  (Spark: %s)",
        config.FLASK_HOST,
        config.FLASK_PORT,
        config.SPARK_URL,
    )
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
