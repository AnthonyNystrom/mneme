"""Central configuration for the showcase.

Override any value via environment variables (`MNEME_SHOWCASE_*`).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Spark / Ollama ---

SPARK_URL = os.environ.get("MNEME_SHOWCASE_SPARK_URL", "http://spark-245d.local:11434")
LLM_MODEL = os.environ.get("MNEME_SHOWCASE_MODEL", "nemotron-3-nano:latest")
LLM_TIMEOUT_SEC = float(os.environ.get("MNEME_SHOWCASE_LLM_TIMEOUT", "60"))

# --- Embedder ---

EMBEDDER_MODEL = os.environ.get(
    "MNEME_SHOWCASE_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDER_DIM = 384  # all-MiniLM-L6-v2

# --- Cache ---

CACHE_DB = Path(__file__).parent / "cache.db"
# 0.65 was empirically chosen with calibrate.py against the seed corpus +
# all-MiniLM-L6-v2. At this threshold the precision stays at 1.0 on the
# corpus's distractor pairs while still catching the close paraphrases
# (e.g. "forgot my password" / "reset password"). Tweak per workload.
SIMILARITY_THRESHOLD = float(os.environ.get("MNEME_SHOWCASE_SIM_THRESHOLD", "0.65"))
VECTOR_DTYPE = os.environ.get("MNEME_SHOWCASE_DTYPE", "float16")  # float32, float16, int8

# --- Classification ---

INTENT_LABELS = (
    "billing",
    "technical",
    "refund",
    "account",
    "how_to",
    "complaint",
    "other",
)

# Approximate per-call LLM latency seed; updated as a rolling mean from real calls.
APPROX_LLM_SECONDS_DEFAULT = 0.5

# Per-namespace LRU caps; multi-tenant page exercises these.
NAMESPACE_QUOTAS = {
    "default": 5_000,
    "support": 5_000,
    "tenant_a": 200,
    "tenant_b": 200,
}

# --- Flask ---

FLASK_HOST = os.environ.get("MNEME_SHOWCASE_HOST", "127.0.0.1")
# 5001 not 5000 — macOS Monterey+ binds AirPlay Receiver to *:5000 on
# both IPv4 and IPv6, which steals browser requests to 127.0.0.1:5000
# (Chrome resolves localhost over IPv6) and returns HTTP 403.
FLASK_PORT = int(os.environ.get("MNEME_SHOWCASE_PORT", "5001"))
FLASK_DEBUG = os.environ.get("MNEME_SHOWCASE_DEBUG", "0") == "1"

# Recent-queries ring buffer size on the dashboard.
RECENT_QUERIES_KEEP = 50
