"""Cache-aware wrappers for the four secondary use cases on the showcase.

Each wrapper mirrors the pattern in ``examples/use_cases/`` but operates against
the same shared ``SemanticCache`` the showcase already runs. Namespaces keep the
data partitioned so the dashboard breakdown stays meaningful.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from nemotron_client import NemotronClient

from mneme import SemanticCache


def _llm_failed(text: str) -> bool:
    """True when the NemotronClient returned its error sentinel instead of
    real content. Failed responses must NOT be cached — a transient Ollama
    outage would otherwise poison the cache until the namespace is cleared."""
    return text.startswith("[error]")


# ---------------------------------------------------------------------------
# Result dataclasses (per use case)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupResult:
    content: str
    is_duplicate: bool
    similarity: float | None
    layer: str  # "exact" / "semantic" / "novel"
    latency_ms: float


@dataclass(frozen=True)
class TranslateResult:
    source: str
    target_lang: str
    translation: str
    layer: str  # "exact" / "semantic" / "miss"
    similarity: float | None
    latency_ms: float
    llm_seconds: float | None  # None on cache hit


@dataclass(frozen=True)
class PlanResult:
    task: str
    agent_id: str
    plan: str
    layer: str  # "exact" / "semantic" / "miss"
    similarity: float | None
    latency_ms: float
    llm_seconds: float | None


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    layer: str  # "exact" / "semantic" / "miss"
    similarity: float | None
    latency_ms: float
    llm_seconds: float | None


# ---------------------------------------------------------------------------
# 1. Semantic deduplication (no LLM; pure cache + similarity)
# ---------------------------------------------------------------------------


_DEDUP_NAMESPACE = "dedup"
_DEDUP_SIMILAR_ENOUGH = 0.85


class Deduplicator:
    """Stream of incoming content → ``is_duplicate`` decision via similarity.

    Uses a sentinel ``"seen"`` response so the cache's default validator
    (which rejects empty Layer-2 candidates) doesn't filter legitimate hits.
    """

    def __init__(self, cache: SemanticCache) -> None:
        self._cache = cache

    def check(self, content: str) -> DedupResult:
        t0 = time.monotonic()
        hit = self._cache.get(content, namespace=_DEDUP_NAMESPACE)
        if hit is not None and hit.layer == "exact":
            return DedupResult(
                content=content,
                is_duplicate=True,
                similarity=1.0,
                layer="exact",
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        if hit is not None and hit.layer == "semantic" and hit.similarity >= _DEDUP_SIMILAR_ENOUGH:
            return DedupResult(
                content=content,
                is_duplicate=True,
                similarity=float(hit.similarity),
                layer="semantic",
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        # Not a duplicate (or below threshold). Record + remember.
        self._cache.put(content, "seen", namespace=_DEDUP_NAMESPACE)
        return DedupResult(
            content=content,
            is_duplicate=False,
            similarity=float(hit.similarity) if hit is not None else None,
            layer="novel",
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# 2. Translation (Nemotron + per-language-pair namespace)
# ---------------------------------------------------------------------------


class CachedTranslator:
    """`(text, target_lang)` → translation. One namespace per ``en→<lang>`` pair."""

    def __init__(self, cache: SemanticCache, llm: NemotronClient) -> None:
        self._cache = cache
        self._llm = llm

    @staticmethod
    def _namespace(target_lang: str) -> str:
        return f"translate:en-{target_lang.lower()[:2]}"

    def translate(self, text: str, target_lang: str) -> TranslateResult:
        t0 = time.monotonic()
        ns = self._namespace(target_lang)
        hit = self._cache.get(text, namespace=ns)
        if hit is not None:
            return TranslateResult(
                source=text,
                target_lang=target_lang,
                translation=hit.response,
                layer=hit.layer,
                similarity=float(hit.similarity) if hit.layer == "semantic" else None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                llm_seconds=None,
            )
        llm_resp = self._llm.translate(text, target_lang)
        if not _llm_failed(llm_resp.intent):
            self._cache.put(text, llm_resp.intent, namespace=ns)
        return TranslateResult(
            source=text,
            target_lang=target_lang,
            translation=llm_resp.intent,
            layer="miss",
            similarity=None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            llm_seconds=llm_resp.duration_sec,
        )


# ---------------------------------------------------------------------------
# 3. Agent memory (Nemotron + per-agent namespace + confidence gate)
# ---------------------------------------------------------------------------


class CachedAgent:
    """`task` → execution plan. Per-agent namespace; confidence-gated reuse."""

    def __init__(self, cache: SemanticCache, llm: NemotronClient) -> None:
        self._cache = cache
        self._llm = llm

    @staticmethod
    def _namespace(agent_id: str) -> str:
        return f"agent:{agent_id}"

    def execute(self, task: str, agent_id: str) -> PlanResult:
        t0 = time.monotonic()
        ns = self._namespace(agent_id)
        hit = self._cache.get(task, namespace=ns)
        if hit is not None and hit.confidence >= 0.7:
            return PlanResult(
                task=task,
                agent_id=agent_id,
                plan=hit.response,
                layer=hit.layer,
                similarity=float(hit.similarity) if hit.layer == "semantic" else None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                llm_seconds=None,
            )
        llm_resp = self._llm.generate_plan(task)
        if not _llm_failed(llm_resp.intent):
            self._cache.put(task, llm_resp.intent, namespace=ns)
        return PlanResult(
            task=task,
            agent_id=agent_id,
            plan=llm_resp.intent,
            layer="miss",
            similarity=None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            llm_seconds=llm_resp.duration_sec,
        )


# ---------------------------------------------------------------------------
# 4. RAG retrieval (corpus + cache + Nemotron synthesis)
# ---------------------------------------------------------------------------


_RAG_NAMESPACE = "rag"


@dataclass(frozen=True)
class RAGChunk:
    """A retrievable document fragment."""

    id: str
    text: str


class CachedRAG:
    """`question` → cached `(top-k chunks, synthesized answer)`.

    Retrieval and synthesis run on a miss; a cache hit returns both pieces in
    one shot. Cache stores a JSON-encoded dict so we don't lose structure.
    """

    def __init__(
        self,
        cache: SemanticCache,
        llm: NemotronClient,
        corpus: list[RAGChunk],
        embedder,  # the showcase's shared embedder; provides .embed(str) -> ndarray
        k: int = 3,
    ) -> None:
        self._cache = cache
        self._llm = llm
        self._corpus = corpus
        self._embedder = embedder
        self._k = k
        # Pre-embed corpus once so retrieval is fast.
        self._corpus_vecs = [embedder.embed(c.text) for c in corpus]

    def _retrieve(self, question: str) -> list[RAGChunk]:
        import numpy as np

        q_vec = self._embedder.embed(question)
        # Cosine on already-L2-normalized vectors == dot product.
        scores = [float(np.dot(q_vec, cv)) for cv in self._corpus_vecs]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: self._k]
        return [self._corpus[i] for i in order]

    def ask(self, question: str) -> RAGResult:
        t0 = time.monotonic()
        hit = self._cache.get(question, namespace=_RAG_NAMESPACE)
        if hit is not None:
            payload = json.loads(hit.response)
            return RAGResult(
                question=question,
                answer=payload["answer"],
                contexts=payload["contexts"],
                chunk_ids=payload["chunk_ids"],
                layer=hit.layer,
                similarity=float(hit.similarity) if hit.layer == "semantic" else None,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                llm_seconds=None,
            )
        chunks = self._retrieve(question)
        contexts = [c.text for c in chunks]
        chunk_ids = [c.id for c in chunks]
        llm_resp = self._llm.synthesize_rag(question, contexts)
        payload = {
            "answer": llm_resp.intent,
            "contexts": contexts,
            "chunk_ids": chunk_ids,
        }
        if not _llm_failed(llm_resp.intent):
            self._cache.put(question, json.dumps(payload), namespace=_RAG_NAMESPACE)
        return RAGResult(
            question=question,
            answer=llm_resp.intent,
            contexts=contexts,
            chunk_ids=chunk_ids,
            layer="miss",
            similarity=None,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            llm_seconds=llm_resp.duration_sec,
        )


# ---------------------------------------------------------------------------
# Default RAG corpus (small customer-support FAQ; ~12 chunks)
# ---------------------------------------------------------------------------


DEFAULT_RAG_CORPUS: list[RAGChunk] = [
    RAGChunk(
        "faq-1",
        "To reset your password, click 'Forgot password' on the login page. "
        "We'll email you a reset link within 5 minutes; check your spam folder if "
        "it doesn't arrive.",
    ),
    RAGChunk(
        "faq-2",
        "We offer full refunds within 30 days of purchase. Open a refund request "
        "from your account's Orders page and our team responds within 2 business days.",
    ),
    RAGChunk(
        "faq-3",
        "Two-factor authentication can be enabled under Account Settings → Security. "
        "We support TOTP apps (Google Authenticator, 1Password, Authy) and hardware "
        "keys (YubiKey).",
    ),
    RAGChunk(
        "faq-4",
        "Export your data as CSV from the Reports tab. Larger exports (>100 MB) are "
        "delivered via email as a download link valid for 24 hours.",
    ),
    RAGChunk(
        "faq-5",
        "If the app crashes on login, first try clearing the app cache (Settings → "
        "Storage → Clear cache). If that doesn't help, uninstall and reinstall — "
        "your data is on the server, so nothing is lost.",
    ),
    RAGChunk(
        "faq-6",
        "Billing runs on the 1st of each month. You can update your payment method "
        "under Account Settings → Billing. We accept Visa, MasterCard, and ACH transfer.",
    ),
    RAGChunk(
        "faq-7",
        "Our API rate limits are 100 requests per minute on the free tier and 5000 "
        "rpm on paid plans. Headers ``X-RateLimit-Remaining`` and "
        "``X-RateLimit-Reset`` tell you your current state.",
    ),
    RAGChunk(
        "faq-8",
        "Webhooks are configurable under Settings → Integrations. We retry failed "
        "deliveries 5 times with exponential backoff, then mark the webhook disabled "
        "and email you.",
    ),
    RAGChunk(
        "faq-9",
        "To delete your account permanently, go to Account Settings → Privacy → "
        "Delete account. Deletion is processed within 30 days, after which all "
        "data is irreversibly removed.",
    ),
    RAGChunk(
        "faq-10",
        "Mobile apps are available for iOS 15+ and Android 10+. The desktop app "
        "supports macOS 12+ and Windows 10+. Linux users can use the web app, "
        "which is fully feature-equivalent.",
    ),
    RAGChunk(
        "faq-11",
        "Single sign-on (SSO) via SAML 2.0 is included on the Enterprise plan. "
        "We support Okta, Azure AD, Google Workspace, and any standard SAML "
        "identity provider.",
    ),
    RAGChunk(
        "faq-12",
        "Our uptime SLA is 99.9% for paid plans (excluding scheduled maintenance). "
        "Status updates are at status.example.com; subscribe via RSS or email "
        "for incident notifications.",
    ),
]
