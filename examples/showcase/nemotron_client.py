"""Thin Ollama HTTP client for Nemotron classification.

Uses ``/api/chat`` with ``think: false`` so the reasoning preamble is
suppressed (Nemotron-3-nano is a reasoning model — without
``think: false`` the entire token budget goes into a ``thinking`` field
and the actual content is empty). Also passes ``format: "json"`` to
constrain the output shape.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import requests
from config import LLM_MODEL, LLM_TIMEOUT_SEC, SPARK_URL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    intent: str
    raw: str
    duration_sec: float


_SYSTEM_PROMPT = (
    "You are an intent classifier for customer-support messages. "
    "Respond ONLY with valid JSON in the format "
    '{"intent": "<one of: billing, technical, refund, account, how_to, complaint, other>"} '
    "with no additional fields, reasoning, or commentary."
)

_STRICTER_SYSTEM_PROMPT = (
    "Output exactly one JSON object and nothing else. The object has one field: "
    '"intent". The value must be exactly one of these strings: '
    '"billing", "technical", "refund", "account", "how_to", "complaint", "other". '
    "Do not include any other text. Example output: "
    '{"intent": "billing"}'
)


class NemotronClient:
    """Synchronous Ollama client. Thread-safe for concurrent requests
    (each call uses a fresh ``requests`` HTTP request)."""

    def __init__(
        self,
        url: str = SPARK_URL,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT_SEC,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.RequestException:
            return []

    def classify(self, query: str) -> LLMResponse:
        """Classify ``query`` into one of the seven intents.

        On a parse failure, retries once with a stricter system prompt. On a
        second failure, returns ``intent="other"`` and logs.
        """
        for attempt, system in enumerate((_SYSTEM_PROMPT, _STRICTER_SYSTEM_PROMPT)):
            t0 = time.monotonic()
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "think": False,  # suppress Nemotron's reasoning preamble
                "format": "json",  # constrain output shape
                "options": {
                    "temperature": 0.0,
                    "num_predict": 40,
                },
            }
            try:
                resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
            except requests.RequestException as exc:
                logger.warning("Nemotron call failed: %s", exc)
                return LLMResponse(intent="other", raw=str(exc), duration_sec=time.monotonic() - t0)

            elapsed = time.monotonic() - t0
            content = body.get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
                intent = parsed.get("intent", "")
            except (ValueError, AttributeError):
                if attempt == 0:
                    logger.info("Retrying with stricter prompt; got: %r", content[:120])
                    continue
                logger.warning("Stricter prompt also failed; got: %r", content[:120])
                return LLMResponse(intent="other", raw=content, duration_sec=elapsed)

            if intent in {
                "billing",
                "technical",
                "refund",
                "account",
                "how_to",
                "complaint",
                "other",
            }:
                return LLMResponse(intent=intent, raw=content, duration_sec=elapsed)

            if attempt == 0:
                logger.info("Bad intent label %r; retrying", intent)
                continue
            logger.warning("Final attempt produced bad intent label: %r", intent)
            return LLMResponse(intent="other", raw=content, duration_sec=elapsed)

        # Unreachable: the loop returns or continues into the second attempt.
        return LLMResponse(intent="other", raw="", duration_sec=0.0)

    def _chat_text(self, system: str, user: str, *, num_predict: int = 256) -> tuple[str, float]:
        """Generic chat call returning ``(content, duration_sec)`` as plain text.

        No JSON parsing; the caller decides how to interpret the response.
        Used for translation, plan generation, and RAG synthesis.
        """
        t0 = time.monotonic()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": num_predict,
            },
        }
        try:
            resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            logger.warning("Nemotron chat failed: %s", exc)
            return f"[error] {exc}", time.monotonic() - t0
        elapsed = time.monotonic() - t0
        return body.get("message", {}).get("content", "").strip(), elapsed

    def translate(self, text: str, target_lang: str) -> LLMResponse:
        """Translate ``text`` from English to ``target_lang`` (full language name)."""
        system = (
            f"You are a professional translator. Translate the user's English text into "
            f"{target_lang}. Output ONLY the translation, no quotes, no commentary, no "
            f"explanation. Preserve punctuation and tone."
        )
        content, elapsed = self._chat_text(system, text, num_predict=200)
        return LLMResponse(intent=content, raw=content, duration_sec=elapsed)

    def generate_plan(self, task: str) -> LLMResponse:
        """Generate a numbered execution plan for ``task``."""
        system = (
            "You are a software engineering agent. Given a task, produce a concise "
            "numbered plan (3-6 steps) for executing it. Each step is one short line. "
            "Output ONLY the numbered list, no preamble, no commentary."
        )
        content, elapsed = self._chat_text(system, task, num_predict=300)
        return LLMResponse(intent=content, raw=content, duration_sec=elapsed)

    def synthesize_rag(self, question: str, contexts: list[str]) -> LLMResponse:
        """Answer ``question`` grounded in the provided ``contexts``."""
        joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        system = (
            "Answer the user's question using ONLY the provided context. Cite sources "
            "as [1], [2], etc. If the context doesn't contain the answer, say so. "
            "Be concise (2-4 sentences)."
        )
        user = f"Context:\n{joined}\n\nQuestion: {question}"
        content, elapsed = self._chat_text(system, user, num_predict=300)
        return LLMResponse(intent=content, raw=content, duration_sec=elapsed)
