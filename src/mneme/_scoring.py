"""Default confidence and validator functions per PRD §11.4 / §11.7.

Both are pure stateless functions; users may override either via the
``confidence_fn`` / ``validator`` constructor kwargs on ``SemanticCache``.
"""

from __future__ import annotations

from typing import Any

# 24-hour half-life: a 24h-old hit at perfect similarity scores 0.5.
_DEFAULT_HALF_LIFE_SECONDS = 86_400.0


def default_confidence(similarity: float, age_seconds: int, metadata: dict[str, Any]) -> float:
    """Combine similarity with an exponential freshness decay.

    ``confidence = similarity * 0.5 ** (age_seconds / 24h)``.

    The PRD-mandated cutoff ``confidence >= 0.7`` (enforced in the cache
    layer) maps to: at perfect similarity, hits stay above the bar for
    roughly the first ~12 hours; at 0.85 similarity, ~10 hours.
    """
    del metadata  # default scoring ignores metadata
    if age_seconds < 0:
        age_seconds = 0
    freshness = 0.5 ** (age_seconds / _DEFAULT_HALF_LIFE_SECONDS)
    return float(similarity) * float(freshness)


def default_validator(response: str) -> bool:
    """Reject empty / whitespace-only / obvious-error responses.

    Returns False for: empty string, whitespace-only, or a string starting
    with ``"[LLM Error]"`` or ``"[ERROR]"``. Otherwise True. Domain-specific
    validation (JSON shape, code compilability, etc.) is the user's job.
    """
    if not response or not response.strip():
        return False
    return not (response.startswith("[LLM Error]") or response.startswith("[ERROR]"))


__all__ = ["default_confidence", "default_validator"]
