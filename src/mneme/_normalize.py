"""Query normalization and hashing for the exact-match layer."""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str, *, casefold: bool = True) -> str:
    """Collapse whitespace, strip, and (optionally) casefold.

    The default is intentionally conservative: only operations that preserve
    semantic identity are applied. Unicode normalization, accent stripping,
    punctuation removal, and stemming are out of scope.
    """
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    if casefold:
        text = text.casefold()
    return text


def hash_query(normalized: str) -> str:
    """Return a 64-char lowercase hex SHA-256 digest of the UTF-8 bytes."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["hash_query", "normalize"]
