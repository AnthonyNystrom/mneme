"""Shared toy embedder for the use-case demos.

A pure-NumPy 'feature hashing' bag-of-words embedder. Real production code
should swap in sentence-transformers, OpenAI, or another semantic embedder
(see ``examples/reference_embedders/``). The toy below produces similar
vectors for queries that share words — enough to demonstrate the cache's
Layer-2 semantic-match behavior in stdout without pulling in torch.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np


class TokenBagEmbedder:
    """Bag-of-words via feature hashing. ~256-dim, deterministic.

    Similar word sets -> similar vectors. Perfect-paraphrase recall is
    nowhere near a real embedder, but token-overlapping queries do
    cluster, which is enough to make the cache's Layer-2 behavior
    visible in the demos.
    """

    dim = 256
    fingerprint = "toy:tokenbag:v1"

    def embed(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokens:
            h = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
