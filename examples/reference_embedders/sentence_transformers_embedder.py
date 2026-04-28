"""Reference embedder: sentence-transformers (any local model).

Documentation only. Copy into your own code; ``mneme`` never imports this.

Local models are great for cost-free embedding, offline ops, and privacy.
GPU acceleration is automatic if torch is installed with CUDA / MPS.

Install:  pip install sentence-transformers numpy
"""

from __future__ import annotations

import numpy as np


class SentenceTransformersEmbedder:
    """Sync embedder backed by a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,             # "cuda", "mps", "cpu", or None=auto
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        # SentenceTransformer exposes the dim via get_sentence_embedding_dimension()
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        # Model name + normalize flag + dim catches accidental mismatches.
        return f"sentence-transformers:{self._model_name}:n{int(self._normalize)}:dim{self._dim}"

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        # encode() returns float32 by default; ensure dtype.
        return v.astype(np.float32, copy=False)


# Usage:
#
# from mneme import SemanticCache
#
# embedder = SentenceTransformersEmbedder("all-MiniLM-L6-v2")
# with SemanticCache(path="cache.db", embedder=embedder) as cache:
#     hit = cache.get("How do I reset my password?")
#
# For an async cache, wrap with ``to_async_embedder``:
#
# from mneme import AsyncSemanticCache, to_async_embedder
#
# async_embedder = to_async_embedder(embedder)
# async with AsyncSemanticCache(path="cache.db", embedder=async_embedder) as cache:
#     hit = await cache.get("...")
