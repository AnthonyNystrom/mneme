"""sentence-transformers embedder satisfying mneme's ``Embedder`` Protocol."""

from __future__ import annotations

import numpy as np
from config import EMBEDDER_DIM, EMBEDDER_MODEL


class SentenceTransformersEmbedder:
    """Local-CPU embedder via sentence-transformers.

    Uses ``all-MiniLM-L6-v2`` (384-dim) by default — small, fast, and
    accurate enough that the cache earns its keep on real paraphrases
    while staying under 100 MB of memory.
    """

    def __init__(self, model_name: str = EMBEDDER_MODEL, *, normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._normalize = normalize
        # Newer sentence-transformers (>=5) renamed the method; support both.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        if self._dim != EMBEDDER_DIM:
            # Soft check: we configured EMBEDDER_DIM=384 for MiniLM; if
            # someone swaps the model to a 768-dim one, EMBEDDER_DIM lies.
            # We still trust the actual model.
            pass

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"sentence-transformers:{self._model_name}:n{int(self._normalize)}:dim{self._dim}"

    def embed(self, text: str) -> np.ndarray:
        v = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return v.astype(np.float32, copy=False)
