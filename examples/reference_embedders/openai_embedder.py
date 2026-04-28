"""Reference embedder: OpenAI ``text-embedding-3-*``.

Documentation only. Copy into your own code; ``mneme`` never imports this.

Sync + async pair. Both produce 1-D float32 numpy arrays of the configured
dimension. The fingerprint includes model name + dimensions so the cache
detects accidental embedder changes (and refuses to mix incompatible
vectors via ``EmbedderMismatchError`` / ``EmbedderDimensionError``).

Install:  pip install openai numpy
"""

from __future__ import annotations

import numpy as np


class OpenAIEmbedder:
    """Sync embedder. Wrap an ``openai.OpenAI`` client."""

    def __init__(
        self,
        client,                                       # noqa: ANN001 — openai.OpenAI
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        # `dimensions` lets you ask OpenAI for a smaller cut of the full
        # vector (only supported on text-embedding-3-*); when None, you
        # get the model's native length.
        self._dimensions = dimensions

    @property
    def dim(self) -> int:
        # Models the dimensions known at the time of writing. If you pass
        # `dimensions=`, that wins. Otherwise hard-code the model native size.
        if self._dimensions is not None:
            return self._dimensions
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }[self._model]

    @property
    def fingerprint(self) -> str:
        return f"openai:{self._model}:dim{self.dim}"

    def embed(self, text: str) -> np.ndarray:
        kwargs = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = self._client.embeddings.create(**kwargs)
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
        # OpenAI 3.x returns L2-normalized vectors already, but normalize
        # defensively in case that ever changes.
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class OpenAIAsyncEmbedder:
    """Async embedder. Wrap an ``openai.AsyncOpenAI`` client."""

    def __init__(
        self,
        client,                                       # noqa: ANN001 — openai.AsyncOpenAI
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @property
    def dim(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        return {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }[self._model]

    @property
    def fingerprint(self) -> str:
        return f"openai:{self._model}:dim{self.dim}"

    async def embed(self, text: str) -> np.ndarray:
        kwargs = {"model": self._model, "input": text}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        resp = await self._client.embeddings.create(**kwargs)
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


# Usage:
#
# import openai
# from mneme import SemanticCache
#
# client = openai.OpenAI()
# embedder = OpenAIEmbedder(client, model="text-embedding-3-small")
#
# with SemanticCache(path="cache.db", embedder=embedder) as cache:
#     hit = cache.get("How do I reset my password?")
