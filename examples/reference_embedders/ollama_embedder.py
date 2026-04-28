"""Reference embedder: Ollama (local self-hosted embeddings).

Documentation only. Copy into your own code; ``mneme`` never imports this.

Ollama runs models locally on a small HTTP server (default port 11434).
Sync + async variants below; use the async one with ``AsyncSemanticCache``
or wrap the sync one with ``mneme.to_async_embedder``.

Install:  pip install requests httpx numpy
"""

from __future__ import annotations

import numpy as np


# Default dims for popular Ollama embedding models. Add yours here.
_OLLAMA_MODEL_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "snowflake-arctic-embed:s": 384,
    "snowflake-arctic-embed:m": 768,
    "snowflake-arctic-embed:l": 1024,
    "bge-m3": 1024,
}


class OllamaEmbedder:
    """Sync embedder for Ollama via HTTP."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        url: str = "http://localhost:11434",
        dim: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        import requests

        self._requests = requests
        self._model = model
        self._url = url.rstrip("/")
        self._timeout = timeout
        if dim is not None:
            self._dim = dim
        elif model in _OLLAMA_MODEL_DIMS:
            self._dim = _OLLAMA_MODEL_DIMS[model]
        else:
            raise ValueError(
                f"OllamaEmbedder: dim for model {model!r} unknown. "
                "Remediation: pass dim= explicitly."
            )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self._model}:dim{self._dim}"

    def embed(self, text: str) -> np.ndarray:
        resp = self._requests.post(
            f"{self._url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        v = np.asarray(resp.json()["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class OllamaAsyncEmbedder:
    """Async embedder for Ollama via httpx."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        url: str = "http://localhost:11434",
        dim: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=timeout)
        self._model = model
        self._url = url.rstrip("/")
        if dim is not None:
            self._dim = dim
        elif model in _OLLAMA_MODEL_DIMS:
            self._dim = _OLLAMA_MODEL_DIMS[model]
        else:
            raise ValueError(
                f"OllamaAsyncEmbedder: dim for model {model!r} unknown. "
                "Remediation: pass dim= explicitly."
            )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self._model}:dim{self._dim}"

    async def embed(self, text: str) -> np.ndarray:
        resp = await self._client.post(
            f"{self._url}/api/embeddings",
            json={"model": self._model, "prompt": text},
        )
        resp.raise_for_status()
        v = np.asarray(resp.json()["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    async def aclose(self) -> None:
        await self._client.aclose()


# Usage:
#
# from mneme import SemanticCache
#
# embedder = OllamaEmbedder("nomic-embed-text")
# with SemanticCache(path="cache.db", embedder=embedder) as cache:
#     hit = cache.get("How do I reset my password?")
