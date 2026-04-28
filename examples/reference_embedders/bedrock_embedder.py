"""Reference embedder: AWS Bedrock (Titan / Cohere via boto3).

Documentation only. Copy into your own code; ``mneme`` never imports this.

Bedrock invokes are synchronous HTTP calls; use ``BedrockAsyncEmbedder``
or wrap with ``mneme.to_async_embedder`` if you need an async cache.

Install:  pip install boto3 numpy
"""

from __future__ import annotations

import json

import numpy as np


class BedrockTitanEmbedder:
    """Sync embedder for Amazon Titan Text Embeddings v2."""

    def __init__(
        self,
        client,                                       # noqa: ANN001 — boto3 bedrock-runtime client
        *,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,                       # 256, 512, or 1024 for v2
        normalize: bool = True,
    ) -> None:
        if dimensions not in (256, 512, 1024):
            raise ValueError(
                f"BedrockTitanEmbedder: dimensions={dimensions!r} not supported; "
                "Titan v2 supports 256, 512, or 1024."
            )
        self._client = client
        self._model_id = model_id
        self._dim = dimensions
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"bedrock:{self._model_id}:dim{self._dim}:n{int(self._normalize)}"

    def embed(self, text: str) -> np.ndarray:
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self._dim,
                "normalize": self._normalize,
            }
        )
        resp = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(resp["body"].read())
        v = np.asarray(payload["embedding"], dtype=np.float32)
        if not self._normalize:
            n = float(np.linalg.norm(v))
            v = v / n if n > 0 else v
        return v


class BedrockCohereEmbedder:
    """Sync embedder for Cohere Embed via Bedrock."""

    def __init__(
        self,
        client,                                       # noqa: ANN001 — boto3 bedrock-runtime client
        *,
        model_id: str = "cohere.embed-english-v3",
        input_type: str = "search_document",          # or "search_query", "classification", "clustering"
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._input_type = input_type
        # Cohere Embed v3 returns 1024-dim vectors.
        self._dim = 1024

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"bedrock:{self._model_id}:dim{self._dim}:type:{self._input_type}"

    def embed(self, text: str) -> np.ndarray:
        body = json.dumps(
            {
                "texts": [text],
                "input_type": self._input_type,
            }
        )
        resp = self._client.invoke_model(
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        payload = json.loads(resp["body"].read())
        v = np.asarray(payload["embeddings"][0], dtype=np.float32)
        # Cohere Bedrock embeddings are not always L2-normalized; do it explicitly.
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


# Usage:
#
# import boto3
# from mneme import SemanticCache
#
# client = boto3.client("bedrock-runtime", region_name="us-east-1")
# embedder = BedrockTitanEmbedder(client, dimensions=512)
# with SemanticCache(path="cache.db", embedder=embedder) as cache:
#     hit = cache.get("How do I reset my password?")
