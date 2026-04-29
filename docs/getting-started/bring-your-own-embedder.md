# Bring your own embedder

`mneme` is bring-your-own-embedder. The library never imports an embedder package; you wire one up against the `Embedder` Protocol.

## The contract

A sync embedder satisfies the `Embedder` Protocol if it provides three things:

```python
class MyEmbedder:
    @property
    def dim(self) -> int:
        """Length of the vector returned by embed()."""

    @property
    def fingerprint(self) -> str:
        """A stable string identifying the (model, dim, normalization) tuple.
        Reused across runs; the cache validates it on open()."""

    def embed(self, text: str) -> numpy.ndarray:
        """Return a 1-D float32 vector of shape (self.dim,)."""
```

For async, the same shape with `async def embed(text) -> numpy.ndarray`.

The full Protocol - both sync and async - is documented in the [API reference](../reference/types.md).

## Conventions that pay off

Three properties that make the cache happy:

1. **L2-normalize the vector before returning.** Cosine similarity is the dot product of unit vectors; the cache assumes you've normalized. If you skip this, hits land further from `1.0` than they should and your threshold needs to drift.
2. **Make `fingerprint` deterministic and specific.** Include the model name, the dimension, and any normalization or instruction-prefix flags. The cache refuses to mix incompatible vectors via [`EmbedderMismatchError`](../reference/exceptions.md) - that's only useful if your fingerprint actually changes when the model changes.
3. **Treat `dim` as a property of the model, not a parameter.** It should match `embed()`'s output exactly. If you can configure the model to return a smaller vector (e.g. OpenAI's `dimensions=` parameter), bake the chosen dim into the fingerprint so the cache notices.

## Thread- and task-safety

`SemanticCache` serializes every public method through a single `RLock`, but it intentionally **drops that lock around your embedder call** so concurrent gets can fan out. That means the cache assumes:

- **Sync `Embedder.embed()` is thread-safe.** Multiple threads may call it concurrently.
- **Async `AsyncEmbedder.embed()` is task-safe.** Multiple `asyncio` tasks may await it concurrently.

Most network-backed embedders (OpenAI, Bedrock, Ollama HTTP) are fine — each call opens its own client connection. Local model embedders are where this bites:

- **`sentence-transformers` on CPU** — generally thread-safe.
- **`sentence-transformers` on a single GPU** — concurrent calls can race the GPU stream and produce garbage vectors. Wrap in a single-thread executor or an `asyncio.Semaphore(1)`.
- **Batch embedders that mutate internal state** — also serialize.

If your embedder isn't safe under concurrency, the simplest fix is a serializing wrapper:

```python
import asyncio

class SerializedAsyncEmbedder:
    """Wraps an AsyncEmbedder so concurrent embed() calls run one at a time."""
    def __init__(self, inner):
        self._inner = inner
        self._sem = asyncio.Semaphore(1)

    @property
    def dim(self): return self._inner.dim

    @property
    def fingerprint(self): return self._inner.fingerprint

    async def embed(self, text):
        async with self._sem:
            return await self._inner.embed(text)
```

The cost is no concurrency on the embed step itself — but the rest of the cache (Layer-1 hash lookup, Layer-2 matvec, store writes) still runs concurrently across requests, so this only matters if your embedder is the bottleneck.

## Reference embedders

Each reference snippet is a real-world starting point. Copy into your own code; `mneme` never imports them. Sources live at [`examples/reference_embedders/`](https://github.com/anthonynystrom/mneme/tree/main/examples/reference_embedders).

=== "OpenAI"

    `text-embedding-3-small` (1536-dim) or `text-embedding-3-large` (3072-dim). Sync + async pair.

    ```python
    import numpy as np


    class OpenAIEmbedder:
        def __init__(self, client, *, model="text-embedding-3-small", dimensions=None):
            self._client = client
            self._model = model
            self._dimensions = dimensions

        @property
        def dim(self) -> int:
            if self._dimensions is not None:
                return self._dimensions
            return {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}[self._model]

        @property
        def fingerprint(self) -> str:
            return f"openai:{self._model}:dim{self.dim}"

        def embed(self, text: str) -> np.ndarray:
            kwargs = {"model": self._model, "input": text}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            resp = self._client.embeddings.create(**kwargs)
            v = np.asarray(resp.data[0].embedding, dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v
    ```

    Async version replaces the client with `openai.AsyncOpenAI` and `embed()` becomes `async def`. See [`examples/reference_embedders/openai_embedder.py`](https://github.com/anthonynystrom/mneme/blob/main/examples/reference_embedders/openai_embedder.py) for both.

=== "sentence-transformers"

    Local CPU/GPU model, no network call. Great for cost-free, offline, or privacy-sensitive deployments.

    ```python
    import numpy as np


    class SentenceTransformersEmbedder:
        def __init__(self, model_name="all-MiniLM-L6-v2", *, device=None, normalize=True):
            from sentence_transformers import SentenceTransformer

            self._model_name = model_name
            self._model = SentenceTransformer(model_name, device=device)
            self._normalize = normalize
            self._dim = int(self._model.get_sentence_embedding_dimension())

        @property
        def dim(self) -> int:
            return self._dim

        @property
        def fingerprint(self) -> str:
            return f"sentence-transformers:{self._model_name}:n{int(self._normalize)}:dim{self._dim}"

        def embed(self, text: str) -> np.ndarray:
            v = self._model.encode(text, normalize_embeddings=self._normalize, convert_to_numpy=True)
            return v.astype(np.float32, copy=False)
    ```

    For an async cache, wrap with `to_async_embedder` (see [Async quickstart](quickstart-async.md)).

=== "AWS Bedrock"

    Titan Text Embeddings v2 (256/512/1024 dim) or Cohere Embed (1024 dim). Sync only via `boto3`.

    ```python
    import json
    import numpy as np


    class BedrockTitanEmbedder:
        def __init__(self, client, *, model_id="amazon.titan-embed-text-v2:0", dimensions=1024, normalize=True):
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
            body = json.dumps({"inputText": text, "dimensions": self._dim, "normalize": self._normalize})
            resp = self._client.invoke_model(modelId=self._model_id, body=body)
            payload = json.loads(resp["body"].read())
            v = np.asarray(payload["embedding"], dtype=np.float32)
            if not self._normalize:
                n = float(np.linalg.norm(v))
                v = v / n if n > 0 else v
            return v
    ```

    See [`examples/reference_embedders/bedrock_embedder.py`](https://github.com/anthonynystrom/mneme/blob/main/examples/reference_embedders/bedrock_embedder.py) for Cohere too.

=== "Ollama"

    Self-hosted local model. Default port 11434.

    ```python
    import numpy as np
    import requests


    class OllamaEmbedder:
        def __init__(self, model="nomic-embed-text", *, url="http://localhost:11434", dim=768):
            self._model = model
            self._url = url.rstrip("/")
            self._dim = dim

        @property
        def dim(self) -> int:
            return self._dim

        @property
        def fingerprint(self) -> str:
            return f"ollama:{self._model}:dim{self._dim}"

        def embed(self, text: str) -> np.ndarray:
            resp = requests.post(
                f"{self._url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            v = np.asarray(resp.json()["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v
    ```

    Async version uses `httpx.AsyncClient`. See [`examples/reference_embedders/ollama_embedder.py`](https://github.com/anthonynystrom/mneme/blob/main/examples/reference_embedders/ollama_embedder.py).

## Picking a model dimension

Bigger isn't always better. Higher-dim embeddings:

- Cost more memory in the in-memory matrix (`d × n × 4 bytes` for fp32).
- Cost more memory bandwidth on every `search()` (`d × n × 4 bytes` to read + matvec).
- Are not necessarily more discriminative for short queries.

For chatbot intent classification or paraphrase detection on short queries (10–30 tokens), 384–768 dim usually beats 1536 on cost without losing meaningful accuracy. For long-context semantic search, 1024+ helps.

Trade memory for either smaller models (`all-MiniLM-L6-v2` is 384) or [int8 quantization](../concepts/quantization.md) of larger ones.

## What if I get the contract wrong

| Mistake | What happens |
| --- | --- |
| `embed()` returns a list, not a `numpy.ndarray` | The cache will `np.asarray()` it but slowly. Convert once at the source. |
| Vector dtype is `float64` | Forced down to `float32` on insert. Costs a copy per `put`/`get`. Cast at the source. |
| Vector is not L2-normalized | Threshold needs to drift; calibration produces lower thresholds. Normalize. |
| `fingerprint` is the same after a model change | The cache happily mixes vectors from two models. Garbage matches. Always change the fingerprint when the model changes. |
| `dim` doesn't match `embed()`'s output | `cache.put()` raises a shape error. Make `dim` a property of the actual model. |

## Where to go next

- **[Your first cached LLM](your-first-cached-llm.md)** - wrapping an actual LLM call.
- **[Embedders concept page](../concepts/embedders.md)** - fingerprints, dim mismatches, the rebuild-on-mismatch path.
- **[Calibration](../guides/calibration.md)** - finding the right `similarity_threshold` for your embedder.
