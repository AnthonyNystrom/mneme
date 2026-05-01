"""High-dimensional embeddings with int8 / float16 quantization.

Persistence is always float32 (the source of truth). The in-memory matrix
can be downcast per ``vector_dtype=`` to cut RAM 2x or 4x. This is the
right knob for high-dim models (1024+) on memory-constrained hosts.

NOTE: int8 search latency on pure NumPy at d>=1536 is bandwidth-bound
(see README "Performance baseline"). For low-latency *and* small
footprint, pair int8 with ``index_backend="hnsw"``.

Run:
    python examples/high_dim_quantized.py
"""

from __future__ import annotations

import numpy as np

from mneme import MemoryStore, SemanticCache


class HighDimEmbedder:
    """1536-dim Gaussian embedder. Toy; see reference_embedders/ for real ones."""

    def __init__(self, dim: int = 1536, fingerprint: str = "toy:gaussian:1536") -> None:
        self.dim = dim
        self.fingerprint = fingerprint
        self._rng = np.random.default_rng(42)

    def embed(self, text: str) -> np.ndarray:
        # Seed by query hash so identical queries embed identically.
        seed = abs(hash(text)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def main() -> None:
    embedder = HighDimEmbedder(dim=1536)

    for dtype in ("float32", "float16", "int8"):
        cache = SemanticCache(
            store=MemoryStore(),
            embedder=embedder,
            vector_dtype=dtype,  # type: ignore[arg-type]
        )
        try:
            for i in range(1000):
                cache.put(f"q{i}", f"r{i}")
            s = cache.stats()
            mb = s.memory_bytes_estimate / (1024 * 1024)
            print(f"dtype={dtype:8s}  entries={s.entries}  in-memory matrix ~{mb:5.1f} MB")
        finally:
            cache.close()


if __name__ == "__main__":
    main()
