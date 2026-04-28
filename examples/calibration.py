"""Calibrate the similarity threshold for your paraphrase / distractor data.

The cache treats two queries as a semantic match when their cosine
similarity exceeds ``similarity_threshold`` (default 0.85). The right
value depends on your embedder *and* the ``vector_dtype`` you'll use in
production. This script sweeps a grid and picks the threshold that
maximizes F1 (or maximizes precision / recall under a constraint).

Run:
    python examples/calibration.py
"""

from __future__ import annotations

import hashlib

import numpy as np

from mneme.tools.calibrate import find_threshold, precision_recall_curve


class ToyEmbedder:
    """Deterministic-by-text 64-dim embedder. Replace with a real one."""

    dim = 64
    fingerprint = "toy:demo:v1"

    def embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((self.dim + len(digest) - 1) // len(digest)))[: self.dim]
        v = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32) - 128.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


# Pairs that *should* match (paraphrases of the same intent). In real use
# these come from production logs you've labeled (or LLM-generated paraphrases).
PARAPHRASES = [
    ("How do I reset my password?", "I forgot my password, what do I do?"),
    ("Where do I cancel my subscription?", "How can I unsubscribe?"),
    ("What's your refund policy?", "Can I get my money back?"),
    ("How do I contact support?", "Where can I reach customer service?"),
    ("How do I update my billing info?", "Where do I change my credit card?"),
]

# Pairs that should NOT match (different intents). Real distractors come
# from random pairings of unrelated queries.
DISTRACTORS = [
    ("How do I reset my password?", "What's the weather today?"),
    ("Where do I cancel my subscription?", "Tell me a joke"),
    ("What's your refund policy?", "How do I bake bread?"),
    ("How do I contact support?", "Translate this to French"),
    ("How do I update my billing info?", "What's 2 + 2?"),
]


def main() -> None:
    embedder = ToyEmbedder()

    # Method 1: just give me the best threshold under a constraint.
    print("=== find_threshold (F1, min_precision=0.9) ===")
    for dtype in ("float32", "int8"):
        result = find_threshold(
            paraphrase_pairs=PARAPHRASES,
            distractor_pairs=DISTRACTORS,
            embedder=embedder,
            target_metric="f1",
            min_precision=0.9,
            vector_dtype=dtype,  # type: ignore[arg-type]
        )
        print(
            f"  dtype={dtype:8s}  threshold={result.threshold:.3f}  "
            f"precision={result.precision:.3f}  recall={result.recall:.3f}  "
            f"f1={result.f1:.3f}"
        )

    # Method 2: get the full precision-recall curve to plot or inspect.
    print("\n=== precision_recall_curve (float32) ===")
    curve = precision_recall_curve(
        paraphrase_pairs=PARAPHRASES,
        distractor_pairs=DISTRACTORS,
        embedder=embedder,
        vector_dtype="float32",
    )
    print(f"  {len(curve)} grid points")
    for t, p, r in curve[::10]:  # sample every 10th
        print(f"  threshold={t:.2f}  precision={p:.3f}  recall={r:.3f}")


if __name__ == "__main__":
    main()
