"""Calibrate ``similarity_threshold`` for the showcase's embedder + corpus.

Builds paraphrase / distractor pairs from ``seed_data.py`` and runs them
through ``mneme.tools.calibrate.find_threshold``. Prints the recommended
threshold + precision/recall curve. Update ``config.SIMILARITY_THRESHOLD``
to the recommended value if you change embedder or corpus.

Run:
    python calibrate.py
"""

from __future__ import annotations

import config
from embedder import SentenceTransformersEmbedder
from seed_data import distractor_pairs, paraphrase_pairs

from mneme.tools.calibrate import find_threshold, precision_recall_curve


def main() -> None:
    print(f"loading {config.EMBEDDER_MODEL} …")
    embedder = SentenceTransformersEmbedder()
    print(f"  dim={embedder.dim}, fingerprint={embedder.fingerprint}")

    pos = paraphrase_pairs()
    neg = distractor_pairs()
    # Cap negatives at 5x positives so the search is reasonable.
    if len(neg) > 5 * len(pos):
        neg = neg[: 5 * len(pos)]
    print(f"  paraphrase pairs: {len(pos)}")
    print(f"  distractor pairs: {len(neg)}")

    print("\n=== find_threshold (target=f1, vector_dtype=float16) ===")
    result = find_threshold(
        paraphrase_pairs=pos,
        distractor_pairs=neg,
        embedder=embedder,
        target_metric="f1",
        vector_dtype=config.VECTOR_DTYPE,  # type: ignore[arg-type]
    )
    print(
        f"  recommended threshold: {result.threshold:.3f}\n"
        f"  precision={result.precision:.3f}  recall={result.recall:.3f}  "
        f"f1={result.f1:.3f}"
    )

    print("\n=== precision_recall_curve (every 10th grid point) ===")
    curve = precision_recall_curve(
        paraphrase_pairs=pos,
        distractor_pairs=neg,
        embedder=embedder,
        vector_dtype=config.VECTOR_DTYPE,  # type: ignore[arg-type]
    )
    for t, p, r in curve[::5]:
        print(f"  threshold={t:.2f}  precision={p:.3f}  recall={r:.3f}")

    print(
        f"\nIf the recommended threshold differs meaningfully from "
        f"config.SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}, "
        "update config.py."
    )


if __name__ == "__main__":
    main()
