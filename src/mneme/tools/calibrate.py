"""Threshold calibration for ``similarity_threshold``.

Given labeled paraphrase pairs (positives) and distractor pairs (negatives),
sweep a grid of thresholds and report the one that optimizes the target
metric (f1 by default).

Calibrate against the same ``vector_dtype`` you'll run in production: int8
shifts the precision/recall curve relative to fp32, and a fp32-tuned
threshold transferred to int8 will degrade quality.

Public API: ``find_threshold``, ``precision_recall_curve``,
``CalibrationResult``. CLI: ``python -m mneme.tools.calibrate ...``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from .._quantization import dequantize, quantize

if TYPE_CHECKING:
    from .._types import Embedder, VectorDtype

_DEFAULT_GRID = [round(0.5 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    pr_curve: list[tuple[float, float, float]] = field(default_factory=list)


def _l2_normalize(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32, copy=False)


def _embed_with_dtype(text: str, embedder: Embedder, dtype: VectorDtype) -> npt.NDArray[np.float32]:
    """Embed, L2-normalize, then quantize+dequantize so the score reflects
    the dtype used in production."""
    raw = embedder.embed(text).astype(np.float32, copy=False)
    raw = _l2_normalize(raw)
    if dtype == "float32":
        return raw
    qv = quantize(raw, dtype)
    return dequantize(qv, dtype)


def _score_pair(a: str, b: str, embedder: Embedder, dtype: VectorDtype) -> float:
    va = _embed_with_dtype(a, embedder, dtype)
    vb = _embed_with_dtype(b, embedder, dtype)
    return float(va @ vb)


def _scores(pairs: list[tuple[str, str]], embedder: Embedder, dtype: VectorDtype) -> list[float]:
    return [_score_pair(a, b, embedder, dtype) for a, b in pairs]


def _metrics_at(
    pos_scores: list[float], neg_scores: list[float], threshold: float
) -> tuple[float, float, float]:
    tp = sum(1 for s in pos_scores if s >= threshold)
    fn = len(pos_scores) - tp
    fp = sum(1 for s in neg_scores if s >= threshold)
    # No positives predicted -> precision is undefined; pick 1.0 so the
    # threshold is preserved as a candidate when ``min_precision`` is set.
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0.0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def precision_recall_curve(
    paraphrase_pairs: list[tuple[str, str]],
    distractor_pairs: list[tuple[str, str]],
    embedder: Embedder,
    *,
    grid: list[float] | None = None,
    vector_dtype: VectorDtype = "float32",
) -> list[tuple[float, float, float]]:
    """Sweep ``grid`` (default 0.50..0.99 step 0.01) and return one tuple
    ``(threshold, precision, recall)`` per grid point."""
    g = grid if grid is not None else list(_DEFAULT_GRID)
    pos_scores = _scores(paraphrase_pairs, embedder, vector_dtype)
    neg_scores = _scores(distractor_pairs, embedder, vector_dtype)
    out: list[tuple[float, float, float]] = []
    for t in g:
        p, r, _f = _metrics_at(pos_scores, neg_scores, t)
        out.append((float(t), p, r))
    return out


def find_threshold(
    paraphrase_pairs: list[tuple[str, str]],
    distractor_pairs: list[tuple[str, str]],
    embedder: Embedder,
    *,
    target_metric: Literal["f1", "precision", "recall"] = "f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
    grid: list[float] | None = None,
    vector_dtype: VectorDtype = "float32",
) -> CalibrationResult:
    """Pick the threshold that maximizes ``target_metric`` subject to the
    optional ``min_precision`` and ``min_recall`` constraints."""
    if not paraphrase_pairs:
        raise ValueError("calibrate: paraphrase_pairs cannot be empty.")
    if not distractor_pairs:
        raise ValueError("calibrate: distractor_pairs cannot be empty.")
    g = grid if grid is not None else list(_DEFAULT_GRID)
    pos_scores = _scores(paraphrase_pairs, embedder, vector_dtype)
    neg_scores = _scores(distractor_pairs, embedder, vector_dtype)

    pr_curve: list[tuple[float, float, float]] = []
    best: tuple[float, float, float, float] | None = None  # (score, t, P, R)
    for t in g:
        p, r, f1 = _metrics_at(pos_scores, neg_scores, t)
        pr_curve.append((float(t), p, r))
        if min_precision is not None and p < min_precision:
            continue
        if min_recall is not None and r < min_recall:
            continue
        score = {"f1": f1, "precision": p, "recall": r}[target_metric]
        if best is None or score > best[0]:
            best = (score, float(t), p, r)
    if best is None:
        raise ValueError(
            f"No threshold satisfied min_precision={min_precision} and "
            f"min_recall={min_recall}. Remediation: relax the constraints "
            f"or expand the grid."
        )
    score, t, p, r = best
    f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
    return CalibrationResult(
        threshold=t,
        precision=p,
        recall=r,
        f1=f1,
        pr_curve=pr_curve,
    )


# --- CLI ---


_EXIT_OK = 0
_EXIT_ARGS = 1
_EXIT_FILE_NOT_FOUND = 2
_EXIT_EMBEDDER_IMPORT = 3
_EXIT_NO_THRESHOLD = 4
_EXIT_BAD_JSONL = 5


def _import_embedder(spec: str) -> Embedder:
    """Import an Embedder by ``module:attr`` spec."""
    if ":" not in spec:
        raise ValueError(f"Embedder spec {spec!r} must be 'module:attr'.")
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import module {module_name!r}: {exc}.") from exc
    if not hasattr(module, attr):
        raise AttributeError(f"Module {module_name!r} has no attribute {attr!r}.")
    obj = getattr(module, attr)
    if callable(obj) and not (hasattr(obj, "dim") and hasattr(obj, "fingerprint")):
        # Allow factories that produce embedders.
        obj = obj()
    return obj  # type: ignore[no-any-return]


def _load_jsonl_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON ({exc}).") from exc
            if "a" not in obj or "b" not in obj:
                raise ValueError(f"{path}:{lineno}: each line must have 'a' and 'b' string fields.")
            pairs.append((str(obj["a"]), str(obj["b"])))
    return pairs


def _format_human(result: CalibrationResult) -> str:
    lines = [
        f"Optimal threshold: {result.threshold:.3f}",
        f"Precision:         {result.precision:.3f}",
        f"Recall:            {result.recall:.3f}",
        f"F1:                {result.f1:.3f}",
        "",
        "Precision-recall curve (every 0.05):",
        "  threshold  precision  recall",
    ]
    for t, p, r in result.pr_curve:
        if abs(round(t * 20) - t * 20) < 1e-9:  # multiples of 0.05
            lines.append(f"  {t:5.2f}      {p:.3f}      {r:.3f}")
    return "\n".join(lines)


def _format_json(result: CalibrationResult) -> str:
    return json.dumps(
        {
            "threshold": result.threshold,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "pr_curve": [list(t) for t in result.pr_curve],
        },
        indent=2,
    )


def _format_csv(result: CalibrationResult) -> str:
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["threshold", "precision", "recall", "f1"])
    for t, p, r in result.pr_curve:
        f1 = 0.0 if (p + r) == 0 else 2.0 * p * r / (p + r)
        w.writerow([t, p, r, f1])
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mneme.tools.calibrate",
        description="Find the optimal similarity threshold for a semantic cache.",
    )
    parser.add_argument(
        "--paraphrases",
        required=True,
        type=Path,
        help="JSONL file of paraphrase pairs (positive examples).",
    )
    parser.add_argument(
        "--distractors",
        required=True,
        type=Path,
        help="JSONL file of distractor pairs (negative examples).",
    )
    parser.add_argument(
        "--embedder",
        required=True,
        type=str,
        help="Python import path to an Embedder, e.g. 'myproject:embedder'.",
    )
    parser.add_argument("--target-metric", choices=["f1", "precision", "recall"], default="f1")
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--vector-dtype", choices=["float32", "float16", "int8"], default="float32")
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Comma-separated thresholds. Default: 0.50..0.99 step 0.01.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=["json", "csv", "human"], default="human")
    args = parser.parse_args(argv)

    try:
        paraphrases = _load_jsonl_pairs(args.paraphrases)
        distractors = _load_jsonl_pairs(args.distractors)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc}", file=sys.stderr)
        return _EXIT_FILE_NOT_FOUND
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_BAD_JSONL

    try:
        embedder = _import_embedder(args.embedder)
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"error: failed to import embedder: {exc}", file=sys.stderr)
        return _EXIT_EMBEDDER_IMPORT

    grid: list[float] | None = None
    if args.grid is not None:
        try:
            grid = [float(x.strip()) for x in args.grid.split(",") if x.strip()]
        except ValueError as exc:
            print(f"error: bad --grid: {exc}", file=sys.stderr)
            return _EXIT_ARGS

    try:
        result = find_threshold(
            paraphrases,
            distractors,
            embedder,
            target_metric=args.target_metric,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            grid=grid,
            vector_dtype=args.vector_dtype,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_NO_THRESHOLD

    if args.format == "human":
        out = _format_human(result)
    elif args.format == "json":
        out = _format_json(result)
    else:
        out = _format_csv(result)

    if args.output is not None:
        args.output.write_text(out, encoding="utf-8")
    else:
        print(out)
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())


__all__ = [
    "CalibrationResult",
    "find_threshold",
    "main",
    "precision_recall_curve",
]
