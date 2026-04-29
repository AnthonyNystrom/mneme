# Calibration

The default `similarity_threshold=0.85` is a starting point, not a recommendation. The right value depends on your embedder, your corpus, and the `vector_dtype` you'll run in production. **Calibrate before you ship.**

`mneme.tools.calibrate` does this in one call.

## The core idea

You give the calibrator two lists of query pairs:

- **Paraphrase pairs** - pairs that *should* match. Same intent, different wording.
- **Distractor pairs** - pairs that should *not* match. Different intents.

It computes the cosine similarity for each pair under your embedder + dtype, then sweeps thresholds and reports precision/recall/F1 at each. You pick the threshold that matches your tolerance for false positives vs misses.

## Quick start

```python
from mneme.tools.calibrate import find_threshold

paraphrases = [
    ("How do I reset my password?", "I forgot my password, what now?"),
    ("Where do I cancel my subscription?", "How can I unsubscribe?"),
    ("What's your refund policy?", "Can I get my money back?"),
    # ... 50–500 of these for stable results
]

distractors = [
    ("How do I reset my password?", "What's the weather today?"),
    ("Where do I cancel?", "Tell me a joke"),
    # ... 5–10x more than paraphrases
]

result = find_threshold(
    paraphrase_pairs=paraphrases,
    distractor_pairs=distractors,
    embedder=my_embedder,
    target_metric="f1",
    min_precision=0.95,            # never accept below 95% precision
    vector_dtype="float16",        # the dtype you'll run in prod
)

print(f"threshold: {result.threshold:.3f}")
print(f"precision: {result.precision:.3f}")
print(f"recall:    {result.recall:.3f}")
print(f"f1:        {result.f1:.3f}")
```

Wire `result.threshold` into `SemanticCache(similarity_threshold=...)` and you're done.

## Sourcing pairs

Where the pairs come from matters more than how many you have. A few good sources:

| Source | Quality | Effort |
| --- | --- | --- |
| Production logs hand-labeled into intent buckets | best | high |
| LLM-generated paraphrases of seed queries | good | low |
| Public paraphrase datasets (PAWS, MRPC, Quora) | varies - domain mismatch hurts | low |
| Self-similarity (a query paired with itself rephrased) | poor - too easy | trivial |

The showcase's [seed_data.py](https://github.com/anthonynystrom/mneme/blob/main/examples/showcase/seed_data.py) shows the LLM-generated pattern: 7 intent clusters, ~10 paraphrases each, then automatic in-cluster pairs (paraphrases) + cross-cluster pairs (distractors). 73 messages → 345 paraphrase pairs + 1725 distractor pairs.

## Distractor count matters

Real workloads have far more distractor pairs than paraphrase pairs (most random query pairs are unrelated). If you give the calibrator equal counts, you'll over-estimate the threshold's precision. **Aim for 5–10× more distractors than paraphrases.**

The calibrator caps the search anyway; very-large distractor lists slow the sweep without changing the answer materially.

## Calibrate against the production dtype

`int8` quantization shifts cosine scores 1–3% on most embedders. A threshold calibrated on `float32` is wrong for `int8` and vice versa. Always pass `vector_dtype=` to match what you'll run:

```python
result = find_threshold(..., vector_dtype="int8")
```

If you're unsure which dtype you'll run, calibrate against the most-quantized option you'd consider - that gives you a threshold that holds up everywhere.

## Inspect the curve

`precision_recall_curve` returns the full sweep so you can pick a non-F1-optimal point:

```python
from mneme.tools.calibrate import precision_recall_curve

curve = precision_recall_curve(
    paraphrase_pairs=paraphrases,
    distractor_pairs=distractors,
    embedder=embedder,
    vector_dtype="float16",
)
for t, p, r in curve[::5]:                # every 5th grid point
    print(f"threshold={t:.2f}  precision={p:.3f}  recall={r:.3f}")
```

Use this when:

- **Precision must be ≥ X.** Find the highest-recall point on the curve where precision is still above X.
- **Recall must be ≥ Y.** Find the highest-precision point where recall is still above Y.
- **You want to plot it.** The curve is a list of `(threshold, precision, recall)` triples; pipe it into matplotlib / Chart.js / your dashboard.

## CLI

For one-shot calibration runs from a shell:

```bash
python -m mneme.tools.calibrate \
    --paraphrases fixtures/paraphrases.jsonl \
    --distractors fixtures/distractors.jsonl \
    --embedder my_module:embedder_factory \
    --vector-dtype int8 \
    --target f1 \
    --min-precision 0.95
```

Where `--embedder my_module:embedder_factory` is an importable factory function returning your embedder. JSONL files have one `{"a": "...", "b": "..."}` per line.

`python -m mneme.tools.calibrate --help` for the full flag list.

## When the answer is "all thresholds are bad"

Sometimes the calibrator returns a low F1 (say <0.5) with no obvious sweet spot. That's a signal - usually one of:

1. **Embedder is too weak for the task.** A 384-dim model can't separate "How do I reset my password?" from "How do I change my username?" - they're both account questions. Switch to a bigger model.
2. **Paraphrase pairs aren't actually paraphrases.** The corpus has cross-cluster pairs labeled as paraphrases (a common mistake when you grep by topic but two distinct intents share a topic).
3. **Distractor pairs are too easy.** If every distractor is wildly off-topic, the calibrator finds a high precision/recall solution that doesn't reflect production traffic. Mix in near-misses.
4. **Same-domain noise.** Customer-support corpora often have queries that *should* match but use different platform jargon ("renew" vs "extend" vs "auto-pay"). No threshold will catch these without domain-specific embeddings or fine-tuning.

Calibration is a diagnostic tool, not just a tuning knob. Low F1 means the cache won't help much for *that* embedder/corpus pair regardless of threshold.

## Re-calibrate when

- You change the embedder model.
- You change the embedder dimension (e.g. OpenAI `dimensions=` parameter).
- You change `vector_dtype`.
- Your corpus drifts substantially (new product lines, new languages).

## Where to go next

- **[examples/calibration.py](https://github.com/anthonynystrom/mneme/blob/main/examples/calibration.py)** - runnable example with a toy embedder.
- **[Confidence and validators](../concepts/confidence-and-validators.md)** - calibration picks the threshold; confidence picks the trust gate.
- **[Performance tuning](performance-tuning.md)** - what changes when threshold moves.
