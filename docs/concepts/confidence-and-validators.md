# Confidence and validators

`mneme` ships two pluggable hooks that decide whether a cached entry is *trustworthy* (`confidence_fn`) and whether a candidate response is *acceptable* on insert (`validator`). Both have sensible defaults; you override either when the defaults don't match your domain.

## Confidence

Every `Hit` carries a `confidence: float` between 0 and 1. The cache itself never blocks on confidence - the value rides along so the *caller* can decide what to do with a low-confidence hit.

```python
hit = cache.get(query)
if hit and hit.confidence < 0.5:
    response = call_llm(query)              # cached value too stale; refresh
    cache.put(query, response)
elif hit:
    response = hit.response                 # confident enough; use it
else:
    response = call_llm(query)
    cache.put(query, response)
```

### Default scorer: 24-hour half-life

The default confidence function decays cached entries by half every 24 hours:

```python
def default_confidence(similarity: float, age_seconds: int, metadata: dict) -> float:
    half_life = 24 * 60 * 60
    decay = 0.5 ** (age_seconds / half_life)
    return similarity * decay
```

So a Layer-2 hit with `similarity=0.9` returned 36 hours after insert has confidence `0.9 × 0.5^1.5 ≈ 0.32`. A 1-minute-old exact hit has confidence ~`1.0`.

### Custom scorers

Pass `confidence_fn=` to `SemanticCache(...)`:

```python
def stricter_confidence(similarity: float, age_seconds: int, metadata: dict) -> float:
    # Decay faster: 6-hour half-life. And drop to 0 after 7 days.
    if age_seconds > 7 * 86400:
        return 0.0
    return similarity * (0.5 ** (age_seconds / (6 * 60 * 60)))


cache = SemanticCache(..., confidence_fn=stricter_confidence)
```

The scorer can read `metadata` you stored on the original `put`:

```python
cache.put(query, response, metadata={"source_model": "gpt-4o", "ts": time.time()})

def model_aware_confidence(similarity, age_seconds, metadata):
    if metadata.get("source_model") == "gpt-3.5-turbo":
        return 0.0           # outdated model; force refresh
    return similarity * (0.5 ** (age_seconds / 86400))

cache = SemanticCache(..., confidence_fn=model_aware_confidence)
```

This is how you turn the cache into a domain-aware policy layer without giving up its lookup speed.

## Validators

A `Validator` decides whether a response is *worth caching* on `put`. It runs after the LLM produced the response but before the cache writes it.

```python
def is_real_answer(response: str) -> bool:
    if not response.strip():
        return False
    if response.startswith("[ERROR]"):
        return False
    if response.startswith("[LLM Error]"):
        return False
    return True


cache = SemanticCache(..., validator=is_real_answer)
```

### Default validator

The shipped default rejects:

- Empty strings.
- Responses starting with `[LLM Error]` or `[ERROR]`.

That's the minimum useful. Real production validators usually add:

- Length bounds (caching a 50KB hallucination wastes bytes).
- Schema checks (if the response is JSON, parse it; reject malformed).
- Refusal-text patterns ("As an AI language model, I can't…" - don't cache).
- Confidence-from-the-LLM checks (if the model self-reported low confidence, skip).

### Behavior on rejection

`cache.put(query, response)` with a response the validator rejects is a **silent no-op** - no exception, no cache entry. The metrics hook fires with `event="put_rejected"` so you can monitor rejection rates. Subsequent `get(query)` calls will still miss until the caller `put()`s a response that passes.

This is intentional: the caller has already done the LLM work; failing the put loudly would just propagate the validator's pickiness back to the call site. Soft rejection lets the cache stay clean without changing the call pattern.

## Putting them together

A typical production `get` wrapper that uses both:

```python
def cached_classify(query: str) -> str:
    hit = cache.get(query, namespace="support")
    if hit and hit.confidence >= 0.7:        # confidence gate
        return hit.response

    intent = call_llm(query)                 # LLM call
    cache.put(query, intent)                 # validator runs; bad responses skipped
    return intent
```

- `hit.confidence >= 0.7` skips stale or low-similarity hits.
- The validator silently drops `[ERROR]`-shaped responses if the LLM fails partway.

## Where to go next

- **[API reference: types](../reference/types.md)** - the exact `ConfidenceFn` and `Validator` signatures.
- **[Calibration](../guides/calibration.md)** - picking the threshold, separately from picking the confidence cutoff.
- **[Metrics](../guides/metrics.md)** - observe `put_rejected` and per-namespace confidence distributions.
