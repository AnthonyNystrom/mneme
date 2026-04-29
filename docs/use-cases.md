# Use cases

`mneme` is marketed as an LLM cache because that's the sharpest hook today. Underneath, it's **semantic memoization**: any expensive function `f(input) → output` where the input is embeddable and small variations should reuse the same result.

This page collects five patterns that fit the same shape. Each one is a real workload that benefits from the same machinery, and each one has a runnable example under [`examples/use_cases/`](https://github.com/anystrom/mneme/tree/main/examples/use_cases) you can execute against a checkout of the repo.

## 1. RAG retrieval result caching

**What it is.** Cache "user question → top-k document chunks" so paraphrases of the same question reuse the same retrieval result.

**Why mneme fits.** Cross-encoder rerankers are 10–100× the cost of a cache lookup. Many production RAG pipelines re-rank the same intent under slightly different phrasings dozens of times per minute. The first call pays the cost; subsequent paraphrases hit Layer 2 in milliseconds.

```python
import json
from mneme import SemanticCache

def retrieve(question: str, namespace: str = "rag") -> list[dict]:
    """Cache-aware retrieval. Replace `expensive_retrieve` with your own
    vector DB query + reranker pipeline."""
    hit = cache.get(question, namespace=namespace)
    if hit is not None:
        return json.loads(hit.response)         # cached top-k, <5 ms

    chunks = expensive_retrieve(question)       # vector search + rerank, 100-500 ms
    cache.put(question, json.dumps(chunks), namespace=namespace)
    return chunks
```

Pair with [confidence-and-validators](concepts/confidence-and-validators.md) to refresh stale retrievals - e.g. `confidence_fn` that drops to 0 after the document index version changes.

Runnable: [`examples/use_cases/rag_retrieval.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/rag_retrieval.py)

## 2. Translation caching

**What it is.** Cache "source text → translated text" with semantic match on the source. Multiple phrasings of the same idea reuse one paid translation.

**Why mneme fits.** Translation APIs charge per character. High-paraphrase corpora (chat logs, support transcripts, user-generated content) repeat themselves heavily; an exact-match cache catches duplicates but misses paraphrases. Layer 2 catches the paraphrases.

```python
def translate(text: str, target_lang: str) -> str:
    namespace = f"translate:en-{target_lang}"
    hit = cache.get(text, namespace=namespace)
    if hit is not None:
        return hit.response                     # cached translation

    translated = call_translation_api(text, target_lang)   # billed per character
    cache.put(text, translated, namespace=namespace)
    return translated
```

Each `(source_lang, target_lang)` pair gets its own namespace so a German cache hit can't leak into a French request. Per-namespace LRU quotas (see [Multi-tenant](concepts/multi-tenant.md)) cap each language pair independently.

Runnable: [`examples/use_cases/translation.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/translation.py)

## 3. Semantic deduplication

**What it is.** Different shape: you don't care about the response, you care about the *similarity score*. The cache becomes a "have I seen something this close before?" detector.

**Why mneme fits.** `Hit.similarity` is exposed on every Layer 2 hit. Set a threshold, treat hits as duplicates, treat misses as novel. Persistence means the dedup state survives restarts.

```python
SIMILAR_ENOUGH = 0.92

def is_duplicate(article_text: str, namespace: str = "dedup") -> bool:
    """Returns True if a near-duplicate has already been processed."""
    hit = cache.get(article_text, namespace=namespace)
    if hit is not None and hit.similarity >= SIMILAR_ENOUGH:
        return True

    cache.put(article_text, "", namespace=namespace)   # response is irrelevant
    return False


# In a data pipeline:
for article in incoming_articles:
    if is_duplicate(article.body):
        skipped_dup_count += 1
        continue
    process(article)
```

Useful for:

- **News / RSS pipelines** - drop wire-service duplicates that got reworded by the syndicator.
- **Customer feedback ingestion** - same complaint phrased twenty ways shouldn't fan out to twenty tickets.
- **Document-submission deduplication** - academic plagiarism, contract templates.
- **Anomaly detection (inverted)** - *misses* are interesting; novel content far from anything seen before is the signal.

The default threshold here (0.92) is stricter than the LLM-cache default (0.85) because false positives in dedup discard real content. Calibrate it; see [Calibration](guides/calibration.md).

!!! note "Validator gotcha"

    The cache's default `Validator` rejects empty responses on Layer-2 lookup, so storing `cache.put(content, "")` will not work for the dedup pattern - the entry exists but Layer-2 hits skip it. Use a non-empty marker like `"seen"` (see the runnable example) or pass a custom `validator=` that accepts empty strings.

Runnable: [`examples/use_cases/dedup.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/dedup.py)

## 4. Classification result caching

**What it is.** Any "input → label" task: spam/ham, sentiment, content moderation, language detection, intent, support-ticket category. The classifier doesn't have to be an LLM.

**Why mneme fits.** Classifiers - sklearn, fastText, BERT-based, even rules engines + scoring - produce stable labels for stable inputs. mneme caches the label. Paraphrased inputs reuse the cached label without re-running the model.

```python
def classify(text: str, namespace: str = "moderation") -> str:
    hit = cache.get(text, namespace=namespace)
    if hit is not None:
        return hit.response                     # "safe" / "spam" / "abusive" / ...

    label = my_classifier.predict([text])[0]    # any model
    cache.put(text, label, namespace=namespace)
    return label
```

Combined with [validators](concepts/confidence-and-validators.md), you can refuse to cache uncertain predictions (e.g. classifier returned `"unknown"`) so the next paraphrase falls through to the model.

This works for **anything that takes text in and returns a category**, not just LLM-based classification. The showcase ([Showcase](showcase.md)) is the LLM variant; the same pattern applies to a sklearn pipeline you trained five years ago.

Runnable: [`examples/use_cases/classification.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/classification.py)

## 5. Agent memory

**What it is.** LLM-driven agents need to remember prior decisions so they're consistent on similar inputs. mneme provides "task embedding → outcome/plan" lookup.

**Why mneme fits.** Agents face the same paraphrase problem as users - "summarize this PR" and "give me a summary of this PR" should land on the same plan. Semantic match catches it; persistence means the agent's memory survives restarts. Per-agent namespacing keeps memories isolated.

```python
def execute_task(task_description: str, agent_id: str) -> str:
    namespace = f"agent:{agent_id}"
    hit = cache.get(task_description, namespace=namespace)
    if hit is not None and hit.confidence >= 0.7:
        # Reuse the prior plan; consistency over re-derivation.
        return hit.response

    plan = run_agent_loop(task_description)            # expensive
    cache.put(task_description, plan, namespace=namespace)
    return plan
```

The `confidence >= 0.7` gate (see [Confidence](concepts/confidence-and-validators.md)) drops stale memories - useful when the agent's environment changes (new tools available, policy update, etc.). Custom `confidence_fn=` can encode "drop memories older than the most recent agent version".

Runnable: [`examples/use_cases/agent_memory.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/agent_memory.py)

## What ties them together

Every pattern above is the same three lines:

```python
hit = cache.get(input_, namespace=...)
if hit is not None:
    return hit.response
result = expensive_thing(input_)
cache.put(input_, result, namespace=...)
return result
```

The differences are:

- **What `expensive_thing` is** - LLM, search, translator, classifier, agent.
- **Whether you read `hit.response` or `hit.similarity`** - cache lookup vs dedup detection.
- **How aggressive the threshold is** - calibrated per use case, not one global default.
- **How namespaces partition** - per tenant, per language, per agent.

Every other piece - multi-process coordination, persistence, quotas, metrics, calibration, sync/async parity - is shared across all five patterns because the shape of the problem is the same.

## Runnable examples

Each pattern above has a self-contained script under [`examples/use_cases/`](https://github.com/anystrom/mneme/tree/main/examples/use_cases). They share a small toy embedder ([`_embedder.py`](https://github.com/anystrom/mneme/blob/main/examples/use_cases/_embedder.py)) - a feature-hashing bag-of-words - so the demos run with no extra dependencies and produce visible Layer-1 / Layer-2 transitions in their stdout.

```bash
# After `pip install -e .` from a checkout:
python examples/use_cases/rag_retrieval.py
python examples/use_cases/translation.py
python examples/use_cases/dedup.py
python examples/use_cases/classification.py
python examples/use_cases/agent_memory.py
```

In production, swap the toy embedder for a real one - see [Bring your own embedder](getting-started/bring-your-own-embedder.md). The cache wrapper code stays the same.

## Where to go next

- **[Your first cached LLM](getting-started/your-first-cached-llm.md)** - the canonical pattern in detail.
- **[Multi-tenant](concepts/multi-tenant.md)** - namespaces are the lever for "different request, different cache slice".
- **[Calibration](guides/calibration.md)** - picking the right threshold per use case.
- **[Showcase](showcase.md)** - pattern #4 (classification) live, with a UI.
