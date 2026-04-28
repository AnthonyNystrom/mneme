"""LRU eviction with namespace-quota precedence per PRD §11.6.

After each ``put``:

1. If ``namespace_quotas[namespace]`` is set and ``count(namespace) > quota``,
   evict the bottom 10% within the namespace (minimum 1, capped at the
   excess so we don't over-evict).
2. Else if global ``max_entries`` is set and total count exceeds it, evict
   the bottom 10% globally.

Namespace quotas take precedence over the global cap.
"""

from __future__ import annotations

from collections.abc import Callable

from ._types import Store

OnEvicted = Callable[[int], None]
"""Callback invoked with each successfully evicted row id. The cache layer
uses this to keep its in-memory index in sync with store deletes."""

_DEFAULT_BATCH_PCT = 0.10
_DEFAULT_MIN_BATCH = 1


def _compute_batch(target: int, excess: int) -> int:
    """Eviction batch size: large enough to restore the cap, plus a 10% of
    target buffer to amortize eviction cost; floored at ``MIN_BATCH``.

    Examples:
    - target=100, excess=20 -> max(20, 10, 1) = 20 (excess dominates)
    - target=1000, excess=50 -> max(50, 100, 1) = 100 (10% buffer dominates)
    - target=2, excess=1 -> max(1, 0, 1) = 1 (MIN_BATCH floor)
    """
    pct_batch = int(target * _DEFAULT_BATCH_PCT)
    return max(excess, pct_batch, _DEFAULT_MIN_BATCH)


def _evict_ids(store: Store, ids: list[int], on_evicted: OnEvicted | None) -> int:
    deleted = 0
    for id_ in ids:
        if store.delete_by_id(id_):
            deleted += 1
            if on_evicted is not None:
                on_evicted(id_)
    return deleted


def evict_for_namespace(
    store: Store,
    namespace: str,
    quota: int,
    *,
    on_evicted: OnEvicted | None = None,
) -> int:
    """Evict LRU entries within ``namespace`` until count <= quota.

    ``on_evicted(row_id)`` is called for each successfully removed row so the
    caller can keep auxiliary structures (e.g. an in-memory index) in sync.
    """
    count = store.count(namespace)
    if count <= quota:
        return 0
    batch = _compute_batch(quota, count - quota)
    ids = list(store.iter_lru_ids(batch, namespace=namespace))
    return _evict_ids(store, ids, on_evicted)


def evict_global(
    store: Store,
    max_entries: int,
    *,
    on_evicted: OnEvicted | None = None,
) -> int:
    """Evict LRU entries globally until total count <= max_entries."""
    count = store.count()
    if count <= max_entries:
        return 0
    batch = _compute_batch(max_entries, count - max_entries)
    ids = list(store.iter_lru_ids(batch))
    return _evict_ids(store, ids, on_evicted)


def maybe_evict(
    store: Store,
    namespace: str,
    *,
    namespace_quotas: dict[str, int] | None = None,
    max_entries: int | None = None,
    on_evicted: OnEvicted | None = None,
) -> dict[str, int]:
    """Apply the §11.6 policy to one ``put``.

    Returns a mapping of ``{namespace -> evictions}`` (empty if nothing was
    evicted) so the cache layer can emit metrics events. Per-id side effects
    flow through ``on_evicted``.
    """
    out: dict[str, int] = {}
    if namespace_quotas and namespace in namespace_quotas:
        evicted = evict_for_namespace(
            store, namespace, namespace_quotas[namespace], on_evicted=on_evicted
        )
        if evicted:
            out[namespace] = evicted
        return out
    if max_entries is not None:
        evicted = evict_global(store, max_entries, on_evicted=on_evicted)
        if evicted:
            # Global eviction can touch any namespace; report under the
            # triggering namespace for metric attribution.
            out[namespace] = evicted
    return out


__all__ = ["OnEvicted", "evict_for_namespace", "evict_global", "maybe_evict"]
