"""Phase-7 sync cache tests: end-to-end orchestration of layers + lifecycle."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mneme import (
    CacheClosedError,
    EmbedderDimensionError,
    Hit,
    MemoryStore,
    SemanticCache,
    SQLiteStore,
)

from .fakes import FakeEmbedder, FlakyEmbedder, ParaphraseEmbedder
from .stores.inmemory_store import InMemoryStore

# --- Construction ---


def test_path_or_store_required_exactly_one(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with pytest.raises(ValueError, match="exactly one"):
        SemanticCache(embedder=e)
    with pytest.raises(ValueError, match="exactly one"):
        SemanticCache(path=tmp_path / "c.db", embedder=e, store=MemoryStore())


def test_embedder_required(tmp_path: Path):
    with pytest.raises(ValueError, match="embedder"):
        SemanticCache(path=tmp_path / "c.db")


def test_path_creates_default_sqlite_store(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    cache = SemanticCache(path=tmp_path / "c.db", embedder=e)
    try:
        assert isinstance(cache._store, SQLiteStore)
    finally:
        cache.close()


def test_custom_store_is_used_directly(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    store = InMemoryStore()
    cache = SemanticCache(store=store, embedder=e)
    try:
        assert cache._store is store
    finally:
        cache.close()


# --- Layer 1: exact match ---


def test_put_then_get_exact(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("hello", "world")
        hit = cache.get("hello")
        assert hit is not None
        assert hit.layer == "exact"
        assert hit.response == "world"
        assert hit.similarity == 1.0


def test_exact_match_does_not_call_embedder():
    """Layer 1 must short-circuit before calling the embedder."""
    e = FlakyEmbedder(dim=8, fail_on=2)  # fail on 2nd call
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("hi", "there")  # call 1 (put always embeds)
        # After put, embedder has been called once. A subsequent get on the
        # same exact query goes through layer 1 (no embed) — no failure.
        hit = cache.get("hi")
        assert hit is not None
        assert hit.layer == "exact"


def test_get_miss_returns_none(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        assert cache.get("never put") is None


def test_get_normalizes_query_by_default(tmp_path: Path):
    """Default normalize=True should make 'Hello' and 'hello' collide."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("Hello", "world")
        hit = cache.get("hello")  # casefolded match
        assert hit is not None
        assert hit.layer == "exact"


def test_get_normalize_off_keeps_case(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e, normalize=False) as cache:
        cache.put("Hello", "world")
        # Different case: layer 1 misses; layer 2 may not match for FakeEmbedder
        # (different seed -> different vector).
        hit = cache.get("hello")
        assert hit is None or hit.layer == "semantic"


# --- Layer 2: semantic match ---


def test_semantic_match_for_paraphrases():
    e = ParaphraseEmbedder(dim=32)
    with SemanticCache(store=MemoryStore(), embedder=e, similarity_threshold=0.5) as cache:
        cache.put("how do i reset password", "click forgot password")
        hit = cache.get("how do i reset my password")
        assert hit is not None
        # First put hit may be exact if normalized matches. The test query has
        # an extra word, so it's not an exact match — must come from layer 2.
        assert hit.layer == "semantic"
        assert hit.similarity >= 0.5
        assert hit.response == "click forgot password"


def test_below_threshold_returns_none():
    e = ParaphraseEmbedder(dim=32)
    with SemanticCache(store=MemoryStore(), embedder=e, similarity_threshold=0.99) as cache:
        cache.put("how do i reset password", "click forgot password")
        # An unrelated query won't clear a 0.99 threshold.
        assert cache.get("what is the weather") is None


def test_caller_supplied_embedding_short_circuits_embedder():
    e = FlakyEmbedder(dim=8, fail_on=1)
    pre_embedded = np.zeros(8, dtype=np.float32)
    pre_embedded[0] = 1.0
    with SemanticCache(store=MemoryStore(), embedder=FakeEmbedder(dim=8)) as cache:
        cache.put("query", "response", embedding=pre_embedded)
    # FlakyEmbedder is constructed but never called via embed() since both
    # put and get receive pre-embedded vectors.
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("query", "response", embedding=pre_embedded)
        hit = cache.get("query", embedding=pre_embedded)
        assert hit is not None


def test_dim_mismatch_at_get_returns_miss():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("q", "r")
        # Caller-supplied wrong-shape embedding -> dim_mismatch miss.
        bad = np.zeros(16, dtype=np.float32)
        assert cache.get("never-existed", embedding=bad) is None


def test_dim_mismatch_at_put_raises():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        bad = np.zeros(16, dtype=np.float32)
        with pytest.raises(EmbedderDimensionError, match="dim"):
            cache.put("q", "r", embedding=bad)


# --- Bypass ---


def test_bypass_returns_none_and_increments_miss(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("hi", "there")
        before = cache.stats().misses
        assert cache.get("hi", bypass=True) is None
        assert cache.stats().misses == before + 1


# --- TTL ---


def test_ttl_expiration_at_layer_1(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("hi", "there", ttl=1)
        time.sleep(1.1)
        assert cache.get("hi") is None  # expired -> miss


def test_default_ttl_applied_when_per_put_omits():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, default_ttl=1) as cache:
        cache.put("hi", "there")
        time.sleep(1.1)
        assert cache.get("hi") is None


def test_per_put_ttl_overrides_default_ttl():
    """A per-put ttl=longer must outlast the cache's default."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, default_ttl=1) as cache:
        cache.put("hi", "there", ttl=3600)
        time.sleep(1.1)
        # Default would have expired; per-put ttl keeps it alive.
        hit = cache.get("hi")
        assert hit is not None


# --- Validator ---


def test_validator_rejects_poisoned_response_at_put():
    """Default validator does not gate puts (puts are user-trusted)."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("q", "[LLM Error] timeout")
        # The default validator runs on layer-2 candidates, not layer-1.
        # An exact-match hit returns the bad response unchanged.
        hit = cache.get("q")
        assert hit is not None
        assert hit.response == "[LLM Error] timeout"


def test_custom_validator_filters_layer_2_candidates():
    e = ParaphraseEmbedder(dim=16)

    def reject_anything(_response: str) -> bool:
        return False

    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        similarity_threshold=0.3,
        validator=reject_anything,
    ) as cache:
        cache.put("how reset password", "answer")
        # Layer 1 hits (exact) — validator does not gate exact matches in this
        # implementation. Use a different query so we go to layer 2.
        hit = cache.get("how reset my password")
        assert hit is None


# --- Eviction ---


def test_max_entries_evicts_lru_globally():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, max_entries=5) as cache:
        for i in range(8):
            cache.put(f"q{i}", f"r{i}")
            time.sleep(0.001)  # ensure monotonic last_accessed_at
        # 8 puts with cap=5 -> evictions occurred. count <= 5.
        assert cache.list_namespaces() == ["default"]
        assert cache.stats().entries <= 5
        assert cache.stats().evictions > 0


def test_namespace_quota_evicts_within_namespace_only():
    e = FakeEmbedder(dim=8)
    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        namespace_quotas={"tenant_a": 3},
    ) as cache:
        for i in range(6):
            cache.put(f"a{i}", "r", namespace="tenant_a")
            time.sleep(0.001)
        for i in range(4):
            cache.put(f"b{i}", "r", namespace="tenant_b")
        assert cache.stats(namespace="tenant_a").entries <= 3
        assert cache.stats(namespace="tenant_b").entries == 4


# --- vacuum, delete, clear_namespace ---


def test_vacuum_removes_expired_only(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("alive", "r")
        cache.put("dead", "r", ttl=1)
        time.sleep(1.1)
        removed = cache.vacuum()
        assert removed == 1
        assert cache.get("alive") is not None
        assert cache.get("dead") is None


def test_delete_returns_true_when_removed(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("hi", "there")
        assert cache.delete("hi") is True
        assert cache.get("hi") is None


def test_delete_returns_false_when_missing():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        assert cache.delete("never put") is False


def test_clear_namespace_removes_only_that_namespace():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("q1", "r", namespace="a")
        cache.put("q2", "r", namespace="a")
        cache.put("q3", "r", namespace="b")
        cleared = cache.clear_namespace("a")
        assert cleared == 2
        assert cache.stats(namespace="a").entries == 0
        assert cache.stats(namespace="b").entries == 1


def test_clear_wipes_everything_across_namespaces():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("q1", "r", namespace="a")
        cache.put("q2", "r", namespace="a")
        cache.put("q3", "r", namespace="b")
        cache.put("q4", "r", namespace="default")
        assert cache.stats().entries == 4

        cleared = cache.clear()

        assert cleared == 4
        assert cache.stats().entries == 0
        assert cache.list_namespaces() == []
        # Subsequent gets are misses (cache is genuinely empty, not just
        # namespace-scoped empty).
        assert cache.get("q1", namespace="a") is None
        assert cache.get("q3", namespace="b") is None


def test_clear_on_empty_cache_returns_zero():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        assert cache.clear() == 0
        assert cache.stats().entries == 0


def test_clear_bumps_version_counter():
    """Cross-process readers rely on ``version_counter`` ticking on every
    write — including the global clear."""
    e = FakeEmbedder(dim=8)
    store = MemoryStore()
    with SemanticCache(store=store, embedder=e) as cache:
        cache.put("q1", "r", namespace="a")
        cache.put("q2", "r", namespace="b")
        before = store.read_version_counter()
        cache.clear()
        after = store.read_version_counter()
        # Two namespaces cleared → counter bumped at least twice.
        assert after >= before + 2


def test_clear_then_put_assigns_fresh_ids():
    """After clear(), the in-memory index is rebuilt empty so subsequent
    puts don't trip into stale-row territory."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("q1", "r1")
        cache.clear()
        cache.put("q1", "r2")
        hit = cache.get("q1")
        assert hit is not None
        assert hit.response == "r2"
        assert hit.layer == "exact"


# --- similarity_threshold mutation ---


def test_set_similarity_threshold_updates_value():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        original = cache.similarity_threshold
        cache.set_similarity_threshold(0.5)
        assert cache.similarity_threshold == 0.5
        assert cache.similarity_threshold != original


def test_set_similarity_threshold_rejects_out_of_range():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        for bad in (-1.5, 1.5, 100.0, float("inf"), float("nan")):
            # nan compares False against everything, so the range check fails
            # on it the same way as out-of-range numbers.
            with pytest.raises(ValueError, match="similarity_threshold"):
                cache.set_similarity_threshold(bad)


def test_set_similarity_threshold_changes_match_behavior():
    """Lowering the threshold turns a previous miss into a hit; raising it
    flips a hit back to a miss."""
    e = ParaphraseEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, similarity_threshold=0.99) as cache:
        cache.put("how do I cancel", "use the cancel button", namespace="t")
        # At 0.99, even the close paraphrase is below threshold.
        assert cache.get("how can I cancel", namespace="t") is None
        cache.set_similarity_threshold(0.50)
        hit = cache.get("how can I cancel", namespace="t")
        assert hit is not None
        assert hit.layer == "semantic"


# --- Stats / Health / list_namespaces ---


def test_stats_per_namespace_isolated():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("a", "r", namespace="t1")
        cache.put("b", "r", namespace="t2")
        cache.get("a", namespace="t1")  # 1 exact hit on t1
        cache.get("missing", namespace="t2")  # miss on t2
        assert cache.stats(namespace="t1").hits_exact == 1
        assert cache.stats(namespace="t1").misses == 0
        assert cache.stats(namespace="t2").hits_exact == 0
        assert cache.stats(namespace="t2").misses == 1


def test_stats_aggregate_sums_all_namespaces():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("a", "r", namespace="t1")
        cache.put("b", "r", namespace="t2")
        cache.get("a", namespace="t1")
        cache.get("b", namespace="t2")
        agg = cache.stats()  # namespace=None
        assert agg.hits_exact == 2
        assert agg.entries == 2


def test_health_reports_consistent_state(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    with SemanticCache(path=tmp_path / "c.db", embedder=e) as cache:
        cache.put("hi", "there")
        h = cache.health()
        assert h.healthy is True
        assert h.entries == 1
        assert h.namespaces == 1
        assert h.embedder_fingerprint_match is True
        assert h.integrity_ok is True
        assert h.index_backend == "numpy"
        assert h.store_backend == "sqlite"


def test_list_namespaces_returns_sorted_distinct():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("a", "r", namespace="zzz")
        cache.put("b", "r", namespace="aaa")
        cache.put("c", "r", namespace="mmm")
        assert cache.list_namespaces() == ["aaa", "mmm", "zzz"]


# --- Lifecycle ---


def test_context_manager_closes_on_exit(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    cache = SemanticCache(path=tmp_path / "c.db", embedder=e)
    with cache:
        cache.put("hi", "there")
    # After __exit__, cache is closed.
    with pytest.raises(CacheClosedError):
        cache.get("hi")


def test_close_is_idempotent(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    cache = SemanticCache(path=tmp_path / "c.db", embedder=e)
    cache.close()
    cache.close()  # second close is no-op


def test_use_after_close_raises(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    cache = SemanticCache(path=tmp_path / "c.db", embedder=e)
    cache.close()
    with pytest.raises(CacheClosedError):
        cache.put("a", "b")
    with pytest.raises(CacheClosedError):
        cache.get("a")


def test_counters_persist_across_open_close(tmp_path: Path):
    e = FakeEmbedder(dim=8)
    db = tmp_path / "c.db"
    with SemanticCache(path=db, embedder=e) as cache:
        cache.put("hi", "there")
        cache.get("hi")
        cache.get("hi")
    with SemanticCache(path=db, embedder=e) as cache:
        s = cache.stats()
        assert s.hits_exact == 2  # carried over from previous session


# --- Embedder failure handling ---


def test_embedder_failure_during_get_returns_miss(caplog):
    e = FlakyEmbedder(dim=8, fail_on=1)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        # Skip put (embedder would fail) — directly test get on cold cache.
        with caplog.at_level(logging.WARNING, logger="mneme.cache"):
            assert cache.get("anything") is None
        assert any("Embedder failed" in record.message for record in caplog.records)


def test_embedder_failure_during_put_raises():
    e = FlakyEmbedder(dim=8, fail_on=1)
    with (
        SemanticCache(store=MemoryStore(), embedder=e) as cache,
        pytest.raises(RuntimeError, match="forced failure"),
    ):
        cache.put("q", "r")


# --- Metrics hook integration ---


def test_metrics_hook_called_for_hit_and_miss():
    events: list[tuple[str, dict[str, Any]]] = []
    e = FakeEmbedder(dim=8)
    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        metrics_hook=lambda ev, attrs: events.append((ev, attrs)),
    ) as cache:
        cache.put("hi", "there")
        cache.get("hi")  # hit
        cache.get("missing")  # miss
        # Filter to the cache events.
        names = [e for e, _ in events]
        assert "hit" in names
        assert "miss" in names


# --- Index backend selection ---


def test_index_backend_explicit_numpy():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e, index_backend="numpy") as cache:
        assert cache.health().index_backend == "numpy"


# --- Size limits ---


def test_query_too_large_raises():
    e = FakeEmbedder(dim=8)
    with (
        SemanticCache(store=MemoryStore(), embedder=e, max_query_bytes=10) as cache,
        pytest.raises(ValueError, match="max_query_bytes"),
    ):
        cache.put("a" * 20, "r")


def test_response_too_large_raises():
    e = FakeEmbedder(dim=8)
    with (
        SemanticCache(store=MemoryStore(), embedder=e, max_response_bytes=10) as cache,
        pytest.raises(ValueError, match="max_response_bytes"),
    ):
        cache.put("q", "r" * 20)


def test_metadata_too_large_raises():
    e = FakeEmbedder(dim=8)
    with (
        SemanticCache(store=MemoryStore(), embedder=e, max_metadata_bytes=20) as cache,
        pytest.raises(ValueError, match="max_metadata_bytes"),
    ):
        cache.put("q", "r", metadata={"k": "x" * 100})


# --- Requantize ---


def test_requantize_changes_dtype():
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("hi", "there")
        assert cache.health().vector_dtype == "float32"
        cache.requantize("int8")
        assert cache.health().vector_dtype == "int8"


# --- Custom store via Protocol ---


def test_custom_store_works_via_protocol():
    """Cache must work with any Store-protocol implementation."""
    e = FakeEmbedder(dim=8)
    store = InMemoryStore()
    with SemanticCache(store=store, embedder=e) as cache:
        cache.put("hi", "there")
        hit = cache.get("hi")
        assert isinstance(hit, Hit)
        assert hit.response == "there"


# --- HNSW index selection paths ---


def test_index_backend_explicit_hnsw():
    """index_backend='hnsw' should produce an HnswIndex when [hnsw] is installed."""
    pytest.importorskip("hnswlib")
    e = FakeEmbedder(dim=8)
    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        index_backend="hnsw",
    ) as cache:
        assert cache.health().index_backend == "hnsw"


def test_index_backend_hnsw_missing_extra_raises(monkeypatch):
    """index_backend='hnsw' without the extra raises IndexBackendUnavailableError."""
    from mneme._exceptions import IndexBackendUnavailableError

    def fake_import(*_args: Any, **_kwargs: Any) -> Any:
        raise IndexBackendUnavailableError("fake: hnsw extra missing")

    monkeypatch.setattr("mneme._index_hnsw._import_hnswlib", fake_import)
    e = FakeEmbedder(dim=8)
    with pytest.raises(IndexBackendUnavailableError):
        SemanticCache(store=MemoryStore(), embedder=e, index_backend="hnsw")


# --- Layer-2 filtering paths (validator + confidence) ---


def test_layer_2_validator_rejection_drops_candidate():
    """A poisoned response must be removed from the cache during layer-2 search."""
    e = ParaphraseEmbedder(dim=32)

    def reject_starts_with_poison(r: str) -> bool:
        return not r.startswith("[POISON]")

    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        similarity_threshold=0.3,
        validator=reject_starts_with_poison,
    ) as cache:
        cache.put("how reset password", "[POISON] bad answer")
        # Different query forces layer-2; validator rejects; entry removed.
        hit = cache.get("how do i reset my password")
        assert hit is None
        # The poisoned entry should be gone from the cache now.
        assert cache.delete("how reset password") is False


def test_layer_2_confidence_cutoff_skips_low_confidence():
    """A confidence_fn returning < 0.7 must skip the candidate."""
    e = ParaphraseEmbedder(dim=32)

    def always_low(_sim: float, _age: int, _meta: dict[str, Any]) -> float:
        return 0.5  # always below the 0.7 cutoff

    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        similarity_threshold=0.3,
        confidence_fn=always_low,
    ) as cache:
        cache.put("how reset password", "answer")
        # Layer-2 candidate exists with high similarity, but confidence_fn
        # returns 0.5 < 0.7 cutoff -> skip -> no_match.
        assert cache.get("how do i reset my password") is None


# --- TTL expiration path at layer 1 ---


def test_layer_1_ttl_expiration_emits_expired_event():
    """When layer-1 finds an expired entry, it removes it and emits 'expired'."""
    events: list[tuple[str, dict[str, Any]]] = []
    e = FakeEmbedder(dim=8)
    with SemanticCache(
        store=MemoryStore(),
        embedder=e,
        metrics_hook=lambda ev, attrs: events.append((ev, attrs)),
    ) as cache:
        cache.put("q", "r", ttl=1)
        time.sleep(1.1)
        cache.get("q")  # triggers TTL expiration on layer 1
        assert any(name == "expired" for name, _ in events)


# --- Checkpoint smoke tests ---


def test_dumps_on_memory_store_raises_checkpoint_error(tmp_path: Path):
    """MemoryStore can't checkpoint; SemanticCache.dumps surfaces
    the underlying CheckpointError."""
    from mneme import CheckpointError

    e = FakeEmbedder(dim=8)
    with (
        SemanticCache(store=MemoryStore(), embedder=e) as cache,
        pytest.raises(CheckpointError),
    ):
        cache.dumps(tmp_path / "snap.tar.gz")


def test_loads_missing_source_raises(tmp_path: Path):
    from mneme import CheckpointError

    with pytest.raises(CheckpointError, match="does not exist"):
        SemanticCache.loads(
            tmp_path / "src.tar.gz",
            tmp_path / "dst.db",
            FakeEmbedder(dim=8),
        )


# --- Index/store divergence handling ---


def test_search_skips_stale_index_pointer_to_missing_store_id():
    """If the store has gaps in its id space (e.g. crash recovery), search
    should clean up rather than crash."""
    e = ParaphraseEmbedder(dim=32)
    with SemanticCache(store=MemoryStore(), embedder=e, similarity_threshold=0.1) as cache:
        cache.put("how reset password", "answer")
        # Manually corrupt: delete from store but leave in index.
        cache._store.get_by_hash(
            "default", (cache._store.list_namespaces() and "default") or "default"
        )
        # Easier: directly delete the store row by id without touching the index.
        ids = list(cache._store.iter_lru_ids(10))
        assert len(ids) == 1
        cache._store.delete_by_id(ids[0])
        # Now index has a stale row_id. A get should not raise.
        hit = cache.get("how do i reset password")
        # No crash; either None or a valid hit (after stale-skip).
        assert hit is None or isinstance(hit, Hit)


# --- Compact / RAM reclaim ---


def test_compact_reclaims_tombstoned_index_rows():
    """After deletes, index.tombstone_count > 0; compact() drives it back to 0."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        for i in range(50):
            cache.put(f"q{i}", f"r{i}")
        for i in range(40):
            cache.delete(f"q{i}")
        s_before = cache.stats()
        # 10 live entries, 40 tombstones in the in-memory index.
        assert s_before.entries == 10
        assert s_before.index_tombstone_count == 40
        reclaimed = cache.compact()
        assert reclaimed == 40
        s_after = cache.stats()
        assert s_after.entries == 10
        assert s_after.index_tombstone_count == 0
        # Live entries still queryable after compact.
        assert cache.get("q45") is not None


def test_compact_no_tombstones_is_cheap_noop():
    """compact() on a fresh cache should be a no-op returning 0."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("a", "1")
        cache.put("b", "2")
        assert cache.compact() == 0
        assert cache.stats().index_tombstone_count == 0


def test_vacuum_auto_compacts_by_default():
    """After vacuum() with default compact=True, no tombstones remain."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        cache.put("alive", "r")
        for i in range(20):
            cache.put(f"dead_{i}", "r", ttl=1)
        time.sleep(1.1)
        removed = cache.vacuum()
        assert removed == 20
        # Auto-compact ran; index has no tombstones.
        assert cache.stats().index_tombstone_count == 0


def test_vacuum_compact_false_leaves_tombstones():
    """compact=False lets the caller schedule compaction independently."""
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        for i in range(5):
            cache.put(f"dead_{i}", "r", ttl=1)
        time.sleep(1.1)
        removed = cache.vacuum(compact=False)
        assert removed == 5
        # Tombstones still present until an explicit compact().
        assert cache.stats().index_tombstone_count == 5
        cache.compact()
        assert cache.stats().index_tombstone_count == 0


def test_stats_exposes_actual_index_memory_bytes():
    """index_memory_bytes is the matrix's actual nbytes, not the live estimate.

    NumpyIndex's _INITIAL_CAPACITY is 256; the matrix doubles on overflow. So
    600 puts grows it to 1024 rows; deleting 590 leaves 10 live but the matrix
    still holds 1024 until compact(), which rebuild_from-shrinks back to the
    floor (max(initial_capacity=256, live=10) = 256).
    """
    e = FakeEmbedder(dim=8)
    with SemanticCache(store=MemoryStore(), embedder=e) as cache:
        for i in range(600):
            cache.put(f"q{i}", "r")
        idx_bytes_before = cache.stats().index_memory_bytes
        assert idx_bytes_before is not None
        # 600 puts forces two doublings: 256 -> 512 -> 1024.
        assert idx_bytes_before == 1024 * 8 * 4
        # Drop most rows; matrix capacity unchanged until compact().
        for i in range(590):
            cache.delete(f"q{i}")
        assert cache.stats().index_memory_bytes == idx_bytes_before
        # After compact, capacity drops to the initial-capacity floor.
        cache.compact()
        bytes_after = cache.stats().index_memory_bytes
        assert bytes_after is not None
        assert bytes_after < idx_bytes_before
        assert bytes_after == 256 * 8 * 4
