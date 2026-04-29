"""Re-embed migration.

Iterate entries from ``source_path``, re-embed each query with
``new_embedder``, write to ``dest_path``. Source unchanged. Used when the
embedder model, dimension, or fingerprint changes — no in-place migration
is supported because changing the embedder fingerprint invalidates the
existing vector matrix.

The migrated cache preserves: entries, namespaces, metadata, TTL,
created_at, last_accessed_at, access_count. Counters DO NOT carry over —
they reflect the OLD embedder's hit/miss history.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .._store_sqlite import SQLiteStore
from .._types import StoredEntry

if TYPE_CHECKING:
    from .._types import AsyncEmbedder, Embedder

logger = logging.getLogger("mneme.tools.migrate")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


def reembed(
    source_path: str | Path,
    dest_path: str | Path,
    new_embedder: Embedder,
    *,
    batch_size: int = 64,
    progress: bool = False,
    namespaces: list[str] | None = None,
) -> int:
    """Re-embed ``source_path`` into ``dest_path`` using ``new_embedder``.

    Returns the count of migrated entries. Source is not modified. The
    destination is created (or replaced) at ``dest_path``.
    """
    src_path = Path(source_path)
    dst_path = Path(dest_path)
    if not src_path.exists():
        raise FileNotFoundError(f"reembed: source {src_path} does not exist.")
    if src_path == dst_path:
        raise ValueError(f"reembed: source and dest must differ; got both = {src_path}.")
    if dst_path.exists():
        dst_path.unlink()

    source = SQLiteStore(src_path)
    # Open with the source's stored fingerprint so we can read entries; we
    # don't care about the fingerprint contract here because we're MOVING
    # to a new embedder.
    src_fp = _peek_meta(src_path, "embedder_fingerprint")
    src_dim = int(_peek_meta(src_path, "embedder_dim") or "0")
    if src_fp is None:
        raise ValueError(
            f"reembed: source {src_path} has no embedder_fingerprint; not a valid mneme cache."
        )
    source.open(src_fp, src_dim)
    try:
        return _migrate_sync(source, dst_path, new_embedder, batch_size, progress, namespaces)
    finally:
        source.close()


async def areembed(
    source_path: str | Path,
    dest_path: str | Path,
    new_embedder: AsyncEmbedder,
    *,
    batch_size: int = 64,
    progress: bool = False,
    concurrency: int = 8,
    namespaces: list[str] | None = None,
) -> int:
    """Async ``reembed``: ``new_embedder.embed`` is awaited; up to
    ``concurrency`` embeddings run in parallel via ``asyncio.gather``."""
    src_path = Path(source_path)
    dst_path = Path(dest_path)
    # These are local-filesystem stat/unlink ops in the milliseconds range —
    # using anyio here would be ceremony without payoff. ASYNC240 silenced
    # per-line; embedder work is the actual await point.
    if not src_path.exists():  # noqa: ASYNC240
        raise FileNotFoundError(f"areembed: source {src_path} does not exist.")
    if src_path == dst_path:
        raise ValueError(f"areembed: source and dest must differ; got {src_path}.")
    if dst_path.exists():  # noqa: ASYNC240
        dst_path.unlink()  # noqa: ASYNC240

    source = SQLiteStore(src_path)
    src_fp = _peek_meta(src_path, "embedder_fingerprint")
    src_dim = int(_peek_meta(src_path, "embedder_dim") or "0")
    if src_fp is None:
        raise ValueError(f"areembed: source {src_path} has no embedder_fingerprint.")
    source.open(src_fp, src_dim)
    try:
        return await _migrate_async(
            source, dst_path, new_embedder, batch_size, progress, concurrency, namespaces
        )
    finally:
        source.close()


def _peek_meta(db_path: Path, key: str) -> str | None:
    """Read a single meta value without going through SemanticCache (which
    would need an embedder)."""
    import sqlite3

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (key,))
            row = cur.fetchone()
            return None if row is None else str(row[0])
    except sqlite3.Error:
        return None


def _select_entries(source: SQLiteStore, namespaces: list[str] | None) -> list[StoredEntry]:
    if namespaces is None:
        return list(source.iter_all())
    keep = set(namespaces)
    return [e for e in source.iter_all() if e.namespace in keep]


def _open_dest(dst_path: Path, new_embedder: Embedder | AsyncEmbedder) -> SQLiteStore:
    dest = SQLiteStore(dst_path)
    dest.open(new_embedder.fingerprint, new_embedder.dim)
    return dest


def _migrate_sync(
    source: SQLiteStore,
    dst_path: Path,
    new_embedder: Embedder,
    batch_size: int,
    progress: bool,
    namespaces: list[str] | None,
) -> int:
    entries = _select_entries(source, namespaces)
    total = len(entries)
    dest = _open_dest(dst_path, new_embedder)
    migrated = 0
    try:
        for i in range(0, total, batch_size):
            batch = entries[i : i + batch_size]
            for e in batch:
                vec = new_embedder.embed(e.query)
                vec = _l2_normalize(vec.astype(np.float32, copy=False))
                migrated_entry = StoredEntry(
                    id=0,
                    namespace=e.namespace,
                    query_hash=e.query_hash,
                    query=e.query,
                    response=e.response,
                    embedding=vec.tobytes(),
                    metadata=e.metadata,
                    created_at=e.created_at,
                    last_accessed_at=e.last_accessed_at,
                    ttl=e.ttl,
                    access_count=e.access_count,
                )
                dest.insert(migrated_entry)
                migrated += 1
            if progress:
                _print_progress(migrated, total)
    finally:
        dest.close()
    return migrated


async def _migrate_async(
    source: SQLiteStore,
    dst_path: Path,
    new_embedder: AsyncEmbedder,
    batch_size: int,
    progress: bool,
    concurrency: int,
    namespaces: list[str] | None,
) -> int:
    entries = _select_entries(source, namespaces)
    total = len(entries)
    dest = _open_dest(dst_path, new_embedder)
    migrated = 0
    try:
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _embed_one(e: StoredEntry) -> tuple[StoredEntry, np.ndarray]:
            async with sem:
                vec = await new_embedder.embed(e.query)
            return e, _l2_normalize(np.asarray(vec, dtype=np.float32))

        for i in range(0, total, batch_size):
            batch = entries[i : i + batch_size]
            results = await asyncio.gather(*(_embed_one(e) for e in batch))
            for e, vec in results:
                dest.insert(
                    StoredEntry(
                        id=0,
                        namespace=e.namespace,
                        query_hash=e.query_hash,
                        query=e.query,
                        response=e.response,
                        embedding=vec.tobytes(),
                        metadata=e.metadata,
                        created_at=e.created_at,
                        last_accessed_at=e.last_accessed_at,
                        ttl=e.ttl,
                        access_count=e.access_count,
                    )
                )
                migrated += 1
            if progress:
                _print_progress(migrated, total)
    finally:
        dest.close()
    return migrated


def _print_progress(done: int, total: int) -> None:
    if total <= 0:
        return
    pct = 100.0 * done / total
    msg = f"\r  reembed: {done}/{total} ({pct:.1f}%)"
    sys.stderr.write(msg)
    sys.stderr.flush()
    if done >= total:
        sys.stderr.write("\n")


__all__ = ["areembed", "reembed"]
