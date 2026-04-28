"""``SQLiteStore``: WAL-mode, transactional SQLite backend. Default for production."""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from ._exceptions import (
    CacheClosedError,
    CorruptCacheError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from ._migrations import apply_migrations
from ._types import StoredEntry

_FILE_MODE = 0o600


class SQLiteStore:
    """SQLite-backed Store. Single-host, crash-safe via WAL.

    Per the PRD's invariants: ``check_same_thread=False`` (the cache's RLock
    serializes), all SQL is parameterized, multi-statement writes are wrapped
    in ``with conn:`` (auto BEGIN / COMMIT / rollback-on-exception), and the
    ``version_counter`` is incremented in the same transaction as the data
    write.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 5.0,
        cache_size_kb: int = 10_000,
    ) -> None:
        self._path = Path(path) if str(path) != ":memory:" else None
        self._raw_path = str(path)
        self._timeout = timeout
        self._cache_size_kb = cache_size_kb
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        if self._closed:
            raise CacheClosedError("SQLiteStore was closed. Remediation: create a new instance.")
        if self._conn is not None:
            return  # idempotent

        is_memory = self._raw_path == ":memory:"
        new_db = is_memory or (self._path is not None and not self._path.exists())

        try:
            self._conn = sqlite3.connect(
                self._raw_path,
                timeout=self._timeout,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise StoreBackendError(
                f"Failed to open SQLite database at {self._raw_path!r}. "
                f"Remediation: check file permissions and disk space."
            ) from exc

        self._conn.row_factory = sqlite3.Row

        if not is_memory and new_db and self._path is not None:
            with contextlib.suppress(OSError):
                self._path.chmod(_FILE_MODE)

        # PRAGMAs (must run outside transactions).
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute(f"PRAGMA cache_size=-{self._cache_size_kb}")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.close()

        try:
            apply_migrations(self._conn)
        except Exception:
            self._conn.close()
            self._conn = None
            raise

        existing_fp = self._read_meta_internal("embedder_fingerprint")
        if existing_fp is None:
            self._write_meta_internal("embedder_fingerprint", embedder_fingerprint)
            self._write_meta_internal("embedder_dim", str(embedder_dim))
        else:
            if existing_fp != embedder_fingerprint:
                raise EmbedderMismatchError(
                    f"Stored fingerprint {existing_fp!r} does not match "
                    f"supplied {embedder_fingerprint!r}. Remediation: open "
                    f"with the original embedder, or use "
                    f"mneme.tools.migrate.reembed() to migrate."
                )
            stored_dim = int(self._read_meta_internal("embedder_dim") or "0")
            if stored_dim != embedder_dim:
                raise EmbedderDimensionError(
                    f"Stored embedder_dim={stored_dim} does not match "
                    f"supplied {embedder_dim}. Remediation: use reembed() "
                    f"to change dimension."
                )

        if self._read_mp_state("version_counter") is None:
            self._write_mp_state("version_counter", "0")

    def close(self) -> None:
        if self._conn is None:
            self._closed = True
            return
        with contextlib.suppress(sqlite3.OperationalError):
            self._conn.execute("PRAGMA wal_checkpoint(FULL)")
        self._conn.close()
        self._conn = None
        self._closed = True

    def _conn_or_fail(self) -> sqlite3.Connection:
        if self._closed:
            raise CacheClosedError("SQLiteStore is closed.")
        if self._conn is None:
            raise CacheClosedError("SQLiteStore not opened. Remediation: call open() first.")
        return self._conn

    # --- Internal helpers ---

    def _read_meta_internal(self, key: str) -> str | None:
        cur = self._conn_or_fail().execute("SELECT value FROM schema_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return None if row is None else str(row["value"])

    def _write_meta_internal(self, key: str, value: str) -> None:
        with self._conn_or_fail() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    def _read_mp_state(self, key: str) -> str | None:
        cur = self._conn_or_fail().execute(
            "SELECT value FROM multi_process_state WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return None if row is None else str(row["value"])

    def _write_mp_state(self, key: str, value: str) -> None:
        with self._conn_or_fail() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO multi_process_state (key, value) VALUES (?, ?)",
                (key, value),
            )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> StoredEntry:
        return StoredEntry(
            id=int(row["id"]),
            namespace=str(row["namespace"]),
            query_hash=str(row["query_hash"]),
            query=str(row["query"]),
            response=str(row["response"]),
            embedding=bytes(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            created_at=int(row["created_at"]),
            last_accessed_at=int(row["last_accessed_at"]),
            ttl=None if row["ttl"] is None else int(row["ttl"]),
            access_count=int(row["access_count"]),
        )

    @staticmethod
    def _wrap_op_error(exc: sqlite3.Error) -> StoreBackendError:
        msg = str(exc).lower()
        hint = (
            "free disk, set max_entries, run vacuum()"
            if "disk" in msg or "full" in msg
            else "see __cause__ for the underlying SQLite error"
        )
        return StoreBackendError(f"SQLite error: {exc}. Remediation: {hint}.")

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        cur = self._conn_or_fail().execute(
            "SELECT * FROM entries WHERE namespace = ? AND query_hash = ?",
            (namespace, query_hash),
        )
        row = cur.fetchone()
        return None if row is None else self._row_to_entry(row)

    def get_by_id(self, id: int) -> StoredEntry | None:
        cur = self._conn_or_fail().execute("SELECT * FROM entries WHERE id = ?", (id,))
        row = cur.fetchone()
        return None if row is None else self._row_to_entry(row)

    def count(self, namespace: str | None = None) -> int:
        conn = self._conn_or_fail()
        if namespace is None:
            cur = conn.execute("SELECT COUNT(*) AS n FROM entries")
        else:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM entries WHERE namespace = ?",
                (namespace,),
            )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def list_namespaces(self) -> list[str]:
        cur = self._conn_or_fail().execute(
            "SELECT DISTINCT namespace FROM entries ORDER BY namespace"
        )
        return [str(r["namespace"]) for r in cur.fetchall()]

    def iter_index_rows(self) -> Iterator[tuple[int, bytes, str]]:
        """Bulk-read just (id, embedding_bytes, namespace) for index rebuild.

        Skips the JSON metadata parse and full ``StoredEntry`` construction
        that ``iter_all`` does — at 100k entries this is the difference
        between a sub-100ms open and a 500ms+ open. Used by
        ``SemanticCache._rebuild_index_from_store`` via duck-typing.
        """
        cur = self._conn_or_fail().execute(
            "SELECT id, embedding, namespace FROM entries ORDER BY id ASC"
        )
        for row in cur.fetchall():
            yield int(row["id"]), bytes(row["embedding"]), str(row["namespace"])

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        conn = self._conn_or_fail()
        if n <= 0:
            return iter(())
        if namespace is None:
            cur = conn.execute(
                "SELECT id FROM entries ORDER BY last_accessed_at ASC, id ASC LIMIT ?",
                (n,),
            )
        else:
            cur = conn.execute(
                "SELECT id FROM entries WHERE namespace = ? "
                "ORDER BY last_accessed_at ASC, id ASC LIMIT ?",
                (namespace, n),
            )
        return iter([int(r["id"]) for r in cur.fetchall()])

    def iter_all(self) -> Iterator[StoredEntry]:
        cur = self._conn_or_fail().execute("SELECT * FROM entries ORDER BY id ASC")
        return iter([self._row_to_entry(r) for r in cur.fetchall()])

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        cur = self._conn_or_fail().execute(
            "SELECT * FROM entries WHERE id > ? ORDER BY id ASC", (last_id,)
        )
        return iter([self._row_to_entry(r) for r in cur.fetchall()])

    # --- Write (transactional with version_counter bump) ---

    def insert(self, entry: StoredEntry) -> int:
        conn = self._conn_or_fail()
        try:
            with conn:
                cur = conn.execute(
                    "SELECT id FROM entries WHERE namespace = ? AND query_hash = ?",
                    (entry.namespace, entry.query_hash),
                )
                existing = cur.fetchone()
                if existing is not None:
                    row_id = int(existing["id"])
                    conn.execute(
                        "UPDATE entries SET query=?, response=?, embedding=?, "
                        "metadata=?, created_at=?, last_accessed_at=?, ttl=?, "
                        "access_count=? WHERE id=?",
                        (
                            entry.query,
                            entry.response,
                            entry.embedding,
                            json.dumps(entry.metadata),
                            entry.created_at,
                            entry.last_accessed_at,
                            entry.ttl,
                            entry.access_count,
                            row_id,
                        ),
                    )
                else:
                    ins = conn.execute(
                        "INSERT INTO entries (namespace, query_hash, query, "
                        "response, embedding, metadata, created_at, "
                        "last_accessed_at, ttl, access_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry.namespace,
                            entry.query_hash,
                            entry.query,
                            entry.response,
                            entry.embedding,
                            json.dumps(entry.metadata),
                            entry.created_at,
                            entry.last_accessed_at,
                            entry.ttl,
                            entry.access_count,
                        ),
                    )
                    rid = ins.lastrowid
                    if rid is None:
                        raise StoreBackendError(
                            "INSERT did not return a lastrowid. Remediation: report this as a bug."
                        )
                    row_id = int(rid)
                conn.execute(
                    "UPDATE multi_process_state SET value = "
                    "CAST(value AS INTEGER) + 1 WHERE key = 'version_counter'"
                )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc
        return row_id

    def update_access(self, id: int, now: int) -> None:
        conn = self._conn_or_fail()
        try:
            with conn:
                conn.execute(
                    "UPDATE entries SET last_accessed_at = ?, "
                    "access_count = access_count + 1 WHERE id = ?",
                    (now, id),
                )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc

    def delete_by_id(self, id: int) -> bool:
        conn = self._conn_or_fail()
        try:
            with conn:
                cur = conn.execute("DELETE FROM entries WHERE id = ?", (id,))
                deleted = cur.rowcount > 0
                if deleted:
                    conn.execute(
                        "UPDATE multi_process_state SET value = "
                        "CAST(value AS INTEGER) + 1 WHERE key = 'version_counter'"
                    )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc
        return deleted

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        conn = self._conn_or_fail()
        try:
            with conn:
                if namespace is None:
                    cur = conn.execute(
                        "DELETE FROM entries WHERE ttl IS NOT NULL AND created_at + ttl <= ?",
                        (now,),
                    )
                else:
                    cur = conn.execute(
                        "DELETE FROM entries WHERE ttl IS NOT NULL "
                        "AND created_at + ttl <= ? AND namespace = ?",
                        (now, namespace),
                    )
                count = int(cur.rowcount)
                if count > 0:
                    conn.execute(
                        "UPDATE multi_process_state SET value = "
                        "CAST(value AS INTEGER) + 1 WHERE key = 'version_counter'"
                    )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc
        return count

    def clear_namespace(self, namespace: str) -> int:
        conn = self._conn_or_fail()
        try:
            with conn:
                cur = conn.execute("DELETE FROM entries WHERE namespace = ?", (namespace,))
                count = int(cur.rowcount)
                if count > 0:
                    conn.execute(
                        "UPDATE multi_process_state SET value = "
                        "CAST(value AS INTEGER) + 1 WHERE key = 'version_counter'"
                    )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc
        return count

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        conn = self._conn_or_fail()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO namespace_quotas (namespace, max_entries) "
                    "VALUES (?, ?) "
                    "ON CONFLICT(namespace) DO UPDATE SET max_entries = excluded.max_entries",
                    (namespace, max_entries),
                )
        except sqlite3.Error as exc:
            raise self._wrap_op_error(exc) from exc

    def get_namespace_quota(self, namespace: str) -> int | None:
        cur = self._conn_or_fail().execute(
            "SELECT max_entries FROM namespace_quotas WHERE namespace = ?",
            (namespace,),
        )
        row = cur.fetchone()
        return None if row is None else int(row["max_entries"])

    # --- Coordination ---

    def read_version_counter(self) -> int:
        value = self._read_mp_state("version_counter")
        return 0 if value is None else int(value)

    def read_meta(self, key: str) -> str | None:
        return self._read_meta_internal(key)

    def write_meta(self, key: str, value: str) -> None:
        self._write_meta_internal(key, value)

    # --- Health ---

    def integrity_check(self) -> bool:
        try:
            cur = self._conn_or_fail().execute("PRAGMA integrity_check")
            row = cur.fetchone()
            ok = row is not None and str(row[0]).lower() == "ok"
        except sqlite3.Error as exc:
            raise CorruptCacheError(
                f"Integrity check raised: {exc}. Remediation: restore from a "
                f"checkpoint via SemanticCache.loads(), or reset the cache files."
            ) from exc
        return ok

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        conn = self._conn_or_fail()
        dest = Path(dest_path)
        try:
            target = sqlite3.connect(str(dest))
            try:
                with target:
                    conn.backup(target)
            finally:
                target.close()
        except sqlite3.Error as exc:
            raise StoreBackendError(
                f"Snapshot to {dest} failed. Remediation: check disk space "
                f"and destination directory permissions."
            ) from exc
        with contextlib.suppress(OSError):
            dest.chmod(_FILE_MODE)

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> SQLiteStore:
        source = Path(source_path)
        dest = Path(dest_path)
        if not source.exists():
            raise StoreBackendError(
                f"Restore source {source} does not exist. Remediation: verify the path."
            )
        try:
            shutil.copy2(source, dest)
        except OSError as exc:
            raise StoreBackendError(
                f"Restore copy from {source} to {dest} failed. Remediation: "
                f"check destination directory permissions."
            ) from exc
        with contextlib.suppress(OSError):
            dest.chmod(_FILE_MODE)
        return cls(dest)


__all__ = ["SQLiteStore"]
