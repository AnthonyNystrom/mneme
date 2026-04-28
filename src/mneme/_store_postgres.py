"""``PostgresStore``: Postgres-backed Store. Optional ``[postgres]`` extra.

Reference implementation per PRD §8.13.4. Cross-host shared cache is supported:
multiple processes on multiple hosts may share a single Postgres instance.
``snapshot_to`` / ``restore_from`` raise ``CheckpointError`` — use ``pg_dump``
externally if you need backups (the library does not subprocess).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._exceptions import (
    CacheClosedError,
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
    StoreBackendError,
)
from ._migrations import CURRENT_SCHEMA_VERSION
from ._types import StoredEntry

if TYPE_CHECKING:
    import psycopg as _psycopg_t


def _import_psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise StoreBackendError(
            "PostgresStore requires the optional 'postgres' extra. "
            "Remediation: pip install mneme[postgres]"
        ) from exc
    return psycopg


def _ddl(schema: str) -> str:
    """Schema DDL. The schema name is validated against ``_validate_schema_name``
    before formatting; SQL injection is impossible because identifiers are not
    parameter-bindable in SQL."""
    return f"""
CREATE SCHEMA IF NOT EXISTS "{schema}";

CREATE TABLE IF NOT EXISTS "{schema}".schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "{schema}".entries (
    id BIGSERIAL PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    query_hash TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    metadata JSONB NOT NULL,
    created_at BIGINT NOT NULL,
    last_accessed_at BIGINT NOT NULL,
    ttl BIGINT,
    access_count BIGINT NOT NULL DEFAULT 0,
    UNIQUE(namespace, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_entries_ns_hash
    ON "{schema}".entries(namespace, query_hash);
CREATE INDEX IF NOT EXISTS idx_entries_ns_lru
    ON "{schema}".entries(namespace, last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_entries_created_at
    ON "{schema}".entries(created_at);

CREATE TABLE IF NOT EXISTS "{schema}".cache_counters (
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    value BIGINT NOT NULL,
    PRIMARY KEY (namespace, name)
);

CREATE TABLE IF NOT EXISTS "{schema}".namespace_quotas (
    namespace TEXT PRIMARY KEY,
    max_entries BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS "{schema}".multi_process_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _validate_schema_name(schema: str) -> None:
    """Reject non-identifier schema names to make DDL formatting safe."""
    if not schema or not all(c.isalnum() or c == "_" for c in schema):
        raise ValueError(
            f"PostgresStore: schema {schema!r} must be alphanumeric or underscore. "
            f"Remediation: pick a Postgres-identifier-safe schema name."
        )


class PostgresStore:
    """Postgres-backed Store. Provide a DSN, a connection, or a connection pool."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection: _psycopg_t.Connection | None = None,
        pool: Any = None,
        schema: str = "mneme",
    ) -> None:
        provided = sum(x is not None for x in (dsn, connection, pool))
        if provided != 1:
            raise ValueError(
                "PostgresStore: provide exactly one of `dsn`, `connection`, "
                "or `pool`. Remediation: pick a single connection source."
            )
        _validate_schema_name(schema)
        self._dsn = dsn
        self._user_conn = connection
        self._pool = pool
        self._schema = schema
        self._owns_conn = False
        self._conn: _psycopg_t.Connection | None = connection
        self._closed = False

    # --- Lifecycle ---

    def open(self, embedder_fingerprint: str, embedder_dim: int) -> None:
        if self._closed:
            raise CacheClosedError("PostgresStore was closed. Remediation: create a new instance.")
        if self._conn is None:
            psycopg = _import_psycopg()
            try:
                if self._dsn is not None:
                    self._conn = psycopg.connect(self._dsn, autocommit=True)
                    self._owns_conn = True
                else:
                    self._conn = self._pool.getconn()
                    # Pool-supplied connections may be either mode; force
                    # autocommit so our explicit ``with conn.transaction():``
                    # blocks always represent top-level transactions.
                    self._conn.autocommit = True
            except Exception as exc:
                raise StoreBackendError(
                    f"Failed to connect to Postgres at {self._dsn!r}. "
                    f"Remediation: verify the DSN, network, and credentials."
                ) from exc

        try:
            with self._conn.transaction():
                self._conn.execute(_ddl(self._schema))
                self._conn.execute(
                    f'INSERT INTO "{self._schema}".schema_meta (key, value) '
                    "VALUES (%s, %s) "
                    "ON CONFLICT (key) DO NOTHING",
                    ("schema_version", str(CURRENT_SCHEMA_VERSION)),
                )
        except Exception as exc:
            raise StoreBackendError(
                f"Failed to apply Postgres DDL in schema {self._schema!r}. "
                f"Remediation: ensure the role has CREATE privilege."
            ) from exc

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
        if self._conn is not None:
            with contextlib.suppress(Exception):
                if self._owns_conn:
                    self._conn.close()
                elif self._pool is not None:
                    self._pool.putconn(self._conn)
        self._conn = None
        self._closed = True

    def _conn_or_fail(self) -> _psycopg_t.Connection:
        if self._closed:
            raise CacheClosedError("PostgresStore is closed.")
        if self._conn is None:
            raise CacheClosedError("PostgresStore not opened. Remediation: call open() first.")
        return self._conn

    # --- Internal helpers ---

    def _q(self, sql: str) -> str:
        """Return SQL with the schema interpolated. Schema is whitelist-validated."""
        return sql.format(schema=self._schema)

    def _read_meta_internal(self, key: str) -> str | None:
        cur = self._conn_or_fail().execute(
            self._q('SELECT value FROM "{schema}".schema_meta WHERE key = %s'),
            (key,),
        )
        row = cur.fetchone()
        return None if row is None else str(row[0])

    def _write_meta_internal(self, key: str, value: str) -> None:
        with self._conn_or_fail().transaction():
            self._conn_or_fail().execute(
                self._q(
                    'INSERT INTO "{schema}".schema_meta (key, value) VALUES (%s, %s) '
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ),
                (key, value),
            )

    def _read_mp_state(self, key: str) -> str | None:
        cur = self._conn_or_fail().execute(
            self._q('SELECT value FROM "{schema}".multi_process_state WHERE key = %s'),
            (key,),
        )
        row = cur.fetchone()
        return None if row is None else str(row[0])

    def _write_mp_state(self, key: str, value: str) -> None:
        with self._conn_or_fail().transaction():
            self._conn_or_fail().execute(
                self._q(
                    'INSERT INTO "{schema}".multi_process_state (key, value) VALUES (%s, %s) '
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ),
                (key, value),
            )

    @staticmethod
    def _row_to_entry(row: tuple[Any, ...]) -> StoredEntry:
        return StoredEntry(
            id=int(row[0]),
            namespace=str(row[1]),
            query_hash=str(row[2]),
            query=str(row[3]),
            response=str(row[4]),
            embedding=bytes(row[5]),
            metadata=row[6] if isinstance(row[6], dict) else json.loads(row[6]),
            created_at=int(row[7]),
            last_accessed_at=int(row[8]),
            ttl=None if row[9] is None else int(row[9]),
            access_count=int(row[10]),
        )

    _SELECT_COLS = (
        "id, namespace, query_hash, query, response, embedding, metadata, "
        "created_at, last_accessed_at, ttl, access_count"
    )

    # --- Read ---

    def get_by_hash(self, namespace: str, query_hash: str) -> StoredEntry | None:
        cur = self._conn_or_fail().execute(
            self._q(
                f'SELECT {self._SELECT_COLS} FROM "{{schema}}".entries '
                "WHERE namespace = %s AND query_hash = %s"
            ),
            (namespace, query_hash),
        )
        row = cur.fetchone()
        return None if row is None else self._row_to_entry(row)

    def get_by_id(self, id: int) -> StoredEntry | None:
        cur = self._conn_or_fail().execute(
            self._q(f'SELECT {self._SELECT_COLS} FROM "{{schema}}".entries WHERE id = %s'),
            (id,),
        )
        row = cur.fetchone()
        return None if row is None else self._row_to_entry(row)

    def count(self, namespace: str | None = None) -> int:
        conn = self._conn_or_fail()
        if namespace is None:
            cur = conn.execute(self._q('SELECT COUNT(*) FROM "{schema}".entries'))
        else:
            cur = conn.execute(
                self._q('SELECT COUNT(*) FROM "{schema}".entries WHERE namespace = %s'),
                (namespace,),
            )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def list_namespaces(self) -> list[str]:
        cur = self._conn_or_fail().execute(
            self._q('SELECT DISTINCT namespace FROM "{schema}".entries ORDER BY namespace')
        )
        return [str(row[0]) for row in cur.fetchall()]

    def iter_lru_ids(self, n: int, namespace: str | None = None) -> Iterator[int]:
        if n <= 0:
            return iter(())
        conn = self._conn_or_fail()
        if namespace is None:
            cur = conn.execute(
                self._q(
                    'SELECT id FROM "{schema}".entries '
                    "ORDER BY last_accessed_at ASC, id ASC LIMIT %s"
                ),
                (n,),
            )
        else:
            cur = conn.execute(
                self._q(
                    'SELECT id FROM "{schema}".entries WHERE namespace = %s '
                    "ORDER BY last_accessed_at ASC, id ASC LIMIT %s"
                ),
                (namespace, n),
            )
        return iter([int(row[0]) for row in cur.fetchall()])

    def iter_all(self) -> Iterator[StoredEntry]:
        cur = self._conn_or_fail().execute(
            self._q(f'SELECT {self._SELECT_COLS} FROM "{{schema}}".entries ORDER BY id ASC')
        )
        return iter([self._row_to_entry(r) for r in cur.fetchall()])

    def iter_since(self, last_id: int) -> Iterator[StoredEntry]:
        cur = self._conn_or_fail().execute(
            self._q(
                f'SELECT {self._SELECT_COLS} FROM "{{schema}}".entries '
                "WHERE id > %s ORDER BY id ASC"
            ),
            (last_id,),
        )
        return iter([self._row_to_entry(r) for r in cur.fetchall()])

    # --- Write ---

    def insert(self, entry: StoredEntry) -> int:
        conn = self._conn_or_fail()
        try:
            with conn.transaction():
                cur = conn.execute(
                    self._q(
                        'SELECT id FROM "{schema}".entries WHERE namespace = %s AND query_hash = %s'
                    ),
                    (entry.namespace, entry.query_hash),
                )
                existing = cur.fetchone()
                metadata_json = json.dumps(entry.metadata)
                if existing is not None:
                    row_id = int(existing[0])
                    conn.execute(
                        self._q(
                            'UPDATE "{schema}".entries SET query=%s, response=%s, '
                            "embedding=%s, metadata=%s::jsonb, created_at=%s, "
                            "last_accessed_at=%s, ttl=%s, access_count=%s WHERE id=%s"
                        ),
                        (
                            entry.query,
                            entry.response,
                            entry.embedding,
                            metadata_json,
                            entry.created_at,
                            entry.last_accessed_at,
                            entry.ttl,
                            entry.access_count,
                            row_id,
                        ),
                    )
                else:
                    cur = conn.execute(
                        self._q(
                            'INSERT INTO "{schema}".entries (namespace, query_hash, '
                            "query, response, embedding, metadata, created_at, "
                            "last_accessed_at, ttl, access_count) "
                            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s) "
                            "RETURNING id"
                        ),
                        (
                            entry.namespace,
                            entry.query_hash,
                            entry.query,
                            entry.response,
                            entry.embedding,
                            metadata_json,
                            entry.created_at,
                            entry.last_accessed_at,
                            entry.ttl,
                            entry.access_count,
                        ),
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise StoreBackendError(
                            "INSERT did not return an id. Remediation: report this as a bug."
                        )
                    row_id = int(row[0])
                conn.execute(
                    self._q(
                        'UPDATE "{schema}".multi_process_state '
                        "SET value = (CAST(value AS BIGINT) + 1)::TEXT "
                        "WHERE key = 'version_counter'"
                    )
                )
        except StoreBackendError:
            raise
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres insert failed: {exc}. Remediation: see __cause__."
            ) from exc
        return row_id

    def update_access(self, id: int, now: int) -> None:
        try:
            with self._conn_or_fail().transaction():
                self._conn_or_fail().execute(
                    self._q(
                        'UPDATE "{schema}".entries SET last_accessed_at = %s, '
                        "access_count = access_count + 1 WHERE id = %s"
                    ),
                    (now, id),
                )
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres update_access failed: {exc}. Remediation: see __cause__."
            ) from exc

    def delete_by_id(self, id: int) -> bool:
        try:
            with self._conn_or_fail().transaction():
                cur = self._conn_or_fail().execute(
                    self._q('DELETE FROM "{schema}".entries WHERE id = %s'),
                    (id,),
                )
                deleted = cur.rowcount > 0
                if deleted:
                    self._conn_or_fail().execute(
                        self._q(
                            'UPDATE "{schema}".multi_process_state '
                            "SET value = (CAST(value AS BIGINT) + 1)::TEXT "
                            "WHERE key = 'version_counter'"
                        )
                    )
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres delete_by_id failed: {exc}. Remediation: see __cause__."
            ) from exc
        return deleted

    def delete_expired(self, now: int, namespace: str | None = None) -> int:
        try:
            with self._conn_or_fail().transaction():
                if namespace is None:
                    cur = self._conn_or_fail().execute(
                        self._q(
                            'DELETE FROM "{schema}".entries '
                            "WHERE ttl IS NOT NULL AND created_at + ttl <= %s"
                        ),
                        (now,),
                    )
                else:
                    cur = self._conn_or_fail().execute(
                        self._q(
                            'DELETE FROM "{schema}".entries '
                            "WHERE ttl IS NOT NULL AND created_at + ttl <= %s "
                            "AND namespace = %s"
                        ),
                        (now, namespace),
                    )
                count = int(cur.rowcount)
                if count > 0:
                    self._conn_or_fail().execute(
                        self._q(
                            'UPDATE "{schema}".multi_process_state '
                            "SET value = (CAST(value AS BIGINT) + 1)::TEXT "
                            "WHERE key = 'version_counter'"
                        )
                    )
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres delete_expired failed: {exc}. Remediation: see __cause__."
            ) from exc
        return count

    def clear_namespace(self, namespace: str) -> int:
        try:
            with self._conn_or_fail().transaction():
                cur = self._conn_or_fail().execute(
                    self._q('DELETE FROM "{schema}".entries WHERE namespace = %s'),
                    (namespace,),
                )
                count = int(cur.rowcount)
                if count > 0:
                    self._conn_or_fail().execute(
                        self._q(
                            'UPDATE "{schema}".multi_process_state '
                            "SET value = (CAST(value AS BIGINT) + 1)::TEXT "
                            "WHERE key = 'version_counter'"
                        )
                    )
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres clear_namespace failed: {exc}. Remediation: see __cause__."
            ) from exc
        return count

    # --- Quotas ---

    def set_namespace_quota(self, namespace: str, max_entries: int) -> None:
        try:
            with self._conn_or_fail().transaction():
                self._conn_or_fail().execute(
                    self._q(
                        'INSERT INTO "{schema}".namespace_quotas (namespace, max_entries) '
                        "VALUES (%s, %s) "
                        "ON CONFLICT (namespace) DO UPDATE SET max_entries = EXCLUDED.max_entries"
                    ),
                    (namespace, max_entries),
                )
        except Exception as exc:
            raise StoreBackendError(
                f"Postgres set_namespace_quota failed: {exc}. Remediation: see __cause__."
            ) from exc

    def get_namespace_quota(self, namespace: str) -> int | None:
        cur = self._conn_or_fail().execute(
            self._q('SELECT max_entries FROM "{schema}".namespace_quotas WHERE namespace = %s'),
            (namespace,),
        )
        row = cur.fetchone()
        return None if row is None else int(row[0])

    # --- Coordination ---

    def read_version_counter(self) -> int:
        v = self._read_mp_state("version_counter")
        return 0 if v is None else int(v)

    def read_meta(self, key: str) -> str | None:
        return self._read_meta_internal(key)

    def write_meta(self, key: str, value: str) -> None:
        self._write_meta_internal(key, value)

    # --- Health ---

    def integrity_check(self) -> bool:
        try:
            self._conn_or_fail().execute("SELECT 1").fetchone()
        except Exception:
            return False
        return True

    # --- Backup ---

    def snapshot_to(self, dest_path: str | Path) -> None:
        del dest_path
        raise CheckpointError(
            "PostgresStore.snapshot_to is not implemented in v1; the library "
            "does not subprocess to pg_dump. Remediation: run `pg_dump` "
            "externally and copy the resulting file."
        )

    @classmethod
    def restore_from(cls, source_path: str | Path, dest_path: str | Path) -> PostgresStore:
        del source_path, dest_path
        raise CheckpointError(
            "PostgresStore.restore_from is not implemented in v1. Remediation: "
            "run `pg_restore` externally, then construct PostgresStore pointing "
            "at the restored database."
        )


__all__ = ["PostgresStore"]
