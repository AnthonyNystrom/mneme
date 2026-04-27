"""Schema migration registry for SQL-backed stores.

Each entry maps a target schema version to a callable that mutates a SQLite
connection. Migrations are applied forward-only, in ascending version order,
each inside its own transaction. A database opened at a *newer* schema than
this library knows about raises ``SchemaMigrationError``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from ._exceptions import SchemaMigrationError

CURRENT_SCHEMA_VERSION = 1

_INITIAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL DEFAULT 'default',
    query_hash TEXT NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    embedding BLOB NOT NULL,
    metadata TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL,
    ttl INTEGER,
    access_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(namespace, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_entries_ns_hash ON entries(namespace, query_hash);
CREATE INDEX IF NOT EXISTS idx_entries_ns_lru ON entries(namespace, last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_entries_created_at ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_id ON entries(id);

CREATE TABLE IF NOT EXISTS cache_counters (
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,
    value INTEGER NOT NULL,
    PRIMARY KEY (namespace, name)
);

CREATE TABLE IF NOT EXISTS namespace_quotas (
    namespace TEXT PRIMARY KEY,
    max_entries INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS multi_process_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_INITIAL_SCHEMA_SQL)


Migration = Callable[[sqlite3.Connection], None]
_MIGRATIONS: dict[int, Migration] = {
    1: _migrate_to_v1,
}


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply pending forward migrations. Returns the final schema_version.

    Bootstraps ``schema_meta`` if the database is brand-new. Each migration runs
    in its own transaction, with the schema_version row updated atomically with
    the migration body. If a migration fails, the transaction rolls back and
    ``SchemaMigrationError`` is raised with the original cause chained.

    A database whose recorded schema_version exceeds ``CURRENT_SCHEMA_VERSION``
    raises ``SchemaMigrationError`` with guidance to upgrade the library.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    current = int(row[0]) if row else 0

    if current > CURRENT_SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"Database schema_version={current} is newer than this library "
            f"supports (max {CURRENT_SCHEMA_VERSION}). "
            f"Remediation: upgrade the mneme library."
        )

    for target in sorted(_MIGRATIONS):
        if target <= current:
            continue
        try:
            with conn:
                _MIGRATIONS[target](conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(target)),
                )
        except Exception as exc:
            raise SchemaMigrationError(
                f"Migration to schema_version={target} failed. Remediation: "
                f"inspect logs for the underlying error, fix the issue, or "
                f"restore from a pre-upgrade checkpoint."
            ) from exc

    return CURRENT_SCHEMA_VERSION


__all__ = ["CURRENT_SCHEMA_VERSION", "apply_migrations"]
