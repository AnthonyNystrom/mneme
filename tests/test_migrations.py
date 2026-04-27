"""Phase-3a migration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mneme._exceptions import SchemaMigrationError
from mneme._migrations import (
    CURRENT_SCHEMA_VERSION,
    apply_migrations,
)


def test_apply_migrations_on_empty_db_creates_schema(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "fresh.db"))
    try:
        version = apply_migrations(conn)
        assert version == CURRENT_SCHEMA_VERSION
        # Required tables exist
        names = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "schema_meta",
            "entries",
            "cache_counters",
            "namespace_quotas",
            "multi_process_state",
        } <= names
    finally:
        conn.close()


def test_apply_migrations_is_idempotent(tmp_path: Path):
    db = tmp_path / "idem.db"
    conn = sqlite3.connect(str(db))
    try:
        v1 = apply_migrations(conn)
        v2 = apply_migrations(conn)
        v3 = apply_migrations(conn)
        assert v1 == v2 == v3 == CURRENT_SCHEMA_VERSION
    finally:
        conn.close()


def test_apply_migrations_records_schema_version(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "ver.db"))
    try:
        apply_migrations(conn)
        cur = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
        assert cur.fetchone()[0] == str(CURRENT_SCHEMA_VERSION)
    finally:
        conn.close()


def test_database_at_newer_schema_raises(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "future.db"))
    try:
        # Manually plant a schema_version higher than the library knows.
        conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        future_version = CURRENT_SCHEMA_VERSION + 5
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(future_version)),
        )
        conn.commit()
        with pytest.raises(SchemaMigrationError, match="upgrade"):
            apply_migrations(conn)
    finally:
        conn.close()


def test_migration_bootstraps_schema_meta_when_absent(tmp_path: Path):
    """A brand-new DB has no tables; migrations must bootstrap from zero."""
    conn = sqlite3.connect(str(tmp_path / "bare.db"))
    try:
        # Sanity: no tables at all.
        names = list(conn.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        assert names == []
        version = apply_migrations(conn)
        assert version >= 1
    finally:
        conn.close()


def test_initial_schema_includes_required_indexes(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "idx.db"))
    try:
        apply_migrations(conn)
        idx_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            if row[0] is not None and row[0].startswith("idx_")
        }
        assert {
            "idx_entries_ns_hash",
            "idx_entries_ns_lru",
            "idx_entries_created_at",
            "idx_entries_id",
        } <= idx_names
    finally:
        conn.close()


def test_initial_schema_unique_constraint_on_namespace_query_hash(tmp_path: Path):
    conn = sqlite3.connect(str(tmp_path / "uniq.db"))
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO entries (namespace, query_hash, query, response, "
            "embedding, metadata, created_at, last_accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ns", "h", "q", "r", b"", "{}", 0, 0),
        )
        conn.commit()
        # Second insert with the same (namespace, query_hash) violates UNIQUE.
        insert_dup = lambda: conn.execute(  # noqa: E731
            "INSERT INTO entries (namespace, query_hash, query, response, "
            "embedding, metadata, created_at, last_accessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ns", "h", "q2", "r2", b"", "{}", 0, 0),
        )
        with pytest.raises(sqlite3.IntegrityError):
            insert_dup()
    finally:
        conn.close()
