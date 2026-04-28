"""Checkpoint export/import per PRD §14.

Archive format (tar.gz):

    manifest.json                # metadata (fingerprint, dim, dtype, ...)
    store/                       # store-specific snapshot (SQLiteStore: cache.db)

Per PRD §21 Q10: ``loads`` rejects a fingerprint mismatch unconditionally —
no ``force=True`` escape hatch. Use ``mneme.tools.migrate.reembed()`` to
migrate between embedders instead.

Only ``SQLiteStore`` is supported as a checkpoint source/sink in v1
(``MemoryStore`` raises ``CheckpointError`` per its docs; ``RedisStore``
and ``PostgresStore`` direct users to ``redis-cli BGSAVE`` / ``pg_dump``
externally — the library does not subprocess per §30 invariant #14).
"""

from __future__ import annotations

import contextlib
import json
import tarfile
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._exceptions import (
    CheckpointError,
    EmbedderDimensionError,
    EmbedderMismatchError,
)

if TYPE_CHECKING:
    from ._cache import SemanticCache
    from ._types import Embedder

CHECKPOINT_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_STORE_SUBDIR = "store"


def dumps(cache: SemanticCache, dest: str | Path) -> None:
    """Write a checkpoint archive of ``cache`` to ``dest`` (tar.gz)."""
    if cache._closed:
        raise CheckpointError("Cannot checkpoint a closed cache. Remediation: dump before close().")
    dest = Path(dest)

    # Persist counters first so they're inside the snapshot.
    cache._persist_counters()

    health = cache.health()
    manifest: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "schema_version": health.schema_version,
        "embedder_fingerprint": cache._embedder.fingerprint,
        "embedder_dim": cache._embedder.dim,
        "index_backend": health.index_backend,
        "store_backend": health.store_backend,
        "vector_dtype": cache._vector_dtype,
        "created_at": int(time.time()),
        "namespaces": cache._store.list_namespaces(),
    }

    with tempfile.TemporaryDirectory(prefix="mneme_dump_") as td:
        td_path = Path(td)
        store_target = td_path / _STORE_SUBDIR
        try:
            cache._store.snapshot_to(store_target)
        except CheckpointError:
            # MemoryStore / RedisStore / PostgresStore raise this — propagate.
            raise
        except Exception as exc:
            raise CheckpointError(
                f"Store snapshot failed: {exc}. Remediation: see __cause__ for "
                f"the underlying error."
            ) from exc

        manifest_path = td_path / _MANIFEST_NAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        try:
            with tarfile.open(dest, "w:gz") as tar:
                tar.add(manifest_path, arcname=_MANIFEST_NAME)
                tar.add(store_target, arcname=_STORE_SUBDIR)
        except OSError as exc:
            raise CheckpointError(
                f"Failed to write checkpoint archive {dest}: {exc}. "
                f"Remediation: check disk space and write permissions."
            ) from exc

    # Best-effort: limit checkpoint readability to the owner.
    with contextlib.suppress(OSError):
        dest.chmod(0o600)


def restore(
    source: str | Path,
    path: str | Path,
    embedder: Embedder,
) -> dict[str, Any]:
    """Restore a checkpoint archive into a fresh store at ``path``.

    Validates the manifest's embedder fingerprint and dim against the
    supplied ``embedder``. Returns the parsed manifest so callers can use
    its defaults (e.g. ``vector_dtype``).
    """
    from ._store_sqlite import SQLiteStore

    src_path = Path(source)
    dest_path = Path(path)
    if not src_path.exists():
        raise CheckpointError(
            f"Checkpoint source {src_path} does not exist. Remediation: verify the path."
        )

    with tempfile.TemporaryDirectory(prefix="mneme_load_") as td:
        td_path = Path(td)
        try:
            with tarfile.open(src_path, "r:gz") as tar:
                # Python 3.12+: use the strict 'data' filter to refuse
                # path-traversal entries; fall through on older versions.
                # S202 suppressed: archives originate from our own dumps()
                # output and the 'data' filter rejects unsafe entries.
                try:
                    tar.extractall(td_path, filter="data")
                except TypeError:  # pragma: no cover - py<3.12 fallback
                    tar.extractall(td_path)  # noqa: S202
        except (tarfile.TarError, OSError) as exc:
            raise CheckpointError(
                f"Failed to read checkpoint archive {src_path}: {exc}. "
                f"Remediation: check that the archive is a mneme dumps() output."
            ) from exc

        manifest_path = td_path / _MANIFEST_NAME
        if not manifest_path.exists():
            raise CheckpointError(
                f"Checkpoint archive missing {_MANIFEST_NAME}. "
                f"Remediation: regenerate the archive via dumps()."
            )
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest: dict[str, Any] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Could not parse {_MANIFEST_NAME}: {exc}.") from exc

        # Per §21 Q10: no force flag — fingerprint mismatch always raises.
        manifest_fp = manifest.get("embedder_fingerprint")
        if manifest_fp != embedder.fingerprint:
            raise EmbedderMismatchError(
                f"Checkpoint fingerprint {manifest_fp!r} does not match "
                f"supplied embedder {embedder.fingerprint!r}. Remediation: "
                f"open with the original embedder, or run "
                f"mneme.tools.migrate.reembed() against the source cache "
                f"BEFORE re-checkpointing."
            )
        manifest_dim = int(manifest.get("embedder_dim", 0))
        if manifest_dim != embedder.dim:
            raise EmbedderDimensionError(
                f"Checkpoint embedder_dim={manifest_dim} != embedder.dim "
                f"{embedder.dim}. Remediation: use reembed() to change dim."
            )

        store_backend = manifest.get("store_backend", "sqlite")
        if store_backend != "sqlite":
            raise CheckpointError(
                f"Cannot restore store_backend={store_backend!r} in v1; "
                f"only sqlite checkpoints round-trip. Remediation: use "
                f"backend-native tools (pg_restore, redis-cli) for "
                f"{store_backend} backups."
            )

        store_dir = td_path / _STORE_SUBDIR
        if not store_dir.exists():
            raise CheckpointError(
                f"Checkpoint archive missing {_STORE_SUBDIR}/. Remediation: regenerate via dumps()."
            )
        # SQLiteStore.snapshot_to writes a single .db at the target path; we
        # asked for store_dir as the path, so that's where the snapshot file is.
        try:
            restored = SQLiteStore.restore_from(store_dir, dest_path)
        except Exception as exc:
            raise CheckpointError(
                f"Failed to restore SQLite store from checkpoint: {exc}."
            ) from exc
        # restore_from returns an unopened store; close it so the cache
        # constructor can open it cleanly afterwards.
        with contextlib.suppress(Exception):
            restored.close()

    return manifest


def loads(
    source: str | Path,
    path: str | Path,
    embedder: Embedder,
    **kwargs: Any,
) -> SemanticCache:
    """Restore a checkpoint and return a freshly-opened ``SemanticCache``.

    The manifest's ``vector_dtype`` is used as the default for the new cache
    unless overridden in ``kwargs``. Other settings (``similarity_threshold``,
    quotas, etc.) come from ``kwargs`` or their defaults.
    """
    from ._cache import SemanticCache

    manifest = restore(source, path, embedder)
    defaults: dict[str, Any] = {
        "vector_dtype": manifest.get("vector_dtype", "float32"),
    }
    defaults.update(kwargs)
    return SemanticCache(path=path, embedder=embedder, **defaults)


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "dumps",
    "loads",
    "restore",
]
