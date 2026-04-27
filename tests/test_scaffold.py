"""Phase-0 smoke tests: package importable, version exposed, py.typed present."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import mneme


def test_package_importable() -> None:
    assert mneme.__name__ == "mneme"


def test_version_string_present() -> None:
    assert isinstance(mneme.__version__, str)
    assert mneme.__version__.count(".") == 2  # e.g. "0.1.0"


def test_py_typed_marker_ships_with_package() -> None:
    """PEP 561 requires the marker file to be packaged."""
    package_root = Path(importlib.resources.files("mneme"))  # type: ignore[arg-type]
    assert (package_root / "py.typed").is_file()


def test_tools_subpackage_importable() -> None:
    import mneme.tools  # noqa: F401


def test_adapters_subpackage_importable() -> None:
    import mneme.adapters  # noqa: F401
