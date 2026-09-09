# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

"""Tests for optional GeoRacoon data caching."""

from pathlib import Path
import tempfile

from riogrande._data import _cache_dir


def test_cache_dir_uses_explicit_destination(monkeypatch):
    """Use the explicit destination ahead of the environment setting."""
    monkeypatch.setenv("GEORACOON_DATA_DIR", "/environment/cache")

    assert _cache_dir("/explicit/cache") == Path("/explicit/cache")


def test_cache_dir_uses_environment_destination(monkeypatch):
    """Use the environment destination when none is passed explicitly."""
    monkeypatch.setenv("GEORACOON_DATA_DIR", "/environment/cache")

    assert _cache_dir() == Path("/environment/cache")


def test_cache_dir_uses_temporary_destination(monkeypatch):
    """Use the system temporary directory when no destination is configured."""
    monkeypatch.delenv("GEORACOON_DATA_DIR", raising=False)

    assert _cache_dir() == Path(tempfile.gettempdir()) / "georacoon"
