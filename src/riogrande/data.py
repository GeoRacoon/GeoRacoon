# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

"""Download and cache optional GeoRacoon example and test data."""

import os
import tempfile
from pathlib import Path

DOI = "10.5281/zenodo.22307203"
REGISTRY = {
    "examples.zip": (
        "sha256:cf5590c944132821d9c534c13b81a859e460cadeff286697f854217788306aad"
    ),
    "test.zip": (
        "sha256:f44ca574c9933497bfb37ff6c4c12bf2dbfdfb75a0636d2d9a1913a9a5e02b4f"
    ),
}


def _cache_dir(destination: str | Path | None = None) -> Path:
    """Resolve the directory used for downloaded archives and extracted data."""
    return Path(
        destination
        or os.environ.get("GEORACOON_DATA_DIR")
        or Path(tempfile.gettempdir()) / "georacoon"
    )


def fetch(name: str, destination: str | Path | None = None) -> str:
    """Return the path to a downloaded GeoRacoon data fixture.

    Parameters
    ----------
    name : str
        Path of the requested file inside ``examples.zip`` or ``test.zip``.
    destination : str or pathlib.Path, optional
        Directory for downloaded archives and extracted data. If omitted,
        ``GEORACOON_DATA_DIR`` is used when set; otherwise, data is cached in
        a ``georacoon`` directory under the system temporary directory.

    Returns
    -------
    str
        Absolute path to the requested extracted file.

    Raises
    ------
    ImportError
        If the optional ``pooch`` dependency is not installed.
    """
    try:
        import pooch
    except ImportError as error:
        raise ImportError(
            "Fetching data requires the optional dependency. "
            "Install GeoRacoon with 'GeoRacoon[data]'."
        ) from error

    cache_dir = _cache_dir(destination)
    archive = name.split("/", 1)[0] + ".zip"
    fetcher = pooch.create(
        path=cache_dir,
        base_url=f"doi:{DOI}/",
        registry=REGISTRY,
        retry_if_failed=3,
    )
    extracted = fetcher.fetch(
        archive,
        processor=pooch.Unzip(members=[name], extract_dir=cache_dir),
    )
    return os.path.normpath(extracted[0])
