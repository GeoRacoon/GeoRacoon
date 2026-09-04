"""Fetch GeoRacoon's example and test raster fixtures from Zenodo.

The rasters used by ``examples/`` and ``tests/`` are not shipped in this git
repository (see ``data/README.md`` for provenance and licensing of each
file). They are hosted on Zenodo as two zip archives, ``examples.zip`` and
``test.zip``. Only the requested member is extracted from the archive,
straight into this ``data/`` directory -- so the file ends up at the same ``data/examples/...`` /
``data/test/...`` path the rest of the codebase already expects, and
existing relative paths keep working once a file has been fetched once.
"""

import os
import pooch

HERE = os.path.dirname(os.path.abspath(__file__))

# Concept DOI (always resolves to the latest version)
DOI = "10.5281/zenodo.22307203"

# sha256 of each archive, computed locally after download. Zenodo's own page
# only displays a md5 checksum, but pooch just needs a hash to verify
# -- the algorithm doesn't need to match Zenodo's.
REGISTRY = {
    "examples.zip":
        "sha256:cf5590c944132821d9c534c13b81a859e460cadeff286697f854217788306aad",
    "test.zip":
        "sha256:f44ca574c9933497bfb37ff6c4c12bf2dbfdfb75a0636d2d9a1913a9a5e02b4f",
}

# The Pooch object we can use to fetch later
FETCHER = pooch.create(
    path=HERE,
    base_url=f"doi:{DOI}/",
    registry=REGISTRY,
)


def fetch(name: str) -> str:
    """Return the local path to a GeoRacoon raster fixture.

    Downloads and hash-checks the containing zip archive (``examples.zip``
    or ``test.zip``, inferred from *name*) on first use, then extracts only
    the requested member. Later calls reuse the cached, already-extracted
    file without re-downloading or re-extracting.

    Parameters
    ----------
    name : str
        Path of the file *inside* the archive, e.g.
        ``"examples/alps_elevation-mean_GLO90DEM_sinusoidal.tif"`` or
        ``"test/switzerland_lc-8-reclass_2012_CLC_epsg3035.tif"``.

    Returns
    -------
    str
        Absolute path to the local, extracted file.
    """
    archive = name.split("/", 1)[0] + ".zip"
    extracted = FETCHER.fetch(
        archive,
        processor=pooch.Unzip(members=[name], extract_dir=".")
    )
    return os.path.normpath(extracted[0])
