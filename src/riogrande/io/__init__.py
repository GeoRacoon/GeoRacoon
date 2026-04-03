"""
Public API for riogrande I/O.

Key classes and functions are importable directly from this package:

Classes
-------
- :class:`~riogrande.io.models.Source` : Represents a raster source file.
- :class:`~riogrande.io.models.Band` : Represents a single band within a source.

Functions
---------
- :func:`~riogrande.io.core.load_block` : Load a spatial block from a raster file.
- :func:`~riogrande.io.core.write_band` : Write a band to a new raster file.
- :func:`~riogrande.io.core.update_band` : Update an existing band in a raster file.
- :func:`~riogrande.io.core.get_bands_by_tag` : Find bands across files by tag.
- :func:`~riogrande.io.core.coregister_raster` : Reproject a raster to match a reference.
- :func:`~riogrande.io.core.compress_tif` : LZW-compress a GeoTIFF file.

Exceptions
----------
- :mod:`~riogrande.io.exceptions` : Custom exceptions for I/O operations.
"""

from .core import (get_bands_by_tag,
                   load_block,
                   write_band,
                   update_band,
                   compress_tif,
                   coregister_raster,
                   _set_tags,
                   _get_tags,
                   _find_bidxs,
                   _get_bidx_by_tag,
                   _export_to_tif,
                   )
from .models import Source, Band
from . import exceptions

__all__ = [
    'Source', 'Band',  # the modules
    'exceptions',  # all io related excepitons
    # useful core functions
    'get_bands_by_tag', 'load_block', 'write_band', 'update_band', 'coregister_raster',
    'compress_tif',
    # other core functions we might want to remove again
    '_set_tags', '_get_tags', '_find_bidxs', '_get_bidx_by_tag', '_export_to_tif',
    ]
