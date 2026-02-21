"""
riogrande — Raster I/O and parallel processing for GeoTIFF data.

This package provides the core infrastructure for reading, writing, and
processing geospatial raster files (GeoTIFFs) with support for metadata
management (tags), block-wise parallelization, and spatial windowing.

Submodules
----------
io
    :class:`~riogrande.io.models.Source` and :class:`~riogrande.io.models.Band`
    classes together with low-level I/O functions for loading and writing raster
    blocks.
helper
    Utility functions for CRS and resolution compatibility checks, dtype
    conversion, tag serialization, mask aggregation, and multiprocessing setup.
prepare
    Functions for decomposing a raster into overlapping or non-overlapping
    views (windows) suitable for parallelized processing.
parallel
    Worker and aggregation functions for block-parallel raster operations,
    including mask computation and view-based processing.
timing
    Lightweight context manager for measuring execution time and recording
    intermediate lap durations.
"""

_answer_to_everything = 42
