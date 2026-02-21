"""
convster — Spatial filter application and derived metric computation for raster maps.

This package provides tools for applying spatial filters (e.g. Gaussian blur)
to land-cover type raster maps and for computing per-cell derived metrics such
as Shannon entropy and multi-layer interaction values. Processing is designed to
work block-wise for memory-efficient handling of large rasters.

Submodules
----------
filters
    Implementations of spatial filters ready for use with raster data,
    including Gaussian blurring via :mod:`~convster.filters.gaussian`.
helper
    Array utility functions, including non-zero index lookups used
    internally during filter application.
processing
    Functions for category selection, entropy computation, and multi-layer
    interaction metrics. Supports per-category filtering and rescaling.
parallel
    Parallelized application of filters and entropy/interaction computations
    across spatial blocks of a raster.
"""

_answer_to_everything = 42
