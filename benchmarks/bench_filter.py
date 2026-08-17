"""Benchmarks for the parallel Gaussian filter (``apply_filter`` + ``bpgaussian``).

The sweep axes are:

- ``n_jobs``: number of worker processes (``2 .. ncpu - 1``).
- ``block_fraction``: block size as a fraction of the raster's total size
  (``1, 1/2, ..., 1/20``).

The border-preserving Gaussian filter requires a border (halo) around each
block whose size is determined by the Gaussian kernel (``sigma``/``truncate``).
Combinations where the block is not larger than that border are invalid
(``create_views`` would produce overlapping, out-of-bounds views) and are
skipped in ``setup``.
"""

import os

import numpy as np
import rasterio as rio

from convster.parallel import apply_filter
from convster.filters import bpgaussian
from convster.filters.gaussian import compatible_border_size

from .common import (
    Machine,
    pretty_name,
    peak_rss_while,
    data_path,
    raster_size,
    block_size_from_fraction,
    make_temp_output,
    synthetic_tif,
)

machine = Machine()
N_JOBS = machine.get_njobs()
BLOCK_FRACTIONS = machine.get_block_fractions()
SIZES = machine.get_sizes()

_SIGMA = machine.get_gaussian_sigma()
_TRUNCATE = machine.get_gaussian_truncate()
FILTER_PARAMS = dict(sigma=_SIGMA, truncate=_TRUNCATE, preserve_range=True)
_BORDER = compatible_border_size(
    sigma=_SIGMA, truncate=_TRUNCATE, preserve_range=True
)


def _skip_if_block_too_small(block_size):
    """Skip block/fraction combos whose block is not larger than the border."""
    if block_size[0] <= _BORDER[0] or block_size[1] <= _BORDER[1]:
        raise NotImplementedError(
            f"block size {block_size} is not larger than the Gaussian border "
            f"{_BORDER}; invalid for this filter"
        )


class _FilterBase:
    """Shared work function and teardown for the filter benchmarks."""

    timeout = 1200

    def _run(self):
        return apply_filter(
            source=self.source_path,
            output_file=self.output_file,
            block_size=self.block_size,
            data_as_dtype=np.float32,
            data_output_range=None,
            img_filter=bpgaussian,
            filter_params=FILTER_PARAMS,
            filter_output_range=None,
            output_dtype=np.float32,
            output_range=None,
            selector_band=None,
            n_jobs=self.n_jobs,
        )

    def teardown(self, *args):
        output_file = getattr(self, "output_file", None)
        if output_file is not None:
            try:
                os.remove(output_file)
            except OSError:
                pass


class _RasterFilterBase(_FilterBase):
    """Setup on the fixed Swiss NDVI raster (the docs-figure data)."""

    params = (N_JOBS, BLOCK_FRACTIONS)
    param_names = ["n_jobs", "block_fraction"]

    def setup(self, n_jobs, block_fraction):
        self.n_jobs = n_jobs
        self.source_path = data_path(machine.get_ndvi())
        width, height = raster_size(self.source_path)
        self.block_size = block_size_from_fraction(
            block_fraction, width, height
        )
        _skip_if_block_too_small(self.block_size)
        self.output_file = make_temp_output(prefix="georacoon_filter_")


class _SyntheticFilterBase(_FilterBase):
    """Setup on deterministic synthetic rasters of increasing size."""

    params = (SIZES, N_JOBS, BLOCK_FRACTIONS)
    param_names = ["size", "n_jobs", "block_fraction"]

    def setup(self, size, n_jobs, block_fraction):
        self.n_jobs = n_jobs
        self.source_path = synthetic_tif(size, seed=0)
        self.block_size = block_size_from_fraction(block_fraction, size, size)
        _skip_if_block_too_small(self.block_size)
        self.output_file = make_temp_output(prefix="georacoon_filter_synth_")


class TimeFilter(_RasterFilterBase):
    """Wall time of the parallel Gaussian filter on the Swiss NDVI raster."""

    @pretty_name("Wall time: apply_filter (bpgaussian)")
    def time_apply_filter_bpgaussian(self, n_jobs, block_fraction):
        self._run()


class PeakMemFilter(_RasterFilterBase):
    """Peak process-tree memory of the parallel Gaussian filter (Swiss NDVI)."""

    unit = "bytes"

    @pretty_name("Peak memory: apply_filter (bpgaussian)")
    def track_apply_filter_bpgaussian_peakmem(self, n_jobs, block_fraction):
        return peak_rss_while(self._run)


class TimeFilterScaling(_SyntheticFilterBase):
    """Wall time of the Gaussian filter on synthetic rasters (size sweep)."""

    @pretty_name("Wall time: apply_filter on synthetic raster")
    def time_apply_filter_bpgaussian(self, size, n_jobs, block_fraction):
        self._run()


class PeakMemFilterScaling(_SyntheticFilterBase):
    """Peak process-tree memory of the Gaussian filter (size sweep)."""

    unit = "bytes"

    @pretty_name("Peak memory: apply_filter on synthetic raster")
    def track_apply_filter_bpgaussian_peakmem(
        self, size, n_jobs, block_fraction
    ):
        return peak_rss_while(self._run)


class _NativeFilterBase:
    """Native (single-process) Gaussian filter: read full band, blur, write."""

    timeout = 1200

    def _run_native(self):
        with rio.open(self.source_path) as src:
            profile = src.profile.copy()
            data = src.read(1)

        data = np.squeeze(data)
        filtered = bpgaussian(data.astype(np.float32), **FILTER_PARAMS)
        filtered = filtered.astype(np.float32)

        profile.update(dtype="float32", count=1)
        with rio.open(self.output_file, "w", **profile) as dst:
            dst.write(filtered, 1)
        return self.output_file

    def teardown(self, *args):
        output_file = getattr(self, "output_file", None)
        if output_file is not None:
            try:
                os.remove(output_file)
            except OSError:
                pass


class _NativeRasterFilterBase(_NativeFilterBase):
    """Native setup on the fixed Swiss NDVI raster (reported as ``n_jobs=1``)."""

    params = ([1],)
    param_names = ["n_jobs"]

    def setup(self, n_jobs):
        self.source_path = data_path(machine.get_ndvi())
        self.output_file = make_temp_output(prefix="georacoon_filter_native_")


class _NativeSyntheticFilterBase(_NativeFilterBase):
    """Native setup on deterministic synthetic rasters (reported as ``n_jobs=1``)."""

    params = (SIZES, [1])
    param_names = ["size", "n_jobs"]

    def setup(self, size, n_jobs):
        self.source_path = synthetic_tif(size, seed=0)
        self.output_file = make_temp_output(
            prefix="georacoon_filter_native_synth_"
        )


class TimeFilterNative(_NativeRasterFilterBase):
    """Wall time of the native (no-mpc) Gaussian filter on the Swiss NDVI."""

    @pretty_name("Wall time: native apply_filter (bpgaussian)")
    def time_apply_filter_bpgaussian(self, n_jobs):
        self._run_native()


class PeakMemFilterNative(_NativeRasterFilterBase):
    """Peak memory of the native (no-mpc) Gaussian filter on the Swiss NDVI."""

    unit = "bytes"

    @pretty_name("Peak memory: native apply_filter (bpgaussian)")
    def track_apply_filter_bpgaussian_peakmem(self, n_jobs):
        return peak_rss_while(self._run_native)


class TimeFilterNativeScaling(_NativeSyntheticFilterBase):
    """Wall time of the native Gaussian filter on synthetic rasters."""

    @pretty_name("Wall time: native apply_filter on synthetic raster")
    def time_apply_filter_bpgaussian(self, size, n_jobs):
        self._run_native()


class PeakMemFilterNativeScaling(_NativeSyntheticFilterBase):
    """Peak memory of the native Gaussian filter on synthetic rasters."""

    unit = "bytes"

    @pretty_name("Peak memory: native apply_filter on synthetic raster")
    def track_apply_filter_bpgaussian_peakmem(self, size, n_jobs):
        return peak_rss_while(self._run_native)
