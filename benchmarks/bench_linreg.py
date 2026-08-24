"""Benchmarks for the parallel multiple linear regression (``compute_weights``).

The sweep axes are:

- ``n_jobs``: number of worker processes (``2 .. ncpu - 1``).
- ``block_fraction``: block size as a fraction of the raster's total size
  (``1, 1/2, ..., 1/20``).

``compute_weights`` runs the full parallel workflow (selector mask, ``X.T @ X``,
rank check, inversion, and optimal weights), spawning a worker pool per step.
"""

import numpy as np

from coonfit.parallel import compute_weights
from coonfit.inference import (
    partial_X,
    partial_response,
    get_optimal_weights,
)
from riogrande.io import Source, Band
from riogrande.helper import aggregated_selector

from .common import (
    Machine,
    pretty_name,
    peak_rss_while,
    data_path,
    raster_size,
    block_size_from_fraction,
    synthetic_tif,
    coregistered_tif,
)

machine = Machine()
N_JOBS = machine.get_njobs()
BLOCK_FRACTIONS = machine.get_block_fractions()
SIZES = machine.get_sizes()


def _make_band(path, category=None):
    """Build a :class:`~riogrande.io.Band` from a raster path."""
    source = Source(path=path)
    source.import_profile()
    if category is not None:
        source.set_tags(bidx=1, tags=dict(category=category))
        return source.get_band(category=category)
    return Band(source=source, bidx=1)


class _LinregBase:
    """Shared work function and band construction for the regression benchmarks."""

    timeout = 1800

    def _run(self):
        return compute_weights(
            response=self.response,
            predictors=self.predictors,
            block_size=self.block_size,
            include_intercept=False,
            as_dtype=np.float32,
            limit_contribution=0.0,
            no_data=np.nan,
            sanitize_predictors=True,
            return_linear_dependent_predictors=True,
            verbose=False,
            n_jobs=self.n_jobs,
        )


class _RasterLinregBase(_LinregBase):
    """Setup on the fixed Swiss NDVI response and coregistered landcover predictor."""

    params = (N_JOBS, BLOCK_FRACTIONS)
    param_names = ["n_jobs", "block_fraction"]

    def setup(self, n_jobs, block_fraction):
        self.n_jobs = n_jobs
        response_path = data_path(machine.get_ndvi())
        predictor_path = coregistered_tif(
            data_path(machine.get_landcover()), response_path
        )
        width, height = raster_size(response_path)
        self.block_size = block_size_from_fraction(
            block_fraction, width, height
        )
        self.response = _make_band(response_path)
        self.predictors = [
            _make_band(predictor_path, category="landcover")
        ]


class _SyntheticLinregBase(_LinregBase):
    """Setup on deterministic synthetic response/predictor rasters."""

    params = (SIZES, N_JOBS, BLOCK_FRACTIONS)
    param_names = ["size", "n_jobs", "block_fraction"]

    def setup(self, size, n_jobs, block_fraction):
        self.n_jobs = n_jobs
        self.block_size = block_size_from_fraction(block_fraction, size, size)
        self.response = _make_band(synthetic_tif(size, seed=1))
        self.predictors = [
            _make_band(synthetic_tif(size, seed=2), category="predictor")
        ]


class TimeComputeWeights(_RasterLinregBase):
    """Wall time of the full parallel regression workflow on the Swiss data."""

    @pretty_name("Wall time: compute_weights")
    def time_compute_weights(self, n_jobs, block_fraction):
        self._run()


class PeakMemComputeWeights(_RasterLinregBase):
    """Peak process-tree memory of the parallel regression (Swiss data)."""

    unit = "bytes"

    @pretty_name("Peak memory: compute_weights")
    def track_compute_weights_peakmem(self, n_jobs, block_fraction):
        return peak_rss_while(self._run)


class TimeComputeWeightsScaling(_SyntheticLinregBase):
    """Wall time of the parallel regression on synthetic rasters (size sweep)."""

    @pretty_name("Wall time: compute_weights on synthetic rasters")
    def time_compute_weights(self, size, n_jobs, block_fraction):
        self._run()


class PeakMemComputeWeightsScaling(_SyntheticLinregBase):
    """Peak process-tree memory of the parallel regression (size sweep)."""

    unit = "bytes"

    @pretty_name("Peak memory: compute_weights on synthetic rasters")
    def track_compute_weights_peakmem(self, size, n_jobs, block_fraction):
        return peak_rss_while(self._run)


class _NativeLinregBase:
    """Native (single-process) exact normal equations on the full data."""

    timeout = 1800

    def _run_native(self):
        masks = []
        for band in [self.response, *self.predictors]:
            mask_reader = band.get_mask_reader()
            with mask_reader() as read_mask:
                masks.append(np.squeeze(read_mask()))
        selector = aggregated_selector(masks, logic="all")

        X = partial_X(
            self.predictors,
            window=None,
            selector=selector,
            include_intercept=False,
            as_dtype=np.float32,
        )
        y = partial_response(self.response, window=None, selector=selector)
        return get_optimal_weights(X, y)


class _NativeRasterLinregBase(_NativeLinregBase):
    """Native setup on the Swiss NDVI response and coregistered landcover."""

    params = ([1],)
    param_names = ["n_jobs"]

    def setup(self, n_jobs):
        response_path = data_path(machine.get_ndvi())
        predictor_path = coregistered_tif(
            data_path(machine.get_landcover()), response_path
        )
        self.response = _make_band(response_path)
        self.predictors = [
            _make_band(predictor_path, category="landcover")
        ]


class _NativeSyntheticLinregBase(_NativeLinregBase):
    """Native setup on deterministic synthetic response/predictor rasters."""

    params = (SIZES, [1])
    param_names = ["size", "n_jobs"]

    def setup(self, size, n_jobs):
        self.response = _make_band(synthetic_tif(size, seed=1))
        self.predictors = [
            _make_band(synthetic_tif(size, seed=2), category="predictor")
        ]


class TimeComputeWeightsNative(_NativeRasterLinregBase):
    """Wall time of the native (no-mpc) regression on the Swiss data."""

    @pretty_name("Wall time: native compute_weights")
    def time_compute_weights(self, n_jobs):
        self._run_native()


class PeakMemComputeWeightsNative(_NativeRasterLinregBase):
    """Peak memory of the native (no-mpc) regression on the Swiss data."""

    unit = "bytes"

    @pretty_name("Peak memory: native compute_weights")
    def track_compute_weights_peakmem(self, n_jobs):
        return peak_rss_while(self._run_native)


class TimeComputeWeightsNativeScaling(_NativeSyntheticLinregBase):
    """Wall time of the native regression on synthetic rasters."""

    @pretty_name("Wall time: native compute_weights on synthetic rasters")
    def time_compute_weights(self, size, n_jobs):
        self._run_native()


class PeakMemComputeWeightsNativeScaling(_NativeSyntheticLinregBase):
    """Peak memory of the native regression on synthetic rasters."""

    unit = "bytes"

    @pretty_name("Peak memory: native compute_weights on synthetic rasters")
    def track_compute_weights_peakmem(self, size, n_jobs):
        return peak_rss_while(self._run_native)
