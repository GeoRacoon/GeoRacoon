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
from riogrande.io import Source, Band

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

    @staticmethod
    def _make_band(path, category=None):
        source = Source(path=path)
        source.import_profile()
        if category is not None:
            source.set_tags(bidx=1, tags=dict(category=category))
            return source.get_band(category=category)
        return Band(source=source, bidx=1)


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
        self.response = self._make_band(response_path)
        self.predictors = [
            self._make_band(predictor_path, category="landcover")
        ]


class _SyntheticLinregBase(_LinregBase):
    """Setup on deterministic synthetic response/predictor rasters."""

    params = (SIZES, N_JOBS, BLOCK_FRACTIONS)
    param_names = ["size", "n_jobs", "block_fraction"]

    def setup(self, size, n_jobs, block_fraction):
        self.n_jobs = n_jobs
        self.block_size = block_size_from_fraction(block_fraction, size, size)
        self.response = self._make_band(synthetic_tif(size, seed=1))
        self.predictors = [
            self._make_band(synthetic_tif(size, seed=2), category="predictor")
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
