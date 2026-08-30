"""Peak-memory scaling with fixed block size and worker count.

These benchmarks vary the side length of deterministic synthetic rasters while
keeping the processing configuration fixed. Synthetic input creation happens
in ASV's ``setup`` phase and is therefore excluded from the measurement.
"""

import os
import sys
import tempfile

import numpy as np
from convster.filters import bpgaussian
from convster.filters.gaussian import compatible_border_size
from convster.parallel import apply_filter
from coonfit.parallel import compute_weights
from riogrande.io import Band, Source

from .common import (
    Machine,
    make_temp_output,
    peak_rss_while,
    synthetic_filter_tif,
    synthetic_tif,
)

machine = Machine()
SIZES = machine.get_scaling_out_sizes()
N_JOBS = 4
BLOCK_SIZE = (1000, 1000)

print(
    f"scaling-out benchmark temporary directory: {tempfile.gettempdir()}",
    file=sys.stderr,
    flush=True,
)

FILTER_PARAMS = dict(
    sigma=machine.get_gaussian_sigma(),
    truncate=machine.get_gaussian_truncate(),
    preserve_range=True,
)
FILTER_BORDER = compatible_border_size(**FILTER_PARAMS)


def _make_band(path, category=None):
    """Build a Band from a raster path."""
    source = Source(path=path)
    source.import_profile()
    if category is not None:
        source.set_tags(bidx=1, tags=dict(category=category))
        return source.get_band(category=category)
    return Band(source=source, bidx=1)


class PeakMemFilterScalingOut:
    """Peak filter memory as raster size increases."""

    params = (SIZES,)
    param_names = ["size"]
    unit = "bytes"
    timeout = 1200

    def setup(self, size):
        if any(block <= border for block, border in zip(BLOCK_SIZE, FILTER_BORDER)):
            raise NotImplementedError(
                f"block size {BLOCK_SIZE} is not larger than filter border "
                f"{FILTER_BORDER}"
            )
        self.source_path = synthetic_filter_tif(size, seed=0)
        self.output_file = make_temp_output(
            prefix="georacoon_filter_scaling_out_"
        )

    def track_peak_memory(self, size):
        return peak_rss_while(self._run)

    def _run(self):
        return apply_filter(
            source=self.source_path,
            output_file=self.output_file,
            block_size=BLOCK_SIZE,
            data_as_dtype=np.float32,
            data_output_range=None,
            img_filter=bpgaussian,
            filter_params=FILTER_PARAMS,
            filter_output_range=(0.0, 1.0),
            output_dtype=np.float32,
            output_range=(0.0, 1.0),
            selector_band=None,
            n_jobs=N_JOBS,
        )

    def teardown(self, *args):
        try:
            os.remove(self.output_file)
        except OSError:
            pass


class PeakMemMLRScalingOut:
    """Peak MLR memory as raster size increases."""

    params = (SIZES,)
    param_names = ["size"]
    unit = "bytes"
    timeout = 1800

    def setup(self, size):
        self.response = _make_band(synthetic_tif(size, seed=1))
        self.predictors = [
            _make_band(synthetic_tif(size, seed=2), category="predictor")
        ]

    def track_peak_memory(self, size):
        return peak_rss_while(self._run)

    def _run(self):
        return compute_weights(
            response=self.response,
            predictors=self.predictors,
            block_size=BLOCK_SIZE,
            include_intercept=False,
            as_dtype=np.float32,
            limit_contribution=0.0,
            no_data=np.nan,
            sanitize_predictors=True,
            return_linear_dependent_predictors=True,
            verbose=False,
            n_jobs=N_JOBS,
        )
