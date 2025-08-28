"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
# is_needed
# needs_work (the module is too big!)
# not_tested (partially)
# usedin_both (should be split up!)
from __future__ import annotations

import math
import warnings
from typing import Any
from collections.abc import Callable, Collection

from copy import copy

from typing import Union

import numpy as np
import rasterio as rio

from multiprocessing import (Queue, Manager)
from numpy.typing import NDArray

from .io_ import Source, Band
from ._helper import (view_to_window,
                      output_filename,
                      reduced_mask,
                      aggregated_selector,
                      check_compatibility,
                      check_rank_deficiency,
                      convert_to_dtype,
                      get_or_set_context,
                      get_nbr_workers, )
from .timing import TimedTask
from .plotting import plot_entropy
from .processing import (
    view_blurred,
    view_entropy,
    view_filtered,
    view_interaction
)
from ._prepare import create_views, update_view
from .filters.gaussian import compatible_border_size
from .inference import (
    transposed_product,
    get_optimal_weights_source)
from .io import write_band
from ._exceptions import InvalidPredictorError

