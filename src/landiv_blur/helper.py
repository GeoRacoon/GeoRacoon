"""This module defines functions that might be helpful when working
with rasterio
"""
from __future__ import annotations

import os
import json
import warnings

import numpy as np
import rasterio as rio

from rasterio.windows import Window

from typing import Any, Union, Dict, List

from collections.abc import Collection

from numpy.typing import NDArray

from decimal import Decimal

import multiprocessing as mpc
from multiprocessing import context as _context_module
from typing import Optional






def usable_pixels_info(all_pixels, data_pixels):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Prints the fraction of usable pixels
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    print(f"Of {all_pixels=} there are {data_pixels=}, i.e. "
          f"{round(100 * data_pixels/all_pixels, 2)}% are usable")


def usable_pixels_count(selector):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Count the number of usable pixels determined by the selector"""
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    vals, counts = np.unique(selector, return_counts=True)
    # vals: [True, False] or inv. in any case ok
    try:
        return int(counts[vals][0])
    except IndexError:
        return 0







