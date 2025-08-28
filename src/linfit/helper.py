
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


def check_rank_deficiency(array, return_by_issue_type: bool=False) -> dict[int, str] | dict[str, list[int]]:
    # TODO: is_needed - no_work - is_tested - usedin_linfit
    """Check if matrix is rank deficient and extract the dependent columns (linear combination of other columns.
    Returns a dictionary with column (key) and issue description (value). Lenght of dictionary is rank-deficiency + 1,
    Empyt dictionary indicates that no rank deficiency was detected

    Parameters
    ----------
    array : np.ndarray
        Matrix to check for rank deficiency
    return_by_issue_type: bool
        If desired, a nested dictionary may be returned separating the type of issue:
        "all_zero" and "linear dependent"
    """
    # is_needed
    # needs_work (formatting)
    # is_tested
    # usedin_linfit
    all_zero_cols = {}
    rank_deficient_cols = {}
    _, num_columns = array.shape
    rank = np.linalg.matrix_rank(array)

    if rank == num_columns:
        return dict()

    for col in range(num_columns):
        column_vector = array[:, col]

        if np.all(column_vector == 0):
            all_zero_cols[col] = "All zero column"
        else:
            # drop focus column
            sub_array = np.delete(array, col, axis=1)

            # does removing a column increase the rank?
            if np.linalg.matrix_rank(sub_array) == rank:
                rank_deficient_cols[col] = "Linear dependent column"

    if return_by_issue_type:
        return dict(linear_dependent=[l for l in rank_deficient_cols.keys()],
                    all_zero=[z for z in all_zero_cols.keys()])
    else:
        return {**rank_deficient_cols, **all_zero_cols}


def usable_pixels_info(all_pixels, data_pixels):
    # TODO: is_needed - no_work - not_tested - usedin_linfit
    """Prints the fraction of usable pixels
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    print(f"Of {all_pixels=} there are {data_pixels=}, i.e. "
          f"{round(100 * data_pixels/all_pixels, 2)}% are usable")


def usable_pixels_count(selector):
    # TODO: is_needed - no_work - not_tested - usedin_linfit
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
