"""
This module contains various helper functions to process numpy arrays
"""

import numpy as np

# TODO: should we move both of these somewhere? or bring them together with the helpers (or split the latter a bit)


def first_nonzero(data, axis=0, no_value=-1):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """Return the first non-zero value along an axis
    """
    # is_needed
    # usedin_processing
    # no_work
    # not_tested (used in other test)
    mask = data != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), no_value)


def last_nonzero(data, axis=0, no_value=-1):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """Return the last non-zero value along an axis
    """
    # is_needed
    # usedin_processing
    # no_work
    # not_tested (used in other test)
    mask = data != 0
    loc = data.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), loc, no_value)
