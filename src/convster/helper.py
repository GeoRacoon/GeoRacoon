"""
Array utility functions for the convster package.

This module provides low-level helper functions for inspecting and manipulating
NumPy arrays. Currently it exposes utilities for locating the first and last
non-zero element along a given axis, which are used internally during filter
application to determine valid data ranges within raster bands.
"""

import numpy as np
from numpy.typing import NDArray


def first_nonzero(data: NDArray, axis: int = 0, no_value: int = -1) -> NDArray:
    """
    Return the index of the first non-zero value along the given axis.

    Parameters
    ----------
    data : NDArray
        Input array to examine.
    axis : int
        Array axis along which to search for the first non-zero. Default is 0.
    no_value : int
        Value to return when no non-zero entries are found along an axis.
        Default is -1.

    Returns
    -------
    indices
        Array indices of the first non-zero values along the specified axis.
        If no non-zero is found, returns `no_value` for that slice.

    See Also
    --------
    :func:`last_nonzero` : Return the index of the last non-zero value.

    Examples
    --------
    >>> a = np.array([
    ...     [0, 0, 3, 0],
    ...     [1, 0, 0, 0],
    ...     [0, 2, 0, 0],
    ...     [0, 0, 0, 4]
    ... ])
    >>> first_nonzero(a, axis=1)
    array([2, 0, 1, 3])
    >>> first_nonzero(a, axis=0)
    array([1, 2, 0, 3])
    """
    mask = data != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), no_value)


def last_nonzero(data: NDArray, axis: int = 0, no_value: int = -1) -> NDArray:
    """
    Return the index of the last non-zero value along the given axis.

    Parameters
    ----------
    data : NDArray
        Input array to examine.
    axis : int
        Array axis along which to search for the last non-zero. Default is 0.
    no_value : int
        Value to return when no non-zero entries are found along an axis.
        Default is -1.

    Returns
    -------
    indices
        Array indices of the last non-zero values along the specified axis.
        If no non-zero is found, returns `no_value` for that slice.

    See Also
    --------
    :func:`first_nonzero` : Return the index of the first non-zero value.

    Examples
    --------
    >>> a = np.array([
    ...     [0, 0, 3, 0],
    ...     [1, 0, 0, 0],
    ...     [0, 2, 0, 0],
    ...     [0, 0, 0, 4]
    ... ])
    >>> last_nonzero(a, axis=1)
    array([2, 0, 1, 3])
    >>> last_nonzero(a, axis=0)
    array([1, 2, 2, 3])
    """
    mask = data != 0
    loc = data.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), loc, no_value)
