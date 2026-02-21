"""
Helper functions for the linear fitting
"""

from __future__ import annotations
import numpy as np


def check_rank_deficiency(array: np.ndarray, return_by_issue_type: bool = False,
                          ) -> dict[int, str] | dict[str, list[int]]:
    """Check if matrix is rank deficient and identify problematic columns.

    Returns a dictionary with column indices (key) and issue description (value).
    An empty dictionary indicates that no rank deficiency was detected.

    Parameters
    ----------
    array : NDArray
        Matrix to check for rank deficiency
    return_by_issue_type : bool, optional
        If True, returns nested dictionary separating issues by type:
        {"linear_dependent": [...], "all_zero": [...]}

    Returns
    -------
    dict[int, str] or dict[str, list[int]]
        Problematic columns and their issues. Uses :func:`numpy.linalg.matrix_rank`
        to determine the rank of the array.

    See Also
    --------
    :func:`~coonfit.parallel.get_XT_X_dependency` : Check predictors for linear dependency.
    """
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


def usable_pixels_info(all_pixels: int, data_pixels: int) -> None:
    """Print the fraction of usable pixels.

    Parameters
    ----------
    all_pixels : int
        Total number of pixels in the dataset
    data_pixels : int
        Number of pixels that contain usable data

    See Also
    --------
    :func:`usable_pixels_count` : Count the number of usable pixels.

    Examples
    --------
    >>> usable_pixels_info(1000, 750)
    Of all_pixels=1000 there are data_pixels=750, i.e. 75.0% are usable
    """
    print(f"Of {all_pixels=} there are {data_pixels=}, i.e. "
          f"{round(100 * data_pixels/all_pixels, 2)}% are usable")


def usable_pixels_count(selector):
    """Count the number of usable pixels determined by the selector.

    Parameters
    ----------
    selector : NDArray
        Boolean array where True indicates a usable pixel and False
        indicates a pixel to be excluded

    Returns
    -------
    int
        Number of True values in the selector array (count of usable pixels).
        Uses :func:`numpy.unique` to count occurrences.

    See Also
    --------
    :func:`usable_pixels_info` : Print the fraction of usable pixels.

    Examples
    --------
    >>> selector = np.array([True, True, False, True, False])
    >>> usable_pixels_count(selector)
    3
    """
    vals, counts = np.unique(selector, return_counts=True)
    # vals: [True, False] or inv. in any case ok
    try:
        return int(counts[vals][0])
    except IndexError:
        return 0
