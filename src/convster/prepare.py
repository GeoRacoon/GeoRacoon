"""
Add sth here
"""
from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

import math


def get_blur_params(diameter=None, sigma=None, truncate=None):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """
    .. note::
        The default of truncate is 3

    """
    # is_needed
    # needs_work (docs)
    # not_tested (used in tests)
    # usedin_processing

    # use default value of 3 for truncate
    truncate = truncate or 3
    if diameter:
        if sigma:
            truncate = 0.5 * diameter / sigma
        else:
            if truncate:
                sigma = 0.5 * diameter / truncate
    else:
        if sigma:
            diameter = 2 * sigma * truncate
        else:
            # TODO: this test should be done when parsing the input arguments
            raise TypeError("Either the `diameter` or the `sigma` parameter "
                            " need to be provided. We got:\n"
                            f"- {diameter=}\n- {sigma=}")
    return dict(diameter=diameter, sigma=sigma, truncate=truncate)