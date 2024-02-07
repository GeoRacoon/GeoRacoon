"""
This module provides various functions to facilitate the usage of
`skimage.filters.gaussian`
"""
import numpy as np
from skimage.filters import gaussian

from .. import ap
from ..processing import apply_filter


def get_kernel_diameter(sigma, **params):
    """Compute the kernel diameter in number of cells

    Parameters
    ----------
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel

        .. note::

          Currently only single vales are supported

    """
    msize = int(10 * sigma)
    # make sure it is odd
    if not msize & 1:
        msize += 1
    diameter = msize
    while diameter == msize:
        msize = int(round(1.1 * msize))
        tmap = np.zeros((msize, msize))
        half = int(0.5 * (msize - 1))
        tmap[half, half] = 1
        blurred = apply_filter(tmap, gaussian, sigma=sigma, **params)
        diameter = max(np.unique(ap.last_nonzero(blurred, axis=1)
                       - ap.first_nonzero(blurred, axis=1))) + 1
    return diameter


def get_kernel_size(sigma, **params):
    """Return the distance from center to boarder of a Gaussian kernel
    """
    return int(0.5 * (get_kernel_diameter(sigma, **params) - 1))
