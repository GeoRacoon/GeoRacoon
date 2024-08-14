from __future__ import annotations
"""
This module provides various functions to facilitate the usage of
`skimage.filters.gaussian`
"""
import numpy as np
from skimage.filters import gaussian

from .. import ap

def get_kernel_diameter(sigma, **params):
    """Compute the kernel diameter in number of cells

    Parameters
    ----------
    sigma: scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel

        .. note::

          Currently only single values are supported

    """
    msize = int(10 * sigma)
    # make sure it is odd
    if not msize & 1:
        msize += 1
    diameter = msize
    while diameter == msize:
        msize += int(max(1, round(0.1 * msize)))
        tmap = np.zeros((msize, msize), dtype=np.uint8)
        half = int(0.5 * (msize - 1))
        tmap[half, half] = 255
        blurred = gaussian(tmap, sigma=sigma, **params)
        diameter = max(np.unique(ap.last_nonzero(blurred, axis=1)
                       - ap.first_nonzero(blurred, axis=1))) + 1
    return diameter


def get_kernel_size(sigma, **params):
    """Return the distance from center to boarder of a Gaussian kernel
    """
    return int(0.5 * (get_kernel_diameter(sigma, **params) - 1))


def compatible_border_size(sigma:float|int, border:tuple[int, int]|None=None,
                           **params)->tuple[int,int]:
    """Assert that the border size is compatible with the specified parameter

    This method asserts that the kernel size determined by `sigma` and further
    parametrization is smaller than the border. 
    If no border is provided, then the minimal border size (in number of 
    pixels) is returned.

    Parameters
    ----------
    sigma: scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel
    border: tuple of int
      The border size (width, height) in number of pixels along each axis

    Returns
    -------
    tuple:
      The border (width, height) compatible with the specified parameters.
      If a border was provided already, it is returned again, if no border
      was provided, the smallest compatible border is returned.

    """
    ks = get_kernel_size(sigma=sigma, **params)
    if border:
        assert all(ks <= b for b in border), f"A dimension of {border=} "\
                f"exceeds the kernel size {ks}"
        bs = border
    else:
        bs = (ks + 1,) * 2
    return bs   
