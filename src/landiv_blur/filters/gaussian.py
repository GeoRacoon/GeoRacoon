from __future__ import annotations
"""
This module provides various functions to facilitate the usage of
`skimage.filters.gaussian`
"""
import numpy as np
from skimage.filters import gaussian

from numpy.typing import NDArray

from .. import ap

# we abstract the specific filter so that wie can to:
# from landiv_blur.filters.<some_filter> import img_filter
img_filter = gaussian

def get_kernel_diameter(sigma, **params):
    """Compute the kernel diameter in number of cells

    Parameters
    ----------
    sigma: scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel

        .. note::

          Currently only single values are supported

    """
    # not_needed (but could be useful - keep it!)
    # no_work
    # not_tested (used in tests)
    # usedin_processing (if at all)
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
    # is_needed (for tests only)
    # needs_work (better docstring)
    # is_tested
    # usedin_processing
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
    # is_needed
    # no_work
    # is_tested
    # usedin_processing
    ks = get_kernel_size(sigma=sigma, **params)
    if border:
        assert all(ks <= b for b in border), f"A dimension of {border=} "\
                f"exceeds the kernel size {ks}"
        bs = border
    else:
        bs = (ks + 1,) * 2
    return bs   


def bpgaussian(data:NDArray, **filter_params)->NDArray:
    """Applies a border-preserving Gaussian filter

    The approach considers a Gaussian blur to be a weighted average over
    all pixels within the kernel diameter with the weight being given by
    the Gaussian function.
    Pixels close to `np.nan` values should simply "ignore" `np.nan` pixels
    and perform the weighted average over all non-`np.nan` pixels within
    the Gaussian kernel.
    This can be achieved with a normal Gaussian filter in a three step process:

    1. Perform a Gaussian filter on the data with `np.nan` substituted by
       the neutral element in terms of addition, i.e. 0.0.
       This leads to a weighted average that is not properly normalized as
       the former `np.nan` pixels do not contribute to the value, but are
       considered in the normalizing sum of the weights.
    2. To properly normalize all pixels, create a binary array in the same shape
       as `data` with `np.nan` values becoming `0` and all other pixels `1`.
       Apply the same Gaussian filter to this binary array which results in an
       array holding the sum of weights for the weighted average.
    3. Dividing the blurred array from point 1. by the sum-of-weights array form
       step 2 results in a properly normalized weighted average and thus a blurred
       version of the input data with preserved borders.


    Parameters
    ----------
    data:
      Array to apply the Gaussian filter on.
    filter_params:
      Parametrization of the Gaussian filter.
      
      sigma:
        Standard deviation for Gaussian kernel
      truncate:
        Number of standard deviation after which the filter is truncated

      See `skimage.filters.gaussian` for further parameters

    """
    # is_needed
    # no_workj
    # is_tested
    # usedin_both

    # ###
    # 1.
    # ###
    # Substitute `np.nan`s with 0.0 and apply filter
    _data_nonnan = np.where(np.isnan(data), 0.0, data)
    _data_nonnan_blurred = gaussian(image=_data_nonnan, **filter_params)
    # ###
    # 2.
    # ###
    _data_binary = np.where(np.isnan(data), 0.0, 1.0)
    _data_binary_blurred = gaussian(image=_data_binary, **filter_params)
    # ###
    # 3.
    # ###
    _blurred_data = np.divide(_data_nonnan_blurred, _data_binary_blurred,
                             out=np.full(data.shape, np.nan),
                             where=~np.isnan(data))
    return _blurred_data
