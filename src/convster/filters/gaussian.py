"""
This module provides various functions to facilitate the usage of
`skimage.filters.gaussian`
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from skimage.filters import gaussian

from ..helper import (first_nonzero,
                      last_nonzero)

# we abstract the specific filters:
img_filter = gaussian


def get_kernel_diameter(sigma: float, **params) -> int:
    """
    Compute the effective Gaussian kernel diameter in number of cells.

    The diameter is estimated by applying a Gaussian filter to a single impulse
    in the center of a zero image, and measuring the spread of the nonzero
    region. The result is always odd to ensure symmetry.

    Parameters
    ----------
    sigma :
        Standard deviation for the Gaussian kernel.
        Currently only single scalar values are supported.
    **params
        Additional keyword arguments passed to the Gaussian filter.

    Returns
    -------
    int
        Estimated kernel diameter in pixels (always an odd number).

    Notes
    -----
    This function determines the kernel diameter adaptively. Starting from
    an initial guess of `10 * sigma`, the size is increased until the blurred
    impulse response fits within the kernel. Uses :func:`skimage.filters.gaussian`
    internally to compute the impulse response.

    See Also
    --------
    :func:`get_kernel_size` : Return the radius (half-diameter) of the Gaussian kernel.
    :func:`compatible_border_size` : Assert or compute a border size compatible with a kernel.

    Examples
    --------
    >>> get_kernel_diameter(1)
    7
    >>> get_kernel_diameter(2)
    15
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
        diameter = max(np.unique(last_nonzero(blurred, axis=1)
                                 - first_nonzero(blurred, axis=1))) + 1
    return diameter


def get_kernel_size(sigma, **params):
    """
    Return the radius of a Gaussian kernel (center to border distance).

    The kernel size is defined as half of the kernel diameter minus one,
    i.e. size = (diameter - 1) / 2 where `diameter` is determined
    by :func:`get_kernel_diameter`.

    Parameters
    ----------
    sigma :
        Standard deviation for the Gaussian kernel.
    **params
        Additional keyword arguments passed to :func:`get_kernel_diameter`
        (e.g. `truncate`, `mode`).

    Returns
    -------
    int
        The radius of the Gaussian kernel in pixels.

    See Also
    --------
    :func:`get_kernel_diameter` : Compute the full kernel diameter.
    :func:`compatible_border_size` : Assert or compute a border size compatible with a kernel.

    Examples
    --------
    >>> get_kernel_size(1)
    3
    >>> get_kernel_size(2)
    7
    """
    return int(0.5 * (get_kernel_diameter(sigma, **params) - 1))


def compatible_border_size(sigma: float | int, border: tuple[int, int] | None = None,
                           **params) -> tuple[int, int]:
    """
    Assert that the border size is compatible with the specified parameter

    This method asserts that the kernel size determined by `sigma` and further
    parametrization is smaller than the border.
    If no border is provided, then the minimal border size (in number of
    pixels) is returned.

    Parameters
    ----------
    sigma:
        Standard deviation for Gaussian kernel
    border:
      The border size (width, height) in number of pixels along each axis
    **params
      Additional keyword arguments passed to :func:`get_kernel_size`.

    Returns
    -------
    tuple:
      The border (width, height) compatible with the specified parameters.
      If a border was provided already, it is returned again, if no border
      was provided, the smallest compatible border is returned.

    See Also
    --------
    :func:`get_kernel_size` : Compute the kernel radius used in compatibility checks.
    :func:`get_kernel_diameter` : Compute the full diameter of the Gaussian kernel.
    """
    ks = get_kernel_size(sigma=sigma, **params)
    if border:
        assert all(ks <= b for b in border), f"A dimension of {border=} " \
                                             f"exceeds the kernel size {ks}"
        bs = border
    else:
        bs = (ks + 1,) * 2
    return bs


def bpgaussian(data: NDArray, **filter_params) -> NDArray:
    """Applies a border-preserving Gaussian filter

    The approach considers a Gaussian blur to be a weighted average over
    all pixels within the kernel diameter with the weight being given by
    the Gaussian function.
    Pixels close to `np.nan` values should simply "ignore" `np.nan` pixels
    and perform the weighted average over all non-`np.nan` pixels within
    the Gaussian kernel.
    This can be achieved with a normal Gaussian filter in a three-step process:

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
    **filter_params : dict
      Additional keyword arguments passed to :func:`skimage.filters.gaussian`.
      Common parameters include:

        - ``sigma`` : float
            Standard deviation for Gaussian kernel.
        - ``truncate`` : float
            Truncate filter at this many standard deviations.

      See :func:`skimage.filters.gaussian` for further parameters.

    See Also
    --------
    :func:`get_kernel_diameter` : Compute the effective kernel diameter for a given sigma.
    :func:`get_blur_params` : Compute Gaussian blur parameters from diameter or sigma.
    """
    # Substitute `np.nan`s with 0.0 and apply filter
    _data_nonnan = np.where(np.isnan(data), 0.0, data)
    _data_nonnan_blurred = gaussian(image=_data_nonnan, **filter_params)

    _data_binary = np.where(np.isnan(data), 0.0, 1.0)
    _data_binary_blurred = gaussian(image=_data_binary, **filter_params)

    _blurred_data = np.divide(_data_nonnan_blurred, _data_binary_blurred,
                              out=np.full(data.shape, np.nan),
                              where=~np.isnan(data))
    return _blurred_data


def get_blur_params(diameter: float | None = None,
                    sigma: float | None = None,
                    truncate: float = 3) -> dict[str, float]:
    """
    Compute Gaussian blur parameters from either `diameter` or `sigma`.

    Either `diameter` or `sigma` must be provided. Missing values are inferred
    from the others, and `truncate` is used or recomputed to maintain consistency.

    Parameters
    ----------
    diameter
        Kernel diameter. If provided with `sigma`, `truncate` is recomputed.
    sigma
        Standard deviation of the Gaussian kernel. If provided with `diameter`,
        `truncate` is recomputed.
    truncate
        Number of standard deviations at which to truncate the kernel.
        Default is 3. Ignored if both `diameter` and `sigma` are provided
        (recomputed).

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - `diameter`: computed kernel diameter
        - `sigma`: computed standard deviation
        - `truncate`: final truncate value

    Raises
    ------
    TypeError
        If neither `diameter` nor `sigma` is provided.

    Notes
    -----
    The function ensures that `diameter`, `sigma`, and `truncate` are consistent
    according to Gaussian kernel conventions.

    See Also
    --------
    :func:`get_kernel_diameter` : Compute the effective kernel diameter from sigma.
    :func:`get_kernel_size` : Compute the kernel radius from sigma.
    :func:`compatible_border_size` : Assert or compute a border compatible with the kernel.

    Examples
    --------
    >>> get_blur_params(diameter=15)
    {'diameter': 15, 'sigma': 2.5, 'truncate': 3}
    >>> get_blur_params(sigma=2.0)
    {'diameter': 12.0, 'sigma': 2.0, 'truncate': 3}
    >>> get_blur_params(diameter=15, sigma=3)
    {'diameter': 15, 'sigma': 3, 'truncate': 2.5}
    """
    if diameter is None and sigma is None:
        raise TypeError("Either the `diameter` or the `sigma` parameter "
                        f"must be provided. \nGot: {diameter=}, {sigma=}")

    if diameter:
        if sigma:
            truncate = 0.5 * diameter / sigma
        else:
            if truncate:
                sigma = 0.5 * diameter / truncate
    else:
        if sigma:
            diameter = 2 * sigma * truncate

    return dict(diameter=diameter, sigma=sigma, truncate=truncate)
