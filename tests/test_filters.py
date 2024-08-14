import pytest

from itertools import product

import numpy as np

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur.filters import _filters
from landiv_blur.filters import _get_kernel_diam
from landiv_blur.filters import _get_kernel_size
from landiv_blur.filters.gaussian import gaussian, compatible_border_size

from .conftest import ALL_MAPS, get_file


def test_kernel_scaling():
    for get_ks in _get_kernel_size:
        default_sigma = 1.0
        default_size = get_ks(sigma=default_sigma)
        expected_ks = []
        kernels = []
        for scale in [0.5, 1, 1.5, 2, 10, 100]:
            sigma = scale * default_sigma
            kernels.append(get_ks(sigma))
            # print(f"{sigma=} - kernel size {kernels[-1]}")
            expected_ks.append(default_size*scale)
        assert kernels == expected_ks, 'Gaussian kernel size does not scale' \
               'linearly with sigma!'


@ALL_MAPS
def test_filter_signal_preservation(datafiles):
    """Filtering shouldn't lose more than 0.1% of the initial signal in the map
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = lbio.load_map(ch_map_tif)['data']
    lctypes = lbproc.get_categories(ch_data)
    sigma = 10
    truncate = 3
    params = dict(
        sigma=sigma,
        truncate=truncate
    )
    for cat in lctypes:
        dtype = np.uint8
        dmax = np.iinfo(dtype).max
        category_data = lbproc.get_category_data(ch_data, category=cat,
                                              output_dtype=dtype)
        filtered_data = lbproc.get_category_data(ch_data, category=cat,
                                                 img_filter=gaussian,
                                                 filter_params=params)
        signal = (category_data.astype(float)/dmax).sum()
        diff = abs(signal - filtered_data.astype(float).sum())
        assert diff/signal <= 0.001


def test_signal_preservation():
    """Make sure the overall signal is preserved.

    In particular we want that the sum over all pixel remains approx. constant.
    """
    square = np.zeros((501, 501), dtype=int)
    square[246:257, 246:257] = 1
    signal = square.sum()
    # convert it to the expected format
    data = lbproc.select_category(data=square, category=1, as_dtype=np.uint8)
    sigma = 3
    truncate = 3
    for _filter, _get_kd, _get_ks in zip(_filters,
                                         _get_kernel_diam,
                                         _get_kernel_size):
        kd = _get_kd(sigma=sigma, truncate=truncate)
        ks = _get_ks(sigma=sigma, truncate=truncate)
        # print(f"kernel diameter {kd=}")
        # print(f"kernel size {ks=}")
        gsquare = _filter(data, sigma=sigma, truncate=truncate)
        assert round(signal, 2) == round(gsquare.sum(), 2)

def test_border_checking():
    """Make sure that we identify border sizes that truncate a kernel.
    """
    square = np.zeros((501, 501), dtype=int)
    center = 250
    square[center, center] = 1
    sigmas = [1, 3, 5, 10]
    truncates = [1,2,3,4,5]
    borders = [1, 5, 10, 20, 50]
    for sigma, t, b in product(sigmas, truncates, borders):
        border = (b, b)
        blurred = gaussian(square, sigma=sigma, truncate=t)
        if blurred[center + b + 1, center] == 0:
            # if the pixel outside the border is 0 we are good
            bs = compatible_border_size(sigma=sigma, border=border, truncate=t)
        else:
            # otherwise, the compatible_border_size should complain
            with pytest.raises(AssertionError):
                bs = compatible_border_size(sigma=sigma, border=border,
                                            truncate=t)
                assert blurred[center + bs[1] + 1, center] == 0

