import numpy as np
from skimage.filters import gaussian

from convster import array_processing as ap
from convster import processing as csproc


def test_sigma_absolut():
    """Test if sigma scales with array size
    """
    kernel_dims = []
    for msize in [11, 51, 101, 501, 1001, 5001]:
        z = np.zeros((msize, msize), dtype=np.uint8)
        half = int(0.5 * (msize - 1))
        z[half, half] = 255
        sigma = 1
        blurred = csproc._apply_filter(z, gaussian, sigma=sigma)
        diameter = max(np.unique(ap.last_nonzero(blurred, axis=1)
                       - ap.first_nonzero(blurred, axis=1)))
        kernel_dims.append(diameter)
    assert len(list(set(kernel_dims))) == 1, "sigma scales with the map size!"


def test_auc_constance():
    """Check if the area under the kernel is affected by parameter choices
    """
    msize = 1001
    sigma = 1
    z = np.zeros((msize, msize), dtype=np.uint8)
    sums = []
    for trunc in [0.01, 0.1, 0.5, 1, 1.5, 2, 3, 5]:
        half = int(0.5 * (msize - 1))
        z[half, half] = 255
        blurred = csproc._apply_filter(z, gaussian, sigma=sigma, truncate=trunc)
        sums.append(np.round(np.sum(blurred), 8))
    assert sums[0] == 1.0, "AUC is not 1!"
    assert len(np.unique(sums)) == 1, \
           "The `truncate` parameter affects the AUC"
