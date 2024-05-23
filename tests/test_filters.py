import numpy as np

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur.filters import _filters
from landiv_blur.filters import _get_kernel_diam
from landiv_blur.filters import _get_kernel_size
from landiv_blur.filters.gaussian import gaussian

from .config import ALL_MAPS


def test_kernel_scaling():
    for get_ks in _get_kernel_size:
        default_sigma = 1.0
        default_size = get_ks(sigma=default_sigma)
        expected_ks = []
        kernels = []
        for scale in [0.5, 1, 1.5, 2, 10, 100]:
            sigma = scale * default_sigma
            kernels.append(get_ks(sigma))
            print(f"{sigma=} - kernel size {kernels[-1]}")
            expected_ks.append(default_size*scale)
        assert kernels == expected_ks, 'Gaussian kernel size does not scale' \
               'linearly with sigma!'


@ALL_MAPS
def test_filter_signal_preservation(datafiles):
    """Filtering shouldn't lose more than 0.1% of the initial signal in the map
    """
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_data = lbio.load_map(ch_map_tif)['data']
    lctypes = lbproc.get_lct(ch_data)
    sigma = 10
    truncate = 3
    params = dict(
        sigma=sigma,
        truncate=truncate
    )
    for layer in lctypes:
        dtype = np.uint8
        dmax = np.iinfo(dtype).max
        layer_data = lbproc.get_layer_data(ch_data, layer=layer,
                                           output_dtype=dtype)
        filtered_data = lbproc.get_layer_data(ch_data, layer=layer,
                                              img_filter=gaussian,
                                              params=params)
        signal = (layer_data.astype(float)/dmax).sum()
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
    data = lbproc.filter_for_layer(data=square, layer=1, as_dtype=np.uint8)
    sigma = 3
    truncate = 3
    for _filter, _get_kd, _get_ks in zip(_filters,
                                         _get_kernel_diam,
                                         _get_kernel_size):
        kd = _get_kd(sigma=sigma, truncate=truncate)
        ks = _get_ks(sigma=sigma, truncate=truncate)
        print(f"kernel diameter {kd=}")
        print(f"kernel size {ks=}")
        gsquare = _filter(data, sigma=sigma, truncate=truncate)
        assert round(signal, 2) == round(gsquare.sum(), 2)
