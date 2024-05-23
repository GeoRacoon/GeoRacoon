import numpy as np
from skimage.filters import gaussian
from scipy.stats import entropy

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbf_gauss

from .config import ALL_MAPS


def test_filter_for_layer():
    """Filter an matrix of integers for a specific value
    """
    # Create a random matrix with integers in [1, 8]
    rand_map = np.random.randint(8, size=(100, 200)) + 1
    layer = 4  # the layer we want to get
    dtype = np.float16  # special case, normal would be np.uint8
    # set the values for a map and a miss
    is_v = np.finfo(dtype).max
    not_v = np.finfo(dtype).min
    target_layer = lbproc.filter_for_layer(
        data=rand_map,
        layer=layer,
        as_dtype=dtype
    )
    # make sure only the maps and misses are present
    assert set(np.unique(target_layer)) == {is_v, not_v}


def test_apply_filter_gaussian():
    """Test the application of a gaussian filter
    """
    from skimage.filters import gaussian
    msize = 101
    point_map = np.zeros((msize, msize))
    center = int(0.5 * (msize - 1))
    point_map[center, center] = 1
    # parameters for the gaussian filter
    sigma = 1
    blurred = lbproc.apply_filter(point_map, gaussian, sigma=sigma)
    # make sure that the center point has still the maximal value
    assert np.unravel_index(np.argmax(blurred), blurred.shape)
    # make sure we do not just have 1's and 0's
    assert set(np.unique(blurred)) != {1, 0}


@ALL_MAPS
def test_nbr_lct(datafiles):
    """Make sure the detection of land-cover types works as expected
    """
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_data = lbio.load_map(ch_map_tif)['data']
    lctypes = lbproc.get_lct(ch_data)
    unique_values = np.unique(lctypes)
    unique_values.sort()
    np.testing.assert_array_equal(lctypes, unique_values)


@ALL_MAPS
def test_single_layer_filter(datafiles):
    """Make sure the detection of land-cover types works as expected
    """
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_data = lbio.load_map(ch_map_tif)['data']
    lctypes = lbproc.get_lct(ch_data)
    lctypes = np.unique(lctypes)
    lctypes.sort()  # those are our layers
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    scale = 100  # meter per pixel
    sigma = real_sigma / scale  # in pixel
    lct_blurred = lbproc.get_filtered_layers(ch_data, layers=lctypes,
                                             img_filter=gaussian,
                                             sigma=sigma,
                                             truncate=truncate)
    for lct, data in lct_blurred.items():
        assert np.nanmax(data) >= 0.1


@ALL_MAPS
def test_entropy_normalization_conversion(datafiles):
    """Test the normalization of the entropy along with casting to unsigned int
    """
    map_tif = list(datafiles.iterdir())[0]
    map = lbio.load_map(map_tif)
    data = map['data']
    lctypes = lbproc.get_lct(data)
    entropy_layer = lbproc.get_entropy(data, lctypes,
                                       normed=False,
                                       img_filter=gaussian)
    normed_entropy_layer = lbproc.get_entropy(data, lctypes,
                                              normed=True,
                                              img_filter=gaussian)
    rescaled_entropy_layer = lbproc.get_entropy(data, lctypes,
                                                normed=True,
                                                dtype=np.uint8,
                                                img_filter=gaussian)
    max_entropy = lbproc.get_max_entropy(len(lctypes))
    assert np.nanmax(entropy_layer) <= max_entropy, \
           'Maximal entropy is exceeded'
    assert np.nanmax(normed_entropy_layer) == \
           np.nanmax(entropy_layer)/max_entropy, 'Normalization is faulty'
    assert np.nanmax(rescaled_entropy_layer) <= 255
    assert rescaled_entropy_layer.dtype == np.uint8
