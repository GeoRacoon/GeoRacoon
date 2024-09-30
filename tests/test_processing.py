import numpy as np
from skimage.filters import gaussian
from scipy.stats import entropy

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbf_gauss

from .conftest import ALL_MAPS, get_file


def test_select_category():
    """Filter an matrix of integers for a specific value
    """
    # Create a random matrix with integers in [1, 8]
    rand_map = np.random.randint(8, size=(100, 200)) + 1
    category = 4  # the category we want to get
    dtype = np.float16  # special case, normal would be np.uint8
    # set the values for a map and a miss
    is_v = np.finfo(dtype).max
    not_v = np.finfo(dtype).min
    target_data = lbproc.select_category(
        data=rand_map,
        category=category,
        as_dtype=dtype
    )
    # make sure only the maps and misses are present
    assert set(np.unique(target_data)) == {is_v, not_v}


def test_apply_filter_gaussian():
    """Test the application of a Gaussian filter
    """
    from skimage.filters import gaussian
    msize = 101
    point_map = np.zeros((msize, msize))
    center = int(0.5 * (msize - 1))
    point_map[center, center] = 1
    # parameters for the gaussian filter
    sigma = 1
    blurred = lbproc._apply_filter(point_map, gaussian, sigma=sigma)
    # make sure that the center point has still the maximal value
    assert np.unravel_index(np.argmax(blurred), blurred.shape)
    # make sure we do not just have 1's and 0's
    assert set(np.unique(blurred)) != {1, 0}


@ALL_MAPS
def test_nbr_lct(datafiles):
    """Make sure the detection of land-cover types works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = lbio.load_map(ch_map_tif)['data']
    categories = lbproc.get_categories(ch_data)
    unique_values = np.unique(categories)
    unique_values.sort()
    np.testing.assert_array_equal(categories, unique_values)


@ALL_MAPS
def test_single_category_filter(datafiles):
    """Make sure the detection of categories works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = lbio.load_map(ch_map_tif)['data']
    categories = lbproc.get_categories(ch_data)
    categories = np.unique(categories)
    categories.sort()  # those are our categories
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    scale = 100  # meter per pixel
    sigma = real_sigma / scale  # in pixel
    lct_blurred = lbproc.get_filtered_categories(ch_data, categories=categories,
                                                 img_filter=gaussian,
                                                 filter_params=dict(
                                                     sigma=sigma,
                                                     truncate=truncate
                                                 ))
    for _, data in lct_blurred.items():
        assert np.nanmax(data) >= 0.1


@ALL_MAPS
def test_filter_data(datafiles):
    """Make sure the detection of categories works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = lbio.load_map(ch_map_tif)['data']
    categories = lbproc.get_categories(ch_data)
    categories = np.unique(categories)
    categories.sort()  # those are our categories
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    scale = 100  # meter per pixel
    sigma = real_sigma / scale  # in pixel
    filter_params=dict(
        sigma=sigma,
        truncate=truncate
    )
    # compute in one go
    lct_blurred = lbproc.get_filtered_categories(ch_data, categories=categories,
                                                 img_filter=gaussian,
                                                 filter_params=filter_params,
                                                 output_dtype=np.uint8)
    # compute binary maps first then blur
    lct_binary = lbproc.get_filtered_categories(ch_data, categories=categories)
    for cat, data in lct_blurred.items():
        filtered_data = lbproc.filter_data(data=lct_binary[cat], img_filter=gaussian,
                                           filter_params=filter_params,
                                           filter_output_range=[0.,1.],
                                           output_dtype=np.uint8)
        np.testing.assert_equal(data, filtered_data)

@ALL_MAPS
def test_entropy_normalization_conversion(datafiles):
    """Test the normalization of the entropy along with casting to unsigned int
    """
    map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    map_data = lbio.load_map(map_tif)
    data = map_data['data']
    categories = lbproc.get_categories(data)
    entropy_data = lbproc.get_entropy(data, categories=categories,
                                       normed=False,
                                       img_filter=gaussian)
    normed_entropy_data = lbproc.get_entropy(data, categories=categories,
                                              normed=True,
                                              img_filter=gaussian)
    rescaled_entropy_data = lbproc.get_entropy(data, categories=categories,
                                                normed=True,
                                                output_dtype=np.uint8,
                                                img_filter=gaussian)
    max_entropy = lbproc.get_max_entropy(len(categories))
    assert np.nanmax(entropy_data) <= max_entropy, \
           'Maximal entropy is exceeded'
    assert np.nanmax(normed_entropy_data) == \
           np.nanmax(entropy_data)/max_entropy, 'Normalization is faulty'
    assert np.nanmax(rescaled_entropy_data) <= 255
    assert rescaled_entropy_data.dtype == np.uint8
