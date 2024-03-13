import numpy as np
from landiv_blur import processing as lbproc


def test_filter_for_layer():
    """Filter an matrix of integers for a specific value
    """
    # Create a random matrix with integers in [1, 8]
    rand_map = np.random.randint(8, size=(100, 200)) + 1
    layer = 4  # the layer we want to get
    is_v, not_v = 1, 0  # set the values for a map and a miss
    target_layer = lbproc.filter_for_layer(
        data=rand_map,
        layer=layer,
        is_value=is_v,
        not_value=not_v
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
