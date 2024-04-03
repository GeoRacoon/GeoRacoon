import numpy as np

from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbgauss


def test_view_definition():
    """Make sure that the views returned are correct
    """
    size = (9, 12)
    border = (1, 2)
    nbr_views = (3, 3)
    view_size = list(map(lambda i: int(size[i] / nbr_views[i]),
                         range(len(size))))
    print(view_size)
    # in a matrix of 9x9 with 3 views and a border size of 1 we expect a.o.:
    expected_views = [
        # in the reference corner:
        (0, 0, view_size[0] + border[0], view_size[1] + border[1]),
        (2, 2, 5, 8),  # view towards the center
        (2, 0, 5, 6),  # view vertically centered on horizontal left border
        (5, 6, 4, 6),  # view opposite to the reference corner
    ]
    views, inner_views = lbprep.create_views(nbr_views, border, size)
    for eb in expected_views:
        assert eb in views, f"View {eb} not in {views=}"
    inner_expected_views = [
        (0, 0, 3, 4),  # view in the reference corner
        (3, 4, 3, 4),  # view towards center
        (3, 0, 3, 4),  # view vertically centered on horizontal left border
        (6, 8, 3, 4),  # view opposite to the reference corner
    ]
    for ieb in inner_expected_views:
        assert ieb in inner_views, f"View {ieb} not in inner {inner_views=}"


def test_recombination():
    """Assert that a per-view filter application and recombination is
       equivalent to an overall application
    """
    from skimage.filters import gaussian
    mapshape = (120, 100)
    sigma = 1.5
    point_map = np.zeros(mapshape)
    output = np.zeros(mapshape)
    # create a random map with 0s and a few 1s
    vpos = np.random.randint(0, high=mapshape[0], size=4)
    hpos = np.random.randint(0, high=mapshape[1], size=4)
    point_map[vpos, hpos] = 1
    nbr_views = (20, 10)  # determine how many blocks along each axis
    # get the required border size (i.e. filter kernel size)
    ksize = lbgauss.get_kernel_size(sigma)
    border = (ksize, ksize)
    views, inner_views = lbprep.create_views(nbr_views, border, size=mapshape)
    desired_output = lbproc.apply_filter(
        point_map,
        gaussian,
        sigma=sigma,
    )
    # for each view apply the filter and write it back to
    output = np.zeros(mapshape)
    blocks = []
    for i, view in enumerate(views):
        # vslice = slice(block[0], block[0] + block[2])
        # hslice = slice(block[1], block[1] + block[3])
        _output = lbproc.apply_filter(
            lbprep.get_view(point_map, view),
            # point_map[vslice, hslice],
            gaussian,
            sigma=sigma,
        )
        # remove the border area of the block
        _usable_block = np.copy(
            lbprep.get_view(_output,
                            lbprep.relative_view(view, inner_views[i]))
        )
        blocks.append((_usable_block, inner_views[i]))
        lbprep.update_view(
            output,
            inner_views[i],
            lbprep.get_view(_output,
                            lbprep.relative_view(view, inner_views[i]))
            )
    # test the partial recombination
    np.testing.assert_array_equal(
       desired_output,
       output,
       'View-wise application fails to reproduce the desired output'
    )
    # test the recombination function
    np.testing.assert_array_equal(
        desired_output,
        lbprep.recombine_blocks(blocks, np.zeros(mapshape))
    )
