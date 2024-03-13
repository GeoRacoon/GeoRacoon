import numpy as np

from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbgauss


def test_view_definition():
    """Make sure that the views returned are correct
    """
    msize = 9
    size = (msize, msize)
    border = 1
    nbr_views = 3
    view_size = int(msize / nbr_views)
    # in a matrix of 9x9 with 3 views and a border size of 1 we expect a.o.:
    expected_views = [
        (0, 0, view_size + border, view_size + border),  # in the reference corner
        (2, 2, 5, 5),  # view in the center computing the center 3x3 square
        (2, 0, 5, 4),  # view vertically centered on horizontal border
        (5, 5, 4, 4),  # view opposite to the reference corner
    ]
    views, inner_views = lbprep.create_views(nbr_views, border, size)
    for eb in expected_views:
        assert eb in views, f"View {eb} not in {views=}"
    inner_expected_views = [
        (0, 0, 3, 3),  # view in the reference corner
        (3, 3, 3, 3),  # view in the center computing the center 3x3 square
        (3, 0, 3, 3),  # view vertically centered on horizontal border
        (6, 6, 3, 3),  # view opposite to the reference corner
    ]
    for ieb in inner_expected_views:
        assert ieb in inner_views, f"View {ieb} not in inner {views=}"


def test_recombination():
    """Assert that a per-view filter application and recombination is
       equivalent to an overall application
    """
    from skimage.filters import gaussian
    msize = 120
    sigma = 1.5
    point_map = np.zeros((msize, msize))
    output = np.zeros((msize, msize))
    # create a random map with 0s and a few 1s
    vpos = np.random.randint(0, high=msize, size=4)
    hpos = np.random.randint(0, high=msize, size=4)
    point_map[vpos, hpos] = 1
    nbr_views = 20
    # get the required border size (i.e. filter kernel size)
    border = lbgauss.get_kernel_size(sigma)
    views, inner_views= lbprep.create_views(nbr_views, border, size=(msize, msize))
    desired_output = lbproc.apply_filter(
        point_map,
        gaussian,
        sigma=sigma,
    )
    # for each view apply the filter and write it back to
    output = np.zeros((msize, msize))
    blocks = []
    for i, view in enumerate(views):
        # vslice = slice(block[0], block[0] + block[2])
        # hslice = slice(block[1], block[1] + block[3])
        _output = lbproc.apply_filter(
            lbprep.get_view(point_map, view),
            #point_map[vslice, hslice],
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
        lbprep.recombine_blocks(blocks, np.zeros((msize, msize)))
    )
