import numpy as np
import rasterio as rio

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from .config import ALL_MAPS
from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbgauss


def test_view_definition():
    """Make sure that the views returned are correct
    """
    size = (9, 12)
    border = (1, 2)
    view_size = (3, 4)
    # in a matrix of 9x9 with 3 views and a border size of 1 we expect a.o.:
    expected_views = [
        # in the reference corner:
        (0, 0, view_size[0] + border[0], view_size[1] + border[1]),
        (2, 2, 5, 8),  # view towards the center
        (2, 0, 5, 6),  # view vertically centered on horizontal left border
        (5, 6, 4, 6),  # view opposite to the reference corner
    ]
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=size)
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
    view_size = tuple(map(lambda x: int(x[0]/x[1]), zip(mapshape, nbr_views)))
    # get the required border size (i.e. filter kernel size)
    ksize = lbgauss.get_kernel_size(sigma)
    border = (ksize, ksize)
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=mapshape)
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
    # TODO: see #33
    # # test the partial recombination
    # np.testing.assert_array_equal(
    #    desired_output,
    #    output,
    #    'View-wise application fails to reproduce the desired output'
    # )
    # # test the recombination function
    # np.testing.assert_array_equal(
    #     desired_output,
    #     lbprep.recombine_blocks(blocks, np.zeros(mapshape))
    # )


def test_visualize_recombination_coverage():
    mapshape = (55, 56)
    sigma = 0.25
    point_map = np.zeros(mapshape)
    # create a random map with 0s and a few 1s
    vpos = np.random.randint(0, high=mapshape[0], size=4)
    hpos = np.random.randint(0, high=mapshape[1], size=4)
    point_map[vpos, hpos] = 1
    # nbr_views = (20, 10)  # determine how many blocks along each axis
    # view_size = list(map(lambda x: int(x[0]/x[1]), zip(mapshape, nbr_views)))
    view_size = (10, 10)
    # get the required border size (i.e. filter kernel size)
    ksize = lbgauss.get_kernel_size(sigma)
    border = (ksize, ksize)
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=mapshape)
    fig, ax = plt.subplots(figsize=(16, 16))
    boxes = [Rectangle(xy=(0, 0), width=mapshape[0], height=mapshape[1])]
    inner_boxes = []
    for i, view in enumerate(views):
        boxes.append(Rectangle(xy=(view[0:2]),
                               width=view[2],
                               height=view[3]))
        inner_boxes.append(Rectangle(xy=(inner_views[i][0:2]),
                                     width=inner_views[i][2],
                                     height=inner_views[i][3]))
    # Create patch collection with specified colour/alpha
    alpha = 0.3
    facecolor = 'none'
    edgecolor = 'red'
    pc_inner = PatchCollection(inner_boxes, facecolor=facecolor, alpha=alpha,
                               edgecolor=edgecolor)
    pc = PatchCollection(boxes, facecolor='none', alpha=1.0,
                         edgecolor='black')

    # Add collection to axes
    ax.add_collection(pc)
    ax.add_collection(pc_inner)
    ax.set_xlim(-1, mapshape[0]+1)
    ax.set_ylim(mapshape[1]+1, -1)
    ax.set_aspect('equal', adjustable='box')
    # # TODO: visually compare image
    # plt.show()
    # fig.savefig('testw.pdf')



@ALL_MAPS
def test_lct_coverage(datafiles):
    """Check if all cells have a land-cover type
    """
    from skimage.filters import gaussian
    # test_tif = 'data/reclass_GLC_FCS30_2015_utm32U.tif'
    test_tif = list(datafiles.iterdir())[0]
    sigma = 0.5
    truncate = 3
    with rio.open(test_tif) as src:
        profile = src.profile
        width = src.width
        height = src.height
        data = src.read(indexes=1)
    haslct = np.zeros((height, width), dtype=np.bool_)
    lcts = lbproc.get_lct(data)
    print(f"{profile=}")
    print(f"{width=}")
    print(f"{height=}")
    print(f"{data.shape=}")
    print(f"{haslct.shape=}")
    print(f"{lcts=}")
    blurred_layers = lbproc.get_filtered_layers(data, layers=lcts,
                               img_filter=gaussian,
                               filter_params=dict(
                                   sigma=sigma,
                                   truncate=truncate
                               ),
                               output_dtype=np.uint8
                               )
    entropy_data = lbproc.compute_entropy(blurred_layers, normed=True,
                                         output_dtype=np.uint8)
    for lct in lcts:
        print(f"{lct=}")
        lct_data = lbproc.get_layer_data(data, layer=lct,
                                         img_filter=gaussian,
                                         filter_params=dict(
                                             sigma=sigma,
                                             truncate=truncate
                                         ),
                                         output_dtype=np.uint8)
        print(f"\t{lct_data.dtype=}")
        print(f"\t{lct_data.shape=}")
        print(f"\t{np.unique(lct_data)=}")
        haslct = np.where(lct_data != 0, 1, haslct)
        vals, counts = np.unique(haslct, return_counts=True)
        print(f"\t{vals=}")
        print(f"\t{counts=}")
    # now make sure that we do not have any unassigned cells
    vals, counts = np.unique(haslct, return_counts=True)
    assert len(vals) == 1, f"We have cells without lct: {vals=}, {counts=}"

