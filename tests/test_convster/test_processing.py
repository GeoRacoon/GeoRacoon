import pytest

import itertools
import random

import numpy as np
import rasterio as rio

from skimage.filters import gaussian

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from riogrande import helper as rghelp
from riogrande import io as rgio
from riogrande import prepare as rgprep

from convster import processing as csproc
from convster import prepare as csprep
from convster.filters import gaussian as lbgauss

from .conftest import (
    ALL_MAPS,
    get_file,
)


def test_select_category():
    """Filter an matrix of integers for a specific value
    """
    # Create a random matrix with integers in [1, 8]
    rand_map = np.random.randint(8, size=(100, 200)) + 1
    category = 4  # the category we want to get
    dtype = "float16"  # special case, normal would be np.uint8
    # set the values for a map and a miss
    is_v = np.finfo(dtype).max
    not_v = np.finfo(dtype).min
    target_data = csproc.select_category(
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
    blurred = csproc._apply_filter(point_map, gaussian, sigma=sigma)
    # make sure that the center point has still the maximal value
    assert np.unravel_index(np.argmax(blurred), blurred.shape)
    # make sure we do not just have 1's and 0's
    assert set(np.unique(blurred)) != {1, 0}


@ALL_MAPS
def test_nbr_lct(datafiles):
    """Make sure the detection of land-cover types works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = rgio.load_block(ch_map_tif)['data']
    categories = csproc.get_categories(ch_data)
    unique_values = np.unique(categories)
    unique_values.sort()
    np.testing.assert_array_equal(categories, unique_values)


@ALL_MAPS
def test_single_category_filter(datafiles):
    """Make sure the detection of categories works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = rgio.load_block(ch_map_tif)['data']
    categories = csproc.get_categories(ch_data)
    categories = np.unique(categories)
    categories.sort()  # those are our categories
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    scale = 100  # meter per pixel
    sigma = real_sigma / scale  # in pixel
    lct_blurred = csproc.get_filtered_categories(ch_data, categories=categories,
                                                 img_filter=gaussian,
                                                 filter_params=dict(
                                                     sigma=sigma,
                                                     truncate=truncate,
                                                     preserve_range=False,
                                                 ),
                                                 filter_output_range=(0,1),
                                                 )
    for _, data in lct_blurred.items():
        assert np.nanmax(data) >= 0.1


@ALL_MAPS
def test_filter_data(datafiles):
    """Make sure the detection of categories works as expected
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = rgio.load_block(ch_map_tif)['data']
    categories = csproc.get_categories(ch_data)
    categories = np.unique(categories)
    categories.sort()  # those are our categories
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    scale = 100  # meter per pixel
    sigma = real_sigma / scale  # in pixel
    filter_params=dict(
        sigma=sigma,
        truncate=truncate,
        preserve_range=False,
    )
    # compute in one go
    lct_blurred = csproc.get_filtered_categories(ch_data, categories=categories,
                                                 img_filter=gaussian,
                                                 filter_params=filter_params,
                                                 filter_output_range=(0, 1),
                                                 output_dtype="uint8")
    # compute binary maps first then blu
    lct_binary = csproc.get_filtered_categories(ch_data, categories=categories)

    for cat, data in lct_blurred.items():
        filtered_data = csproc.filter_data(data=lct_binary[cat], img_filter=gaussian,
                                           filter_params=filter_params,
                                           filter_output_range=(0., 1.),
                                           as_dtype="uint8")
        np.testing.assert_equal(data, filtered_data)

@ALL_MAPS
def test_filter_data_float(datafiles):
    """Make sure that filter works with floats (continous) data as well
    """
    ch_f_map_tif = get_file(pattern="Switzerland_area_frac_*.tif", datafiles=datafiles)
    diameter = 10000  # 10km
    scale = 1000  # meter per pixel
    truncate = 3  # after 3 sigma
    real_sigma = 0.5 * diameter / truncate
    filter_output_range = (0, 1)
    filter_params=dict(
        sigma=real_sigma / scale,  # in pixel,
        truncate=truncate,
        preserve_range=False,
    )
    ch_f_data = rgio.load_block(ch_f_map_tif)['data']  # this will automatically take the first band (okay here)
    # First without replacint NaN values (so the image will be cropped)
    with pytest.warns(UserWarning) as record:
        filtered_data = csproc.filter_data(data=ch_f_data,
                                           img_filter=gaussian,
                                           filter_params=filter_params,
                                           filter_output_range=filter_output_range,
                                           as_dtype="float32",
                                           output_range=(0,1)
                                           )
        assert str(record[0].message).startswith("Raster array has NaN")
    # Check that the nan's have been 'eating up area'
    assert np.isnan(filtered_data).sum() >= np.isnan(ch_f_data).sum()

    # Second replace the NaNs with 0s
    filtered_data = csproc.filter_data(data=ch_f_data,
                                       img_filter=gaussian,
                                       replace_nan_with=0,
                                       filter_params=filter_params,
                                       filter_output_range=filter_output_range,
                                       as_dtype="float32",
                                       output_range=(0, 1)
                                       )
    # Check that the nan's have been 'eating up area'
    assert np.isnan(filtered_data).sum() == np.isnan(ch_f_data).sum()

@ALL_MAPS
def test_entropy_normalization_conversion(datafiles):
    """Test the normalization of the entropy along with casting to unsigned int
    """
    # TODO: these tests become a bit unnecessary as get_entropy should be removed
    map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    map_data = rgio.load_block(map_tif)
    data = map_data['data']
    blur_output_dtype = "uint8"  # convert the blurred data to uint8 before
    categories = csproc.get_categories(data)

    filter_output_range = (0, 1)
    filter_params = dict(
        preserve_range=False
    )

    max_entropy = csproc.get_max_entropy(len(categories))
    print(f'{max_entropy=}')

    with pytest.warns(expected_warning=UserWarning, match='bounded'):
        # without normalization we cannot set the output_range (as we have not
        # fixed input range > unbounded).
        entropy_data = csproc.get_entropy(data=data,
                                          categories=categories,
                                          normed=False,
                                          output_range=(0,1),  # this should lead to a warning
                                          img_filter=gaussian,
                                          filter_params=filter_params,
                                          blur_output_dtype=blur_output_dtype,
                                          filter_output_range=filter_output_range,
                                          )

    assert np.nanmax(entropy_data) <= max_entropy, \
           'Maximal entropy is exceeded'

    with pytest.warns(expected_warning=UserWarning, match='without rescaling'):
        # we convert but do not normalize > no rescaling possible, only type
        # conversion
        _= csproc.get_entropy(data=data,
                              categories=categories,
                              normed=False,
                              as_dtype="uint8",  # this should lead to a warning
                              img_filter=gaussian,
                              filter_params=filter_params,
                              blur_output_dtype=blur_output_dtype,
                              filter_output_range=filter_output_range,)

    entropy_data = csproc.get_entropy(data=data,
                                     categories=categories,
                                     normed=False,
                                     as_dtype="float64",
                                     img_filter=gaussian,
                                     filter_params=filter_params,
                                     blur_output_dtype=blur_output_dtype,
                                     filter_output_range=filter_output_range,
                                     )
    print(f"{np.nanmax(entropy_data)=}")
    print(f"{np.nanmin(entropy_data)=}")
    normed_entropy_data = csproc.get_entropy(data=data, categories=categories,
                                             normed=True,
                                             img_filter=gaussian,
                                             filter_params=filter_params,
                                             filter_output_range=filter_output_range,
                                             blur_output_dtype=blur_output_dtype,
                                             )

    print(f"{np.nanmax(normed_entropy_data)=}")
    print(f"{np.nanmin(normed_entropy_data)=}")
    assert np.nanmax(normed_entropy_data) == \
           np.nanmax(entropy_data)/max_entropy, 'Normalization is faulty'

    normed_set_maximum_entropy_data = csproc.get_entropy(
        data=data,
        categories=categories,
        normed=True,
        max_entropy_categories=(len(categories) * 2),
        img_filter=gaussian,
        filter_params=filter_params,
        filter_output_range=filter_output_range,
        blur_output_dtype=blur_output_dtype,
    )
    assert np.nanmax(normed_set_maximum_entropy_data) <= \
           np.nanmax(entropy_data)/ csproc.get_max_entropy(len(categories)*2)

    rescaled_entropy_data = csproc.get_entropy(data, categories=categories,
                                               normed=True,
                                               as_dtype="uint8",
                                               img_filter=gaussian,
                                               filter_params=filter_params,
                                               filter_output_range=filter_output_range,
                                               blur_output_dtype=blur_output_dtype,
                                               )
    print(f"{np.nanmax(rescaled_entropy_data)=}")
    print(f"{np.nanmin(rescaled_entropy_data)=}")
    assert np.nanmax(rescaled_entropy_data) <= 255
    assert rescaled_entropy_data.dtype == np.uint8


@ALL_MAPS
def test_interaction_computation(datafiles):
    """Test the computation of the interaction on a general bassis
    """
    # TODO: also test this for float64 - which fails until now as the dtype_range function return inf for _max
    # see issue #115 (https://gitlab.uzh.ch/landiv/landiv/-/issues/115) for more
    test_dtype = np.uint8

    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = rgio.load_block(ch_map_tif)['data']
    categories = csproc.get_categories(ch_data)
    diameter = 1000  # 1km
    truncate = 3  # after 3 sigma
    scale = 100  # meter per pixel
    blurred_categories = csproc.get_filtered_categories(ch_data,
                                                        categories=categories,
                                                        img_filter=gaussian,
                                                        filter_params=dict(
                                                            sigma=(0.5 * diameter / truncate) / scale,  # in pixel
                                                            truncate=truncate,
                                                            preserve_range=True),
                                                        output_dtype=test_dtype)
    # Pairs
    # TODO: write a better test here (it still fails for i >= 3
    for i in range(2, 3):
        all_possible_pairs = [list(x) for x in itertools.combinations(categories, r=i)]
        test_pair = random.choice(all_possible_pairs)
        data_array = [blurred_categories[c] for c in test_pair]
        # Interaction
        interaction_data = csproc.compute_interaction(data_arrays=data_array,
                                                      input_dtype=test_dtype,
                                                      normed=True,
                                                      standardize=False,
                                                      output_dtype=test_dtype)
        # make sure output is correct
        assert interaction_data.dtype == test_dtype
        assert np.nanmax(interaction_data) <= 255
        # Calculate manual result
        manual_result = np.ones(data_array[0].shape, dtype=float)
        for arr in data_array:
            manual_result *= arr.astype(float)
        manual_result = np.ceil((manual_result * len(data_array) ** len(data_array)) / 255).astype(test_dtype)
        # Check if the error (from rounding etc.) is not > 1 in unit8
        assert np.nanmax(np.absolute(interaction_data.astype(float) - manual_result.astype(float))) <= 1

def test_interaction_standardized_arrays():
    # define example arrays
    x = np.array([[0.5, 0.25], [0.0, 0.05]])
    y = np.array([[0.5, 0.25], [0.0, 0.3]])

    # test for float and unit8
    type_test = {255: np.uint8, 1: np.float64}

    for _max, _type in type_test.items():
        a = (x * _max).astype(_type)
        b = (y * _max).astype(_type)

        # manual calculation
        inter = np.ones_like(a, dtype=float)
        for i in [a, b]:
            inter *= (i / _max)
        stand = np.zeros_like(a, dtype=float)
        for j in [a, b]:
            stand += (j / _max)
        result = np.divide(inter, stand, out=np.zeros_like(inter), where=stand != 0)
        result = result / 0.25  # maximum value for 50:50 mixture of 2 types 0.5*0.5
        if _type == np.uint8:
            result = np.ceil(result * _max).astype(_type)
        else:
            result = (result * _max).astype(_type)
        # compare with result from function
        interaction_array = csproc.compute_interaction([a, b],
                                                       input_dtype=_type,
                                                       standardize=True,
                                                       normed=True,
                                                       output_dtype=_type)
        # print(f"{a=}\n{b=}")
        # print(f"{interaction_array=}\n{result=}")
        # Assert arrays are equal
        np.testing.assert_array_equal(interaction_array, result)


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
    views, inner_views = rgprep.create_views(view_size=view_size,
                                             border=border,
                                             size=mapshape)
    desired_output = csproc._apply_filter(
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
        _output = csproc._apply_filter(
            rgprep.get_view(point_map, view),
            # point_map[vslice, hslice],
            gaussian,
            sigma=sigma,
        )
        # remove the border area of the block
        _usable_block = np.copy(
            rgprep.get_view(_output,
                            rgprep.relative_view(view, inner_views[i]))
        )
        blocks.append((_usable_block, inner_views[i]))
        rgprep.update_view(
            output,
            inner_views[i],
            rgprep.get_view(_output,
                            rgprep.relative_view(view, inner_views[i]))
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



@ALL_MAPS
def test_lct_coverage(datafiles):
    """Check if all cells have a land-cover type
    """
    from skimage.filters import gaussian
    # test_tif = 'data/reclass_GLC_FCS30_2015_utm32U.tif'
    test_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    sigma = 0.5
    truncate = 3
    with rio.open(test_tif) as src:
        profile = src.profile
        width = src.width
        height = src.height
        data = src.read(indexes=1)
    haslct = np.zeros((height, width), dtype=np.bool_)
    lcts = csproc.get_categories(data)
    # print(f"{profile=}")
    # print(f"{width=}")
    # print(f"{height=}")
    # print(f"{data.shape=}")
    # print(f"{haslct.shape=}")
    # print(f"{lcts=}")
    blurred_categories = csproc.get_filtered_categories(data, categories=lcts,
                                                        img_filter=gaussian,
                                                        filter_params=dict(
                                                            sigma=sigma,
                                                            truncate=truncate
                                                        ),
                                                        output_dtype='uint8',
                                                        filter_output_range=(0,1),
                                                        )
    entropy_data = csproc.compute_entropy(
        data_arrays=tuple(blurred_categories.values()),
        normed=True,
        as_dtype=np.uint8
    )
    for lct in lcts:
        # print(f"{lct=}")
        lct_data = csproc.get_category_data(data, category=lct,
                                            img_filter=gaussian,
                                            filter_params=dict(
                                                sigma=sigma,
                                                truncate=truncate
                                            ),
                                            filter_output_range=(0,1),
                                            as_dtype='uint8')
        # print(f"\t{lct_data.dtype=}")
        # print(f"\t{lct_data.shape=}")
        # print(f"\t{np.unique(lct_data)=}")
        print(f"{np.unique(lct_data)=}")
        haslct = np.where(lct_data != 0, 1, haslct)
        vals, counts = np.unique(haslct, return_counts=True)
        # print(f"\t{vals=}")
        # print(f"\t{counts=}")
    # now make sure that we do not have any unassigned cells
    vals, counts = np.unique(haslct, return_counts=True)
    assert len(vals) == 1, f"We have cells without lct: {vals=}, {counts=}"



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
    views, inner_views = rgprep.create_views(view_size=view_size,
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
def test_import_export(datafiles):
    """Export per-cell entropy map after categories are blurred, load it and compare.
    """
    start = (1020, 1020)
    size = (700, 700)
    view1 = (*start, *size)
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    block = rgio.load_block(ch_map_tif, view=view1, indexes=1)
    entropy_array = csproc.get_entropy(block['data'], categories=range(8),
                                       normed=True,
                                       img_filter=gaussian)
    outfile = datafiles / 'out.tif'
    rgio._export_to_tif(
        destination=str(outfile),
        data=entropy_array,
        orig_profile=block['orig_profile'],
        # we need the transform from the window from block 1
        transform=block['transform']
    )
    view2 = (0, 0, *size)
    block_2 = rgio.load_block(outfile, view=view2)
    # NOTE: if the arrays contain np.nan then np.all will always be False
    assert np.all(np.nan_to_num(entropy_array,
                  nan=-1) == np.nan_to_num(block_2['data'], nan=-1))
    assert block['transform'] == block_2['transform']

@ALL_MAPS
def test_convert_to_dtype_real_range_handling(datafiles):
    """Make sure datatypes are properly converted
    """
    ch_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = rgio.load_block(ch_tif)['data']
    ch_range = np.nanmin(ch_data), np.nanmax(ch_data)
    print(ch_range)

    lctypes = csproc.get_categories(ch_data)
    sigma = 10
    truncate = 3
    params = dict(
        sigma=sigma,
        truncate=truncate
    )
