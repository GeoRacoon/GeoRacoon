import builtins
from functools import partial

from time import sleep

import numpy as np
import multiprocessing as mproc
import itertools
import random
import rasterio as rio

from landiv_blur import helper as lbhelp
from landiv_blur import io as lbio
from landiv_blur import io_ as lbio_
from landiv_blur import processing as lbproc
from landiv_blur import prepare as lbprep
from landiv_blur import inference as lbinf
from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lbf_gauss
from landiv_blur.helper import rasterio_to_numpy_dtype

from .conftest import ALL_MAPS, get_file, set_mpc_strategy

from matplotlib import pyplot as plt

@ALL_MAPS
def test_extract_categories(datafiles):
    """Make sure the extract categories works as expected
    """
    verbose = True
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    lct_source = lbio_.Source(path=landcover_map)
    source_profile = lct_source.import_profile()
    source_band = lct_source.get_band(bidx=1)
    print(f"{source_profile=}")
    # extract categories without applying a filter
    to_dtype="uint8"
    categories = [1,2,3,4,5]
    category_tif = lbpara.extract_categories(
        source=lct_source,
        categories=categories,
        output_file=str(datafiles / 'category_out.tif'),
        output_dtype=to_dtype,
        block_size=(500, 500),
        compress = True,
        output_params = dict(
            nodata=0,
            dtype=to_dtype
        ),
    )
    category_source = lbio_.Source(category_tif)
    assert len(category_source.get_bands()) == len(categories)
    source_data = source_band.get_data()
    for cat in categories:
        # check for each category whether the extraction
        # is identical to the real data
        cat_band = category_source.get_band(category=cat)
        cat_data = cat_band.get_data()
        _cat_data = np.where(cat_data == 255, 1, 0)
        _category_data = np.where(source_data==cat, 1, 0)
        np.testing.assert_equal(_cat_data, _category_data)

    # check nodata handling
    # creat an output file (with changed nodata)
    to_dtypes = ["float32", "int16", "uint8"]
    nodatas = [np.nan, 0, None]
    for nodata, to_dtype in zip(nodatas, to_dtypes):
        tmp_map = str(datafiles / 'bands_out.tif')
        tmp_source = lbio_.Source(path=tmp_map)
        tmp_profile = source_profile.copy()
        tmp_profile['nodata'] = nodata
        tmp_profile['dtype'] = to_dtype
        tmp_source.profile = tmp_profile
        tmp_source.init_source(overwrite=True)
        # sanity check for Source.init_source resp. Source.open
        with rio.open(tmp_source.path, 'r') as src:
            _profile = src.profile.copy()
        np.testing.assert_equal(_profile['nodata'], nodata)

        tmp_band = lbio_.Band(source=tmp_source, bidx=1)
        # write out data as 
        tmp_band.set_data(source_band.get_data().astype(to_dtype))
        filter_params = dict(
            sigma = 100,
            truncate = 3
        )
        blurred_tif = lbpara.extract_categories(
            source=tmp_source,
            categories=[1,2,3,4,5],
            output_file=str(datafiles / 'blur_out.tif'),
            img_filter=lbf_gauss.gaussian,
            filter_params=filter_params,
            output_dtype=to_dtype,
            block_size=(500, 500),
            compress = True,
            output_params = dict(
                nodata=nodata,
                dtype=to_dtype
            ),
        )
        out_source = lbio_.Source(path=blurred_tif)
        out_profile = out_source.import_profile()
        print(f"{out_profile=}")
        np.testing.assert_equal(out_profile['nodata'], nodata)
        # We need to map GDAL to numpy datatypes
        assert out_profile['dtype'] == to_dtype
        #np.testing.assert_equal(rasterio_to_numpy_dtype(out_profile['dtype']), to_dtype)

@ALL_MAPS
def test_apply_filter(datafiles):
    """Test the parallel application of a filter to one or serveral bands
    """
    blur_para = str(datafiles / 'blur_para.tif')
    blur_single = str(datafiles / 'blur_single.tif')
    bands_out = str(datafiles / 'bands_out.tif')
    diameter = 1000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    block_size = (500, 400)
    output_dtype = np.uint8  # data type to use for the blurred arrays
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    img_filter=lbf_gauss.gaussian
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    categories = [1,2,3]
    # compute in one go
    blurred_tif = lbpara.extract_categories(
        source=str(ch_map_tif),
        categories=categories,
        output_file=blur_single,
        img_filter=img_filter,
        filter_params=filter_params,
        filter_output_range=(0,1),
        output_params=dict(
            as_dtype=output_dtype,
            output_range=output_dtype
        ),
        block_size=block_size,
        compress=True
    )
    # now in two steps:
    # first extract categories
    bands_tif = lbpara.extract_categories(
        source=str(ch_map_tif),
        categories=categories,
        output_file=bands_out,
        img_filter=None,
        filter_params=filter_params,
        output_dtype=output_dtype,
        block_size=block_size,
        verbose=True,
        compress=True
    )
    # apply the filter in parallel
    blurred_para = lbpara.apply_filter(source=bands_tif,
                                       output_file=blur_para,
                                       block_size=block_size,
                                       bands=None,
                                       data_as_dtype=np.uint8,
                                       img_filter=img_filter,
                                       filter_params=filter_params,
                                       filter_output_range=(0.,1.),
                                       output_dtype=output_dtype,
                                       verbose=True)
    for cat in categories:
        b_nope = lbio_.Band(source=lbio_.Source(path=bands_tif),
                            bidx=cat)
        b_twostep = lbio_.Band(source=lbio_.Source(path=blurred_para),
                               bidx=cat)
        b_single = lbio_.Band(source=lbio_.Source(path=blurred_tif),
                              bidx=cat)
        b_nope.import_tags()
        b_twostep.import_tags()
        b_single.import_tags()
        np.testing.assert_equal(b_twostep.get_data(), b_single.get_data())

@ALL_MAPS
def test_reduced_mask(datafiles):
    """Compute a mask from multiple bands in one go and then in parallel
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    source = lbio_.Source(path=ch_map_tif)
    blur_out = str(datafiles / 'blur_out.tif')
    # create the blurred bands
    img_filter = lbf_gauss.gaussian
    output_dtype = np.uint8
    diameter = 5000  # this is in meter
    scale = 100  # meter per pixel
    truncate = 3
    view_size = (500, 400)
    categories = [1, 2, 3, 4, 5]
    _diameter = diameter / scale
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    blurred_tif = lbpara.extract_categories(
        source=source,
        categories=categories,
        output_file=blur_out,
        img_filter=img_filter,
        filter_params=filter_params,
        output_dtype=output_dtype,
        block_size=view_size,
        compress=True
    )
    blurr_source = lbio_.Source(path=blurred_tif)
    initial_mask = blurr_source.get_mask()
    # get the mask loading the entire dataset
    with blurr_source.data_reader(mode='r') as read:
        dataset = read()
    # print(f"{dataset.shape=}")
    mask = lbhelp.reduced_mask(array=dataset)
    # print(f"{mask=}")
    lbpara.compute_mask(source=blurr_source, block_size=view_size)
    updated_mask = blurr_source.get_mask()
    # as get_mask returns [0, 255] mask and mask produces [0, 1] we need to account for that
    # it is important that > 0 is Valid data and needs to be equal
    updated_mask = np.divide(updated_mask, 255)
    # print(f"UNIQUE VALUES: \n mask: {np.unique(mask)}\n updated_mask: {np.unique(updated_mask)}")
    np.testing.assert_array_equal(mask, updated_mask)
    assert not np.array_equal(initial_mask, updated_mask)


@ALL_MAPS
def test_prepare_selector_parallel(datafiles, create_blurred_tif):
    """`parallel.prepare_selector` is equivalent to `inference.prepare_selector`
    """
    block_size = (500, 500)
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio._coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))
    blurred_source = lbio_.Source(path=create_blurred_tif)
    # set the mask
    lbpara.compute_mask(source=blurred_source,
                        block_size=block_size,
                        nodata=0,
                        logic='all',
                        )
    # create the inputs
    response = lbio_.Band(source=lbio_.Source(path=ndvi_map))
    predictors = blurred_source.get_bands()
    # Each band should use the dataset mask:
    for pred_band in predictors:
        pred_band.set_mask_reader(use='source')

    # without the extra masking band, both should lead to the same result
    selector_wo = lbinf.prepare_selector(
        response,
        *predictors,)
    selector_parallel_wo = lbpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
    )
    np.testing.assert_equal(selector_wo, selector_parallel_wo)
    # Create extra masking band
    resp_profile = response.source.import_profile()
    tmp_map = str(datafiles / 'extra_mask_band.tif')
    tmp_source = lbio_.Source(path=tmp_map)
    tmp_profile = resp_profile.copy()
    tmp_profile['nodata'] = 0
    tmp_profile['dtype'] = np.uint8 
    tmp_source.profile = tmp_profile
    tmp_source.init_source(overwrite=True)
    extra_masking_band = lbio_.Band(source=tmp_source, bidx=1)
    # write out data as (mask all)
    extra_mask_data = np.full(shape=response.shape, fill_value=0, dtype=np.uint8)
    extra_masking_band.set_data(data=extra_mask_data)
    selector = lbinf.prepare_selector(
        response,
        *predictors,
        extra_masking_band=extra_masking_band)
    selector_parallel = lbpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
        extra_masking_band=extra_masking_band,
    )
    np.testing.assert_equal(selector, selector_parallel)
    # extra mask none and check that it has no influence
    extra_mask_data = np.full(shape=response.shape, fill_value=255, dtype=np.uint8)
    extra_masking_band.set_data(data=extra_mask_data)
    selector = lbinf.prepare_selector(
        response,
        *predictors,
        extra_masking_band=extra_masking_band,
        )
    selector_para = lbpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
        extra_masking_band=extra_masking_band,
        )
    np.testing.assert_equal(selector,selector_para)


@ALL_MAPS
def test_selector_computation(datafiles, create_blurred_tif):
    """Compare the selector generation in parallel to the full one
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio._coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))

    lct_source = lbio_.Source(path=landcover_map)
    ndvi_source = lbio_.Source(path=ndvi_map)
    blurred_source = lbio_.Source(path=create_blurred_tif)
    # set the mask
    lbpara.compute_mask(source=blurred_source, block_size=(1000, 1000),
                        nodata=0, logic='all')
    # create the inputs
    response = lbio_.Band(source=lbio_.Source(path=ndvi_map))
    predictors = blurred_source.get_bands()
    # Each band should use the dataset mask:
    for pred_band in predictors:
        pred_band.set_mask_reader(use='source')
    #predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), False),)

    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)

    selector_full = lbinf.prepare_selector(response,
                                           *predictors)
    selector_para = lbpara.prepare_selector(response, *predictors, block_size=(1000,1000))
    np.testing.assert_equal(selector_full, selector_para)
