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
def test_blur_recombination(datafiles, set_mpc_strategy):
    """Assert recombined blur of a band is identical to processing entire map
    """
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    output_dtype = "uint8"  # data type to use for the blurred arrays
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                                  truncate=truncate)
    border = (100, 100)
    # load the data
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_map = lbio.load_map(ch_map_tif, indexes=1)
    ch_data = ch_map['data']
    profile = ch_map['orig_profile']
    width = profile['width']
    height = profile['height']
    # we will save each category separately
    profile['count'] = 1
    profile['dtype'] = rio.uint8
    # get the categories 
    categories = lbproc.get_categories(ch_data)
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    # we perform the test for each category 
    # print("CATEGORIES", categories)
    for category in categories:
        # Index needs to be fixed as code does not write the band to equivalent
        # index.
        #This was changed to allow for (1) seleciton of non-sequential lists, (2) lc maps with values [0, 255]
        index = 1
        # perform the blur in a single run
        blurred_data = lbproc.get_category_data(ch_data,
                                                category=category,
                                                img_filter=img_filter,
                                                as_dtype=output_dtype
                                                )
        # use multiprocessing and blur block by block
        # first set the parameters for the recombintion task
        blur_output_file = lbhelp.output_filename(
            base_name=blur_partial,
            out_type=f"blur_lct_{category}",
            blur_params=blur_params
        )
        blur_output_params = dict(
            profile=profile,
            output_file=blur_output_file,
            as_dtype=output_dtype
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(
                source=ch_map_tif,
                view=view,
                inner_view=inner_view,
                categories=[category],
                img_filter=img_filter,
                output_dtype=output_dtype,
            )
            block_params.append(bparams)
        manager = mproc.Manager()
        blur_q = manager.Queue()
        # get number of workers
        nbr_workers = lbhelp.get_nbr_workers()
        pool = set_mpc_strategy.Pool(nbr_workers)
        # start the blurred category writer task
        blur_combiner = pool.apply_async(
            lbpara.combine_blurred_categories,
            (blur_output_params, blur_q,)
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(
                lbpara.runner_call,
                (blur_q,
                 lbproc.view_blurred,
                 bparams)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        blur_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        duration = blur_combiner.get().get_duration()
        # free up the resources
        pool.close()
        pool.join()
        # print(f"job took {duration} seconds")

        # check if tags were set correctly
        with rio.open(blur_output_file) as src:
            tags = lbio.get_tags(src, bidx=index)
            bidx = lbio.get_bidx(src, category=category)
            np.testing.assert_equal(tags['category'], category)
            np.testing.assert_equal(bidx, index)

        # now we can read out the tif with the blurred category and compare
        blurred_cat_map = lbio.load_map(blur_output_file, indexes=index)
        blurred_cat_data = blurred_cat_map['data']

        # plt.imshow(blurred_data)
        # plt.savefig(f'{datafiles}/{category}_single.png')
        # plt.imshow(blurred_cat_data)
        # plt.savefig(f'{datafiles}/{category}_recombined.png')
        # plt.imshow(blurred_cat_data - blurred_data)
        # plt.savefig(f'{datafiles}/{category}_diff.png')
        np.testing.assert_array_equal(
            blurred_data,
            blurred_cat_data,
            f'For {category=} the recombined blurred map is different!'
        )

@ALL_MAPS
def test_entropy_recombination(datafiles, set_mpc_strategy):
    """Assert recombined entropy map is identical to processing the entire map
    """
    # TODO: the use of get_entropy should be avoided anyways.
    #  We should get rid of this test after getting rid of the respective functions
    dtype_tests = {"uint8": (0, 255),
                   "float32": None}
    for test_dtype, test_filter_output_range in dtype_tests.items():
        blur_full = str(datafiles / f'blur_full_{test_dtype}.tif')
        blur_partial = str(datafiles / f'blur_partial_{test_dtype}.tif')
        diameter = 1000  # this is in meter
        scale = 100  # meter per pixel
        _diameter = diameter / scale
        truncate = 3  # property of the gaussian filter
        view_size = (500, 400)
        filter_output_range = test_filter_output_range  # full range of data the filter can produce
        blur_output_dtype = test_dtype  # blurred maps will be saved in this format
        output_dtype = test_dtype
        normed = True
        blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
        min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                                      truncate=truncate)
        border = (50, 50)
        print(f"{min_border=}, {border=}")
        # load the data
        ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
        ch_map = lbio.load_map(ch_map_tif, indexes=1)
        ch_data = ch_map['data']
        profile = ch_map['orig_profile']
        width = profile['width']
        height = profile['height']
        # we will save each category separately
        profile['count'] = 1
        profile['dtype'] = np.dtype(output_dtype)
        # get the categories
        categories = lbproc.get_categories(ch_data)
        max_entropy_categories = len(categories)
        # we partially evaluate the guassian filter to make sure it gets
        # identical parameter everywhere
        # we do not need to pass the diameter to the filter function
        _ = blur_params.pop('diameter')
        img_filter = partial(lbf_gauss.gaussian, **blur_params)
        filter_params = blur_params.copy()
        filter_params.update(dict(preserve_range=True))
        entropy_data = lbproc.get_entropy(
            data=ch_data,
            categories=categories,
            max_entropy_categories=max_entropy_categories,
            normed=normed,
            img_filter=img_filter,
            filter_params=filter_params,
            blur_output_dtype=blur_output_dtype,
            filter_output_range=filter_output_range,
            as_dtype=output_dtype
        )
        # use multiprocessing and blur block by block
        # first set the parameters for the recombintion task
        entropy_output_file = lbhelp.output_filename(
            base_name=blur_partial,
            out_type=f"entropy_lct",
            blur_params=blur_params
        )
        entropy_output_params = dict(
            profile=profile,
            output_dtype=output_dtype,
            output_file=entropy_output_file,
            count=len(categories)
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(
                source=ch_map_tif,
                categories=categories,
                max_entropy_categories=max_entropy_categories,
                normed=normed,
                img_filter=img_filter,
                filter_params=filter_params,
                blur_output_dtype=blur_output_dtype,
                filter_output_range=filter_output_range,
                output_dtype=output_dtype,
                view=view,
                inner_view=inner_view,
            )
            block_params.append(bparams)
        manager = mproc.Manager()
        entropy_q = manager.Queue()
        # get number of workers
        nbr_workers = lbhelp.get_nbr_workers()
        # print(f"using {nbr_workers=}")
        pool = set_mpc_strategy.Pool(nbr_workers)
        # start the blurred category writer task
        blur_combiner = pool.apply_async(
            lbpara.combine_entropy_blocks,
            (entropy_output_params, entropy_q,)
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(
                lbpara.runner_call,
                (entropy_q,
                 lbproc.get_entropy_view,
                 bparams)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        entropy_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        duration = blur_combiner.get().get_duration()
        # free up the resources
        pool.close()
        pool.join()
        # print(f"job took {duration} seconds")

        # check if tags were set correctly
        with rio.open(entropy_output_file) as src:
            tags = lbio.get_tags(src, bidx=1)
            bidx = lbio.get_bidx(src, category="entropy")
            np.testing.assert_equal(tags['category'], "entropy")
            np.testing.assert_equal(bidx, 1)

        # now we can read out the tif with the blurred category and compare
        entropy_recomb_map = lbio.load_map(entropy_output_file, indexes=1)
        entropy_recomb_data = entropy_recomb_map['data']

        # plt.imshow(entropy_data)
        # plt.savefig(f'{datafiles}/entropy_single.png')
        # plt.imshow(entropy_recomb_data)
        # plt.savefig(f'{datafiles}/entropy_recombined.png')
        # plt.imshow(entropy_recomb_data - entropy_data)
        # plt.savefig(f'{datafiles}/entropy_diff.png')
        print(f"{entropy_data.dtype=}")
        print(f"{entropy_recomb_data.dtype=}")
        np.testing.assert_array_equal(
            entropy_data,
            entropy_recomb_data,
            f'The recombined entropy map is different!'
        )


@ALL_MAPS
def test_entropy_2_step(datafiles):
    """Assert that the 2 step approach (blur->entropy) yields identical results
    """
    dtype_tests = {"uint8": (0, 255),
                   "float32": None}
    for test_dtype, test_filter_out_range in dtype_tests.items():
        blur_partial = str(datafiles / f'{test_dtype}_entropy_onego.tif')
        blur_out = str(datafiles / f'{test_dtype}_blur_out.tif')
        entropy_out = str(datafiles / f'{test_dtype}_entropy_twostep.tif')
        diameter = 5000  # this is in meter
        scale = 100  # meter per pixel
        _diameter = diameter / scale
        truncate = 3  # property of the gaussian filter
        view_size = (500, 400)
        output_dtype = test_dtype  # data type to use for the entropy array
        blur_output_dtype = test_dtype
        normed = True
        blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
        min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                                      truncate=truncate)
        border = (50, 50)
        # load the data
        ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
        ch_map = lbio.load_map(ch_map_tif, indexes=1)
        ch_data = ch_map['data']
        profile = ch_map['orig_profile']
        width = profile['width']
        height = profile['height']
        # we will save each category separately
        profile['count'] = 1
        profile['dtype'] = test_dtype
        # get the categories
        categories = lbproc.get_categories(ch_data)
        # Filter params for both
        img_filter = lbf_gauss.gaussian
        filter_params = blur_params.copy()
        filter_params.update(dict(preserve_range=True))
        _ = filter_params.pop('diameter')

        # ###
        # use single step approach
        # ###
        entropy_output_file = lbhelp.output_filename(
            base_name=blur_partial,
            out_type=f"entropy_lct",
            blur_params=blur_params.copy()
        )
        entropy_output_params = dict(
            profile=profile,
            output_dtype=output_dtype,
            output_file=entropy_output_file,
            count=len(categories)
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(
                    source=ch_map_tif,
                    categories=categories,
                    view=view,
                    inner_view=inner_view,
                    img_filter=img_filter,
                    filter_params=filter_params,
                    filter_output_range=test_filter_out_range,
                    output_dtype=output_dtype,
                    normed=normed,
                    blur_output_dtype=blur_output_dtype,)
            block_params.append(bparams)
        manager = mproc.Manager()
        entropy_q = manager.Queue()
        nbr_workers = lbhelp.get_nbr_workers()
        pool = mproc.Pool(nbr_workers)
        # start the blurred category writer task
        blur_combiner = pool.apply_async(
            lbpara.combine_entropy_blocks,
            (entropy_output_params, entropy_q,)
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(
                lbpara.runner_call,
                (entropy_q,
                 lbproc.get_entropy_view,
                 bparams)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        entropy_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        duration = blur_combiner.get().get_duration()
        # free up the resources
        pool.close()
        pool.join()
        # print(f"job took {duration} seconds")

        ###
        # Now calculate first the map with blurred layers and then the entropy
        ###
        source = lbio_.Source(path=ch_map_tif)
        blurred_tif = lbpara.extract_categories(
            source=source,
            categories=categories,
            output_file=blur_out,
            img_filter=img_filter,
            filter_params=filter_params,
            filter_output_range=test_filter_out_range,
            output_dtype=blur_output_dtype,
            block_size=view_size,
            compress = True
        )
        # TODO: this should change in parallel.extract_categories
        # get the somewhat weird output filename
        # check if tags were set correctly for the blurred layers
        with rio.open(blurred_tif) as src:
            tags = lbio.get_tags(src, bidx=1)
            bidx = lbio.get_bidx(src, category=categories[0])
            np.testing.assert_equal(tags['category'], categories[0])
            np.testing.assert_equal(bidx, 1)

        blurred_source = lbio_.Source(path=blurred_tif)
        for bidx in blurred_source.band_indexes:
            b = lbio_.Band(source=blurred_source, bidx=bidx)
            with rio.open(blurred_source.path, 'r') as src:
                data = src.read(indexes=bidx)
                np.testing.assert_equal(data, b.get_data())

        entropy_tif = lbpara.compute_entropy(
            source=blurred_source,
            output_file=entropy_out,
            block_size=view_size,
            blur_params=blur_params.copy(),
            categories=categories,
            output_dtype=output_dtype,
            normed=normed,
        )

        # now we can read out the tif with the blurred category and compare
        entropy_map = lbio.load_map(entropy_output_file, indexes=1)
        entropy_data = entropy_map['data']

        # for the 2-step approach
        entropy_source = lbio_.Source(path=entropy_tif)
        # get the entropy band as a object
        eband = entropy_source.get_band(category='entropy')
        entropy_data_2step = eband.get_data()

        # plt.imshow(entropy_data)
        # plt.savefig(f'{datafiles}/entropy_single.png')
        # plt.imshow(entropy_recomb_data)
        # plt.savefig(f'{datafiles}/entropy_recombined.png')
        # plt.imshow(entropy_recomb_data - entropy_data)
        # plt.savefig(f'{datafiles}/entropy_diff.png')
        # import matplotlib.pyplot as plt
        # plt.imshow(entropy_data)
        # plt.savefig('/home/.../1step.pdf')
        # plt.imshow(entropy_data_2step)
        # plt.savefig('/home/.../2step.pdf')
        # plt.imshow(entropy_data_2step-entropy_data)
        # plt.savefig('/home/.../test.pdf')
        np.testing.assert_array_equal(
            entropy_data,
            entropy_data_2step,
            'The recombined entropy map in the 1 step and the 2 step process ' \
            'are different!'
        )


@ALL_MAPS
def test_interaction_parallel_computation(datafiles, create_blurred_tif):
    """Compare whether parallel and single compute_interaction give the same result
    TODO: only tested for n=2 (test more)
    """
    dtype_tests = {"uint8": (0, 255),
                   "float32": None}
    for test_dtype, _ in dtype_tests.items():
        out_source = lbio_.Source(path=create_blurred_tif)
        categories = [b.tags['category'] for b in out_source.get_bands()]
        # Pairs
        all_possible_pairs = [list(x) for x in itertools.combinations(categories, r=2)]
        test_pair = random.choice(all_possible_pairs)
        test_pair = [1, 3]

        # Interaction (parallel)
        para_interaction_tif = lbpara.compute_interaction(source=out_source,
                                                          output_file=str(datafiles / f'{test_dtype}_interact_out.tif'),
                                                          block_size=(500, 500),
                                                          categories=test_pair,
                                                          blur_params=dict(
                                                              sigma=(0.5 * 5000 / 3) / 100, truncate=3), # irrelevant f test
                                                          output_dtype=test_dtype,                                                          standardize=True,
                                                          normed=True,
                                                          verbose=False)
        int_band = lbio_.Band(source=lbio_.Source(path=para_interaction_tif), bidx=1)
        para_interaction_data = int_band.get_data()

        # Interaction (single process
        band_list = [out_source.get_band(category=c) for c in test_pair]
        data_array = [b.get_data() for b in band_list]
        interaction_data = lbproc.compute_interaction(data_arrays=data_array,
                                                      input_dtype=np.uint8,
                                                      normed=True,
                                                      standardize=True,
                                                      output_dtype=test_dtype)
        np.testing.assert_array_equal(para_interaction_data, interaction_data)


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
    blur_params = rgprep.get_blur_params(diameter=_diameter, truncate=truncate)
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
        b_nope = rgio_.Band(source=rgio_.Source(path=bands_tif),
                            bidx=cat)
        b_twostep = rgio_.Band(source=rgio_.Source(path=blurred_para),
                               bidx=cat)
        b_single = rgio_.Band(source=rgio_.Source(path=blurred_tif),
                              bidx=cat)
        b_nope.import_tags()
        b_twostep.import_tags()
        b_single.import_tags()
        np.testing.assert_equal(b_twostep.get_data(), b_single.get_data())


@ALL_MAPS
def test_prepare_selector_parallel(datafiles, create_blurred_tif):
    """`parallel.prepare_selector` is equivalent to `inference.prepare_selector`
    """
    block_size = (500, 500)
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    rgio._coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))
    blurred_source = rgio_.Source(path=create_blurred_tif)
    # set the mask
    lbpara.compute_mask(source=blurred_source,
                        block_size=block_size,
                        nodata=0,
                        logic='all',
                        )
    # create the inputs
    response = rgio_.Band(source=rgio_.Source(path=ndvi_map))
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
    tmp_source = rgio_.Source(path=tmp_map)
    tmp_profile = resp_profile.copy()
    tmp_profile['nodata'] = 0
    tmp_profile['dtype'] = np.uint8 
    tmp_source.profile = tmp_profile
    tmp_source.init_source(overwrite=True)
    extra_masking_band = rgio_.Band(source=tmp_source, bidx=1)
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
    rgio._coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))

    lct_source = rgio_.Source(path=landcover_map)
    ndvi_source = rgio_.Source(path=ndvi_map)
    blurred_source = rgio_.Source(path=create_blurred_tif)
    # set the mask
    lbpara.compute_mask(source=blurred_source, block_size=(1000, 1000),
                        nodata=0, logic='all')
    # create the inputs
    response = rgio_.Band(source=rgio_.Source(path=ndvi_map))
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

@ALL_MAPS
def test_extract_categories(datafiles):
    """Make sure the extract categories works as expected
    """
    verbose = True
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    lct_source = rgio_.Source(path=landcover_map)
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
    category_source = rgio_.Source(category_tif)
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
        tmp_source = rgio_.Source(path=tmp_map)
        tmp_profile = source_profile.copy()
        tmp_profile['nodata'] = nodata
        tmp_profile['dtype'] = to_dtype
        tmp_source.profile = tmp_profile
        tmp_source.init_source(overwrite=True)
        # sanity check for Source.init_source resp. Source.open
        with rio.open(tmp_source.path, 'r') as src:
            _profile = src.profile.copy()
        np.testing.assert_equal(_profile['nodata'], nodata)

        tmp_band = rgio_.Band(source=tmp_source, bidx=1)
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
        out_source = rgio_.Source(path=blurred_tif)
        out_profile = out_source.import_profile()
        print(f"{out_profile=}")
        np.testing.assert_equal(out_profile['nodata'], nodata)
        # We need to map GDAL to numpy datatypes
        assert out_profile['dtype'] == to_dtype
        #np.testing.assert_equal(rasterio_to_numpy_dtype(out_profile['dtype']), to_dtype)

@ALL_MAPS
def test_reduced_mask(datafiles):
    """Compute a mask from multiple bands in one go and then in parallel
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    source = rgio_.Source(path=ch_map_tif)
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
    blur_params = rgprep.get_blur_params(diameter=_diameter, truncate=truncate)
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
    blurr_source = rgio_.Source(path=blurred_tif)
    initial_mask = blurr_source.get_mask()
    # get the mask loading the entire dataset
    with blurr_source.data_reader(mode='r') as read:
        dataset = read()
    # print(f"{dataset.shape=}")
    mask = rghelp.reduced_mask(array=dataset)
    # print(f"{mask=}")
    lbpara.compute_mask(source=blurr_source, block_size=view_size)
    updated_mask = blurr_source.get_mask()
    # as get_mask returns [0, 255] mask and mask produces [0, 1] we need to account for that
    # it is important that > 0 is Valid data and needs to be equal
    updated_mask = np.divide(updated_mask, 255)
    # print(f"UNIQUE VALUES: \n mask: {np.unique(mask)}\n updated_mask: {np.unique(updated_mask)}")
    np.testing.assert_array_equal(mask, updated_mask)
    assert not np.array_equal(initial_mask, updated_mask)
