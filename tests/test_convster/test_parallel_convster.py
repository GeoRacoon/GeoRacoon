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
