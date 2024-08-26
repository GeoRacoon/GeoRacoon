from functools import partial

from time import sleep

import numpy as np
import multiprocessing as mproc
import rasterio as rio

from rasterio.plot import show as rioshow

from landiv_blur import helper as lbhelp
from landiv_blur import io as lbio
from landiv_blur import io_ as lbio_
from landiv_blur import processing as lbproc
from landiv_blur import prepare as lbprep
from landiv_blur import inference as lbinf
from landiv_blur.filters import gaussian as lbf_gauss

from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lbf_gauss

from .conftest import ALL_MAPS, get_file

from matplotlib import pyplot as plt


@ALL_MAPS
def test_blur_recombination(datafiles):
    """Assert recombined blur of a band is identical to processing entire map
    """
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    output_dtype = np.uint8  # data type to use for the blurred arrays
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
                                                output_dtype=output_dtype
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
            as_int=True,
            output_file=blur_output_file,
            output_dtype=output_dtype
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(source=ch_map_tif,
                           view=view,
                           inner_view=inner_view,
                           categories=[category],
                           img_filter=img_filter,
                           output_dtype=output_dtype,)
            block_params.append(bparams)
        manager = mproc.Manager()
        blur_q = manager.Queue()
        # get number of cpu's
        nbr_cpus = mproc.cpu_count() - 1
        pool = mproc.Pool(nbr_cpus)
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
def test_entropy_recombination(datafiles):
    """Assert recombined entropy map is identical to processing the entire map
    """
    blur_full = str(datafiles / 'blur_full.tif')
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    output_dtype = np.uint8  # data type to use for the entropy array
    blur_as_int = True
    entropy_as_ubyte = True
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
    profile['dtype'] = np.uint8
    # get the categories
    categories = lbproc.get_categories(ch_data)
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    entropy_data = lbproc.get_entropy(ch_data, categories=categories,
                                      normed=normed,
                                      img_filter=img_filter,
                                      output_dtype=output_dtype)
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
        as_ubyte=True,
        output_file=entropy_output_file,
        count=len(categories)
    )
    # now the parameter for the per block blur tasks
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=(width, height))
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=ch_map_tif,
                       categories=categories,
                       view=view,
                       inner_view=inner_view,
                       img_filter=img_filter,
                       entropy_as_ubyte=entropy_as_ubyte,
                       normed=normed,
                       blur_as_int=blur_as_int, )
        block_params.append(bparams)
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    # get number of cpu's
    nbr_cpus = mproc.cpu_count() - 1
    # print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
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
    np.testing.assert_array_equal(
        entropy_data,
        entropy_recomb_data,
        f'The recombined entropy map is different!'
    )


@ALL_MAPS
def test_parallel_transposed_prod(datafiles):
    """Calculate the transposed product of a predictor matrix
    """
    verbose = True
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    lct_source = lbio_.Source(path=landcover_map)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # scale it down to 100x100m (from 30x30)
    lbio.coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)
    # create the predicotrs 
    response = ndvi_map
    # ###
    # compute blurred layers
    blur_out = str(datafiles / 'blur_out.tif')
    diameter = 5000  # this is in meter
    scale = 100  # meter per pixel
    truncate = 3
    _diameter = diameter / scale
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    blurred_tif = lbpara.extract_categories(
        source=lct_source,
        categories=[1,2,3,4,5],
        output_file=blur_out,
        img_filter=lbf_gauss.gaussian,
        filter_params=filter_params,
        blur_as_int=True,
        block_size=(500, 500),
        compress = True
    )
    blurr_source = lbio_.Source(path=blurred_tif)
    # compute the mask
    view_size = (500, 400)
    lbpara.compute_mask(source=blurr_source, block_size=view_size, logic='all')
    # ###
    predictors = blurr_source.get_bands()
    # use the dataset mask
    for pred in predictors:
        pred.set_mask_reader(use='source')
    # first compute the full matrix and calculate XT X
    X, _ = lbinf.prepare_predictors(response,
                                    *predictors,
                                    include_intercept=False,
                                    verbose=verbose,
                                    )
    transprod_full = X.T @ X
    # now compute it in parallel
    # get the size of the response
    with rio.open(response, 'r') as src:
        src_width = src.width
        src_height = src.height

    # get the aggregated selector
    selector = lbinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)

    # create a list of views and put it into runner_params
    size = (src_width, src_height)
    view_size = (500, 400)
    border = (0, 0)
    #  _ is for the inner_views which we do not need
    views, _ = lbprep.create_views(view_size=view_size,
                                   border=border,
                                   size=size)
    part_params = []
    for view in views:
        pparams = dict(predictors=predictors,
                       view=view,
                       selector=selector)
        part_params.append(pparams)
    # create the arguments for the aggregation script
    # start the processes 
    manager = mproc.Manager()
    output_q = manager.Queue()
    nbr_cpus = mproc.cpu_count() - 1
    # print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
    # start the aggregation step
    matrix_aggregator = pool.apply_async(
        lbpara.combine_matrices,
        (output_q,)
    )
    all_jobs = []
    for pparams in part_params:
        all_jobs.append(pool.apply_async(
            lbpara.partial_transposed_product,
            (pparams, output_q)
        ))
    # now lets wait for all of these jobs to finish
    job_timers = []
    for job in all_jobs:
        # await for the jobs to return (i.e. complete) by calling .get
        # get the duration from the timer object that is returned by .get()
        job_timers.append(job.get())
    # send the final kill job to the queue
    output_q.put(dict(signal='kill'))
    # wait for the recombination job to terminate
    recombined_tpX, _ = matrix_aggregator.get()
    # print(f"\n{transprod_full=}\n{recombined_tpX=}\n")
    np.testing.assert_allclose(transprod_full, recombined_tpX, rtol=1e-06)
    # finally in condensed form
    # get the aggregated selector (again)
    selector = lbinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)
    recombtpX = lbpara.get_XT_X(response,
                                *predictors,
                                selector=selector,
                                include_intercept=False,
                                verbose=verbose,
                                view_size=view_size, )
    np.testing.assert_allclose(transprod_full, recombtpX, rtol=1e-06)


@ALL_MAPS
def test_parallel_optimal_weights(datafiles, create_blurred_tif):
    """Calculate the transposed product of a predictor matrix
    """
    as_dtype = np.float64
    include_intercept = True
    verbose = True
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio.coregister_raster(_ndvi_map, landcover_map, output=ndvi_map)
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)
    # create the predicotrs 
    response = ndvi_map

    blurred_source = lbio_.Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    # choose the write mask
    for pred in predictors:
        pred.set_mask_reader(use='source')
    selector = lbinf.prepare_selector(response,
                                      *predictors)
    view_size = (500, 400)
    tpX = lbpara.get_XT_X(response,
                          *predictors,
                          selector=selector,
                          include_intercept=include_intercept,
                          verbose=verbose,
                          view_size=view_size,
                          )
    Y = np.linalg.inv(tpX)
    # print(f"{tpX=}\n{Y=}")
    # print("#####\n#####\n#####")
    betas_dict = lbpara.get_optimal_betas(*predictors,
                                     Y=Y,
                                     response=response,
                                     selector=selector,
                                     include_intercept=include_intercept,
                                     verbose=verbose,
                                     as_dtype=as_dtype,
                                     view_size=view_size,
                                     )
    # compute the betas by loading the entire map
    X, y = lbinf.prepare_predictors(response,
                                    *predictors,
                                    include_intercept=include_intercept,
                                    verbose=verbose,
                                    )
    # round both to the 6th digit
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    betas = np.round(list(betas_dict.values()), 6)
    # print(f"{b=}\n{betas=}")
    np.testing.assert_array_equal(betas, b)
    # test ouput length for correct key, value pairs
    n_predictors = len(predictors)
    n_betas = len(betas_dict.values())
    if include_intercept:
        n_predictors += 1
    np.testing.assert_equal(n_betas, n_predictors,
                            err_msg=f"Number of beta {n_betas=} not equal to prdictors {n_predictors=}")


@ALL_MAPS
def test_entropy_2_step(datafiles):
    """Assert that the 2 step approach (blur->entropy) yields identical results
    """
    blur_partial = str(datafiles / 'entropy_onego.tif')
    blur_out = str(datafiles / 'blur_out.tif')
    entropy_out = str(datafiles / 'entropy_twostep.tif')
    diameter = 5000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    output_dtype = np.uint8  # data type to use for the entropy array
    blur_as_int = True
    entropy_as_ubyte = True
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
    profile['dtype'] = np.uint8
    # get the categories
    categories = lbproc.get_categories(ch_data)

    img_filter = lbf_gauss.gaussian
    # get number of cpu's
    nbr_cpus = mproc.cpu_count() - 1
    # print(f"using {nbr_cpus=}")

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
        as_ubyte=True,
        output_file=entropy_output_file,
        count=len(categories)
    )
    # now the parameter for the per block blur tasks
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=(width, height))
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=ch_map_tif,
                       categories=categories,
                       view=view,
                       inner_view=inner_view,
                       img_filter=img_filter,
                       filter_params=filter_params,
                       entropy_as_ubyte=entropy_as_ubyte,
                       normed=normed,
                       blur_as_int=blur_as_int, )
        block_params.append(bparams)
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    pool = mproc.Pool(nbr_cpus)
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
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    blurred_tif = lbpara.extract_categories(
        source=source,
        categories=categories,
        output_file=blur_out,
        img_filter=img_filter,
        filter_params=filter_params,
        blur_as_int=blur_as_int,
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
        categories=categories, entropy_as_ubyte=entropy_as_ubyte,
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
    print(f"{np.unique(entropy_data)=}")
    print(f"{np.unique(entropy_data_2step)=}")
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
def test_reduced_mask(datafiles):
    """Compute a mask from multiple bands in one go and then in parallel
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    source = lbio_.Source(path=ch_map_tif)
    blur_out = str(datafiles / 'blur_out.tif')
    # create the blurred bands
    img_filter = lbf_gauss.gaussian
    blur_as_int = True
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
        blur_as_int=blur_as_int,
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
def test_selector_computation(datafiles, create_blurred_tif):
    """Compare the selector generation in parallel to the full one
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio.coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))

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
    print(f"{np.unique(selector_full, return_counts=True)=}")
    print(f"{np.unique(selector_para, return_counts=True)=}")
    np.testing.assert_equal(selector_full, selector_para)
