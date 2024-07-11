from functools import partial

from time import sleep

import numpy as np
import multiprocessing as mproc
import rasterio as rio

from rasterio.plot import show as rioshow

from landiv_blur import helper as lbhelp
from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur import prepare as lbprep
from landiv_blur import inference as lbinf
from landiv_blur.filters import gaussian as lbf_gauss

from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lbf_gauss

from .config import ALL_MAPS

from matplotlib import pyplot as plt


@ALL_MAPS
def test_blur_recombination(datafiles):
    """Assert recombined blur of a layer is identical to processing entire map
    """
    blur_full = str(datafiles / 'blur_full.tif')
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000  # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter / scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                                  truncate=truncate)
    border = (100, 100)
    # load the data
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_map = lbio.load_map(ch_map_tif, indexes=1)
    ch_data = ch_map['data']
    profile = ch_map['orig_profile']
    width = profile['width']
    height = profile['height']
    # we will save each layer separately
    profile['count'] = 1
    profile['dtype'] = rio.uint8
    # get the layers
    layers = lbproc.get_lct(ch_data)
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    # we perform the test for each layer
    print("LAYERS", layers)
    for layer in layers:
        # Index needs to be fixed as code does not write layer to equivalent index
        # This was changed to allow for (1) seleciton of non-sequential lists, (2) lc maps with values [0, 255]
        index = 1
        # perform the blur in a single run
        blurred_data = lbproc.get_layer_data(ch_data,
                                             layer=layer,
                                             img_filter=img_filter,
                                             output_dtype=np.uint8
                                             )
        # use multiprocessing and blur block by block
        # first set the parameters for the recombintion task
        blur_output_file = lbhelp.output_filename(
            base_name=blur_partial,
            out_type=f"blur_lct_{layer}",
            blur_params=blur_params
        )
        blur_output_params = dict(
            profile=profile,
            as_int=True,
            output_file=blur_output_file
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(source=ch_map_tif,
                           layers=[layer],
                           view=view,
                           inner_view=inner_view,
                           img_filter=img_filter,
                           # entropy_as_ubyte=True,
                           blur_as_int=True, )
            block_params.append(bparams)
        manager = mproc.Manager()
        blur_q = manager.Queue()
        # get number of cpu's
        nbr_cpus = mproc.cpu_count() - 1
        pool = mproc.Pool(nbr_cpus)
        # start the blurred layer writer task
        blur_combiner = pool.apply_async(
            lbpara.combine_blurred_land_cover_types,
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
        print(f"job took {duration} seconds")

        # now we can read out the tif with the blurred layer and compare
        blurred_layer_map = lbio.load_map(blur_output_file, indexes=index)
        blurred_layer_data = blurred_layer_map['data']

        # plt.imshow(blurred_data)
        # plt.savefig(f'{datafiles}/{layer}_single.png')
        # plt.imshow(blurred_layer_data)
        # plt.savefig(f'{datafiles}/{layer}_recombined.png')
        # plt.imshow(blurred_layer_data - blurred_data)
        # plt.savefig(f'{datafiles}/{layer}_diff.png')
        np.testing.assert_array_equal(
            blurred_data,
            blurred_layer_data,
            f'For {layer=} the recombined blurred map is different!'
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
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                                  truncate=truncate)
    border = (50, 50)
    # load the data
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_map = lbio.load_map(ch_map_tif, indexes=1)
    ch_data = ch_map['data']
    profile = ch_map['orig_profile']
    width = profile['width']
    height = profile['height']
    # we will save each layer separately
    profile['count'] = 1
    profile['dtype'] = np.uint8
    # get the layers
    layers = lbproc.get_lct(ch_data)
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    entropy_data = lbproc.get_entropy(ch_data, layers, normed=True,
                                      img_filter=img_filter,
                                      output_dtype=np.uint8)
    # use multiprocessing and blur block by block
    # first set the parameters for the recombintion task
    entropy_output_file = lbhelp.output_filename(
        base_name=blur_partial,
        out_type=f"entropy_lct",
        blur_params=blur_params
    )
    entropy_output_params = dict(
        profile=profile,
        as_ubyte=True,
        output_file=entropy_output_file,
        count=len(layers)
    )
    # now the parameter for the per block blur tasks
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=(width, height))
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=ch_map_tif,
                       layers=layers,
                       view=view,
                       inner_view=inner_view,
                       img_filter=img_filter,
                       entropy_as_ubyte=True,
                       blur_as_int=True, )
        block_params.append(bparams)
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    # get number of cpu's
    nbr_cpus = mproc.cpu_count() - 1
    print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
    # start the blurred layer writer task
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
    print(f"job took {duration} seconds")

    # now we can read out the tif with the blurred layer anc compare
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
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    # scale it down to 100x100m (from 30x30)
    lbio.coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)
    # create the predicotrs 
    response = ndvi_map
    predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), True),)
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
        src_widht = src.width
        src_height = src.height

    # get the aggregated selector
    selector = lbinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)

    # create a list of views and put it into runner_params
    size = (src_widht, src_height)
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
    print(f"using {nbr_cpus=}")
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
    print(f"\n{transprod_full=}\n{recombined_tpX=}\n")
    np.testing.assert_array_equal(transprod_full, recombined_tpX)
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
    np.testing.assert_array_equal(transprod_full, recombtpX)


@ALL_MAPS
def test_parallel_optimal_weights(datafiles):
    """Calculate the transposed product of a predictor matrix
    """
    as_dtype = np.float64
    include_intercept = False
    verbose = True
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    # scale it down to 100x100m (from 30x30)
    lbio.coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)
    # create the predicotrs 
    response = ndvi_map
    predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), True),)
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
    print(f"{tpX=}\n{Y=}")
    print("#####\n#####\n#####")
    betas = lbpara.get_optimal_betas(*predictors,
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
    # round both the the 6th digit
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    betas = np.round(betas, 6)
    print(f"{b=}\n{betas=}")
    np.testing.assert_array_equal(betas, b)
