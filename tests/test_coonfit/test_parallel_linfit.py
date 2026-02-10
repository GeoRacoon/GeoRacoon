import numpy as np
import multiprocessing as mproc
import random
import rasterio as rio

from skimage.filters import gaussian

from riogrande.io import Source, Band, coregister_raster

from riogrande import helper as rghelp
from riogrande import parallel as rgpara
from riogrande import prepare as rgprep

from convster import parallel as cspara
from convster.filters import get_blur_params

from coonfit import inference as lfinf
from coonfit import parallel as lfpara
from coonfit.parallel_helpers import (
    _combine_matrices,
    _partial_transposed_product
)

from .conftest import (
    ALL_MAPS,
    get_file,
    set_mpc_strategy,
)


@ALL_MAPS
def test_parallel_transposed_prod(datafiles, set_mpc_strategy):
    """Calculate the transposed product of a predictor matrix
    """
    verbose = True
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    lct_source = Source(path=landcover_map)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # scale it down to 100x100m (from 30x30)
    coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))
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
    blur_params = get_blur_params(diameter=_diameter, truncate=truncate)
    filter_params = blur_params.copy()
    _ = filter_params.pop('diameter')
    print(mproc.get_start_method(allow_none=True))
    blurred_tif = cspara.extract_categories(
        source=lct_source,
        categories=[1, 2, 3, 4, 5],
        output_file=blur_out,
        img_filter=gaussian,
        filter_params=filter_params,
        output_dtype=np.uint8,
        block_size=(500, 500),
        compress=True
    )
    blurr_source = Source(path=blurred_tif)
    # compute the mask
    view_size = (500, 400)
    rgpara.compute_mask(source=blurr_source, block_size=view_size, logic='all')
    # ###
    predictors = blurr_source.get_bands()
    # use the dataset mask
    for pred in predictors:
        pred.set_mask_reader(use='source')
    # first compute the full matrix and calculate XT X
    X, _ = lfinf.prepare_predictors(response,
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
    selector = lfinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)

    # create a list of views and put it into runner_params
    size = (src_width, src_height)
    view_size = (500, 400)
    border = (0, 0)
    #  _ is for the inner_views which we do not need
    views, _ = rgprep.create_views(view_size=view_size,
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
    nbr_workers = rghelp.get_nbr_workers()
    # print(f"using {nbr_workers=}")
    pool = set_mpc_strategy.Pool(nbr_workers)
    # start the aggregation step
    matrix_aggregator = pool.apply_async(
        _combine_matrices,
        (output_q,)
    )
    all_jobs = []
    for pparams in part_params:
        all_jobs.append(pool.apply_async(
            _partial_transposed_product,
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
    selector = lfinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)
    recombtpX = lfpara.get_XT_X(response,
                                *predictors,
                                selector=selector,
                                include_intercept=False,
                                verbose=verbose,
                                view_size=view_size, )
    np.testing.assert_allclose(transprod_full, recombtpX, rtol=1e-06)


@ALL_MAPS
def test_model_output(datafiles, create_blurred_tif):
    """Test the parallelized model prediction calculation.
    """
    as_dtype = 'float32'
    blurred_source = Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    coregister_raster(ndvi_map, blurred_source.path, output=str(ndvi_map))
    resp_source = Source(path=ndvi_map)
    resp_profile = resp_source.import_profile()
    resp_profile['count'] = 1
    print('computing weights')
    optimal_weights = lfpara.compute_weights(response=ndvi_map,
                                             predictors=predictors,
                                             block_size=(500, 500),
                                             include_intercept=False,
                                             sanitize_predictors=True)
    print(f'{optimal_weights=}')
    print('done!')
    # perform the computation in parallel 
    model_output_file = str(datafiles / 'model_out.tif')
    verbose = True
    block_size = (500, 400)
    params = dict()
    print('Compute the model prediction')
    model_out = lfpara.compute_model(
        predictors=predictors,
        optimal_weights=optimal_weights,
        output_file=model_output_file,
        block_size=block_size,
        profile=resp_profile,
        verbose=verbose,
        **params)
    print('done!')
    # compute it "manually"
    model_data = np.full(shape=(resp_profile['height'],
                                resp_profile['width']),
                         fill_value=0.0,
                         dtype=as_dtype)
    for pred in predictors:
        model_data += (optimal_weights[pred] * pred.get_data()).astype(as_dtype)

    model_source = Source(model_out)
    model_band = model_source.get_band(bidx=1)
    # make sure we get the same
    print(f"{np.unique(model_band.get_data())=}")
    print(f"{np.unique(model_data)=}")
    np.testing.assert_allclose(model_band.get_data(), model_data)


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
    coregister_raster(_ndvi_map, landcover_map, output=ndvi_map)
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)
    # create the predicotrs 
    response = ndvi_map

    blurred_source = Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    # choose the write mask
    for pred in predictors:
        pred.set_mask_reader(use='source')
    selector = lfinf.prepare_selector(response,
                                      *predictors)
    view_size = (500, 400)
    tpX = lfpara.get_XT_X(response,
                          *predictors,
                          selector=selector,
                          include_intercept=include_intercept,
                          verbose=verbose,
                          view_size=view_size,
                          )
    Y = np.linalg.inv(tpX)
    # print(f"{tpX=}\n{Y=}")
    # print("#####\n#####\n#####")
    betas_dict = lfpara.get_optimal_betas(*predictors,
                                          Y=Y,
                                          response=response,
                                          selector=selector,
                                          include_intercept=include_intercept,
                                          verbose=verbose,
                                          as_dtype=as_dtype,
                                          view_size=view_size,
                                          )
    # compute the betas by loading the entire map
    X, y = lfinf.prepare_predictors(response,
                                    *predictors,
                                    include_intercept=include_intercept,
                                    verbose=verbose,
                                    )
    # round both to the 6th digit
    b = np.round(lfinf.get_optimal_weights(X, y), 6)
    betas = np.round(list(betas_dict.values()), 6)
    # print(f"{b=}\n{betas=}")
    np.testing.assert_allclose(betas, b, rtol=1e-04)
    # test ouput length for correct key, value pairs
    n_predictors = len(predictors)
    n_betas = len(betas_dict.values())
    if include_intercept:
        n_predictors += 1
    np.testing.assert_equal(n_betas, n_predictors,
                            err_msg=f"Number of beta {n_betas=} not equal to prdictors {n_predictors=}")


@ALL_MAPS
def test_get_XT_X_dependency(datafiles, create_blurred_tif):
    """Test wether rank deficiency is captured when layers would be linear dependent
    """
    blur_source = Source(path=create_blurred_tif)
    predictors = blur_source.get_bands()

    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    coregister_raster(ndvi_map, blur_source.path, output=str(ndvi_map))  # rescale to 100m

    # Generally it should be empty (as there is no linear dependency by nature)
    result = lfpara.get_XT_X_dependency(response=ndvi_map,
                                        predictors=predictors,
                                        block_size=(500, 500),
                                        include_intercept=False)
    assert result == dict()

    # Modify one band (to be linear dependent of others)
    pred_sample = random.sample(predictors, 2)
    ref_array = pred_sample[0].get_data()
    with rio.open(blur_source.path, mode='r+') as dst:
        dst.write(ref_array, indexes=pred_sample[1].get_bidx())

    result_issue = lfpara.get_XT_X_dependency(response=ndvi_map,
                                              predictors=predictors,
                                              block_size=(500, 500),
                                              include_intercept=False)
    print(f"{pred_sample=}, {result_issue=}")
    assert set(pred_sample) == set([k for k, v in result_issue.items()])


@ALL_MAPS
def test_compute_weights(datafiles, create_blurred_tif):
    """Test compute weights for different issues:
        1) normal
        2) with rank deficiency and all zero column
    """
    blur_source = Source(path=create_blurred_tif)
    predictors = blur_source.get_bands()

    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    coregister_raster(ndvi_map, blur_source.path, output=str(ndvi_map))  # rescale to 100m

    rgpara.compute_mask(blur_source, block_size=(500, 500))
    for p in predictors:
        p.set_mask_reader(use="source")

    # 1) Normal test
    beta_weights = lfpara.compute_weights(response=ndvi_map,
                                          predictors=predictors,
                                          block_size=(500, 500),
                                          include_intercept=False)
    assert set(predictors) == set([k for k in beta_weights.keys()])

    # 2)
    # 2.1) Linear dependency
    print('Create dependent bands')
    # Modify one band (to be linear dependent of other)
    pred_sample_dep = random.sample(predictors, 2)
    ref_array = pred_sample_dep[0].get_data()
    with rio.open(blur_source.path, mode='r+') as dst:
        for i in range(1, 2):
            dst.write(ref_array, indexes=pred_sample_dep[i].get_bidx())

    # 2.2) All zero band
    print('Create all-zero band')
    # This should get caught by sanitize predictors
    pred_sample_zero = random.sample([p for p in predictors if p not in pred_sample_dep], 1)
    zero_array = np.zeros(ref_array.shape)
    with rio.open(blur_source.path, mode='r+') as dst:
        dst.write(zero_array, indexes=pred_sample_zero[0].get_bidx())

    # 2.3) Run test
    # The All-Zero band should be caught by the sanitize
    beta_weights = lfpara.compute_weights(response=ndvi_map,
                                          predictors=predictors,
                                          block_size=(500, 500),
                                          include_intercept=False,
                                          limit_contribution=0,
                                          sanitize_predictors=True,
                                          return_linear_dependent_predictors=True)
    assert set(beta_weights) == set(pred_sample_dep)
    assert all(['Linear dependent column' == b for b in beta_weights.values()])


@ALL_MAPS
def test_calculate_rmse(datafiles, create_blurred_tif):
    """Test the parallelized RSME calculation.
      """
    block_size = (500, 500)
    blurred_source = Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    coregister_raster(ndvi_map, blurred_source.path, output=str(ndvi_map))
    resp_source = Source(path=ndvi_map)
    resp_profile = resp_source.import_profile()
    resp_profile['count'] = 1

    # Masks (important for latter model fitting)
    for p in predictors:
        p.set_mask_reader(use="source")
    rgpara.compute_mask(source=resp_source, nodata=np.nan, block_size=block_size)

    # Comupte selector (else the data does not overlap)
    print('Calculate Selector')
    selector = rgpara.prepare_selector(resp_source.get_band(bidx=1), *predictors,
                                       block_size=block_size)

    print('computing weights')
    optimal_weights = lfpara.compute_weights(response=ndvi_map,
                                             predictors=predictors,
                                             block_size=block_size,
                                             as_dtype=np.float64,
                                             include_intercept=False,
                                             sanitize_predictors=True)
    print(f'{optimal_weights=}')
    print('done!')
    # perform the computation in parallel
    model_output_file = str(datafiles / 'model_out.tif')
    verbose = False
    params = dict()
    print('Compute the model prediction')
    model_out = lfpara.compute_model(
        predictors=predictors,
        predictors_as_dtype=np.float64,
        optimal_weights=optimal_weights,
        output_file=model_output_file,
        block_size=block_size,
        profile=resp_profile,
        verbose=verbose,
        **params)

    # make sure we get the same
    ndvi_band = resp_source.get_band(bidx=1)
    model_band = Source(path=model_out).get_band(bidx=1)
    ndvi_array = ndvi_band.get_data()
    model_array = model_band.get_data()

    # Selector
    ndvi_array[~selector] = np.nan
    model_array[~selector] = np.nan

    # -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- -
    # NOTE: Test results are very bad (weird) due to blurring border around CH
    # plot images for better understanding

    # RMSE
    residuals = np.subtract(ndvi_array, model_array)
    residuals_pw = np.power(residuals, 2)
    n = np.count_nonzero(~np.isnan(residuals_pw))
    ssr = np.nansum(residuals_pw)
    rmse_manual = np.sqrt((ssr / n))
    print(f"{rmse_manual=}")

    rmse = lfpara.calculate_rmse(response=ndvi_band,
                                 model=model_band,
                                 selector=selector,
                                 block_size=block_size, )
    print(f'{rmse=}')
    np.testing.assert_almost_equal(rmse, rmse_manual, decimal=6)

    # R2
    y_mean = np.nanmean(ndvi_array)
    diff_mean = np.subtract(ndvi_array, y_mean)
    diff_mean_pw = np.power(diff_mean, 2)
    sst = np.nansum(diff_mean_pw)
    r2_manual = 1 - (ssr / sst)
    print(f"{r2_manual=}")

    r2 = lfpara.calculate_r2(response=ndvi_band,
                             model=model_band,
                             selector=selector,
                             block_size=block_size, )
    print(f'{r2=}')
    np.testing.assert_almost_equal(r2, r2_manual, decimal=6)

    # Some Plotting to check whether this is accurate
    # fig, ax = plt.subplots(nrows=1, ncols=3)
    # ax[0].imshow(ndvi_array)
    # ax[1].imshow(model_array)
    # ax[2].imshow(selector)
    #
    # fig2, ax2 = plt.subplots(nrows=1, ncols=2)
    # scale_min = min(np.nanmin(residuals), np.nanmin(diff_mean))
    # scale_max = max(np.nanmax(residuals), np.nanmax(diff_mean))
    # ax2[0].imshow(residuals, vmin=scale_min, vmax=scale_max)
    # ax2[1].imshow(diff_mean, vmin=scale_min, vmax=scale_max)
    # plt.show()


@ALL_MAPS
def test_prepare_selector_parallel(datafiles, create_blurred_tif):
    """`parallel.prepare_selector` is equivalent to `inference.prepare_selector`
    """
    block_size = (500, 500)
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))
    blurred_source = Source(path=create_blurred_tif)
    # set the mask
    rgpara.compute_mask(source=blurred_source,
                        block_size=block_size,
                        nodata=0,
                        logic='all',
                        )
    # create the inputs
    response = Band(source=Source(path=ndvi_map))
    predictors = blurred_source.get_bands()
    # Each band should use the dataset mask:
    for pred_band in predictors:
        pred_band.set_mask_reader(use='source')

    # without the extra masking band, both should lead to the same result
    selector_wo = lfinf.prepare_selector(
        response,
        *predictors, )
    selector_parallel_wo = rgpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
    )
    np.testing.assert_equal(selector_wo, selector_parallel_wo)
    # Create extra masking band
    resp_profile = response.source.import_profile()
    tmp_map = str(datafiles / 'extra_mask_band.tif')
    tmp_source = Source(path=tmp_map)
    tmp_profile = resp_profile.copy()
    tmp_profile['nodata'] = 0
    tmp_profile['dtype'] = np.uint8
    tmp_source.profile = tmp_profile
    tmp_source.init_source(overwrite=True)
    extra_masking_band = Band(source=tmp_source, bidx=1)
    # write out data as (mask all)
    extra_mask_data = np.full(shape=response.shape, fill_value=0, dtype=np.uint8)
    extra_masking_band.set_data(data=extra_mask_data)
    selector = lfinf.prepare_selector(
        response,
        *predictors,
        extra_masking_band=extra_masking_band)
    selector_parallel = rgpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
        extra_masking_band=extra_masking_band,
    )
    np.testing.assert_equal(selector, selector_parallel)
    # extra mask none and check that it has no influence
    extra_mask_data = np.full(shape=response.shape, fill_value=255, dtype=np.uint8)
    extra_masking_band.set_data(data=extra_mask_data)
    selector = lfinf.prepare_selector(
        response,
        *predictors,
        extra_masking_band=extra_masking_band,
    )
    selector_para = rgpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size,
        extra_masking_band=extra_masking_band,
    )
    np.testing.assert_equal(selector, selector_para)


@ALL_MAPS
def test_selector_computation(datafiles, create_blurred_tif):
    """Compare the selector generation in parallel to the full one
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))

    lct_source = Source(path=landcover_map)
    ndvi_source = Source(path=ndvi_map)
    blurred_source = Source(path=create_blurred_tif)
    # set the mask
    rgpara.compute_mask(source=blurred_source, block_size=(1000, 1000),
                        nodata=0, logic='all')
    # create the inputs
    response = Band(source=Source(path=ndvi_map))
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

    selector_full = lfinf.prepare_selector(response,
                                           *predictors)
    selector_para = rgpara.prepare_selector(response, *predictors, block_size=(1000, 1000))
    np.testing.assert_equal(selector_full, selector_para)
