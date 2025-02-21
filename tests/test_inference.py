import pytest

import numpy as np
import rasterio as rio
from numpy.random import Generator, PCG64
# from memory_profiler import profile as mem_profile
from skimage.filters import gaussian

from pydataset import data as pydata

from landiv_blur import exceptions as lbexcept
from landiv_blur import helper as lbhelp
from landiv_blur import io as lbio
from landiv_blur import io_ as lbio_
from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur import parallel as lbpara
from landiv_blur import inference as lbinf
from landiv_blur.filters import gaussian as lbf_gauss

from .conftest import ALL_MAPS, get_file


@ALL_MAPS
def test_preparation(datafiles, create_blurred_tif):
    """Test the preparation of predictors based on a response matrix
    """
    test_data = list(datafiles.iterdir())
    _landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # print(f"{test_data=}")

    landcover_map = str(datafiles / 'lct_coreged.tif')
    lbio.coregister_raster(_landcover_map, ndvi_map, output=landcover_map)
    # work with strings first
    lbinf.prepare_predictors(ndvi_map,
                             landcover_map,
                             include_intercept=True,)
    # now same with Band objects
    lct_source = lbio_.Source(path=landcover_map)
    lct_band = lbio_.Band(source=lct_source, bidx=1)
    ndvi_source = lbio_.Source(path=ndvi_map)
    ndvi_band = lbio_.Band(source=ndvi_source, bidx=1)
    lbinf.prepare_predictors(ndvi_band,
                             lct_band,
                             include_intercept=True,)


def test_optimal_weights_test_data():
    iris = pydata('iris')
    # there are 4 columns of values, followed by the `species` as 5th column
    # first we model the 4th column with all 4 columns, thus the weight for
    # column 4 should be 1 and the others close to 0
    X = iris[iris.columns[:4]].values
    y = iris[iris.columns[3]].values
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    np.testing.assert_array_equal(b, np.array([0, 0, 0, 1]))
    # now we add some white noise to column 4 (as predictor):
    X_noisy = np.copy(X)
    rng = Generator(PCG64(seed=33))
    X_noisy[:, 3] += rng.standard_normal(X_noisy.shape[0]) * 0.02
    # this should affect the quality of the fit (so we round to first digit:
    b_noisy = np.round(lbinf.get_optimal_weights(X_noisy, y), 1)
    np.testing.assert_array_equal(b_noisy, np.array([0, 0, 0, 1]))


def test_weights_computations_test_data():
    """Compare the analytical solution with the approximation"""
    iris = pydata('iris')
    # there are 4 columns of values, followed by the `species` as 5th column
    # first we model the 4th column with all 4 columns, thus the weight for
    # column 4 should be 1 and the others close to 0
    X = iris[iris.columns[:4]].values
    y = iris[iris.columns[3]].values
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    # now use the linear regression estimation
    reg = lbinf.get_approx_weights(X, y)
    b_approx = np.round(reg.coef_)
    # print(f"\n{b=}\n{b_approx=}")
    # print(reg.intercept_)
    np.testing.assert_array_equal(b, b_approx)
    # now in a second round we add some noise
    rng = Generator(PCG64(seed=33))
    X_noisy = np.copy(X)
    X_noisy[:, 3] += rng.standard_normal(X_noisy.shape[0]) * 0.02
    # this should affect the quality of the fit (so we round to first digit:
    b_noisy = np.round(lbinf.get_optimal_weights(X_noisy, y), 1)
    reg_noisy = lbinf.get_approx_weights(X_noisy, y)
    b_noisy_approx = np.round(reg_noisy.coef_)
    # print(f"{b_noisy=}\n{b_noisy_approx=}")
    np.testing.assert_array_equal(b_noisy, b_noisy_approx)


# @mem_profile
@ALL_MAPS
def test_optimal_weights_example_data(datafiles, create_blurred_tif):
    """Test the preparation of predictors based on a response matrix
    """
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

    blurred_source = lbio_.Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    # choose the write mask
    for pred in predictors:
        pred.set_mask_reader(use='source')

    X, y = lbinf.prepare_predictors(ndvi_map,
                                    *predictors,
                                    include_intercept=False,
                                    verbose=True,
                                    )
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    # print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_allclose(b, b_approx, rtol=1e-05)
    # now check when ingnoring the intercept in both cases
    X, y = lbinf.prepare_predictors(ndvi_map,
                                    *predictors,
                                    include_intercept=False,)
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    # print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_allclose(b, b_approx, rtol=1e-05)


# @mem_profile
@ALL_MAPS
def test_transposed_prod_example_data(datafiles, create_blurred_tif):
    """Calculate transposed product from the predictor matrix
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
    lbpara.compute_mask(source=blurred_source, block_size=(500, 500), nodata=0, logic='all')
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

    X, y = lbinf.prepare_predictors(response,
                                    *predictors,
                                    include_intercept=False,
                                    verbose=True,
                                    )
    tpX = X.T @ X
    # calculate it per predictor columns
    selector = lbinf.prepare_selector(response,
                                      *predictors)
    transprodX = lbinf.transposed_product(predictors,
                                          view=None,
                                          selector=selector,
                                          as_dtype="float64")

    # print(f"\n{tpX=}\n{transprodX=}\n")
    np.testing.assert_allclose(tpX, transprodX, rtol=1e-05)

@ALL_MAPS
def test_extra_masking_band(datafiles, create_blurred_tif):
    """Assert that the extra masking band is included correctly
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)

    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio.coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))
    blurred_source = lbio_.Source(path=create_blurred_tif)
    # set the mask
    lbpara.compute_mask(source=blurred_source, block_size=(500, 500), nodata=0, logic='all')
    # create the inputs
    response = lbio_.Band(source=lbio_.Source(path=ndvi_map))
    predictors = blurred_source.get_bands()
    # Each band should use the dataset mask:
    for pred_band in predictors:
        pred_band.set_mask_reader(use='source')

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
    assert True not in np.unique(selector).tolist()
    # extra mask none and check that it has no influence
    extra_mask_data = np.full(shape=response.shape, fill_value=255, dtype=np.uint8)
    extra_masking_band.set_data(data=extra_mask_data)
    selector_wo = lbinf.prepare_selector(
        response,
        *predictors,)
    selector = lbinf.prepare_selector(
        response,
        *predictors,
        extra_masking_band=extra_masking_band)
    np.testing.assert_equal(selector, selector_wo)


@ALL_MAPS
def test_transposed_prod_blurred_example_data(datafiles, create_blurred_tif):
    """Calculate transposed product form a predictor matrix with blurred input
    """
    # get the response data
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # get (compute) the blurred bands
    blurred_source = lbio_.Source(path=create_blurred_tif)
    # parameter setting
    view = None  # use the full maps
    include_intercept = True
    verbose = True
    as_dtype = "float64"
    sigma = 10
    # scale it down to 100x100m (from 30x30)
    ndvi_map = str(datafiles / 'lct_coreged.tif')
    lbio.coregister_raster(_ndvi_map, blurred_source.path, output=ndvi_map)
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)

    # create the inputs
    response = ndvi_map
    predictors = blurred_source.get_bands()
    # choose the write mask
    for pred in predictors:
        pred.set_mask_reader(use='source')

    selector = lbinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)
    riow = lbhelp.view_to_window(view)
    X = lbinf.init_X(predictors,
                     selector=selector,
                     window=riow,
                     include_intercept=include_intercept,
                     as_dtype=as_dtype
                     )

    lbinf.populate_X(X=X,
                     predictors=predictors,
                     window=riow,
                     selector=selector,
                     include_intercept=include_intercept,
                     as_dtype=as_dtype)
    # create transprodX matrix:
    # this is equivalent
    transprodX = X.T @ X

    # print(f"\n{transprodX=}\n")
    # TODO: there is no actual test here
    # now the same but with parallel processing

    # np.testing.assert_array_equal(tpX, transprodX)


@ALL_MAPS
def test_optimal_beta(datafiles, create_blurred_tif):
    """Calculate the optimal beta values analytically
    """
    as_dtype = 'float32'  # we use this since the ndvi map uses it
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    _ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # scale it down to 100x100m (from 30x30)

    ndvi_map = str(datafiles / 'ndvi_coreged.tif')
    lbio.coregister_raster(_ndvi_map, landcover_map, output=str(ndvi_map))
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)

    # create the inputs
    response = ndvi_map
    blurred_source = lbio_.Source(path=create_blurred_tif)
    predictors = blurred_source.get_bands()
    # choose the write mask
    for pred in predictors:
        pred.set_mask_reader(use='source')

    # test for intercept set and not set
    intercept_switch = [True, False]
    for isetup in intercept_switch:
        X, y = lbinf.prepare_predictors(response,
                                        *predictors,
                                        include_intercept=isetup,
                                        verbose=True,
                                        )
        tpX = X.T @ X
        # calculte the inverse
        Y = np.linalg.inv(tpX)
        # calculate optimal betas
        betas = Y @ X.T @ y
        # calculate it per predictor columns
        selector = lbinf.prepare_selector(response,
                                          *predictors)
        tpX_col = lbinf.transposed_product(predictors,
                                           view=None,
                                           include_intercept=isetup,
                                           selector=selector,
                                           as_dtype=as_dtype)
        Y_col = np.linalg.inv(tpX_col)
        betas_col = lbinf.get_optimal_weights_source(Y=Y,
                                                     response=response,
                                                     predictors=predictors,
                                                     view=None,
                                                     include_intercept=isetup,
                                                     selector=selector,
                                                     as_dtype=as_dtype)
        # print(f"{Y=}, {Y_col=}")
        np.testing.assert_allclose(Y, Y_col, rtol=1e-04)
        np.testing.assert_allclose(betas, list(betas_col.values()), rtol=1e-04)

        # test ouput length for correct key, value pairs
        n_predictors = len(predictors)
        n_betas = len(betas_col.values())
        if isetup:
            n_predictors += 1
        np.testing.assert_equal(n_betas, n_predictors,
                                err_msg=f"Number of beta {n_betas=} not equal to predictors {n_predictors=}")
