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
from landiv_blur import processing as lbproc
from landiv_blur import inference as lbinf

from .config import ALL_MAPS


@ALL_MAPS
def test_preparation(datafiles):
    """Test the preparation of predictors based on a response matrix
    """
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    lbio.coregister_raster(landcover_map, ndvi_map, output=str(landcover_map))
    lbinf.prepare_predictors(ndvi_map,
                             (landcover_map, 1, (2, 3, 4), False),
                             include_intercept=True,)
    print(f"{test_data=}")


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
    print(f"\n{b=}\n{b_approx=}")
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
    print(f"{b_noisy=}\n{b_noisy_approx=}")
    np.testing.assert_array_equal(b_noisy, b_noisy_approx)


# @mem_profile
@ALL_MAPS
def test_invalid_predictors_selection(datafiles):
    """Test the preparation of predictors based on a response matrix
    """
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

    # inexistant categorie
    with pytest.raises(lbexcept.InferenceError):
        X, y = lbinf.prepare_predictors(ndvi_map,
                                        (landcover_map, 1, (22, 2, 3, 4, 5), False),
                                        include_intercept=True,
                                        verbose=True,
                                        )
    # using bot exclusive classes and an intercept fit
    with pytest.raises(lbexcept.InferenceError):
        X, y = lbinf.prepare_predictors(ndvi_map,
                                        (landcover_map, 1, (1, 2, 3, 4, 5), True),
                                        include_intercept=True,
                                        verbose=True,
                                        )


# @mem_profile
@ALL_MAPS
def test_optimal_weights_example_data(datafiles):
    """Test the preparation of predictors based on a response matrix
    """
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

    X, y = lbinf.prepare_predictors(ndvi_map,
                                    (landcover_map, 1, (1, 2, 3, 4, 5), True),
                                    include_intercept=False,
                                    verbose=True,
                                    )
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_array_equal(b, b_approx)
    # now check when ingnoring the intercept in both cases
    X, y = lbinf.prepare_predictors(ndvi_map,
                                    (landcover_map, 1, (1, 2, 3, 4, 5, ), True),
                                    include_intercept=False,)
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_array_equal(b, b_approx)


# @mem_profile
@ALL_MAPS
def test_transposed_prod_example_data(datafiles):
    """Calculate transposed product from the predictor matrix
    """
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    # create the inputs
    response = ndvi_map
    predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), False),)

    # scale it down to 100x100m (from 30x30)
    lbio.coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))

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
                                          as_dtype=np.float64)

    print(f"\n{tpX=}\n{transprodX=}\n")
    np.testing.assert_array_equal(tpX, transprodX)


@ALL_MAPS
def test_transposed_prod_blurred_example_data(datafiles):
    """Calculate transposed product form a predictor matrix with blurred input
    """
    view = None  # use the full maps
    include_intercept = True
    verbose = True
    as_dtype = np.float64
    test_data = list(datafiles.iterdir())
    sigma = 10
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    # scale it down to 100x100m (from 30x30)
    lbio.coregister_raster(ndvi_map, landcover_map, output=str(ndvi_map))
    # create a mask for ndvi_map masking the nan's
    with rio.open(ndvi_map, 'r+') as src:
        data = src.read(indexes=1)
        mask = np.where(np.isnan(data), 0, 255)
        src.write_mask(mask)

    # create the inputs
    response = ndvi_map
    predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), False),)

    selector = lbinf.prepare_selector(response,
                                      *predictors,
                                      verbose=verbose)
    riow = lbhelp.view_to_window(view)
    X = lbinf.init_X(*predictors,
                     selector=selector,
                     window=riow,
                     include_intercept=include_intercept)

    # read out the data
    # NOTE: This is taken from lbinf.extract_predictor_data but with added filter
    pred_datas = []
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        if len(predictor) >= 3:
            extract_values = predictor[2]
        else:
            extract_values = None
        with rio.open(pred_file_path, 'r') as psrc:
            pred_data = psrc.read(indexes=band, window=riow)
        if extract_values is not None:
            for value in extract_values:
                pred_datas.append(
                    lbproc.apply_filter(
                        lbproc.select_layer(pred_data, layer=value,
                                            as_dtype=as_dtype,
                                            limits=(1.0, 0.0)),
                        img_filter=gaussian,
                        sigma=sigma,
                    )
                )
        else:
            pred_datas.append(pred_data.astype(as_dtype))

    lbinf.populate_X(X=X,
                     predictor_datas=pred_datas,
                     window=riow,
                     selector=selector,
                     include_intercept=include_intercept)
    # create transprodX matrix:
    # this is equivalent
    transprodX = X.T @ X

    print(f"\n{transprodX=}\n")
    # TODO: there is no actual test here
    # now the same but with parallel processing

    # np.testing.assert_array_equal(tpX, transprodX)


@ALL_MAPS
def test_optimal_beta(datafiles):
    """Calculate the optimal beta values analytically
    """
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

    # create the inputs
    response = ndvi_map
    predictors = ((landcover_map, 1, (1, 2, 3, 4, 5), False),)


    X, y = lbinf.prepare_predictors(response,
                                    *predictors,
                                    include_intercept=False,
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
                                       selector=selector,
                                       as_dtype=np.float64)
    Y_col = np.linalg.inv(tpX_col)
    betas_col = lbinf.get_optimal_weights_source(Y=Y,
                                                 response=response,
                                                 predictors=predictors,
                                                 view=None,
                                                 selector=selector,
                                                 as_dtype=np.float64)
    print(f"{Y=}, {Y_col=}")
    np.testing.assert_array_equal(Y, Y_col)
    print(f"{betas=}, {betas_col=}")
    np.testing.assert_array_equal(betas, betas_col)
