import numpy as np
import rasterio as rio
from numpy.random import Generator, PCG64
# from memory_profiler import profile as mem_profile

from pydataset import data as pydata

from landiv_blur import io as lbio
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
                             (landcover_map, 1, (2, 3, 4)),
                             include_intercept=True,)
    print(test_data)


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
                                    (landcover_map, 1, (1, 2, 3, 4, 5, 6, 7)),
                                    include_intercept=True,)
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_array_equal(b, b_approx)
    # now check when ingnoring the intercept in both cases
    X, y = lbinf.prepare_predictors(ndvi_map,
                                    (landcover_map, 1, (1, 2, 3, 4, 5, 6, 7)),
                                    include_intercept=False,)
    b = np.round(lbinf.get_optimal_weights(X, y), 6)
    reg = lbinf.get_approx_weights(X, y, fit_intercept=False)
    b_approx = np.round(reg.coef_, 6)
    print(f"\n{b=}\n{b_approx=}\n")
    np.testing.assert_array_equal(b, b_approx)
