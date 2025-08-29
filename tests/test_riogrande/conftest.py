import os
import pytest
import glob

import numpy as np
from rasterio.transform import Affine

from riogrande.helper import get_or_set_context

FIXTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'data'
))
lct_map = os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_CLC_2012_reclass8.tif')
lct_float_map = os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_area_frac_grid_1km_CGLS_2015.tif')
landiv_map = os.path.join(FIXTURE_DIR, 'testing', 'ndvi',
                 'Switzerland_NDVI_binning_2015.tif')

ALL_MAPS = pytest.mark.datafiles(lct_map, lct_float_map, landiv_map)


def get_file(pattern:str, datafiles):
    matching_files = list(glob.glob(os.path.join(str(datafiles), pattern)))
    if len(matching_files) != 1:
        raise ValueError(f"Found multiple files matching this {pattern=}:\n{matching_files}")
    return matching_files[0]


@pytest.fixture(scope="session")  # use session since all run on same OS
def set_mpc_strategy():
    return get_or_set_context(method='fork')


def get_example_data(bands=1, size=(240, 180)):
    """Generate some example data similar to the example in the rasterio docs

    The example:
    https://rasterio.readthedocs.io/en/latest/quickstart.html#creating-data

    Returns
    -------
    (bands_data, profile)
    """
    width = size[0]
    height = size[1]
    x = np.linspace(-4.0, 4.0, width)
    y = np.linspace(-3.0, 3.0, height)
    X, Y = np.meshgrid(x, y)
    # create a transform
    res = (x[-1] - x[0]) / 240.0
    transform = Affine.translation(x[0] - res /
                                   2, y[0] - res /
                                   2) * Affine.scale(res, res)
    # create some data for the bands
    Zs = list()
    for band in range(bands):
        Z = 10 * np.exp(-2 * np.log(2) * ((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 1 ** 2)
        Zs.append(Z)
    profile = {
        'count': len(Zs),
        'width': Zs[0].shape[1],
        'height': Zs[0].shape[0],
        'dtype': Zs[0].dtype,
        'driver': 'GTiff',
        'crs': "+proj=latlong",
        'transform': transform,
    }
    return Zs, profile
