import os
import pytest

import numpy as np
from rasterio.transform import Affine

FIXTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'data'
))

ALL_MAPS = pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_CLC_2012_reclass8.tif'),
    os.path.join(FIXTURE_DIR, 'testing', 'ndvi',
                 'Switzerland_NDVI_binning_2015.tif'),
)


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
