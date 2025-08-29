import os
import pytest
import glob

import numpy as np

from rasterio.transform import Affine

from landiv_blur.prepare import get_blur_params
from landiv_blur.parallel import (
    extract_categories,
    compute_mask,
)
from landiv_blur.helper import get_or_set_context
from landiv_blur.io_ import Source, Band
from landiv_blur.filters.gaussian import gaussian

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

@pytest.fixture(scope="function")  # is function scope since datafiles is too
def create_blurred_tif(datafiles):
    """Create blurred single land-cover type layers in uint8 format
    """
    as_dtype = 'uint8'
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    lct_source = Source(path=landcover_map)
    # ###
    # compute blurred layers
    blur_out = str(datafiles / 'blur_out_uint.tif')
    diameter = 5000  # this is in meter
    scale = 100  # meter per pixel
    truncate = 3
    _diameter = diameter / scale
    blur_params = get_blur_params(diameter=_diameter, truncate=truncate)
    filter_params = blur_params.copy()
    filter_params['preserve_range'] = False
    _ = filter_params.pop('diameter')
    blurred_tif = extract_categories(
        source=lct_source,
        categories=[1,3,4,5,6],
        output_file=blur_out,
        img_filter=gaussian,
        filter_params=filter_params,
        filter_output_range=(0,1),
        output_params=dict(
            as_dtype=as_dtype,
        ),
        block_size=(500, 500),
        compress = True,
    )
    blurr_source = Source(path=blurred_tif)
    # compute the mask
    view_size = (500, 400)
    compute_mask(source=blurr_source, block_size=view_size, logic='all')
    # ###
    return blurred_tif

@pytest.fixture(scope="function")  # is function scope since datafiles is too
def create_blurred_tif_float(datafiles):
    """Create blurred single land-cover type layers as float rescaled to [0, 1]
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    lct_source = Source(path=landcover_map)
    # ###
    # compute blurred layers
    blur_out = str(datafiles / 'blur_out_float.tif')
    diameter = 5000  # this is in meter
    scale = 100  # meter per pixel
    truncate = 3
    _diameter = diameter / scale
    blur_params = get_blur_params(diameter=_diameter, truncate=truncate)
    filter_params = blur_params.copy()
    filter_params['preserve_range'] = False
    _ = filter_params.pop('diameter')
    blurred_tif = extract_categories(
        source=lct_source,
        categories=[1,3,4,5,6],
        output_file=blur_out,
        img_filter=gaussian,
        filter_params=filter_params,
        filter_output_range=(0,1),
        output_params=dict(
            as_dtype='float64',
            output_range=(0, 1),
        ),
        block_size=(500, 500),
        compress = True,
    )
    blurr_source = Source(path=blurred_tif)
    # compute the mask
    view_size = (500, 400)
    compute_mask(source=blurr_source, block_size=view_size, logic='all')
    # ###
    print(f"{blurr_source.import_profile()=}")
    return blurred_tif
