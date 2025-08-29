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

# TODO: we need to get rid of this config for testing --> we dont want to be dependent on convster for the linfit package
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