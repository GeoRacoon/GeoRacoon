import os
import pytest
import glob

import numpy as np
import rasterio as rio
from skimage.filters import gaussian

from riogrande.helper import get_or_set_context
from riogrande.io import Source
from riogrande.parallel import (
    compute_mask,)

FIXTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../',
    'data'
))
lct_map = os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_CLC_2012_reclass8.tif')
lct_float_map = os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_area_frac_grid_1km_CGLS_2015.tif')
ndvi_map = os.path.join(FIXTURE_DIR, 'testing', 'ndvi',
                 'Switzerland_NDVI_binning_2015.tif')

ALL_MAPS = pytest.mark.datafiles(lct_map, lct_float_map, ndvi_map)


def get_file(pattern: str, datafiles):
    matching_files = list(glob.glob(os.path.join(str(datafiles), pattern)))
    if len(matching_files) != 1:
        raise ValueError(
            f"Found multiple files matching this {pattern=}:\n{matching_files}"
        )
    return matching_files[0]


@pytest.fixture(scope="session")  # use session since all run on same OS
def set_mpc_strategy():
    return get_or_set_context(method='fork')

@pytest.fixture(scope="function")  # is function scope since datafiles is too
def create_blurred_tif(datafiles):
    """Create blurred single land-cover type layers in uint8 format.
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif",
                             datafiles=datafiles)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    print(f"Using\n- landcover map: {landcover_map}\n- ndvi map {ndvi_map}")

    # Blur parameters (equivalent to get_blur_params(diameter=50, truncate=3))
    diameter = 5000 / 100  # 50 pixels
    truncate = 3
    sigma = 0.5 * diameter / truncate  # ≈ 8.333

    categories = [1, 3, 4, 5, 6]
    blur_out = str(datafiles / 'blur_out_uint_compress.tif')

    with rio.open(landcover_map) as src:
        lct_data = src.read(1)
        out_profile = src.profile.copy()

    out_profile.update(count=len(categories), dtype='uint8', compress='lzw')

    with rio.open(blur_out, 'w', **out_profile) as dst:
        for band_idx, cat in enumerate(categories, start=1):
            binary = np.where(lct_data == cat, 1.0, 0.0)
            blurred = gaussian(binary, sigma=sigma, truncate=truncate,
                               preserve_range=False)
            # Rescale float [0, 1] → uint8 [0, 255]
            scaled = np.clip(blurred * 255, 0, 255).astype(np.uint8)
            dst.write(scaled, band_idx)

    blurr_source = Source(path=blur_out)
    compute_mask(source=blurr_source, block_size=(500, 400), logic='all')
    return blur_out
