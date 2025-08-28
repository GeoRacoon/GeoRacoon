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
