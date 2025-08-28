from __future__ import annotations

import os
import glob
import rasterio


from math import floor

from typing import Any

import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds
)

from shapely.geometry import box as shbox
import geopandas as gpd

from numpy.typing import NDArray

from ._exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
)
from ._helper import (
    check_crs_raster,
    outfile_suffix,
    serialize,
    deserialize,
    sanitize,
    match_all,
    view_to_window,
    get_scale_factor,
)
# is_needed
# needs_work (module should be moved, needs docs)
# is_tested (at least partially)
# usedin_both

# this is our namespace for tags
NS = 'LANDIV'

# TODO: General Idea - maybe we can merge some of these into io_.py class structure - so we avoid having both.
#  --> yet it is nice to have the function by themselves as well without direct need of class structures

