"""This module defines functions that might be helpful when working
with rasterio
"""
from __future__ import annotations

import os
import json
import warnings

import numpy as np
import rasterio as rio

from rasterio.windows import Window

from typing import Any, Union, Dict, List

from collections.abc import Collection

from numpy.typing import NDArray

from decimal import Decimal

import multiprocessing as mpc
from multiprocessing import context as _context_module
from typing import Optional









def get_scale_factor(source, target):
    # TODO: not_needed
    """Get scaling factors (width & height) to match target to source
    """
    # not_needed (used in clipping_and_masking.py example)
    # needs_work (better doc, check if it is not the inverse)
    # not_tested
    # Make sure both have the same projection and linear units
    check_crs(source, target)
    with rio.open(source) as src:
        source_res = src.res
    with rio.open(target) as trg:
        target_res = trg.res
    # calculate the sale factor along each dimension and return it
    return tuple(sres / tres for sres, tres in zip(source_res, target_res))


def nodata_mask_band(source, nodata=None):
    # TODO: not_needed (should be in class actually)
    """Update exiting raster with an added nodata mask band (alpha band)
    0=nodata, 255=valid_data
    Note: it is only possible to set one mask band for all value bands.
    Consequently, if a tif with band_count > 1 is given. Pixels will be masked
    which show the given nodata value in any of the provided bands.

    Parameters
    ----------
    source: str
      The path to the tif file you want to create alpha band for
    nodata: float or int (optional)
      The nodata value to use for the mask (e.g. np.nan or integer)
      if not provided - the nodata value from the source metadata is taken

    Returns
    -------
    None
    """
    # not_needed (used in clip_and_maks_SILA.py example)
    # needs_work (formatting)
    # not_tested
    with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rio.open(source, mode='r+') as src:

            if nodata is None:
                nodata = src.nodata
                if nodata is None:
                    raise ValueError(f"Neither nodata value provided nor inherent in raster metadata")
            if src.count > 1:
                print(f"WARNING: You are creating a mask for multiple bands (n={src.count} (bitwise And condition)")

            for ji, window in src.block_windows(1):
                msk = np.full((window.height, window.width), 255, dtype=np.uint8)
                for i in range(1, src.count + 1):
                    band = src.read(i, window=window)
                    if np.isnan(nodata):
                        msk_band = np.where(np.isnan(band), 0, 255).astype(np.uint8)
                    else:
                        msk_band = np.where(band == nodata, 0, 255).astype(np.uint8)
                    msk = np.bitwise_and(msk, msk_band)
                src.write_mask(msk, window=window)




def usable_pixels_info(all_pixels, data_pixels):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Prints the fraction of usable pixels
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    print(f"Of {all_pixels=} there are {data_pixels=}, i.e. "
          f"{round(100 * data_pixels/all_pixels, 2)}% are usable")


def usable_pixels_count(selector):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Count the number of usable pixels determined by the selector"""
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    vals, counts = np.unique(selector, return_counts=True)
    # vals: [True, False] or inv. in any case ok
    try:
        return int(counts[vals][0])
    except IndexError:
        return 0







