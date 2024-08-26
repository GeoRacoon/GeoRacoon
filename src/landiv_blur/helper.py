"""This module defines functions that might be helpful when working
with rasterio
"""
from __future__ import annotations

import os
import json

import numpy as np
import rasterio as rio

from rasterio.windows import Window

from typing import Any
from numpy.typing import NDArray


def serialize(tags:dict[str,Any])->dict[str,str]:
    """Convert the values of a dict to into JSON
    """
    return {tag: json.dumps(obj=value) 
            for tag, value in tags.items()}


def deserialize(tags:dict[str,str])->dict[str,Any]:
    """Reads python objects from JSON-encoded values of a dict
    """
    return {tag: json.loads(s=value) 
            for tag, value in tags.items()}


def sanitize(tags:dict[str,Any])->Any:
    """Serializes then deserializes values of a dict
    """
    return deserialize(serialize(tags))


def match_all(targets:dict, tags:dict)->bool:
    """Check if all tags in targets are present in tags
    """
    match = True
    for t, v in targets.items():
        if not match:
            break  # stop if the last was no match
        if t in tags:  # if tag is present check for value match
            if tags[t] == v:
                match = True
            else:  # if a value is different it is no match
                match = False
        else:  # if a tag is absent it is no match
            match = False
    return match


def match_any(targets:dict, tags:dict)->bool:
    """Check if any tag in targets is present in tags
    """
    match = False
    for t, v in targets.items():
        if match:
            break  # stop if there was a match
        if t in tags:  # if tag is present check for value match
            if tags[t] == v:
                match = True
            else:  # if a value is different it is no match
                match = False
        else:  # if a tag is absent it is no match
            match = False
    return match


def view_to_window(view: None | tuple[int, int, int, int]):
    """Conerts a view into a rasterio Window

    Parameters
    ----------
    view:
      tuple (x, y, width, height) defining the view of the data array to update
    """
    if view is not None:
        window =  Window(view[0],
                         view[1],
                         view[2],
                         view[3])
    else:
        window = None
    return window


def check_crs_raster(source, reference, verbose=False):
    """Compare coordinate reference systems of two raster datasets"""
    with rio.open(source, mode='r') as src:
        src_crs = str(src.crs)
    with rio.open(reference, mode='r') as ref:
        ref_crs = str(ref.crs)

    if src_crs == ref_crs:
        if verbose:
            print(f"Coordinate systems are the same: {src_crs} --> {ref_crs}")
        return True
    else:
        print(f"CRS CHECK FAILING: {src_crs=} - {ref_crs=}")
        return False


def check_units(*sources):
    """Assert that all sources have the same units
    """
    units = []
    for source in sources:
        with rio.open(source) as src:
            units.append(src.profile['crs'].linear_units.lower())
            if len(set(units)) != 1:
                raise TypeError(f"{source=} has linear units {units[-1]}, "
                                "which is different from the other(s) "
                                f"({units[0]})")
    return units


def check_crs(*sources):
    """Assert that all the sources have the same projection (i.e. same crs)
    """
    crss = []
    for source in sources:
        with rio.open(source) as src:
            crss.append(str(src.profile['crs']))
            if len(set(crss)) != 1:
                raise TypeError(f"{source=} has crs {crss[-1]}, which is "
                                f"different from the other(s) ({crss[0]})")
    return crss


def check_resolution(*sources):
    """Assert that all the sources have the same resolution
    """
    ress = []
    for source in sources:
        with rio.open(source) as src:
            # NOTE: we round 8th digit after the comma here
            ress.append(tuple(map(lambda x: round(x, 8), src.res)))
            if len(set(ress)) != 1:
                raise TypeError(f"{source=} has resolution {ress[-1]}, which "
                                f"is different from the other(s) ({ress[0]})")
    return ress


def check_compatibility(*sources):
    """Assert that all the sources are compatible with each other.

    The checks include:

        - crs
        - units
        - resolution

    """
    units = check_units(*sources)
    crss = check_crs(*sources)
    ress = check_resolution(*sources)
    # print(f"{crss=}, {units=}, {ress=}")
    return crss, units, ress


def get_scale_factor(source, target):
    """Get scaling factors (width & height) to match target to source
    """
    # Make sure both have the same projection and linear units
    check_crs(source, target)
    with rio.open(source) as src:
        source_res = src.res
    with rio.open(target) as trg:
        target_res = trg.res
    # calculate the sale factor along each dimension and return it
    return tuple(sres / tres for sres, tres in zip(source_res, target_res))


def nodata_mask_band(source, nodata=None):
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


def outfile_suffix(filename, suffix):
    """Insert suffix into filename and hand back basename_suffix.extension"""
    base, ext = os.path.splitext(filename)
    return f"{base}_{suffix}{ext}"

def output_filename(base_name: str, out_type: str, blur_params: dict):
    """Construct the filename for the specific output type.

    Parameters
    ----------
    base_name: str
      The basic output name in the form <name>.tif
    out_type: str
      The type of output that will be saved.
      This should be either 'blur' or 'entropy' but any string is accepted
    blur_params: dict
      Output of `get_blur_params`, so 'sigma', 'truncate' and 'diameter'
      are expected keys.

    Returns
    ------
    str:
      The resulting filename of the form
      '<name>_<out_type>_sig_<{sigma}>_diam_<{diameter}>_trunc_<{truncate}>.tif'
    """
    _base_name, _ext = os.path.splitext(base_name)
    # sig = blur_params['sigma']
    # diam = blur_params['diameter']
    # trunc = blur_params['truncate']
    _blur_string = ""
    for name, value in blur_params.items():
        _blur_string += f"_{name}_{round(value)}"
    # _blur_string = f"sig_{sig}_diam_{diam}_trunc_{trunc}"
    return f"{_base_name}_{out_type}{_blur_string}{_ext}"


def usable_pixels_info(all_pixels, data_pixels):
    """Prints the fraction of usable pixels
    """
    print(f"Of {all_pixels=} there are {data_pixels=}, i.e. "
          f"{round(100 * data_pixels/all_pixels, 2)}% are usable")


def usable_pixels_count(selector):
    """Count the number of usable pixels determined by the selector"""
    vals, counts = np.unique(selector, return_counts=True)
    # vals: [True, False] or inv. in any case ok
    try:
        return int(counts[vals][0])
    except IndexError:
        return 0


def dtype_range(dtype):
    """Get the range of the specified dtype
    """
    # avoid issues of object not callable from rasterio
    if hasattr(dtype, 'type'):
        dtype = dtype.type
    try:
        _max = dtype(np.iinfo(dtype).max)
        _min = dtype(np.iinfo(dtype).min)
    except ValueError:
        try:
            _max = dtype(np.finfo(dtype).max)
            _min = dtype(np.finfo(dtype).min)
        except ValueError:
            raise ValueError(f"{dtype=} has no min-/maximal values.")
    return _max, _min


def convert_to_scaled(arr: NDArray,
                      as_dtype,
                      data_range: tuple=None) -> NDArray:
    """Converts a data array to desired as_type and
    scales the array by the maximum value in the dtype of the input array or provided data_range if provided

    Parameters
    ----------
    arr: input numpy NDArray
    as_dtype: desired data type to convert to (e.g. np.float)
    data_range: provide a range of intput array as tuple (e.g. (0, 1)
    """
    if data_range is None:
        _max, _min = dtype_range(arr.dtype)
    else:
        if len(data_range) != 2:
            raise ValueError(f"data_range must be of length 2 (e.g. (0,1), but is {len(data_range)}")
        _min, _max = data_range
    scale = _max - _min
    return np.divide(arr.astype(as_dtype), scale)


def aggregated_selector(masks:list[NDArray], logic:str='all')->NDArray:
    """Turns several rasterio masks into a boolen selector for a numpy array

    Rasterio masks are uint8 numpy arrays where every value > 0 is considered
    a valid cell

    Parameters
    ----------
    masks:
        Arbitrary number of numpy arrays resalting from
        `rasterio.io.DatasetReader.dataset_mask` or
        `rasterio.io.DatasetReader.read_masks`
    logic:
        Determines how the aggreagation should happen.
        If `all` (the default) a cell is only selected if **all** masks
        consider it valid data. `logic="any"` will lead to selecting
        all cells which **at least one** mask considers valid
    """
    selector = masks[0]!=0  # values > 0 are selected (i.e. True)
    if logic == 'any':
        _logic = np.logical_or
    else:
        _logic = np.logical_and
    if len(masks) > 1:
        for mask in masks[1:]:
            _logic(selector, mask!=0, out=selector)
    return selector


def reduced_mask(array:NDArray,
                nodata=0,
                logic:str='all',):
    """Computes a mask based on the value of serveral bands

    Parameters
    ----------
    array:
        3D array holding multiple bands of map data
    logic:
        Allowed strings are:
        - `"any"`: Masekd will be each cell for which any of the bands matches the nodata value
        - `"all"`: Masked will be each cell for which all of the bands match the nodata value
    """
    if logic=='any':
        _logic = np.logical_and
    else:
        _logic = np.logical_or
    return _logic.reduce(array=array!=nodata, axis=0).astype(np.uint8)
