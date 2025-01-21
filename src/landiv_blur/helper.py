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
            crs = src.profile['crs']
            if crs is not None:
                units.append(src.profile['crs'].linear_units.lower())
            else:
                units.append(None)
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
            crss.append(str(src.profile.get('crs', None)))
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


def outfile_suffix(filename, suffix, separator:str='_'):
    """Insert suffix into filename and hand back basename_suffix.extension"""
    base, ext = os.path.splitext(filename)
    return f"{base}{separator}{suffix}{ext}"

def strip_suffix(filename:str, separator:str='_'):
    """Removes the last suffix from the name (i.e. the last part separated by '_')
    """
    base, ext = os.path.splitext(filename)
    if separator in filename:
        _base = separator.join(filename.split(separator)[:-1])
    else:
        _base = base
    return f"{_base}{ext}"

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


def dtype_range(dtype)->tuple[int|float, int|float]:
    """Get the range of the specified dtype

    ..warning::
      This functions returns min or max as either `int` or `floats`.

      Be sure to convert them back into `dtype` if needed!

    """
    # avoid issues of object not callable from rasterio
    if hasattr(dtype, 'type'):
        dtype = dtype.type
    try:
        _max = int(np.iinfo(dtype).max)
        _min = int(np.iinfo(dtype).min)
    except ValueError:
        try:
            _max = float(np.finfo(dtype).max)
            _min = float(np.finfo(dtype).min)
        except ValueError:
            raise ValueError(f"{dtype=} has no min-/maximal values.")
    return _max, _min


def convert_to_dtype(data: NDArray,
                     as_dtype:None|type|np._dtype=None,
                     in_range:None|NDArray|Collection=None,
                     out_range:None|NDArray|Collection=None)->NDArray:
    """Converts data to as_dtype and optionally rescales it.

    ..note::

      The default range for any floating type is `[0,1]`!

      This means:

      - If the output data type, `as_dtype` is any subclass of `np.floating`
        and no `out_range` is defined then the output is scaled to the intervarl
        `[0, 1]`.
      - If data is of any `np.floating` type and the data range lies withing
        `[0, 1]` (and `in_range` is not provided) then `in_range` is set to be
        `[0, 1]`.


    Parameters
    ----------
    data: input numpy NDArray
    as_dtype: desired data type to convert to (e.g. np.float64)
        If not provided then at least the `out_range` needs to be set in
        which case the data type remains unchanges, but the data is
        rescaled.
    in_range:
        an array or list from which min and max will be used as input range

        ..note::
          You might simply provide the same value as for `data` in order to
          use its min an max for scaling
    out_range:
      an array or list from which min and max will be used as limits for the
      output
    """
    in_dtype = data.dtype

    data_min, data_max = np.nanmin(data), np.nanmax(data)

    cut = False

    if in_range is None:
        if np.issubdtype(in_dtype, np.floating) and data_min >= 0.0 and data_max <= 1.0:
            _inmax, _inmin = 1.0, 0.0
        else:
            # TODO: It would actually make sense to raise a warning directly in the user-method
            #      (extract_categories and block_heterogenity (unused?) for parallelized and
            #      get_filtered_categories for single threaded), as convert_to_dtype is in most cases called internally.
            if np.issubdtype(in_dtype, np.floating): 
                warnings.warn(f"Converting the full range of `{ in_dtype }` to `{ as_dtype }`. "
                              "If this is not what you want, specify the input range with `in_range`.")
            _inmax, _inmin = dtype_range(in_dtype)
    else:
        # we convert to float since Decimal cannot handle np.uintX
        _inmax = float(np.nanmax(in_range))
        _inmin = float(np.nanmin(in_range))
        if data_min < _inmin or data_max > _inmax:
            warnings.warn(f"The data-range [{data_min}, {data_max}] extends "
                          f"over the chose input range [{_inmin}, {_inmax}].\n"
                          "Exceeding values will be replaced by the limit values")
            cut = True
    # now the output dtype
    if out_range is None:
        assert as_dtype is not None, f"{out_range=} and {as_dtype=} cannot both be unset!"
        if np.issubdtype(as_dtype, np.floating):
            _outmax, _outmin = 1.0, 0.0
        else:
            _outmax, _outmin = dtype_range(as_dtype)
    else:
        # we convert to float since Decimal cannot handle np.uintX
        _outmax = float(np.nanmax(out_range))
        _outmin = float(np.nanmin(out_range))
        
        if as_dtype is None:
            # We simply rescale the data
            as_dtype = data.dtype
    scale = (Decimal(_outmax) - Decimal(_outmin)) / \
            (Decimal(_inmax) - Decimal(_inmin))

    out_data = np.array(_outmin).astype(as_dtype) + ((data - _inmin) * float(scale)).astype(as_dtype)

    if cut:
        out_data[out_data<_outmin] = _outmin
        out_data[out_data>_outmax] = _outmax

    return out_data


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
    """Computes a mask based on the value of several bands

    Parameters
    ----------
    array:
        3D array holding multiple bands of map data
    logic:
        Allowed strings are:
        - `"any"`: Masked will be each cell for which any of the bands matches the nodata value
        - `"all"`: Masked will be each cell for which all of the bands match the nodata value
    """
    if logic=='any':
        _logic = np.logical_and
    else:
        _logic = np.logical_or
    if np.isnan(nodata):
        return _logic.reduce(array=~np.isnan(array), axis=0).astype(np.uint8)
    else:
        return _logic.reduce(array=array!=nodata, axis=0).astype(np.uint8)

def count_contribution(data:NDArray,
                       selector:NDArray[np.bool_],
                       no_data:Union[int, float]=0)->int:
    """The remaining number of data cells when applying the selector

    Parameters
    ----------
    data:
      The data to cont the contribution in
    selector:
      A boolean array in the shape of `data` selecting the single cells that
      should be considered
    no_data:
      The value that should be considered as invalid.

      .. note::
        You might also provide `np.nan` as no data value.

    """
    if np.isnan(no_data):
        b_vals, b_counts = np.unique(~np.isnan(data[selector]), return_counts=True)
    else:
        b_vals, b_counts = np.unique(data[selector]!=no_data, return_counts=True)
    # b_vals is [True, False] and can be used as selector for b_counts
    # thus returning the count of True
    if True in b_vals:
        return int(b_counts[b_vals][0])
    else:
        return 0


def check_rank_deficiency(array, return_by_issue_type: bool=False) -> dict[int, str] | dict[str, list[int]]:
    """Check if matrix is rank deficient and extract the dependent columns (linear combination of other columns.
    Returns a dictionary with column (key) and issue description (value). Lenght of dictionary is rank-deficiency + 1,
    Empyt dictionary indicates that no rank deficiency was detected

    Parameters
    ----------
    array : np.ndarray
        Matrix to check for rank deficiency
    return_by_issue_type: bool
        If desired, a nested dictionary may be returned separating the type of issue:
        "all_zero" and "linear dependent"
    """
    all_zero_cols = {}
    rank_deficient_cols = {}
    _, num_columns = array.shape
    rank = np.linalg.matrix_rank(array)

    if rank == num_columns:
        return dict()

    for col in range(num_columns):
        column_vector = array[:, col]

        if np.all(column_vector == 0):
            all_zero_cols[col] = "All zero column"
        else:
            # drop focus column
            sub_array = np.delete(array, col, axis=1)

            # does removing a column increase the rank?
            if np.linalg.matrix_rank(sub_array) == rank:
                rank_deficient_cols[col] = "Linear dependent column"

    if return_by_issue_type:
        return dict(linear_dependent=[l for l in rank_deficient_cols.keys()],
                    all_zero=[z for z in all_zero_cols.keys()])
    else:
        return {**rank_deficient_cols, **all_zero_cols}


def rasterio_to_numpy_dtype(rasterio_dtype):
    """
    Map Rasterio actual data types to NumPy data types.

    Rasterio types like rasterio.dtypes.int16, rasterio.dtypes.float32
    are mapped to their NumPy equivalents.
    """
    dtype_mapping = {
        rio.dtypes.int16: np.int16,
        rio.dtypes.int32: np.int32,
        rio.dtypes.uint8: np.uint8,
        rio.dtypes.uint16: np.uint16,
        rio.dtypes.uint32: np.uint32,
        rio.dtypes.float32: np.float32,
        rio.dtypes.float64: np.float64,
    }
    return dtype_mapping.get(rasterio_dtype, None)
