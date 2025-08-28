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


# TODO: Adapt this to rioG (or similar)
# this is our namespace for tags
NS = 'LANDIV'


# TODO: General Idea - maybe we can merge some of these into io_.py class structure - so we avoid having both.
#  --> yet it is nice to have the function by themselves as well without direct need of class structures

def set_tags(src, bidx:int|None=None, ns:str=NS, **tags):
    # is_needed
    # needs_work (should be made internal?)
    # is_tested
    """Update tags for a dataset or a single band of a dataset.

    Since metadata in a tif file is stored as a string the value of each tag is
    serialized and converted to a string with `helper.serialize`.

    ..Example::

      Setting the tags

      - 'category': 1
      - foo: 'bar'

      on band with index 2 in some opened tif file (`src`) is done with:

      ```python
      set_tags(src=src, bidx=2, category=1, foo='bar')
      ```

    ..Note::
      A tag name must satisfy the python variable naming convention and
      must be different from `src`, `bidx` and `ns` as these are reserved
      for the arguments of this function.

    ..Note::
      Existing tags are either kept or updated

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open` in write mode (i.e. "w" or "r+")
    bidx:
      Index of the band to set tags for (starting from 1 as is the convention
      in rasterio). If set to `None` then the tags are set for the entire
      dataset.
    ns:
      The namespace to set the tags in.
      
      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be set as tags.
      The value provided is converted to a string with `helper.serialize`
      before the tag is written to the file.
    """

    if bidx is None:
        bidx = 0
    # serialize the tag values:
    serialized_tags = serialize(tags)
    src.update_tags(ns=ns, bidx=bidx, **serialized_tags)

def get_tags(src, bidx:int|None=None, ns:str=NS):
        # is_needed
        # needs_work (should be internal)
        # is_tested
        """Get all the tags and deserialize the values

        Parameters
        ----------
        src:
          `tif` file openend with `rasterio.open`
        bidx:
          Index of the band to get tags from (starting from 1 as is the convention
          in rasterio). If set to `None` then the tags for the entire dataset are
          returned.
        ns:
          The namespace to get the tags from.
          
          ..Note::
            It is dicouraged to change this value from the default as all tagging
            related methods of this package use the same default namespace.
        """
        if bidx is None:
            bidx = 0  # get the tags for the files metadata
        return deserialize(src.tags(bidx=bidx, ns=ns))

def find_bidxs(src, ns:str=NS, **tags):
    # is_needed
    # neews_work (should be internal)
    # not_tested
    """Find all bands in src for which all tags match

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open`
    ns:
      The namespace to set the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of the bands in the dataset.
    """

    _tags = sanitize(tags) 
    matching_bidxs = []
    for bidx in src.indexes:
        b_tags = get_tags(src=src, bidx=bidx, ns=ns)
        if match_all(targets=_tags, tags=b_tags):
            matching_bidxs.append(bidx)
    return matching_bidxs

def get_bidx(src, ns:str=NS, **tags)->None|int:
    # is_needed
    # needs_work
    # not_tested (used in tests)
    """Get the index of the band with matching tags

    ..Note::
      This function returns (if any) only a single band index!

      If you want to get potentially multiple bands that match
      the criterion use the `get_bands` method instead.

      An exception is when passing `indexes=None`, in which case
      all bands are returned.

    You can specify an arbitrary number of tags by passing keyword arguments
    to this selector.
    Make sure that the provided tags identify one and only one specific band.
    Multiple matching bands lead to a `BandSelectionAmbiguousError`.

    If no band with matching tags if found a
    `BandSelectionNoMatchError` is raised.


    If no tags are provided then the index of the first band is returned.

    ..Note::
      If `indexes` is provided then all other tags are ignored and
      the indexes are directly passed as band indexes to rasterio

    ..Example::

      Get the band with the tags

      - 'category': 1
      - foo: 'bar'

      ```python
      bidx = get_bidx(src=src, category=1, foo='bar')
      ```

    ..Note::
      The values of the provided tags are first serialized and then
      deserialized again with `helper.serialize`, resp. `helper.deserialize`,
      before comparing to the tags from the provided file.

      The reason for this procedure is the fact that the values of tags are
      converted to and stored as strings in the tif metadata.
      Serializing the values with `helper.serialize` allows us to know how
      arbitrary python objects are converted.
      As a consequence, we serialize/deserialize the values of the provided
      tags to bring them into the form they will we when loading them from
      the tif.

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open`
    ns:
      The namespace to set the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of the bands in the dataset.
    """

    if 'indexes' in tags or not tags:
        bidx = tags.get('indexes', 1)  # return 1 if nothing is provided
    else:
        # serialize/deserialize tags
        _tags = sanitize(tags) 
        matching_bidxs = find_bidxs(src=src, ns=ns, **_tags)
        matches = len(matching_bidxs)
        if matches > 1:
            raise BandSelectionAmbiguousError(
                f"The selection\n\t{_tags}\nleads to multiple matches:\n\t{matching_bidxs}"
            )
        elif matches == 0:
            raise BandSelectionNoMatchError(
                f"No band matches the tags: {_tags}"
            )
        bidx = matching_bidxs[0]
    return bidx

def get_bands(source:str, ns:str=NS, **tags)->list[tuple[str,int]]:
    # is_needed
    # needs_work
    # not_tested (used in tests)
    """Find all bands that match specific tags

    This method check the metadata (including those of bands)
    in one or several tif files and returns the file paths, as well as,
    the band indexes for all bands with matching tags.

    Whenever a band has tags that match, the name of the tif file,
    as well as, the band index are added to the list of returned
    matches.

    If the tags are found in the metadata of a dataset the path
    to the file and a band index of None are added to the list of
    returned matches (this different form the rasterio convention to
    that uses bidx=0 for "all bands" - I find that confusing).

    Parameters
    ----------
    source:
      A string that is fed to `glob.glob` leading to (potentially) multiple
      source files that will be checked
    ns:
      The namespace to search the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.

    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of each tif file
    """

    _tags = sanitize(tags)
    _sources = glob.glob(source)
    matches = []
    for source in _sources:
        with rasterio.open(source, "r") as src:
            ds_tags = get_tags(src=src, bidx=None, ns=ns)
            bidxs = find_bidxs(src=src, ns=ns, **_tags)
            if match_all(targets=_tags, tags=ds_tags):
                bidxs.append(None)  # use bidx None to indicate tags of the file
        for bidx in bidxs:
            matches.append((source, bidx))
    return matches


def load_map(source:str, **tags)->dict:
    # is_needed (this is only used in tests)
    # needs_work (replace usage with `load_block` and get rid of it)
    # not_tested (used in tests)
    """Load a specific band from a tif file

    See `load_block` for details

    Returns
    -------
    dict:
       Returns the callback of
       `load_block(source=source, view=None, scaling_params=None, **tags)`
    """
    return load_block(source=source, view=None, scaling_params=None, **tags)

def load_block(source:str,
               view:None|tuple[int,int,int,int]=None,
               scaling_params:dict|None=None,
               **tags)->dict:
    # is_needed
    # needs_work
    # is_tested
    """Get a block from a specific band of a *.tif file along with the transform

    You can select what band(s) to load by passing keyword arguments as tags
    (see `**tags` below) and limit the area to load by bassing a view.

    Parameters
    ----------
    source: str
      The path to the tif file to load
    view:
      An optional tuple (x, y, width, height) defining the area to load.

      If `None` is provided (the default) then the entire file is loaded.

    scaling_params:
      Optional dictionary to set a rescaling of the data.
      If provided, the following keywords are accepted:

      scaling: tuple[float,float]
        Factors to rescale the number of pixels. Values >1 will upscale.
      method: rasterio.enums.Resampling
        The resampling method. If not provided then the bilinear resampling
        is used.

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `get_bidx` for further details

    Returns
    -------
    dict:
       data: holding a numpy array with the actual data
       transform: an ???.Affine object that encodes the transformation used
       orig_meta: The meta information of the original .tif file
       orig_profile: The profile information of the original .tif file
    """
    window=view_to_window(view)
    with rasterio.open(source) as img:
        # TODO: rasterio Window allows using slices. In doing so we could
        #       harmonize what we call blocks and views and just work with
        #       slices.

        bidx = get_bidx(src=img, **tags)
        if window is not None:
            transform = img.window_transform(window)
            width = window.width
            height = window.height
        else:
            transform = img.transform
            width = img.width
            height = img.height
        # perform a re-scaling if needed
        if scaling_params:
            scaling = scaling_params.get('scaling')
            resampling = scaling_params.get('method', Resampling.bilinear)
            out_shape = (
                img.count,
                floor(img.height * scaling[0]),
                floor(img.width * scaling[1])
            )
        else:
            out_shape = None
            resampling = Resampling.nearest
        # read out the desired part
        data = img.read(indexes=bidx,
                        window=window,
                        out_shape=out_shape,
                        resampling=resampling)
        if scaling_params:
            # scale image transform
            transform = transform * transform.scale(
                (width / data.shape[-1]),
                (height / data.shape[-2])
            )
        return {
            'data': data,
            'transform': transform,
            'orig_profile': img.profile.copy()
        }
