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
