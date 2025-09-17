from __future__ import annotations

import os
import glob

from math import floor

import rasterio  # TODO: choose either rio or rasterio as name
import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

from shapely.geometry import box as shbox

from numpy.typing import NDArray

from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
)
from .helper import (
    check_crs,
    outfile_suffix,
    output_filename,
    serialize,
    deserialize,
    sanitize,
    match_all,
    view_to_window,
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

def write_band(src:DatasetWriter,
               data:NDArray,
               bidx:int=1,
               window:Window|None=None,
               **tags):
    # is_needed
    # needs_work
    # not_tested
    """Write data to a specific band of a tif file and set the tags


    Parameters
    ----------
    src:
      An opened file to write into
    data:
      The array to write into the file
    window:
      An optional window to specify an area to write
    **tags:
      Arbitrary number of keyword arguments that will be set as tags.
      The value provided is converted to a string with `helper.serialize`
      before the tag is written to the file.
    """
    src.write(data, indexes=bidx, window=window)
    set_tags(src, bidx=bidx, **tags)

def update_band(src:DatasetWriter,
                data:NDArray,
                window:Window|None=None,
                **tags):
    # not_needed (could be useful though)
    # no_work
    # not_tested
    """Find a specific band and update it with data

    ..Note::
      If no band with the matching tags is found a
      `BandSelectionNoMatchError` is raised.

    Parameters
    ----------
    src:
      An opened file to write into
    data:
      The array to write into the file
    window:
      An optional window to specify an area to write
    **tags:
      Arbitrary number of keyword arguments that will be used to find
      the band to write into
    """
    try:
        bidx = get_bidx(src=src, **tags)
    except BandSelectionNoMatchError as e:
        raise BandSelectionNoMatchError(
            "There was no band with matching tags. "
            "Either adapt the tags or use `write_band` "
            "with a specific band index instead if you "
            "want to write and also update the tags."
        ) from e
    else:
        src.write(data, indexes=bidx, window=window)

def export_to_tif(destination:str,
                  data:NDArray,
                  orig_profile:dict,
                  start=(0, 0),
                  **pparams):
    # not_needed (could be useful though)
    # no_work
    # not_tested
    """Export a np.array to tif, only updating a window if data is smaller

    .. note::
      This function will overwrite the dtype of the destination tif with the
      value provided in `pparams` or the data type of `data`.

    Parameters
    ----------
    destination: str
        location to export save the .tif file
    data: np.array
        The map to export
    start: tuple
      horizontal and vertical starting coordinate
    orig_profile: dict
        the profile of the original map
        (see https://rasterio.readthedocs.io/en/stable/topics/profiles.html)
    **pparams:
        further parameter to be added to the profile
    """
    profile = orig_profile.copy()
    # Note: we no longer update the size automatically as for Windows this is
    # not correct, pass height and width explicitly to update via pparams
    # # update for the correct dimensions
    # profile['height'] = data.shape[1]
    # profile['width'] = data.shape[0]
    # set the dtype explicitly of get it from the data
    profile['dtype'] = pparams.pop('dtype', str(data.dtype))
    profile.update(pparams)
    # write it:
    size = data.shape[::-1]  # since positions are inverted in numpy
    with rasterio.open(destination, "w", **profile) as dest:
        dest.write(data, window=Window(*start, *size), indexes=1)


def _coregister_raster(source, reference, output=None):
    # is_needed (in tests only)
    # needs_work (format doc)
    # not_tested
    """Align raster to have identical resolution.

    Resolution will be calculated automatically from bounds and height/width of reference raster.

    Parameters
    ----------
    source: str
      The path to the tif file you want to co-register
    reference: str
      The path to the tif file with the pixel registration to use as reference for co-registration
    output: str (optional)
      The path to write the co-registered map to

    Returns
    -------
    str:
      The name of the file that holds co-registered map
    """
    check_crs(source, reference)

    if output is None:
        output = output_filename(source, out_type="coreg")

    with rasterio.open(source) as src:
        src_transform = src.transform
        src_nodata = src.nodata

        with rasterio.open(reference) as refsrc:
            dst_crs = refsrc.crs

            (dst_transform,
             dst_width,
             dst_height) = calculate_default_transform(src.crs,
                                                       dst_crs,
                                                       refsrc.width,
                                                       refsrc.height,
                                                       *refsrc.bounds)

        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": src_nodata})

        with rasterio.open(output, "w", **dst_kwargs) as dst:
            for bidx in src.indexes:
                reproject(
                    source=rasterio.band(src, bidx),
                    destination=rasterio.band(dst, bidx),
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return output

def buffer_geometries_metric(geom_geoseries, buffer_meter, source_crs):
    # is_needed (internal only)
    # needs_work (docs)
    # not_tested
    """ Applies a buffer to the geometries in GeoSeries given.

    ..Note: This function re-projects the GeoSeries to the respective UTM zone in order to use metric buffer and best
    distance calculations. Further empty geometries are dropped before handing back the results.

    Parameters
    ----------
    geom_geoseries: GeoPandas GeoSeries
      The geoseries holding the polygons to perform the buffer on
    buffer_meter: float, int
      The buffer in meters to apply to the ecoregion polygon before clipping.
      Needs to be negative for reducing the polygon e.g. -1000
    source_crs:
      The coordinate system of the inptu GeoSeries (taken from GeoDataframe before by user) to project to after buffer.

    Returns
    -------
    GeoSeries object:
      The buffered GeoSeries object
    """
    geom_utm = geom_geoseries.to_crs(geom_geoseries.estimate_utm_crs())
    geom_buff = geom_utm.buffer(buffer_meter,
                                resolution=10, cap_style='round', join_style='round')
    geom_buff = geom_buff[geom_buff.area > 0]
    return geom_buff.to_crs(source_crs)

def compress_tif(source, output:str|None=None, compression:str|None='lzw'):
    # is_needed
    # needs_work (docs)
    # is_tested
    # TODO: once this goes, we may also remove the outfile_suffix I believe (but double-check)
    """Compress tif file with LZW compression

    Parameters
    ----------
    source: str
      The path to the tif file you want to compress
    output:
      Optional path to output file.
      If not set, the resulting file will inherit the filename from `source` and get
      a `_compress` appended to the filename.
      If compression is `'none'`, i.e. no compression the appendix will be '_decompressed'

    Returns
    -------
    str:
      The name of the compressed file
    """
    if compression is None:
        compression = 'none'
    overwrite = False
    if output is None:
        if compression != 'none':
            output = outfile_suffix(source, "compress")
        else:
            output = outfile_suffix(source, "decompressed")
    elif output == source:
        overwrite = True
        output = outfile_suffix(source, 'tmp')

    with rasterio.Env():
        with rasterio.open(source) as src:
            profile = src.profile
            profile.update(compress=compression)

            with rasterio.open(output, 'w', **profile) as dst:
                set_tags(src=dst, bidx=None, **get_tags(src=src, bidx=None))
                for i in range(1, src.count + 1):
                    for ji, window in src.block_windows(i):
                        array = src.read(i, window=window)
                        dst.write(array, i, window=window)
                    tags = get_tags(src, bidx=i)
                    set_tags(dst, bidx=i, **tags)
                    band_names = src.descriptions[(i - 1)]
                    dst.set_band_description(i, band_names)
    if overwrite:
        os.remove(source)
        os.rename(src=output, dst=source)
        output = source
    return output
