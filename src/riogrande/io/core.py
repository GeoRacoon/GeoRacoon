"""
Holds [...] as well as internal functions largely used for Source and Band Classes
"""

from __future__ import annotations

import os
import glob
from typing import Any

from math import floor
from numpy.typing import NDArray

import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.windows import Window
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from ..helper import (
    check_crs,
    output_filename,
    serialize,
    deserialize,
    sanitize,
    match_all,
    view_to_window,
)
from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
)

NS = 'GEORACOON'


def _set_tags(src: DatasetWriter, bidx: int | None = None, ns: str = NS, **tags: Any) -> None:
    """Update tags for a dataset or a single band of a dataset.

    Since metadata in a tif file is stored as a string the value of each tag is
    serialized and converted to a string with :func:`~riogrande.helper.serialize`.
    A tag name must satisfy the python variable naming convention and must be different from ``src``,
    ``bidx`` and ``ns`` as these are reserved for the arguments of this function.
    Existing tags are either kept or updated.

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open` in write mode (i.e. ``"w"`` or ``"r+"``)
    bidx : int or None
        Index of the band to set tags for (starting from 1 as is the convention
        in rasterio). If set to ``None`` then the tags are set for the entire
        dataset.
    ns : str
        The namespace to set the tags in.
        .. note::
          It is dicouraged to change this value from the default as all tagging
          related methods of this package use the same default namespace.

    **tags : Any
        Arbitrary number of keyword arguments that will be set as tags.
        The value provided is converted to a string with :func:`~riogrande.helper.serialize`
        before the tag is written to the file.

    See Also
    --------
    :func:`~riogrande.io.core._get_tags` : Read and deserialize tags from a dataset or band.

    Examples
    ----------
      Setting the tags

      - 'category': 1
      - foo: 'bar'

      on band with index 2 in some opened tif file (`src`) is done with:

      ```python
      set_tags(src=src, bidx=2, category=1, foo='bar')
      ```
    """
    if bidx is None:
        bidx = 0
    serialized_tags = serialize(tags)
    src.update_tags(ns=ns, bidx=bidx, **serialized_tags)


def _get_tags(src: DatasetWriter, bidx: int | None = None, ns: str = NS) -> dict[str, Any]:
    """Get all the tags and deserialize the values

    Reads raw tags from the dataset and deserializes them with
    :func:`~riogrande.helper.deserialize`.

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open`
    bidx : int or None
        Index of the band to get tags from (starting from 1 as is the convention
        in rasterio). If set to ``None`` then the tags for the entire dataset are
        returned.
    ns : str
        The namespace to get the tags from.

        .. note::
          It is dicouraged to change this value from the default as all tagging
          related methods of this package use the same default namespace.

    Returns
    ----------
    dict
        Tags from queried band are returned in a dictionary form.

    See Also
    --------
    :func:`~riogrande.io.core._set_tags` : Write and serialize tags to a dataset or band.
    """
    if bidx is None:
        bidx = 0  # get the tags for the files metadata
    return deserialize(src.tags(bidx=bidx, ns=ns))


def _find_bidxs(src: DatasetWriter, ns: str = NS, **tags: Any) -> list[int]:
    """Find all bands in src for which all tags match

    Iterates over all bands and reads their tags via :func:`~riogrande.io.core._get_tags`,
    then checks for full tag agreement with :func:`~riogrande.helper.match_all`.

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open`
    ns : str
        The namespace to set the tags in.

        .. note::
          It is dicouraged to change this value from the default as all tagging
          related methods of this package use the same default namespace.

    **tags : Any
        Arbitrary number of keyword arguments that will be compared to the tags
        of the bands in the dataset.

    Returns
    ----------
    list[int]
        List of all indexes (integer) for bands where tags match.

    See Also
    --------
    :func:`~riogrande.io.core._get_bidx_by_tag` : Return a single matching band index.
    :func:`~riogrande.io.core.get_bands_by_tag` : Search across multiple source files.
    """
    _tags = sanitize(tags)
    matching_bidxs = []
    for bidx in src.indexes:
        b_tags = _get_tags(src=src, bidx=bidx, ns=ns)
        if match_all(targets=_tags, tags=b_tags):
            matching_bidxs.append(bidx)
    return matching_bidxs


def _get_bidx_by_tag(src: DatasetWriter, ns: str = NS, **tags: Any) -> None | int:
    """Get the index of the band with matching tags

    You can specify an arbitrary number of tags by passing keyword arguments
    to this selector. Make sure that the provided tags identify one and only one specific band,
    as only a single band index is returned. If no band with matching tags is found,
    or if multiple matching bands are found a `BandSelectionNoMatchError` is raised.
    To return potentially multiple bands matching, use the `get_bands` method instead.
    If no tags are provided then the index of the first band is returned.

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open`
    ns : str
        The namespace to set the tags in.
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags : Any
        Arbitrary number of keyword arguments that will be compared to the tags
        of the bands in the dataset. If ``indexes`` is provided as tag key then all other tags are ignored and
        the indexes are directly passed as band indexes to rasterio.

    Returns
    ----------
    int or None
        Band index (integer) of band matching provided tags. If no match was found None is returned.

    Notes
    ----------
    The values of the provided tags are first serialized and then
    deserialized again with :func:`~riogrande.helper.serialize` /
    :func:`~riogrande.helper.deserialize` (i.e. via :func:`~riogrande.helper.sanitize`)
    before comparing to the tags from the provided file.

    The reason for this procedure is the fact that the values of tags are
    converted to and stored as strings in the tif metadata.
    Serializing the values with :func:`~riogrande.helper.serialize` allows us to know how
    arbitrary python objects are converted.
    As a consequence, we serialize/deserialize the values of the provided
    tags to bring them into the form they will we when loading them from
    the tif.

    Raises
    ------
    :exc:`~riogrande.io.exceptions.BandSelectionAmbiguousError`
        If the tags match more than one band.
    :exc:`~riogrande.io.exceptions.BandSelectionNoMatchError`
        If no band matches the provided tags.

    See Also
    --------
    :func:`~riogrande.io.core._find_bidxs` : Return all matching band indexes.
    :func:`~riogrande.io.core.get_bands_by_tag` : Search across multiple source files.

    Examples
    ----------
    Get the band with the tags

    - 'category': 1
    - foo: 'bar'
    >>> bidx = _get_bidx_by_tag(src=src, foo='bar', category=1)
    """
    if 'indexes' in tags or not tags:
        bidx = tags.get('indexes', 1)  # return 1 if nothing is provided
    else:
        _tags = sanitize(tags)
        matching_bidxs = _find_bidxs(src=src, ns=ns, **_tags)
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


def get_bands_by_tag(source: str, ns: str = NS, **tags: Any) -> list[tuple[str, int]]:
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
    source : str
        A glob pattern string passed to :func:`glob.glob`, leading to (potentially)
        multiple source files that will be checked.
    ns : str
        The namespace to search the tags in. It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags : Any
        Arbitrary number of keyword arguments that will be compared to the tags
        of each tif file.

    Returns
    ----------
    list
        A list of tuples with source (path) and bandindex entries in tuples.

    See Also
    --------
    :func:`~riogrande.io.core._get_bidx_by_tag` : Find a single band in one open dataset.
    :func:`~riogrande.io.core._find_bidxs` : Return all matching band indexes in one dataset.
    """
    _tags = sanitize(tags)
    _sources = glob.glob(source)
    matches = []
    for source in _sources:
        with rio.open(source, "r") as src:
            ds_tags = _get_tags(src=src, bidx=None, ns=ns)
            bidxs = _find_bidxs(src=src, ns=ns, **_tags)
            if match_all(targets=_tags, tags=ds_tags):
                bidxs.append(None)  # use bidx None to indicate tags of the file
        for bidx in bidxs:
            matches.append((source, bidx))
    return matches


def load_block(source: str, view: None | tuple[int, int, int, int] = None, scaling_params: dict | None = None,
               **tags: Any) -> dict[str, Any]:
    """Get a block from a specific band of a *.tif file along with the transform

    You can select what band(s) to load by passing keyword arguments as tags
    (see `**tags` below) and limit the area to load by passing a view
    (converted to a :class:`rasterio.windows.Window` via
    :func:`~riogrande.helper.view_to_window`).

    Parameters
    ----------
    source : str
        The path to the tif file to load
    view : tuple[int, int, int, int] or None
        An optional tuple (x, y, width, height) defining the area to load.
        If ``None`` is provided (the default) then the entire file is loaded.

    scaling_params : dict or None
        Optional dictionary to set a rescaling of the data.
        If provided, the following keywords are accepted:

        scaling : tuple[float, float]
            Factors to rescale the number of pixels. Values >1 will upscale.
        method : rasterio.enums.Resampling
            The resampling method. If not provided then
            :attr:`rasterio.enums.Resampling.bilinear` is used.

    **tags : Any
        Arbitrary number of keyword arguments to describe the band to select.
        See :func:`~riogrande.io.core._get_bidx_by_tag` for further details.

    Returns
    -------
    dict
       data: holding a numpy array with the actual data
       transform: an :class:`affine.Affine` object that encodes the transformation used
       orig_profile: The profile information of the original .tif file

    See Also
    --------
    :func:`~riogrande.io.core.write_band` : Write data to a specific band.
    :func:`~riogrande.helper.view_to_window` : Convert a view tuple to a rasterio Window.
    """
    window = view_to_window(view)
    with rio.open(source) as img:
        bidx = _get_bidx_by_tag(src=img, **tags)
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


def write_band(src: DatasetWriter, data: NDArray, bidx: int = 1, window: Window | None = None,
               **tags: Any) -> None:
    """Write data to a specific band of a tif file and set the tags

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open`
    data : NDArray
        The array to write into the file
    bidx : int
        Band index to write into the file
    window : Window or None
        An optional :class:`rasterio.windows.Window` to specify an area to write.
    **tags : Any
        Arbitrary number of keyword arguments that will be set as tags.
        The value provided is converted to a string with :func:`~riogrande.helper.serialize`
        via :func:`~riogrande.io.core._set_tags` before being written to the file.

    Returns
    -------
    None

    See Also
    --------
    :func:`~riogrande.io.core.update_band` : Find a band by tags and update it.
    :func:`~riogrande.io.core.load_block` : Load a block of data from a band.
    """
    src.write(data, indexes=bidx, window=window)
    _set_tags(src, bidx=bidx, **tags)


def update_band(src: DatasetWriter, data: NDArray, window: Window | None = None, **tags: Any) -> None:
    """Find a specific band and update it with data

    This function writes a data array in a band specified with tags,
    identified via :func:`~riogrande.io.core._get_bidx_by_tag`.
    If no band with the matching tags is found a
    :exc:`~riogrande.io.exceptions.BandSelectionNoMatchError` is raised.

    Parameters
    ----------
    src : DatasetWriter
        ``tif`` file opened with :func:`rasterio.open`
    data : NDArray
        The array to write into the file
    window : Window or None
        An optional :class:`rasterio.windows.Window` to specify an area to write.
    **tags : Any
        Arbitrary number of keyword arguments that will be used to find
        the band to write into.

    Returns
    --------
    None

    See Also
    --------
    :func:`~riogrande.io.core.write_band` : Write to a band by explicit index.
    """
    try:
        bidx = _get_bidx_by_tag(src=src, **tags)
    except BandSelectionNoMatchError as e:
        raise BandSelectionNoMatchError(
            "There was no band with matching tags. "
            "Either adapt the tags or use `write_band` "
            "with a specific band index instead if you "
            "want to write and also update the tags."
        ) from e
    else:
        src.write(data, indexes=bidx, window=window)


def _export_to_tif(destination: str, data: NDArray, orig_profile: dict, start=(0, 0), **pparams: Any) -> None:
    """Export a np.array to tif, only updating a window if data is smaller

    This function will overwrite the dtype of the destination tif with the
    value provided in `pparams` or the data type of `data`.

    Parameters
    ----------
    destination : str
        location to export save the .tif file
    data : NDArray
        The map to export
    start : tuple[int, int]
        horizontal and vertical starting coordinate
    orig_profile : dict
        the profile of the original map
        (see https://rasterio.readthedocs.io/en/stable/topics/profiles.html)
    **pparams : Any
        further parameter to be added to the profile

    Returns
    --------
    None
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
    with rio.open(destination, "w", **profile) as dest:
        dest.write(data, window=Window(*start, *size), indexes=1)


def coregister_raster(source: str, reference: str, output: str | None = None) -> str:
    """Align raster to have identical resolution.

    Resolution will be calculated automatically from bounds and height/width of reference raster
    using :func:`rasterio.warp.calculate_default_transform`.
    Reprojection is performed with :func:`rasterio.warp.reproject` and
    :attr:`rasterio.enums.Resampling.nearest`.
    CRS compatibility is verified with :func:`~riogrande.helper.check_crs`.

    Parameters
    ----------
    source : str
        The path to the tif file you want to co-register
    reference : str
        The path to the tif file with the pixel registration to use as reference for co-registration
    output : str or None
        The path to write the co-registered map to. If ``None``, a filename is generated
        by :func:`~riogrande.helper.output_filename`.

    Returns
    -------
    str:
      The name of the file that holds co-registered map
    """
    check_crs(source, reference)

    if output is None:
        output = output_filename(source, out_type="coreg")

    with rio.open(source) as src:
        src_transform = src.transform
        src_nodata = src.nodata

        with rio.open(reference) as refsrc:
            dst_crs = refsrc.crs
            (dst_transform,
             dst_width,
             dst_height) = calculate_default_transform(src.crs, dst_crs, refsrc.width, refsrc.height, *refsrc.bounds)

        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": src_nodata})

        with rio.open(output, "w", **dst_kwargs) as dst:
            for bidx in src.indexes:
                reproject(
                    source=rio.band(src, bidx),
                    destination=rio.band(dst, bidx),
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return output


def compress_tif(source, output: str | None = None, compression: str | None = 'lzw') -> str:
    """Compress tif file with LZW compression

    Band tags are copied band-by-band using :func:`~riogrande.io.core._get_tags`
    and :func:`~riogrande.io.core._set_tags`.

    Parameters
    ----------
    source : str
        The path to the tif file you want to compress
    output : str or None
        Optional path to output file.
        If not set, the resulting file will inherit the filename from ``source`` and get
        a ``_compress`` appendix via :func:`~riogrande.helper.output_filename`.
        If compression is ``'none'``, i.e. no compression the appendix will be ``'_decompressed'``.
    compression : str or None
        Type of compression to use, default is LZW. See GDAL documentation for details.

    Returns
    -------
    str
      The name of the compressed file

    See Also
    --------
    :meth:`~riogrande.io.models.Source.compress` : Convenience wrapper on the :class:`~riogrande.io.models.Source` class.
    """
    if compression is None:
        compression = 'none'
    overwrite = False
    if output is None:
        if compression != 'none':
            output = output_filename(source, "compress")
        else:
            output = output_filename(source, "decompressed")
    elif output == source:
        overwrite = True
        output = output_filename(source, 'tmp')

    with rio.Env():
        with rio.open(source) as src:
            profile = src.profile
            profile.update(compress=compression)

            with rio.open(output, 'w', **profile) as dst:
                _set_tags(src=dst, bidx=None, **_get_tags(src=src, bidx=None))
                for i in range(1, src.count + 1):
                    for ji, window in src.block_windows(i):
                        array = src.read(i, window=window)
                        dst.write(array, i, window=window)
                    tags = _get_tags(src, bidx=i)
                    _set_tags(dst, bidx=i, **tags)
                    band_names = src.descriptions[(i - 1)]
                    dst.set_band_description(i, band_names)
    if overwrite:
        os.remove(source)
        os.rename(src=output, dst=source)
        output = source
    return output
