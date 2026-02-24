"""
Data model classes for GeoTIFF raster sources and bands.

This module defines :class:`Source` and :class:`Band`, the two central data
model classes of the ``riogrande`` package. A :class:`Source` represents a
single GeoTIFF file together with its metadata (tags, profile, namespace),
while a :class:`Band` encapsulates one raster band within a source, including
band-specific tags, a band index, and configurable mask and data readers.

Together these classes form the primary interface through which raster data is
opened, read, written, and tagged throughout the package.
"""

from __future__ import annotations
import os

import numpy as np
from numpy.typing import NDArray

import rasterio as rio
from rasterio.windows import Window

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from collections.abc import Callable

from typing import Union, Any

from ..helper import (
    check_compatibility,
    count_contribution,
)

from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
    InvalidMaskSelectorError,
)
from . import core


# noinspection PyArgumentList
class Source:
    """
    A ``Source`` object represents a file-based dataset (GeoTIFF) along with associated
    tags, profile metadata, and namespace information.

    Parameters
    ----------
    path : str or Path
        Path to the dataset file.
    tags : dict or None
        Optional dictionary of key-value metadata associated with the source.
        Defaults to an empty dictionary.
    profile : dict or None
        Optional dictionary of dataset profile information (e.g. width, height,
        dtype). Defaults to an empty dictionary.
    ns : str
        Namespace string used to distinguish sources. Defaults to "GEORACOON".

    Examples
    --------
    >>> s1 = Source("example.tif", tags={"type": "satellite"})
    >>> s1
    Source(path=example.tif, exists: False)
    >>> s1.tags
    {'type': 'satellite'}
    >>> s1._ns
    'GEORACOON'
    """
    _mode_writing = ('w', 'w+')
    _mode_reading = ('r', 'r+')
    _mode_default = 'r'
    _modes = _mode_reading + _mode_writing

    def __init__(self, path: str | Path,
                 tags: dict | None = None,
                 profile: dict | None = None,
                 ns: str = core.NS):
        """
        Initialize a new Source object.
        """
        self.path = Path(path)
        self.tags = tags or dict()
        self._ns = ns
        self.profile = profile or dict()

    def __repr__(self) -> str:
        """
        Return a string representation of the Source.

        Returns
        -------
        str
            String representation of the object.
        """
        items = [f"path={str(self.path)}", f"exists: {self.exists}"]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self) -> hash:
        """
        Compute a hash value for the Source.

        Returns
        -------
        int
            Hash based on path, namespace, and tag values.
        """
        return hash((self.path, self._ns, *(self.tags.values())))

    def __eq__(self, other: Source) -> bool:
        """
        Test equality between two Source objects.

        Parameters
        ----------
        other : Source
            Source object to compare against.

        Returns
        -------
        bool
         True if both objects are Source instances with the same path,
         tags, and namespace.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.path == other.path and self.tags == other.tags and
                self._ns == other._ns)

    def import_profile(self, update_self: bool = True) -> dict:
        """
        Read the profile from the source file

        Opens the file via :meth:`open` and reads the rasterio profile.

        Parameters
        ----------
        update_self : bool
            If True (default), update the ``profile`` attribute with the values
            fetched from the source file. If False, the object's profile remains
            unchanged.

        Returns
        -------
        dict
            The profile dictionary retrieved from the source file.

        See Also
        --------
        :meth:`init_source` : Create or overwrite the source file using the stored profile.
        """
        with self.open(mode='r') as src:
            profile = src.profile
        if update_self:
            self.profile.update(profile)
        return self.profile

    @property
    def exists(self) -> bool:
        """
        Check whether the source file exists on disk.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        return self.path.is_file()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the numpy shape of the data stored in this Source.

        Requesting the shape will synchronize the profile with the data
        written on disk.

        Returns
        -------
        tuple
            A 2-tuple ``(height, width)`` representing the dataset dimensions.
        """
        self.import_profile()
        height = self.profile['height']
        width = self.profile['width']
        return height, width

    def get_tags(self, bidx: int) -> dict:
        """
        Retrieve all tags for a specific band.

        Reads and deserializes tags via :func:`~riogrande.io.core._get_tags`.

        Parameters
        ----------
        bidx : int
            The band index to query.

        Returns
        -------
        dict
            A dictionary of tag key-value pairs associated with the band.

        See Also
        --------
        :meth:`set_tags` : Write tags back to the source file.
        :meth:`get_tag_values` : Fetch values for a single tag key across multiple bands.
        """
        with self.open(mode='r') as src:
            tags = core._get_tags(src=src, bidx=bidx, ns=self._ns)
        return tags

    def get_tag_values(self, tag: str, bidx: int | list | None = None) -> dict:
        """
        Fetch the value of a tag for one or more bands.

        If a tag is not present for a given band, the value will be ``None``
        and a mapping `{bidx: value}` for all bands will be returned.

        Parameters
        ----------
        tag : str
            The tag key to look up.
        bidx : int or list or None
            Band index (int), list of indices, or None.
            If None, all bands are queried.

        Returns
        -------
        dict
            A mapping of band index to the tag value (or None if missing).
        """
        t_vals = dict()
        with self.open(mode='r') as src:
            if bidx is None:
                bidxs = src.indexes
            elif isinstance(bidx, int):
                bidxs = [bidx, ]
            else:
                bidxs = bidx
            for _bidx in bidxs:
                t_vals[_bidx] = core._get_tags(src=src, bidx=_bidx,
                                          ns=self._ns).get(tag, None)
        return t_vals

    def set_tags(self, bidx: int | None, tags: dict):
        """
        Set one or more tags for a specific band or the dataset.

        Tags are serialized via :func:`~riogrande.helper.serialize` before being written
        to the file by :func:`~riogrande.io.core._set_tags`.

        Parameters
        ----------
        bidx : int or None
            The band index to set tags for. If None, tags are applied to the
            Source metadata (bidx=0).
        tags : dict
            A dictionary of key-value pairs to assign as tags.

        Returns
        -------
        None

        See Also
        --------
        :meth:`get_tags` : Read and deserialize tags for a band.
        """
        with self.open(mode='r+') as src:
            core._set_tags(src=src, bidx=bidx, ns=self._ns, **tags)

    @contextmanager
    def mask_reader(self, **kwargs):
        """
        Context manager for reading the dataset mask.

        Opens the source file and yields its ``dataset_mask`` method, which can
        be called to read the internal mask as an array.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :meth:`open`.

        Yields
        ------
        callable
            A callable that can be used to read the mask.
        """
        mode = kwargs.pop('mode', 'r')
        with self.open(mode=mode, **kwargs) as src:
            yield src.dataset_mask

    def get_mask(self, **kwargs) -> NDArray:
        """
        Read the dataset mask from the file.

        Except `okwargs` all keyword arguments are passed to
        the :meth:`rasterio.io.DatasetReader.dataset_mask` method of the source
        via :meth:`mask_reader`.

        Parameters
        ----------
        **kwargs : dict
            Optional set of keyword arguments to pass to the ``read`` method of the source.
            Notable exception: if ``okwargs`` is present, it is passed to :meth:`open`
            when accessing the source.

        Returns
        -------
        NDArray
            An array representing the dataset mask. Non-zero values indicate
            valid pixels, and 0 indicates masked/invalid pixels.

        See Also
        --------
        :meth:`mask_reader` : Context manager yielding the underlying mask-read callable.
        """
        okwargs = kwargs.pop('okwargs', dict())
        with self.mask_reader(mode='r', **okwargs) as dataset_mask:
            mask = dataset_mask(**kwargs)
        return mask

    @contextmanager
    def mask_writer(self, **kwargs):
        """
        Context manager for writing to the dataset mask.

        Opens the source file with internal TIFF mask support enabled and
        yields the ``write_mask`` method of the underlying dataset.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to :meth:`open`.

        Yields
        ------
        callable
            A callable that can be used to write a mask array.
        """
        mode = kwargs.pop('mode', 'r+')
        with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with self.open(mode=mode, **kwargs) as src:
                yield src.write_mask

    def export_mask(self, mask: NDArray, window: Window):
        """
        Write a mask into the source file.

        Parameters
        ----------
        mask : NDArray
            Array to use as a mask. Values greater than 0 represent valid pixels;
            0 represents invalid/masked pixels.
        window : Window
            The raster window to which the mask should be written.

        Returns
        -------
        None
        """
        with self.open(mode='r+') as src:
            src.write_mask(mask_array=mask, window=window)

    def init_source(self, overwrite: bool = False, **kwargs):
        """Initialize or create the source file.

        This method either creates a new file (empty dataset) on disk or opens an existing one
        via :meth:`open`. If ``overwrite`` is True, the existing file will be replaced.

        Parameters
        ----------
        overwrite : bool
            If True, overwrite an existing file.
        **kwargs : dict
            Additional keyword arguments passed to the :meth:`open` method when
            creating the dataset (e.g., driver options, compression).

        Returns
        -------
        None

        See Also
        --------
        :meth:`import_profile` : Read the profile from an existing source file.
        """
        if overwrite or not self.exists:
            with self.open(mode='w', **self.profile, **kwargs) as _:
                print(f'Initiating empty file\n\t"{self.path}"\n')

    def get_band(self, bidx: int | None = None, **tags) -> Band:
        """
        Retrieve a specific band as a Band object.

        A :class:`Band` can be selected either by its index (`bidx`) or by matching
        tags. If both are provided, they must match the same band.

        Parameters
        ----------
        bidx : int or None
            Optional band index to select.
        **tags : dict
            Optional tag key-value pairs to match the band via
            :func:`~riogrande.io.core._get_bidx_by_tag`.

        Returns
        -------
        :class:`Band`
            A ``Band`` object corresponding to the requested band, including
            associated tags.

        Raises
        ------
        :exc:`~riogrande.io.exceptions.SourceNotSavedError`
            If the source file does not exist on disk.
        :exc:`~riogrande.io.exceptions.BandSelectionNoMatchError`
            If no band matches the provided index or tags.
        AssertionError
            If the index and tag selection refer to different bands.

        See Also
        --------
        :meth:`get_bands` : Return all bands in the dataset.
        """
        if not self.exists:
            raise SourceNotSavedError(
                f"{self.path}:\n\tNot present in the filesystem"
            )
        _bidx = None
        _tb_bidx = None
        band_tags = dict()
        if bidx is not None:
            if not self.has_bidx(bidx):
                raise BandSelectionNoMatchError(
                    f'{bidx=} is not present in file\n- "{self.path=}"' \
                    f"\nPresent are: {self.band_indexes}"
                )
            _bidx = bidx
        if tags:
            with self.open() as src:
                _tb_bidx = core._get_bidx_by_tag(src=src, ns=self._ns, **tags)
            if _bidx:
                assert _tb_bidx == _bidx, \
                    f"The band index matching {tags=} is diffrent from " \
                    f"the provided {bidx=}: Unclear which band to use!"

        found_bidx = _bidx or _tb_bidx
        if found_bidx is not None:
            band_tags.update(self.get_tags(bidx=found_bidx))
            tags.update(band_tags)
        return Band(source=self, bidx=_bidx or _tb_bidx, tags=tags)

    def get_bands(self) -> list[Band]:
        """
        Return all bands present in the dataset.

        Returns
        -------
        list of :class:`Band`
            A list of ``Band`` objects for all bands in the dataset.

        See Also
        --------
        :meth:`get_band` : Retrieve a single band by index or tags.
        """
        bands = []
        with self.open(mode='r') as src:
            for bidx in src.indexes:
                tags = core._get_tags(src=src, bidx=bidx, ns=self._ns)
                _b = Band(source=self, bidx=bidx, tags=tags)
                bands.append(_b)
        return bands

    def _get_source(self, *args, **kwargs):
        """
        Prepare a partially-evaluated ``rasterio.open`` call for this source.

        If the file is opened in writing mode, the source's profile is injected
        automatically.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``rasterio.open``.
        **kwargs : dict
            Keyword arguments forwarded to ``rasterio.open``. The ``mode`` argument
            defaults to the source's default mode.

        Returns
        -------
        DatasetReader or DatasetWriter
            Partially-evaluated rasterio open object ready to be used or called.

        Raises
        ------
        AssertionError
            If the requested mode is not recognized.
        UnknownExtensionError
            If the file extension is not supported (currently only ".tif").
        """
        mode = kwargs.get('mode', args[0] if len(args) else self._mode_default)
        assert mode in self._modes
        if self.path.suffix in ['.tif', ]:
            if mode in self._mode_writing:
                src_open = partial(rio.open, fp=self.path, **self.profile)
            else:
                src_open = partial(rio.open, fp=self.path)
        else:
            raise UnknownExtensionError(
                f'"{self.path.suffix}" is not supported.\nCurrently only ".tif" is.'
            )
        return src_open(*args, **kwargs)

    @contextmanager
    def open(self, *args, **kwargs):
        """
        Open the source file for I/O operations.

        This context manager wraps the underlying rasterio dataset, yielding
        an open dataset object. If the file is openend in writing mode the profile is
        injected into the open function.
        Therefore: **the profile needs to be set when calling this method with writing mode!**

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :meth:`_get_source`.
        **kwargs : dict
            Keyword arguments forwarded to :meth:`_get_source`.

        Yields
        ------
        DatasetReader or DatasetWriter
            Open rasterio dataset object, ready for reading or writing.
        """
        src = self._get_source(*args, **kwargs)
        try:
            yield src
        finally:
            src.close()

    @contextmanager
    def data_reader(self, bands: list[Band] | None = None, **kwargs):
        """
        Context manager for reading multiple bands as a 3D array.

        Opens the source file and prepares a callable that reads the requested
        bands as a 3-dimensional array (band, row, column).

        Parameters
        ----------
        bands : list[Band] or None
            A collection of ``Band`` objects specifying which bands to read.
            If None, all bands in the dataset are used.
        **kwargs : dict
            Additional keyword arguments forwarded to :meth:`open` (e.g., mode,
            driver options).

        Yields
        ------
        callable
            A callable equivalent to ``src.read(indexes=...)`` that returns a
            3D numpy array with shape `(len(bands), height, width)`.
        """
        mode = kwargs.pop('mode', 'r')
        if bands is None:
            bands = self.get_bands()
        bidxs = [band.get_bidx() for band in bands]
        with self.open(mode=mode, **kwargs) as src:
            yield partial(src.read, indexes=bidxs)

    @property
    def band_indexes(self):
        """
        Band indexes available in the source.

        Returns
        -------
        list
          The band indexes present in the dataset.
        """
        with self.open() as src:
            bidxs = src.indexes
        return bidxs

    def has_bidx(self, bidx: int) -> bool:
        """
        Check whether a band index exists in the source.

        Parameters
        ----------
        bidx : int
            The band index to check for (1-based, as in rasterio).

        Returns
        -------
        bool
            True if the band index exists in the dataset.
        """

        has_it = False
        with rio.open(self.path, 'r') as src:
            if bidx in src.indexes:
                has_it = True
        return has_it

    def has_tags(self, tags: dict) -> bool:
        """
        Check whether any band contains all the provided tags.

        Parameters
        ----------
        tags : dict
            Dictionary of tag key–value pairs to look for.

        Returns
        -------
        bool
            True if at least one band contains all provided tags.
        """
        all_tags = []
        with rio.open(self.path, 'r') as src:
            for bidx in src.indexes:
                all_tags.append(core._get_tags(src=src, bidx=bidx, ns=self._ns))
        return any(core.match_all(tags, btags) for btags in all_tags)

    def find_indexes(self, tags: dict, mode: str ='all') -> list:
        """
        Find band indexes matching the given tags.

        Parameters
        ----------
        tags : dict
            Tag key–value pairs to search for.
        mode : str
            Matching mode:

            - 'all': All provided tags must be present.
            - 'any': Any one of the provided tags may match
              (currently not implemented, placeholder).

        Returns
        -------
        list of int
            The list of band indexes matching the tags.
        """
        with self.open() as src:
            if mode == 'any':
                print('WARNING: mode "any" not implemented yet. Emtpy list returned')
                bidxs = []
            else:
                bidxs = core._find_bidxs(src=src, ns=self._ns, **tags)
        return bidxs

    def find_index(self, tags: dict) -> int | None:
        """
        Find a single band index matching the given tags.

        Parameters
        ----------
        tags : dict
            Tag key–value pairs to search for.

        Returns
        -------
        int or None
            The band index if exactly one match is found, None otherwise.
        """
        midx = None
        matching_bidxs = self.find_indexes(tags=tags, mode='all')
        if len(matching_bidxs) != 1:
            print('WARNING: no matching index found')
        else:
            midx = matching_bidxs[0]
        return midx

    def has_band(self, band: Band) -> int:
        """
        Check whether a given Band is present in the source.

        Parameters
        ----------
        band : Band
            The band object to test for.

        Returns
        -------
        bool
            True if the band is present.
        """
        has_it = False
        if band.bidx is not None:
            has_it = self.has_bidx(band.bidx)
        elif band.tags:
            midx = self.find_index(tags=band.tags)
            if midx is not None:
                has_it = True
        return has_it

    def get_bidx(self, band: Band) -> int | None:
        """
        Resolve the band index of a given Band object in the source.

        Attempts to identify the band index (`bidx`) associated with
        the provided Band, based on either its explicit index or its tags. If both
        are given, they must resolve to the same unique band.

        Parameters
        ----------
        band : Band
            The Band object for which to resolve the band index.

        Returns
        -------
        int or None
            The resolved band index if a unique match is found, otherwise None.

        Raises
        ------
        BandSelectionAmbiguousError
           If only tags are provided, and they match multiple bands, or if both
           `bidx` and `tags` are given, but they are inconsistent.
        """
        midx = None
        if band.bidx is not None:
            if self.has_bidx(band.bidx):
                midx = band.bidx
        if band.tags:
            tidxs = self.find_indexes(tags=band.tags)
            missmatch = False
            if len(tidxs) > 1:
                if midx is None:
                    raise BandSelectionAmbiguousError(
                        "Only tags are provided and we have multiple matches"
                    )
                else:
                    print(
                        "WARNING: The tags are not unique, several bands share them"
                    )
                    if not any(tidx == midx for tidx in tidxs):
                        missmatch = True
            elif len(tidxs) == 1:
                if midx is not None and tidxs[0] != midx:
                    missmatch = True
                else:
                    midx = tidxs[0]
            if missmatch:
                raise BandSelectionAmbiguousError(
                    "The band has `bidx` and `tags` set but they do not "
                    "match to a single band in the source file"
                )
        return midx

    def compress(self, output: str | None = None, compression: str | None = 'lzw',
                 keep_original: bool = False) -> None:
        """
        Compress the source file using a given compression algorithm.

        A new compressed GeoTIFF is created via :func:`~riogrande.io.core.compress_tif`.
        By default, the original file is replaced with the compressed one unless
        `keep_original` is set.

        Parameters
        ----------
        output : str or None
            Path to the output file. If None (default), the compressed file
            overwrites the current source path.
        compression : str or None
            Compression algorithm to use. Default is ``'lzw'``.
            See GDAL documentation for valid options.
        keep_original : bool
            If True, the original uncompressed file is preserved.
            If False (default), the uncompressed file is deleted after
            compression.

        Returns
        -------
        None

        Notes
        -----
        - Updates the ``path`` attribute of the Source to point to the new
          compressed file.

        See Also
        --------
        :func:`~riogrande.io.core.compress_tif` : Underlying compression function.
        """
        uncompressed = self.path
        self.path = Path(core.compress_tif(str(self.path), output=output, compression=compression))
        if not keep_original and uncompressed != self.path:
            os.remove(uncompressed)

    def check_compatibility(self, *sources: Source):
        """
        Check whether this source is compatible with one or more other sources.

        Delegates to :func:`~riogrande.helper.check_compatibility`, which verifies
        CRS, linear units, and spatial resolution.

        Parameters
        ----------
        *sources : Source
            One or more additional :class:`Source` objects to check against this one.

        Returns
        -------
        bool
            True if all provided sources are compatible with this one.

        See Also
        --------
        :func:`~riogrande.helper.check_compatibility` : Underlying compatibility check.
        """
        _sources = {self.path, }
        for source in sources:
            _sources.add(source.path)
        return check_compatibility(*_sources)

    def load_block(self, view: None | tuple[int, int, int, int] = None,
                   scaling_params: dict | None = None, **tags) -> dict[str, Any]:
        """
        Load a block of raster data from the source along with the transform.

        Band where data is loaded from needs to be identified with tags.
        If no tags are provided data from bidx=1 is returned.
        See :func:`~riogrande.io.core.load_block` for further details.

        Parameters
        ----------
        view : tuple[int, int, int, int] or None
            The window to read, given as (row_start, row_stop, col_start, col_stop).
            If None (default), the entire raster is read.
        scaling_params : dict, optional
            Parameters controlling rescaling of the data. If provided, the
            dictionary may include:

            - ``scaling`` : tuple of float
              Factors to rescale the raster dimensions. Values > 1 upscale,
              values < 1 downscale.
            - ``method`` : rasterio.enums.Resampling, optional
              Resampling method to use. Defaults to
              :data:`rasterio.enums.Resampling.bilinear`.
        **tags : dict
            Band selection criteria. See :meth:`Source.get_bidx` for details.
            Tags need to present to the source file.

        Returns
        -------
        dict
            A dictionary with the following entries:

            - ``data`` :
              The loaded raster data. Shape depends on band selection and scaling.
            - ``transform`` :
              :class:`affine.Affine` transform mapping array coordinates to spatial coordinates.
            - ``orig_profile`` :
              Copy of the original raster profile metadata.

        See Also
        --------
        :func:`~riogrande.io.core.load_block` : Underlying function with full parameter details.
        """
        return core.load_block(source=str(self.path), view=view, scaling_params=scaling_params, **tags)


# noinspection PyArgumentList
@dataclass
class Band:
    """
    A ``Band`` object encapsulates metadata and configuration for accessing
    a band in a raster dataset. It references a parent :class:`Source` object,
    contains band-specific tags, and holds parameters for reading.

    Parameters
    ----------
    source : Source
        The ``Source`` object from which this band originates.
    tags : dict
        Optional dictionary of key-value metadata associated with the band.
        Defaults to an empty dictionary.
    bidx : int or None
        Band index in the source dataset (1-based). If ``None``, defaults to
        an implicit index or is determined at runtime.
    read_params : dict
        Dictionary of parameters controlling how the band is read (e.g.,
        windowing, resampling). Defaults to an empty dictionary.

    Examples
    --------
    >>> s = Source("example.tif")
    >>> b = Band(source=s, bidx=1, tags={"type": "NIR"})
    >>> b
    Band(tags={'type': 'NIR'})
    >>> b._use_mask
    'self'
    >>> b._ns
    'GEORACOON'
    """
    source: Source
    tags: dict = field(default_factory=dict)
    bidx: int | None = None
    read_params: dict = field(default_factory=dict)
    _use_mask: str = 'self'
    _ns = core.NS

    def __repr__(self) -> str:
        """
        Return a string representation of the Band's tags.

        Returns
        -------
        str
            String representation of the tags.
        """
        items = [f"tags={self.tags}", ]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self) -> hash:
        """
        Compute a hash value for the Band.

        Returns
        -------
        int
            Hash value for the object.
        """
        return hash((self.bidx, *(self.tags.values())))

    @property
    def source_exists(self) -> bool:
        """
        Check whether the parent source file of the Band exists on disk.

        This property queries the parent ``Source`` object to determine
        if the underlying file is present.

        Returns
        -------
        bool
            True if the file exists.
        """
        return self.source.exists

    @property
    def index_exists(self) -> bool:
        """
        Check whether the band's index exists in the source dataset.

        Returns
        -------
        bool
           True if the band's index (`bidx`) is set and exists in the parent
           ``Source``. False if no `bidx` is set or if the index is absent.
        """
        i_exists = False
        if self.bidx is None:
            print(f"No index set for {self}")
        else:
            i_exists = self.source.has_bidx(self.bidx)
        return i_exists

    @property
    def status(self) -> None:
        """
        Print the current status of the Band.

        Reports include:
            - Existence of the source file.
            - Presence of the band's index in the source.
            - Exact and partial matches for the band's tags.
            - Warnings if the index or tags are inconsistent.

        """
        print(f"\n### Status of {self}")
        # check if the resource exist
        print("# Source file")
        if not self.source_exists:
            print(
                f"- not found: {self.source.path}"
            )
        else:
            print(
                f"- exists: {self.source.path}"
            )
            # check the band index
            print("# BAND INDEX")
            if not self.index_exists:
                if not self.bidx is None:
                    print(
                        f'# - {self.bidx=} not present in file\n'
                        f'#   "{self.source.path}"\n'
                        f"# - Present are: {self.source.band_indexes}"
                    )
            else:
                print(
                    f'# - {self.bidx} is present in file\n'
                    f'# - "{self.source.path}"'
                )
            # now check the resources
            print("# TAGS")
            if self.tags:
                print("#   EXACT MATCH")
                # check if a band matches all
                exact_bidxs = self.source.find_indexes(self.tags, mode='all')
                matches = len(exact_bidxs)
                if matches > 1:
                    print(
                        f"# - WARNING:\n#\tThe selection\n#\t{self.tags}\n"
                        f"#\thas multiple exact matches:\n#\t{exact_bidxs=}"
                    )
                elif matches == 0:
                    print(
                        f"# - No exact matche for tags: {self.tags}"
                    )
                else:
                    print(
                        f'# - Band {exact_bidxs[0]} matches the tags: '
                        f"{self.tags}"
                    )
                    # check if self.bindx and found match are ok
                    if self.bidx is not None and self.bidx != exact_bidxs[0]:
                        print(
                            f'# - WARNING: The tags "{self.tags}" match with '
                            f'band index "{exact_bidxs[0]}"\n'
                            f'#            but {self.bidx=}'
                        )
                print("#   PARTIAL MATCH")
                # check if a band matches all
                any_bidxs = self.source.find_indexes(self.tags, mode='any')
                matches = len(any_bidxs)
            else:
                print(
                    f"# Not tags are set for this band"
                )
        print("###")

    def _pair_operation(self, pair_op: Callable, band: Band, out_band: Band=None, **op_kwargs):
        """
        Internal helper to perform element-wise operations between two bands.

        This method reads data from this band and another ``Band`` object
        in blocks, applies the provided operation, and writes the result to
        a destination band.

        Parameters
        ----------
        pair_op : Callable
            Function that performs an element-wise operation on two arrays.
            Should accept two arrays as first arguments and return an array.
        band : Band
            The second band participating in the operation.
        out_band : Band or None
            Destination band for storing the result. If None, this band is
            overwritten.
        **op_kwargs : dict
            Additional keyword arguments to pass to ``pair_op``.
        """
        self.source.check_compatibility(band.source)
        if out_band is None:
            out_band = self
        bidx = self.get_bidx()
        other_bidx = band.get_bidx()
        with self.source.open(mode='r') as src:
            with out_band.data_writer() as write:
                with band.source.open() as src_to_add:
                    for _ji, window, in src.block_windows(bidx):
                        data = src.read(bidx, window=window)
                        other = src_to_add.read(other_bidx, window=window)
                        write(pair_op(data, other, **op_kwargs), window=window)

    def add(self, band: Band, out_band: None | Band = None, **add_kwargs):
        """
        Add the values of another band to this band, element-wise.

        The data from both bands are added using :func:`numpy.add`,
        and the result is stored in ``out_band`` or overwrites this band by default.

        Parameters
        ----------
        band : Band
            :class:`Band` whose values will be added.
        out_band : Band or None
            Destination :class:`Band` for storing the result. If None (default), the
            operation overwrites this band.
        **add_kwargs : dict
            Additional keyword arguments passed to :func:`numpy.add`.

        See Also
        --------
        :meth:`subtract` : Element-wise subtraction of another band.
        """
        return self._pair_operation(pair_op=np.add, band=band,
                                    out_band=out_band, **add_kwargs)

    def subtract(self, band: Band, out_band: None | Band = None, **add_kwargs):
        """
        Subtract another band from this band, element-wise.

        The operation computes ``self - band`` by adding the negative
        of the second band to this band via :func:`numpy.add`.
        The result is stored in ``out_band`` or overwrites this band by default.

        Parameters
        ----------
        band : Band
            :class:`Band` to subtract from this band.
        out_band : Band or None
            Destination :class:`Band` for storing the result. If None (default), the
            operation overwrites this band.
        **sub_kwargs : dict
            Additional keyword arguments passed to the underlying operation.

        See Also
        --------
        :meth:`add` : Element-wise addition of another band.
        """
        def _subtract(data1, data2, **kwargs):
            return np.add(data1, (-1) * data2, **kwargs)

        return self._pair_operation(pair_op=_subtract, band=band,
                                    out_band=out_band, **add_kwargs)

    def export_tags(self, match: str | list | None = None):
        """
        Write the band’s set tags back to the source file.

        The band index is resolved with :meth:`get_bidx` and the tags are
        written via :meth:`~riogrande.io.models.Source.set_tags`.

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to identify a matching band.
            If provided, the routine tries to find a single band in the
            source file for which only the tags specified in this list
            have matching values.
            It can be used if you want to export some new tags or if you
            have updated some tags and want to export these new values.

        Notes
        -----
        - If the band has the ``bidx`` attribute set, ``match`` is ignored.
        - Example:

        >>> b1.export_tags(match=’category’)
        # Identifies the band via the ‘category’ tag and updates other tags

        See Also
        --------
        :meth:`import_tags` : Load tags from the source file into this band object.
        :meth:`get_bidx` : Resolve the band index in the source file.
         """
        bidx = self.get_bidx(match=match)
        self.source.set_tags(bidx=bidx, tags=self.tags)

    def import_tags(self, match: str | list | None = None, keep: bool = True):
        """
        Import tags from the source file (at its band index) into this Band object.

        The band index is resolved with :meth:`get_bidx` and tags are read via
        :meth:`~riogrande.io.models.Source.get_tags`.

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to match a band in the source file.
            Used internally by :meth:`get_bidx` to locate the correct band.
        keep : bool
            If True, update the existing ``tags`` dictionary with the imported
            tags. If False, replace the existing ``tags`` dictionary with the
            imported tags.

        See Also
        --------
        :meth:`export_tags` : Write this band's tags back to the source file.
        """
        bidx = self.get_bidx(match=match)
        tags = self.source.get_tags(bidx)
        if keep:
            self.tags.update(tags)
        else:
            self.tags = tags

    def get_bidx(self, match: str | list | None = None) -> int:
        """
        Determine the correct band index in the source file.

        If the ``bidx`` attribute is already set, it is checked for existence
        in the :class:`Source` file. If ``bidx`` is None, the method tries to infer
        the correct band index using the ``tags`` attribute via
        :meth:`~riogrande.io.models.Source.find_index`. The optional
        ``match`` argument can limit which tags are considered for matching.

        Parameters
        ----------
        match : str or list or None
            Selection of tags to identify the correct band. If an integer is
            provided, it is converted to a single-element list. If None (default),
            all tags in ``self.tags`` are considered. Ignored if ``bidx`` is set.

         Notes
         -----
         - Can be used when exporting or updating tags to identify a single
           band in the source.
         - Example:

           >>> b1.get_bidx(match='category')
           # Finds the band whose 'category' tag matches

        Returns
        -------
        int
         Index of the band in the ``Source`` that matches this band.

        Raises
        ------
        :exc:`~riogrande.io.exceptions.BandSelectionNoMatchError`
         If there is no clear match for the band in the source file or if the
         specified ``bidx`` does not exist.

        See Also
        --------
        :meth:`export_tags` : Write tags using the resolved band index.
        :meth:`import_tags` : Load tags using the resolved band index.
        """
        failed = ''
        bidx = None
        if self.bidx is not None:
            if self.index_exists:
                bidx = self.bidx
            else:
                failed += f"The band has {self.bidx=} which is not present " \
                          f"in  the source file \n'{self.source.path}'\n" \
                          f"Present are:\n{self.source.band_indexes}\n"
        else:
            if isinstance(match, int):
                match = [match, ]
            elif match is None:
                match = list(self.tags)
            matching_tags = {tag: value
                             for tag, value in self.tags.items()
                             if tag in match}
            bidx = self.source.find_index(tags=matching_tags)
            if bidx is None:
                failed += "Unable to find a band that matches the tags:\n" \
                          f"{matching_tags}\n"
        if failed or bidx is None:
            raise BandSelectionNoMatchError(failed)
        return bidx

    def init_source(self, profile: dict, overwrite: bool = False, **kwargs):
        """
         Initialize the source file for this band, creating it if necessary.

         Updates the source's profile and optionally overwrites an existing file.

         Parameters
         ----------
         profile : dict
             Dictionary specifying the profile of the dataset.
             This will update the source's profile before creating or opening the file.
         overwrite : bool
             If True, any existing file at the source path will be overwritten.
             Equivalent to deleting the existing source and creating a new one.
         **kwargs
             Additional keyword arguments passed to the underlying ``Source.init_source``
             method.
         """
        self.source.profile.update(profile)
        return self.source.init_source(overwrite=overwrite, **kwargs)

    def get_data(self, **kwargs) -> NDArray:
        """
        Read the full data array of this band from the source file.

        All keyword arguments except `okwargs` are passed to the ``read`` method
        of the underlying ``Source``. The optional `okwargs` are passed to the
        ``read`` method of the ``Source``.

        Parameters
        ----------
        **kwargs
           Optional keyword arguments for ``Source.read``. Common options include
           `window`, `out_shape`, `resampling`, etc.
           Special keyword: if ``okwargs`` is present, it is passed to
           ``Source.open`` when opening the file.

        Returns
        -------
        NDArray
           A NumPy array containing the data of this band.
        """
        okwargs = kwargs.pop('okwargs', dict())
        with self.source.open(mode='r', **okwargs) as src:
            data = src.read(indexes=self.source.get_bidx(band=self), **kwargs)
        return data

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the band as a tuple (height, width).

        This corresponds to the shape of the NumPy array.

        Returns
        -------
        tuple of int
          Tuple (height, width) representing the number of rows and columns
          in the band.
        """
        return self.source.shape

    def count_valid_pixels(self, selector: NDArray | None, no_data: Union[int, float],
                           limit_count: int = 0) -> int | bool:
        """
        Count the number of valid pixels in the band, optionally under a selector.

        Valid pixels are those that are not equal to `no_data`. If a selector
        mask is provided, only pixels where the selector is True are counted.
        The per-block counting is delegated to :func:`~riogrande.helper.count_contribution`.

        Parameters
        ----------
        selector : NDArray or None
            Boolean :class:`numpy.ndarray` of the same shape as the band, indicating
            which pixels to include in the count. If None, all pixels are considered.
        no_data : int or float
            Pixel value considered invalid (e.g., nodata value).
        limit_count : int
            Optional early-exit threshold. If > 0, the method returns a boolean:
            - True if the number of valid pixels exceeds `limit_count`
            - False otherwise
            If `limit_count` is 0, the method returns the actual count.

        Returns
        -------
        int or bool
            - If `limit_count` is 0, returns the total count of valid pixels.
            - If `limit_count` > 0, returns True/False depending on whether the
              count exceeds the limit.

        See Also
        --------
        :func:`~riogrande.helper.count_contribution` : Per-block pixel counting function.
        :meth:`get_min_max` : Compute the min/max of valid pixels.
        """
        if selector is None:
            self.source.import_profile()
            height = self.source.profile['height']
            width = self.source.profile['width']
            selector = np.full(shape=(height, width), fill_value=True)

        bidx = self.get_bidx()
        count = 0
        with self.source.open() as src:
            for _ji, window, in src.block_windows(bidx):
                data = src.read(bidx, window=window)

                count += count_contribution(data=data,
                                            selector=selector[window.toslices()],
                                            no_data=no_data)
                if limit_count and count > limit_count:
                    return True
        if limit_count:
            return False
        else:
            return count

    def get_min_max(self, no_data: Union[int, float], selector: NDArray | None = None) -> tuple | None:
        """
        Compute the minimum and maximum values of the band, optionally under a selector.

        Only pixels not equal to `no_data` and selected by the `selector` are considered.
        Per-block min/max are computed with :func:`numpy.nanmin` / :func:`numpy.nanmax`.

        Parameters
        ----------
        no_data : int or float
            Value considered invalid; these pixels are ignored.
        selector : NDArray or None
            Boolean :class:`numpy.ndarray` of the same shape as the band, indicating
            which pixels to include. If None, all pixels are considered.

        Returns
        -------
        tuple of (float, float) or None
            Tuple (min_value, max_value) over the selected valid pixels.
            Returns None if no valid pixels are found.

        See Also
        --------
        :meth:`count_valid_pixels` : Count valid (non-nodata) pixels.
        """
        if selector is None:
            self.source.import_profile()
            height = self.source.profile['height']
            width = self.source.profile['width']
            selector = np.full(shape=(height, width), fill_value=True)

        bidx = self.get_bidx()
        _min = []
        _max = []
        with self.source.open() as src:
            for _ji, window, in src.block_windows(bidx):
                data = src.read(bidx, window=window)
                wdw_selector = selector[window.toslices()]
                valid_mask = wdw_selector & (data != no_data)
                valid_data = data[valid_mask]
                if valid_data.size > 0:
                    _min.append(np.nanmin(valid_data))
                    _max.append(np.nanmax(valid_data))

        if _min and _max:
            return min(_min), max(_max)
        else:
            return None

    @contextmanager
    def data_writer(self, match: str | list | None = None, **kwargs):
        """
        Context manager for writing data to this band.

        Opens the underlying ``Source`` for writing and yields a callable
        that writes data to the specified band index.

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to identify a matching band. Ignored if
            the band has ``bidx`` set.
        **kwargs
            Additional keyword arguments passed to ``rasterio.io.DatasetWriter.write``,
            e.g., window specification.

        Yields
        ------
        Callable
         A partial function that writes data to the band.
        """
        mode = kwargs.pop('mode', 'r+' if self.source.exists else 'w')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.write, indexes=bidx)

    @contextmanager
    def data_reader(self, match: str | list | None = None, **kwargs):
        """
        Context manager for reading data from this band.

        Opens the underlying ``Source`` for reading and yields a callable
        that reads data from the specified band index.

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to identify a matching band. Ignored if
            the band has ``bidx`` set.
        **kwargs
            Additional keyword arguments passed to ``rasterio.io.DatasetReader.read``,
            e.g., window, masked, out_dtype.

        Yields
        ------
        Callable
            A partial function that reads data from the band.
        """
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.read, indexes=bidx)

    def set_mask_reader(self, use: str = 'band'):
        """
        Set which mask should be used when reading data for this band.

        Parameters
        ----------
        use : {'self', 'band', 'source', 'mask_none', 'mask_all'}, default 'band'
            Determines how the mask is applied:

            - ``'self'`` or ``'band'``: use the band-specific mask
              (i.e., :meth:`rasterio.io.DatasetReader.read_masks`).
            - ``'source'``: use the dataset mask from the source
              (i.e., :meth:`rasterio.io.DatasetReader.dataset_mask`).
            - ``'mask_none'``: consider all pixels valid (returns an array of 1s).
            - ``'mask_all'``: consider all pixels invalid (returns an array of 0s).

        Raises
        ------
        AssertionError
            If `use` is not one of the allowed options.

        See Also
        --------
        :meth:`get_mask_reader` : Return the mask-reading callable based on the current setting.
        """
        assert use in ['self', 'band', 'source', 'mask_all', 'mask_none'], \
            f'"{use}" is an invalid selector for a mask, options are:' \
            '\n\t- "band": uses the bands own mask (i.e. ' \
            'rasterio.io.DataReader.read_masks)\n\t- "source": uses the ' \
            'dataset mask (i.e. rasterio.io.DataReader.dataset_mask)'
        if use in ['self', 'band']:
            self._use_mask = 'self'
        else:
            self._use_mask = use

    def get_mask_reader(self, ):
        """
        Return the appropriate mask reader for this band based on `_use_mask`.

        Returns
        -------
        Callable
          A callable that reads a mask array when called. Depending on ``_use_mask``,
          it may read the band mask via :meth:`mask_reader`, the dataset mask via
          :meth:`~riogrande.io.models.Source.mask_reader`, or return a constant array
          via :meth:`_mask_full`.

        Raises
        ------
        :exc:`~riogrande.io.exceptions.InvalidMaskSelectorError`
          If `_use_mask` has an invalid value.

        See Also
        --------
        :meth:`set_mask_reader` : Configure which mask to use.
        """
        if self._use_mask is None or self._use_mask == 'self':  # read the band mask
            return self.mask_reader
        elif self._use_mask == 'source':  # read the dataset mask
            return self.source.mask_reader
        elif self._use_mask == 'mask_none':
            # 1 is valid data
            return partial(self._mask_full, fill_value=1)
        elif self._use_mask == 'mask_all':
            # 0 is invalid data
            return partial(self._mask_full, fill_value=0)
        else:
            raise InvalidMaskSelectorError(
                f'"{self._use_mask}" is an invalid selector for a mask,'
                'options are:'
                '\n\t- "self": uses the bands own mask'
                ' (i.e. rasterio.io.DataReader.read_masks)'
                '\n\t- "source": uses the bands own mask'
                ' (i.e. rasterio.io.DataReader.dataset_mask)'
            )

    @contextmanager
    def mask_reader(self, match: str | list | None = None, **kwargs):
        """
        Context manager for reading the band-specific mask.

        Always returns the mask associated with this specific band.

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to identify the band.
        **kwargs
            Keyword arguments passed to `Source.open`.

        Yields
        ------
        Callable
            Partial function to read the band mask.
        """
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.read_masks, indexes=bidx)

    @contextmanager
    def _mask_full(self, match: str | list | None = None, fill_value: int | float | bool = False,
                   **kwargs):
        """
        Context manager for a mocked band mask returning a constant array.

        This can be used to override the actual band or dataset mask with a
        True/False or numeric values.
        The mocked read method first calls the normal band read method
        (i.e. rasterios `read_masks`) in order to assure than transformation,
        rescaling, window, etc. is performed correctly.
        It then returns a similar numpy array holding exclusively
        `True` or `False` as values effectively ignoring the actual mask

        Parameters
        ----------
        match : str or list or None
            Optional selection of tags to identify the band.
        fill_value : int or float or bool
            Value to fill in the mask array. Accepts also True/False.
        **kwargs
            Keyword arguments passed to `Source.open`.

        Yields
        ------
        Callable
            Partial function returning a mask array filled with `fill_value`,
            while respecting windowing, rescaling, and transformations.
        """
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)

        def _mock_all(mask_reader: Callable, *args, **kwargs) -> NDArray:
            _mask = mask_reader(indexes=bidx, *args, **kwargs)
            full_array = np.full(shape=_mask.shape, fill_value=fill_value)
            del _mask
            return full_array

        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(_mock_all, mask_reader=src.read_masks)

    def set_data(self, data: NDArray, overwrite: bool = False, **kwargs):
        """
        Write data to the band in the underlying source file.

        Uses :meth:`data_writer` as the context manager for writing.

        Parameters
        ----------
        data : NDArray
            :class:`numpy.ndarray` containing the data to write.
        overwrite : bool
            If True, overwrite an existing source file. If False and the source
            exists, the file is opened in update mode (``'r+'``).
        **kwargs
            Additional keyword arguments passed to :meth:`data_writer`.

        See Also
        --------
        :meth:`get_data` : Read the full data array of this band.
        :meth:`data_writer` : Context manager used for writing.
        """
        if self.source.exists and not overwrite:
            mode = 'r+'
        else:
            mode = 'w'
        with self.data_writer(mode=mode, **kwargs) as write:
            write(data)

    def load_block(self,
                   view: None | tuple[int, int, int, int] = None,
                   scaling_params: dict | None = None,
                   match: str | list | None = None) -> dict[str, Any]:
        """Load a block of data from this band along with its transform.

        This reads a portion of the band data from the source, optionally applying
        scaling/resampling. The block can be limited to a rectangular region (`view`),
        and the specific band can be selected via `match` (tags or `bidx`).
        Delegates to :func:`~riogrande.io.core.load_block` via :meth:`~riogrande.io.models.Source.load_block`.
        See :func:`~riogrande.io.core.load_block` for further details.

        Parameters
        ----------
        view : tuple[int, int, int, int] or None
            Optional window defined as (x, y, width, height) in pixels.
            If `None`, the entire band is read.
        scaling_params : dict or None
            Optional dictionary specifying rescaling parameters:
              - `scaling`: tuple[float, float], factors to rescale the number of pixels.
                           Values >1 will upscale.
              - `method`: rasterio.enums.Resampling, resampling method (default: bilinear)
        match : str or list or None
            Optional tag(s) or criteria to identify the band in the source.
            If `None`, the current `bidx` or all tags are used.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:

              - ``data``: :class:`numpy.ndarray` with the band data of shape (1, height, width)
              - ``transform``: :class:`affine.Affine` transformation object for the loaded block
              - ``orig_profile``: Original raster profile of the source band

        See Also
        --------
        :meth:`~riogrande.io.models.Source.load_block` : Analogous method on the Source class.
        :func:`~riogrande.io.core.load_block` : Underlying function with full parameter details.
        """
        bidx = self.get_bidx(match=match)
        return self.source.load_block(view=view, scaling_params=scaling_params, indexes=bidx)
