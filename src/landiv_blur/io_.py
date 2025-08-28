from __future__ import annotations
import os

import numpy as np
import rasterio as rio

from rasterio.windows import Window
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from numpy.typing import NDArray
from collections.abc import Callable

from typing import Union

from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
    InvalidMaskSelectorError,
)

from .io import (
    NS,
    get_tags,
    set_tags,
    get_bidx,
    match_all,
    find_bidxs,
    compress_tif,
    load_block
)
from ._helper import (
    check_compatibility as _check_compatibility,
    count_contribution,
)


# TODO: I feel: all is_needed - but needs a bit of work

class Source:
    """Specifies a data source
    """
    _mode_writing = ('w', 'w+')
    _mode_reading = ('r', 'r+')
    _mode_default = 'r'
    _modes = _mode_reading + _mode_writing
    # is_needed
    # needs_work (docs)
    # not_tested
    # usedin_both

    def __init__(self, path:str|Path,
                 tags: dict|None=None,
                 profile: dict|None=None,
                 ns: str=NS):
        # is_needed
        # needs_work (docs)
        # not_tested (used in tests)
        # usedin_both
        self.path = Path(path)
        self.tags = tags or dict()
        self._ns = ns
        self.profile = profile or dict()

    def __repr__(self):
        # is_needed
        # needs_work (docs - revisit what is printed)
        # not_tested (no need)
        # usedin_both
        items = [f"path={str(self.path)}", f"exists: { self.exists }"]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self):
        # is_needed
        # no_work
        # not_tested (no need)
        # usedin_both
        return hash((self.path, self._ns, *(self.tags.values())))

    def __eq__(self, other):
        # is_needed
        # needs_work (docs)0
        # not_tested
        # usedin_both
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.path == other.path and self.tags == other.tags and
                self._ns == other._ns)

    def import_profile(self, update_self:bool=True):
        """Read the profile from the source file

        Parameters
        ----------
        update_self:
          If set, the profile property will be updated with the profile fetched
          from the source file.
        """
        # is_needed
        # needs_work (better docs)
        # not_tested (used in tests)
        # usedin_both
        with self.open(mode='r') as src:
            profile = src.profile
        if update_self:
            self.profile.update(profile)
        return self.profile

    @property
    def exists(self)->bool:
        # is_needed
        # needs_work (make internal, docs)
        # not_tested
        # usedin_both
        return self.path.is_file()

    @property
    def shape(self)->tuple:
        """Return the numpy shape of the data stored in this Source

        .. note::
          Requesting the shape will synchronize the profile with the data
          written on disk.
        """
        # is_needed (not sure - check)
        # needs_work (this should be renamed to avoid confusion w np.array.shape
        # not_tested
        # usedin_both (potentially, if used at all)
        self.import_profile()
        height = self.profile['height']
        width = self.profile['width']
        return (height, width)
    
    def get_tags(self, bidx:int)->dict:
        # is_needed (noly internally)
        # needs_work (docs)
        # not_tested
        # usedin_both (part of IO module)
        with self.open(mode='r') as src:
            tags = get_tags(src=src, bidx=bidx, ns=self._ns)
        return tags

    def get_tag_values(self, tag:str, bidx:int|list|None=None)->dict:
        """Try to fetch for each band the value of this tag
        
        If the tag is not present, None is returned
        """
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_processing (but should be part of the io module)
        t_vals = dict()
        with self.open(mode='r') as src:
            if bidx is None:
                bidxs = src.indexes
            elif isinstance(bidx, int):
                bidxs = [bidx,]
            else:
                bidxs = bidx
            for _bidx in bidxs:
                t_vals[_bidx] = get_tags(src=src,
                                         bidx=_bidx,
                                         ns=self._ns).get(tag, None)
        return t_vals

    def set_tags(self, bidx:int|None, tags:dict):
        # is_needed (noly internally)
        # needs_work (docs)
        # not_tested
        # usedin_both (part of IO module)
        with self.open(mode='r+') as src:
            set_tags(src=src, bidx=bidx, ns=self._ns, **tags)

    @contextmanager
    def mask_reader(self, **kwargs):
        # is_needed
        # needs_work (docs)
        # is_tested
        # usedin_both
        mode = kwargs.pop('mode', 'r')
        with self.open(mode=mode, **kwargs) as src:
            yield src.dataset_mask

    def get_mask(self, **kwargs):
        """Read band dataset mask from the file

        With the exception of `okwargs` all keyword arguments are passed to
        the `dataset_mask` method of the source.

        Parameters
        ----------
        **kwargs:
          Optional set of keyword arguments to pass to the `read` method of the source.
          Notable exception:

          `okwargs`: dict
            These arguments will be passed to the `open` method of the source
        """
        # is_needed
        # no_work
        # is_tested
        # usedin_both
        okwargs = kwargs.pop('okwargs', dict())
        with self.mask_reader(mode='r', **okwargs) as dataset_mask:
            mask = dataset_mask(**kwargs)
        return mask

    @contextmanager
    def mask_writer(self, **kwargs):
        # is_needed
        # needs_work (docs)
        # not_tested (used in tests)
        # usedin_both (part of io)
        mode = kwargs.pop('mode', 'r+')
        with rio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with self.open(mode=mode, **kwargs) as src:
                yield src.write_mask

    def export_mask(self, mask:NDArray, window:Window):
        """Write the mask into the output file

        Parameters
        ----------
        mask:
            Array to use as a mask. Values > 0 representa valid data
        window:
            Optional subset to write to
        """
        # not_needed (useful?)
        # no_work
        # not_tested
        # usedin_both (potentially)
        with self.open(mode='r+') as src:
            src.write_mask(mask_array=mask, window=window)

    def init_source(self, overwrite:bool=False, **kwargs):
        """Create or accesses source file
        """
        # is_needed (only internally for now)
        # needs_work (docs)
        # not_tested (but used in tests)
        # usedin_both (potentially)
        if overwrite or not self.exists:
            with self.open(mode='w', **self.profile, **kwargs) as _:
                print(f'Initiating empty file\n\t"{self.path}"\n')

    def get_band(self, bidx:int|None=None, **tags)->Band:
        """Find the wanted band and return a related band object
        """
        # is_needed
        # needs_work (dcos)
        # is_tested (though no dedicated test)
        # usedin_both
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
                _tb_bidx = get_bidx(src=src, ns=self._ns, **tags)
            if _bidx:
                assert _tb_bidx == _bidx, \
                    f"The band index matching {tags=} is diffrent from " \
                    f"the provided {bidx=}: Unclear which band to use!"

        found_bidx = _bidx or _tb_bidx
        if found_bidx is not None:
            band_tags.update(self.get_tags(bidx=found_bidx))
            tags.update(band_tags)
        return Band(source=self, bidx=_bidx or _tb_bidx, tags=tags)

    def get_bands(self)->list[Band]:
        """Return a list with all Bands present in the dataset
        """
        # is_needed
        # needs_work (dcos)
        # is_tested (though no dedicated test)
        # usedin_both
        bands = []
        with self.open(mode='r') as src:
            for bidx in src.indexes:
                tags = get_tags(src=src, bidx=bidx, ns=self._ns)
                _b = Band(source=self, bidx=bidx, tags=tags)
                bands.append(_b)
        return bands

    def _get_source(self, *args, **kwargs):
        """Partially evaluate `rasterio.open` by passing fp and further kwargs

        .. note::

          If the file is opened in writing mode, the profile is injected
          into rio.open
        """
        # is_needed (internal only)
        # no_work
        # not_tested
        # usedin_both (potentially)
        mode = kwargs.get('mode', args[0] if len(args) else self._mode_default)
        assert mode in self._modes
        if self.path.suffix in ['.tif', ]:
            if mode in self._mode_writing:
                src_open = partial(rio.open, fp=self.path, **self.profile)
            else:
                src_open = partial(rio.open, fp=self.path)
        else:
            raise UnknownExtensionError(
                f'"{self.path.suffix}" is not supported.\nCurrently only '
                '.tif" is.'
            )
        return src_open(*args, **kwargs)

    @contextmanager
    def open(self, *args, **kwargs):
        f"""Opens the file for I/O operations.

        .. note::
          If the file is openend in writing mode the profile is ijected into the open function.

          Therefore: **the profile needs to be set when calling this method with writing mode!**

        """
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_both
        src = self._get_source(*args, **kwargs)
        try:
            yield src
        finally:
            src.close()

    @contextmanager
    def data_reader(self, bands:list[Band]|None=None, **kwargs):
        """Read out from mulitple bands and return a 3D data array

        Parameters
        ----------
        bands:
            Collection of Band objects.
        """
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_both
        mode = kwargs.pop('mode', 'r')
        if bands is None:
            bands = self.get_bands()
        bidxs = [band.get_bidx() for band in bands]
        with self.open(mode=mode, **kwargs) as src:
            yield partial(src.read, indexes=bidxs)

    @property
    def band_indexes(self,):
        # is_needed (only internally)
        # needs_work (docs, make internal?)
        # not_tested
        # usedin_both (potentially)
        with self.open() as src:
            bidxs = src.indexes
        return bidxs

    def has_bidx(self, bidx:int)->bool:
        # is_needed (only internally)
        # needs_work (docs; make internal?)
        # not_tested
        # usedin_both (potentially)
        has_it = False
        with rio.open(self.path, 'r') as src:
            if bidx in src.indexes:
                has_it = True
        return has_it

    def has_tags(self, tags:dict)->bool:
        # not_needed (might be useful if working with tags
        # needs_work (doc)
        # not_tested
        # usedin_both (potentially)
        all_tags = []
        with rio.open(self.path, 'r') as src:
            for bidx in src.indexes:
                all_tags.append(get_tags(src=src, bidx=bidx, ns=self._ns))
        return any(match_all(tags, btags) for btags in all_tags)

    def find_indexes(self, tags:dict, mode='all')->list:
        """Check if one or several bands have matching tags
        """
        # is_needed (only internally)
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially)
        with self.open() as src:
            if mode=='any':
                # TODO: match_any was implemented in !41
                bidxs = []
            else:
                bidxs = find_bidxs(src=src, ns=self._ns, **tags)
        return bidxs

    def find_index(self, tags:dict)->int|None:
        # is_needed (only internally)
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially)
        midx = None
        matching_bidxs = self.find_indexes(tags=tags, mode='all')
        if len(matching_bidxs) != 1:
            print('WARNING: no matching index found')
        else:
            midx = matching_bidxs[0]
        return midx

    def has_band(self, band:Band)->int:
        # not_needed (might be useful if working with tags
        # needs_work (doc)
        # not_tested
        # usedin_both (potentially)
        has_it = False
        if band.bidx is not None:
            has_it = self.has_bidx(band.bidx)
        elif band.tags:
            midx = self.find_index(tags=band.tags)
            if midx is not None:
                has_it = True
        return has_it

    def get_bidx(self, band:Band)->int|None:
        """Find an specific band to write to in the file
        """
        # is_needed (only internally)
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially
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
                        "WARNING: The tags are not unique, several bands "
                        "share them"
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

    def compress(self, output:str|None=None, compression:str|None='lzw', keep_original:bool=False):
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_both
        uncompressed = self.path
        # create a compressed file
        self.path = Path(compress_tif(str(self.path),
                                      output=output,
                                      compression=compression))
        # remove uncompressed file:
        if not keep_original and uncompressed != self.path:
            os.remove(uncompressed)

    def check_compatibility(self, *sources: Source):
        """Make sure the provided bands are compatible with this one

        See `helper.check_compatibility` for details

        """
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_both
        _sources = {self.path,}
        for source in sources:
            _sources.add(source.path)
        return _check_compatibility(*_sources)

    def load_block(self,
                   view:None|tuple[int,int,int,int]=None,
                   scaling_params:dict|None=None,
                   **tags)->dict:
        """Get a block from a specific band along with the transform

        See `io.load_block` for further details
        """
        # is_needed (internal only)
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially)
        return load_block(source=str(self.path),
                          view=view,
                          scaling_params=scaling_params,
                          **tags)

@dataclass
class Band:
    # is_needed
    # needs_work (mostly docs)
    # is_tested (partially)
    # usedin_both
    source: Source
    tags: dict = field(default_factory=dict)
    bidx: int|None = None
    read_params: dict = field(default_factory=dict)
    _use_mask: str = 'self'
    _ns = NS

    def __repr__(self):
        # is_needed
        # needs_work (docs - revisit what is printed)
        # not_tested (no need)
        # usedin_both
        items = [f"tags={self.tags}", ]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self):
        # is_needed
        # no_work
        # not_tested (no need)
        # usedin_both
        return hash((self.bidx, *(self.tags.values())))
            
    @property
    def source_exists(self)->bool:
        # is_needed (internal only)
        # needs_work (docs; make internal?)
        # not_tested
        # usedin_both (potentially)
        return self.source.exists

    @property
    def index_exists(self,)->bool:
        # is_needed (internally)
        # needs_work (docs; make internal)
        # not_tested
        # usedin_both (io module only)
        i_exists = False
        if self.bidx is None:
            print(f"No index set for {self}")
        else:
            i_exists = self.source.has_bidx(self.bidx) 
        return i_exists

    @property
    def status(self):
        # not_needed (might be useful?)
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially)
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
                print("- TODO")
            else:
                print(
                    f"# Not tags are set for this band"
                )
        print("###")

    def _pair_operation(self, pair_op: Callable, band, out_band=None, **op_kwargs):
        # TODO: important note, when out_band is used it does not create a new band (if you target Raster has only
        # one band (bidx 2 does not exist) using out_band will not work as you will not create an extra band here, we
        # may want to implement this.
        """Internal method for performing operations on data arrays
        """
        # is_needed (only internally)
        # needs_work (docs)
        # not_tested (should be as it is a generic method)
        # usedin_both (potentially)
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

    def add(self, band, out_band:None|Band=None, **add_kwargs):
        """Add another band to this one

        This performs the numpy.add operation between the data of the two bands
        and stores the resulting data back in the source of this band.

        Parameters
        ----------
        bans:
          A band object to add
        out_band:
          Optional destination band to store the data in.
          If not provided, then self is used.
        """
        # not_needed (though useful)
        # no_work
        # is_tested
        # usedin_both (potentially)
        return self._pair_operation(pair_op=np.add, band=band,
                                    out_band=out_band, **add_kwargs)

    def subtract(self, band, out_band:None|Band=None, **add_kwargs):
        """Subtract another band from this one

        This performs the numpy.add operation with the data of this band
        and the negative version of the data form the other band
        and stores the resulting data back in the source of this band.

        Parameters
        ----------
        bans:
          A band object to subtract
        out_band:
          Optional destination band to store the data in.
          If not provided, then self is used.
        """
        # not_needed (though useful)
        # no_work
        # is_tested
        # usedin_both (potentially)
        def _subtract(data1, data2, **kwargs):
            return np.add(data1, (-1)*data2, **kwargs)
        return self._pair_operation(pair_op=_subtract, band=band,
                                    out_band=out_band, **add_kwargs)

    def export_tags(self, match:str|list|None=None):
        """Write the defined tags to the band

        Parameters
        ----------
        match:
          Optional selection of tags to identify a matching band.
          If provided, the routine tries to find a single band in the
          source file for which only the tags specified in this list
          have matching values.
          It can be used if you want to export some new tags or if you
          have updated some tags and want to export these new values.

          .. Example::

             To identify the layer in the source via the value of the
             `category` tag and then update all (other) tags:
             ```python
             b1.export_tags(match='category')
             ```

          ..Note::
            If the band has the `bidx` attribute set, `match` will be ignored

        """
        # is_needed
        # no_work
        # is_tested
        # usedin_both
        bidx = self.get_bidx(match=match)
        self.source.set_tags(bidx=bidx, tags=self.tags)

    def import_tags(self, match:str|list|None=None, keep:bool=True):
        """Get the tags form the source file

        Parameters
        ----------
        keep:
          If set to true the `tags` are simply updated with the tags form the
          file. `keep=False` will empty the `tags` before fetching them from
          the source.
        """
        # is_needed (only used in tests)
        # needs_work (docs)
        # is_tested
        # usedin_both
        bidx = self.get_bidx(match=match)
        tags = self.source.get_tags(bidx)
        if keep:
            self.tags.update(tags)
        else:
            self.tags = tags

    def get_bidx(self, match:str|list|None=None)->int:
        """Compare with the source to get the correct band index.

        If the attribute `bidx` is set, it is simply checked if this index
        exists in the source file. If `bidx` is unset (i.e. equal to `None`)
        then the routine tries to infer the right band index based on the
        `tags` attribute. In this case it is possible to limit the considered
        tags to a single tag or a selection of tags specified by the optional
        argument `match`.

        ..Note::
          A `BandSelectionNoMatchError' is raised if there is not clear match
          with a band from the source file.

        Parameters
        ----------
        match:
          Optional selection of tags to identify a matching band.
          If provided, the routine tries to find a single band in the
          source file for which only the tags specified in this list
          have matching values.
          It can be used if you want to export some new tags or if you
          have updated some tags and want to export these new values.

          .. Example::

             To identify the layer in the source via the value of the
             `category` tag and then update all (other) tags:
             ```python
             b1.export_tags(match='category')
             ```

          ..Note::
            If the band has the `bidx` attribute set, `match` will be ignored

        Returns
        -------
        bidx:
            The index of the band in the source that matches this band
        """
        # is_needed
        # no_work
        # not_tested
        # usedin_both (potentially)
        failed = ''
        bidx = None
        if self.bidx is not None:
            if self.index_exists:
                bidx = self.bidx
            else:
                failed += f"The band has {self.bidx=} which is not present " \
                          f"in  the source file \n'{self.source.path}'\n"\
                          f"Present are:\n{self.source.band_indexes}\n"
        else:
            if isinstance(match, int):
                match = [match,]
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
    
    def init_source(self, profile:dict, overwrite:bool=False, **kwargs):
        """Create or accesses source file

        Parameters
        ----------
        overwrite:
          Determines if an existing file should be overwritten

          ..Note::
            Setting this to `True` is equivalent to deleting the existing source
            and creating a new file.
        profile:
          Specifies the profile of the data set (see `rasterio.profile` for an
          example)
        """
        # is_needed
        # needs_work (docs)
        # not_tested (but used in tests)
        # usedin_both (potentially)
        self.source.profile.update(profile)
        return self.source.init_source(overwrite=overwrite, **kwargs)

    def get_data(self, **kwargs)->NDArray:
        """Read band out from the file

        With the exception of `okwargs` all keyword arguments are passed to
        the `read` method of the source.

        Parameters
        ----------
        **kwargs:
          Optional set of keyword arguments to pass to the `read` method of the source.
          Notable exception:

          `okwargs`: dict
            These arguments will be passed to the `open` method of the source
        """
        # is_needed
        # no_work
        # is_tested
        # usedin_both (potentially)
        okwargs = kwargs.pop('okwargs', dict())
        with self.source.open(mode='r', **okwargs) as src:
            data = src.read(indexes=self.source.get_bidx(band=self), **kwargs)
        return data

    @property
    def shape(self):
        """Get the np.array shape of this band
        """
        # is_needed (not sure - check)
        # needs_work (this should be renamed to avoid confusion w np.array.shape
        # not_tested
        # usedin_both (potentially, if used at all)
        return self.source.shape


    def count_valid_pixels(self, selector:NDArray|None, no_data:Union[int,float],
                           limit_count:int=0):
        """Count the number of valid pixels under some selector mask

        Parameters
        ----------
        selector:
          A boolean array in the same shape as the data stored in the band
        no_data:
          Value of a cell considered as invalid value.
        limit_count:
          An optional number to use as limit count. If set, then this method
          returns True/False if the count of valid cells is bigger/smaller than
          this limit value.
        """
        # is_needed
        # needs_work (check if definition in helper or io and import here)
        # is_tested
        # usedin_both
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


    def get_min_max(self, no_data:Union[int,float], selector:NDArray|None=None):
        """Get the minimum and maximum values in a band
        Parameters
        ----------
        selector:
          A boolean array in the same shape as the data stored in the band
        no_data:
          Value of a cell considered as invalid value.
        """
        # not_needed (useful?)
        # no_work
        # is_tested
        # usedin_both (potentially)
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
    def data_writer(self, match:str|list|None=None, **kwargs):
        """

        Parameters
        ----------
        **kwargs:
          Optional keword arguments that will be passed to `rasterio.io.DatasetWriter.write`
        """
        # is_needed
        # needs_work (docs)
        # not_tested (used in tests)
        # usedin_both
        mode = kwargs.pop('mode', 'r+' if self.source.exists else 'w')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.write, indexes=bidx)

    @contextmanager
    def data_reader(self, match:str|list|None=None, **kwargs):
        """

        Parameters
        ----------
        **kwargs:
          Optional keword arguments that will be passed to `rasterio.io.DatasetReader.read`
        """
        # is_needed
        # needs_work (docs; tests)
        # not_tested
        # usedin_both
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.read, indexes=bidx)

    def set_mask_reader(self, use:str='band'):
        """Set what mask should be used, if at all

        Parameters
        ----------
        use:
          Determines the mask to use.

          Options are:

          `'self'` or `'band'`:
            The band mask should be used
          `'source'`:
            The mask of the source file should be used,
            i.e. either `nodata` or the associated mask bandl
          `'mask_none'`:
            This instructs to consider all pixels as valid data, i.e.
            an array containg all `1`s is returned
          `'mask_all'`:
            This simply assumes all values are invalid, i.e. an array
            containing only `0`s is returend.
            It is likely only useful in some edge-cases
        """
        # is_needed (only in tests)
        # needs work(doc)
        # not_tested (used in tests)
        # usedin_both (potentially)
        assert use in ['self', 'band', 'source', 'mask_all', 'mask_none'], \
            f'"{use}" is an invalid selector for a mask, options are:' \
            '\n\t- "band": uses the bands own mask (i.e. ' \
            'rasterio.io.DataReader.read_masks)\n\t- "source": uses the ' \
            'dataset mask (i.e. rasterio.io.DataReader.dataset_mask)'
        if use in ['self', 'band']:
            self._use_mask = 'self'
        else:
            self._use_mask = use
            
        
    def get_mask_reader(self,):
        """Return the mask reader for this band.

        By default mask reader is `rasterio.io.DatasetReader.read_masks` with the corresponding
        band index set. However, other readers can be specified. See `set_mask_reader` for more
        details.

        """
        # is_needed
        # needs_work (docs)
        # is_tested
        # usedin_both

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
    def mask_reader(self, match:str|list|None=None, **kwargs):
        """Get a read method for the band mask.

        ..Note::

          This yields always the band specific mask reader (i.e. the
          'rasterio.io.DatasetReader.read_masks(indexes=<bidx of self>)`

        """
        # is_needed (only internally)
        # needs_work (docs; make internal?)
        # not_tested
        # usedin_both (potentially)
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)
        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(src.read_masks, indexes=bidx)

    @contextmanager
    def _mask_full(self, match:str|list|None=None, fill_value:int|float|bool=False,
                   **kwargs):
        """Mocked maks read method of band mask returning all `True`/`False`

        The mocked read method first calls the normal band read method 
        (i.e. rasterios `read_masks`) in order to assure than transformation,
        rescaling, window, etc. is performed correctly.
        It then returns a similar numpy array holding exclusively
        `True` or `False` as values effectively ignoring the actual mask

        ..Note::

          This uses always the band specific mask reader (i.e. the
          'rasterio.io.DatasetReader.read_masks(indexes=<bidx of self>)`

        Parameters
        ----------
        value:
          Either `True` or `False` (but also numbers are accepted) that will be used
          in the array

        """
        # is_needed (only internally)
        # needs_work (make internal)
        # not_tested
        # usedin_both (io module)
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)

        def _mock_all(mask_reader:Callable, *args, **kwargs)->NDArray:
            _mask = mask_reader(indexes=bidx, *args, **kwargs)
            full_array =  np.full(shape=_mask.shape, fill_value=fill_value)
            del _mask
            return full_array

        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(_mock_all, mask_reader=src.read_masks)


    def set_data(self, data:NDArray, overwrite=False, **kwargs):
        """Write out the data from a band
        """
        # is_needed (tests only)
        # needs_work (docs)
        # not_tested
        # usedin_both (io module)

        # with self.source.open(mode='w', **kwargs) as src:
        if self.source.exists and not overwrite:
            mode = 'r+'
        else:
            mode = 'w'
        with self.data_writer(mode=mode, **kwargs) as write:
            write(data)

    def load_block(self,
                   view:None|tuple[int,int,int,int]=None,
                   scaling_params:dict|None=None,
                   match:str|list|None=None)->dict:
        """Get a block from a specific band along with the transform

        See `io.load_block` for further details
        """
        # is_needed
        # needs_work (docs)
        # not_tested
        # usedin_both (potentially)
        bidx = self.get_bidx(match=match)
        return self.source.load_block(
                          view=view,
                          scaling_params=scaling_params,
                          indexes=bidx)
