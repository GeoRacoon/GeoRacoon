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
from .helper import (
    check_compatibility as _check_compatibility,
    count_contribution,
)


class Source:
    """Specifies a data source
    """
    def __init__(self, path:str|Path,
                 tags: dict|None=None,
                 profile: dict|None=None,
                 ns: str=NS):
        self.path = Path(path)
        self.tags = tags or dict()
        self._ns = ns
        self.profile = profile or dict()

    def __repr__(self):
        items = [f"path={str(self.path)}", f"exists: { self.exists }"]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self):
        return hash((self.path, self._ns, *(self.tags.values())))

    def __eq__(self, other):
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
        with self.open(mode='r') as src:
            profile = src.profile
        if update_self:
            self.profile.update(profile)
        return self.profile

    @property
    def exists(self)->bool:
        return self.path.is_file()
    
    def get_tags(self, bidx:int)->dict:
        with self.open(mode='r') as src:
            tags = get_tags(src=src, bidx=bidx, ns=self._ns)
        return tags

    def get_tag_values(self, tag:str, bidx:int|list|None=None)->dict:
        """Try to fetch for each band the value of this tag
        
        If the tag is not present, None is returned
        """
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
        with self.open(mode='r+') as src:
            set_tags(src=src, bidx=bidx, ns=self._ns, **tags)

    @contextmanager
    def mask_reader(self, **kwargs):
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
        okwargs = kwargs.pop('okwargs', dict())
        with self.mask_reader(mode='r', **okwargs) as dataset_mask:
            mask = dataset_mask(**kwargs)
        return mask

    @contextmanager
    def mask_writer(self, **kwargs):
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
        with self.open(mode='r+') as src:
            src.write_mask(mask_array=mask, window=window)

    def init_source(self, overwrite:bool=False, **kwargs):
        """Create or accesses source file
        """
        if overwrite or not self.exists:
            with self.open(mode='w', **self.profile, **kwargs) as _:
                print(f'Initiating empty file\n\t"{self.path}"\n')

    def get_band(self, bidx:int|None=None, **tags)->Band:
        """Find the wanted band and return a related band object
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
        bands = []
        with self.open(mode='r') as src:
            for bidx in src.indexes:
                tags = get_tags(src=src, bidx=bidx, ns=self._ns)
                _b = Band(source=self, bidx=bidx, tags=tags)
                bands.append(_b)
        return bands

    def _get_source(self, *args, **kwargs):
        if self.path.suffix in ['.tif', ]:
            src_open = partial(rio.open, fp=self.path)
        else:
            raise UnknownExtensionError(
                f'"{self.path.suffix}" is not supported.\nCurrently only '
                '.tif" is.'
            )
        return src_open(*args, **kwargs)

    @contextmanager
    def open(self, *args, **kwargs):
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
        mode = kwargs.pop('mode', 'r')
        if bands is None:
            bands = self.get_bands()
        bidxs = [band.get_bidx() for band in bands]
        with self.open(mode=mode, **kwargs) as src:
            yield partial(src.read, indexes=bidxs)

    @property
    def band_indexes(self,):
        with self.open() as src:
            bidxs = src.indexes
        return bidxs

    def has_bidx(self, bidx:int)->bool:
        has_it = False
        with rio.open(self.path, 'r') as src:
            if bidx in src.indexes:
                has_it = True
        return has_it

    def has_tags(self, tags:dict)->bool:
        all_tags = []
        with rio.open(self.path, 'r') as src:
            for bidx in src.indexes:
                all_tags.append(get_tags(src=src, bidx=bidx, ns=self._ns))
        return any(match_all(tags, btags) for btags in all_tags)

    def find_indexes(self, tags:dict, mode='all')->list:
        """Check if one or several bands have matching tags
        """
        with self.open() as src:
            if mode=='any':
                # TODO: match_any was implemented in !41
                bidxs = []
            else:
                bidxs = find_bidxs(src=src, ns=self._ns, **tags)
        return bidxs

    def find_index(self, tags:dict)->int|None:
        midx = None
        matching_bidxs = self.find_indexes(tags=tags, mode='all')
        if len(matching_bidxs) != 1:
            print('WARNING: no matching index found')
        else:
            midx = matching_bidxs[0]
        return midx

    def has_band(self, band:Band)->int:
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

    def compress(self, output=None):
        uncompressed = self.path
        # create a compressed file
        self.path = Path(compress_tif(str(self.path), output=output))
        # remove uncompressed file:
        if uncompressed != self.path:
            os.remove(uncompressed)

    def check_compatibility(self, *sources: Source):
        """Make sure the provided bands are compatible with this one

        See `helper.check_compatibility` for details

        """
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
        return load_block(source=str(self.path),
                          view=view,
                          scaling_params=scaling_params,
                          **tags)

@dataclass
class Band:
    source: Source
    tags: dict = field(default_factory=dict)
    bidx: int|None = None
    read_params: dict = field(default_factory=dict)
    _use_mask: str = 'self'
    _ns = NS

    def __repr__(self):
        items = [f"tags={self.tags}", ]
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __hash__(self):
        return hash((self.bidx, *(self.tags.values())))
            
    @property
    def source_exists(self)->bool:
        return self.source.exists

    @property
    def index_exists(self,)->bool:
        i_exists = False
        if self.bidx is None:
            print(f"No index set for {self}")
        else:
            i_exists = self.source.has_bidx(self.bidx) 
        return i_exists

    @property
    def status(self):
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
        okwargs = kwargs.pop('okwargs', dict())
        with self.source.open(mode='r', **okwargs) as src:
            data = src.read(indexes=self.source.get_bidx(band=self), **kwargs)
        return data

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

    @contextmanager
    def data_writer(self, match:str|list|None=None, **kwargs):
        """

        Parameters
        ----------
        **kwargs:
          Optional keword arguments that will be passed to `rasterio.io.DatasetWriter.write`
        """
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

        if self._use_mask is None or self._use_mask == 'self':  # read the band mask
            return self.mask_reader
        elif self._use_mask == 'source':  # read the dataset mask
            return self.source.mask_reader
        elif self._use_mask == 'mask_none':
            return partial(self._mask_full, fill_value=0)
        elif self._use_mask == 'mask_all':
            return partial(self._mask_full, fill_value=1)
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
        mode = kwargs.pop('mode', 'r')
        bidx = self.get_bidx(match=match)

        def _mock_all(mask_reader:Callable, *args, **kwargs)->NDArray:
            _mask = mask_reader(indexes=bidx, *args, **kwargs)
            full_array =  np.full(shape=_mask.shape, fill_value=fill_value)
            del _mask
            return full_array

        with self.source.open(mode=mode, **kwargs) as src:
            yield partial(_mock_all, mask_reader=src.read_masks)


    def set_data(self, data:NDArray, **kwargs):
        """Read out the data from a band
        """
        # with self.source.open(mode='w', **kwargs) as src:
        with self.data_writer(mode='w', **kwargs) as src:
            src.write(data)

    def load_block(self,
                   view:None|tuple[int,int,int,int]=None,
                   scaling_params:dict|None=None,
                   match:str|list|None=None)->dict:
        """Get a block from a specific band along with the transform

        See `io.load_block` for further details
        """
        bidx = self.get_bidx(match=match)
        return self.source.load_block(
                          view=view,
                          scaling_params=scaling_params,
                          indexes=bidx)
