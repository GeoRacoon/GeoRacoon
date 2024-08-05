from __future__ import annotations
import os

import rasterio as rio
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from numpy.typing import NDArray

from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
)

from .io import (
    NS,
    get_tags,
    set_tags,
    get_bidx,
    match_all,
    find_bidxs,
    compress_tif,
)

class Source:
    """Specifies a data source
    """
    def __init__(self, path:str|Path,
                 tags: dict|None=None,
                 ns: str=NS):
        self.path = Path(path)
        self.tags = tags or dict()
        self._ns = ns

    def __repr__(self):
        items = [f"path={str(self.path)}", f"exists: { self.exists }"]
        return "{}({})".format(type(self).__name__, ", ".join(items))


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
            for _bidx in src.indexes:
                t_vals[_bidx] = get_tags(src=src,
                                         bidx=_bidx,
                                         ns=self._ns).get(tag, None)
        return t_vals

    def set_tags(self, bidx:int|None, tags:dict):
        with self.open(mode='r+') as src:
            set_tags(src=src, bidx=bidx, ns=self._ns, **tags)

    def extract_band(self, bidx:int|None=None, **tags)->Band:
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
        # TODO: keep the source's own tag
        # keep the band tags
        bands = []
        for bidx in self.band_indexes:
            bands.append(self.extract_band(bidx=bidx))
        # create a compressed file
        self.path = Path(compress_tif(str(self.path), output=output))
        # TODO: export the source's own tag
        # export the band tags
        for band in bands:
            band.export_tags()
        # remove uncompressed file:
        if uncompressed != self.path:
            os.remove(uncompressed)


@dataclass
class Band:
    source: Source
    tags: dict = field(default_factory=dict)
    bidx: int|None = None
    masked: bool = False  # we might want to get a masked array
                          # then return data and the mask directly
    read_params: dict = field(default_factory=dict)
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
        if overwrite or not self.source.exists:
            with self.source.open(mode='w', **profile, **kwargs) as _:
                print(f'Initiating empty file\n\t"{self.source.path}"\n')

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
        if self.masked:
            # separate data and mask
            # TODO
            pass
        return data

    @contextmanager
    def data_writer(self, match:str|list|None=None, *args, **kwargs):
        bidx = self.get_bidx(match=match)
        with self.source.open(*args, **kwargs) as src:
            yield partial(src.write, indexes=bidx)

    def set_data(self, data:NDArray, **kwargs):
        """Read out the data from a band
        """
        # with self.source.open(mode='w', **kwargs) as src:
        with self.data_writer(mode='w', **kwargs) as src:
            src.write(data)
                      
