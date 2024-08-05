import pytest

import numpy as np
import rasterio as rio

from landiv_blur.io_ import Source, Band
from landiv_blur.exceptions import (
    BandSelectionNoMatchError,
)

from .config import ALL_MAPS

@ALL_MAPS
def test_Band_tagging(datafiles):
    test_file = datafiles / 'test.tif'
    b1_tags_0 = dict(category=1)
    with rio.open(test_file, 'w+', width=10, height=10, count=3,
                  dtype=np.float64) as src:
        # setting a tag via rasterio directly
        src.update_tags(1, ns='LANDIV', **b1_tags_0)
    # create source objcet
    mysrc = Source(path=test_file)
    b1 = mysrc.extract_band(bidx=1)
    assert b1.bidx == 1
    assert b1_tags_0 == b1.tags
    # adding a tag to the object
    b1.tags['new_tag'] = 'some_value'
    # propagate tags to the file
    b1.export_tags()
    # get the band in a new object
    b1_1 = mysrc.extract_band(1)
    # make sure the new tag was added
    assert 'new_tag' in b1_1.tags
    assert b1_1.tags['new_tag'] == 'some_value'
    # create a new object without reading from file
    b1_2 = Band(source=mysrc,tags=dict(category=1))
    assert b1_2.bidx is None
    # try to find it matching all tags (i.e. category=1)
    assert b1_2.get_bidx() == b1.bidx
    # add a tag to the object
    b1_2.tags.update(dict(extra_tag=4.4))
    # now matching all tags should fail
    with pytest.raises(BandSelectionNoMatchError):
        b1_2.get_bidx()
    # matching only tag category should work
    assert b1_2.get_bidx(match='category') == b1.bidx
    # make sure exporting tags also works with matching tags
    b1_2.export_tags(match='category')
    # new object with current state in file
    b1_3 = mysrc.extract_band(1)
    # make sure the extra_tag='hello' is there
    assert 'extra_tag' in b1_3.tags
    assert b1_3.tags['extra_tag'] == 4.4
    # make sure the new_tag = 'some_value' was not lost
    assert 'new_tag' in b1_3.tags
    assert b1_3.tags['new_tag'] == 'some_value'
    # update the status of an existing object
    b1.import_tags(match=None, keep=True)
    # make sure we get the new tags
    assert 'new_tag' in b1.tags
    assert b1.tags['new_tag'] == 'some_value'
    # get another band
    b2 = mysrc.extract_band(2)
    # TODO: make sure we cannot select multiple bands with a tag

    # -----------------------------------
    # compress source
    test_file_compressed = datafiles / 'test_compressed.tif'
    mysrc.compress(output=test_file_compressed)
    # updated source path to new file
    assert test_file_compressed == mysrc.path
    # make sure file is compressed
    with rio.open(test_file_compressed) as src:
        prof = src.profile
        assert prof['compress'] == 'lzw'
    # make sure all tags were transferrred
    com_b1 = mysrc.extract_band(1)
    assert 'extra_tag' in com_b1.tags
    assert com_b1.tags['extra_tag'] == 4.4
    assert 'new_tag' in com_b1.tags
    assert com_b1.tags['new_tag'] == 'some_value'
    assert com_b1.get_bidx(match='category') == com_b1.bidx