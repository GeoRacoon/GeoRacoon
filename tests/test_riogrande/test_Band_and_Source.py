import pytest

import rasterio
import numpy as np
import numpy.ma as ma
import rasterio as rio

from riogrande.io_ import Source, Band
from riogrande.io.exceptions import (
    BandSelectionNoMatchError,
)

from .conftest import ALL_MAPS, get_file


@ALL_MAPS
def test_Band_operations(datafiles):
    """Cover some basic operations on Band objects
    """
    test_file = datafiles / 'band_ops_test.tif'
    b1_tags = dict(category=1)
    width = 8
    height = 12
    profile = {
        "count": 3,
        "width": width, "height": height,
        "dtype": "float64",
        "transform": rasterio.Affine(1, 0, 0, 0, 1, 0)
    }
    band = Band(source=Source(test_file), bidx=1, tags=b1_tags)
    band.init_source(profile=profile, overwrite=True)
    band.export_tags()

    # create some data
    init_data = np.full(shape=(height, width), fill_value=0.0)
    band.set_data(init_data)

    # adding some
    add_data = np.full(shape=(height, width), fill_value=0.0)
    add_data[2:4, 2:4] = 1
    adding_band = Band(source=Source(test_file), bidx=2)
    adding_band.set_data(add_data)
    band.add(adding_band)
    np.testing.assert_equal(band.get_data(), add_data)
    band.subtract(adding_band)
    np.testing.assert_equal(band.get_data(), init_data)

    # min max test
    mm_data = np.random.rand(height, width)
    mm_data[2:3, 1:4] = 0
    mm_data[4, 4] = np.nan
    mm_min = np.nanmin(mm_data[mm_data>0])
    mm_max = np.nanmax(mm_data[mm_data>0])
    mm_band = Band(source=Source(test_file), bidx=3)
    mm_band.set_data(mm_data)
    is_min, is_max = mm_band.get_min_max(no_data=0)
    assert mm_min == is_min
    assert mm_max == is_max

    # with another band
    test_file_2 = datafiles / 'band_ops_test_2.tif'
    band_out = Band(source=Source(test_file_2), bidx=1, tags=b1_tags)
    band_out.init_source(profile=profile, overwrite=True)
    band_out.export_tags()
    band.add(adding_band, out_band=band_out)
    np.testing.assert_equal(band_out.get_data(), add_data)
    # orig band should still be the same
    np.testing.assert_equal(band.get_data(), init_data)

@ALL_MAPS
def test_Band_tagging(datafiles):
    test_file = datafiles / 'test.tif'
    b1_tags_0 = dict(category=1)
    profile = {
        "count": 3,
        "width": 10, "height": 10,
        "dtype": "float64",
        "transform": rasterio.Affine(1, 0, 0, 0, 1, 0)
    }
    with rio.open(test_file, 'w+', **profile) as src:
        # setting a tag via rasterio directly
        src.update_tags(1, ns='GEORACOON', **b1_tags_0)
    # create source objcet
    mysrc = Source(path=test_file)
    b1 = mysrc.get_band(bidx=1)
    assert b1.bidx == 1
    assert b1_tags_0 == b1.tags
    # adding a tag to the object
    b1.tags['new_tag'] = 'some_value'
    # propagate tags to the file
    b1.export_tags()
    # get the band in a new object
    b1_1 = mysrc.get_band(1)
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
    b1_3 = mysrc.get_band(1)
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
    b2 = mysrc.get_band(2)
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
    com_b1 = mysrc.get_band(1)
    assert 'extra_tag' in com_b1.tags
    assert com_b1.tags['extra_tag'] == 4.4
    assert 'new_tag' in com_b1.tags
    assert com_b1.tags['new_tag'] == 'some_value'
    assert com_b1.get_bidx(match='category') == com_b1.bidx

@ALL_MAPS
def test_rasterio_band_mask(datafiles):
    """Check if there is a difference between rastreio's band and dataset mask
    """
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    test_file = datafiles / 'test.tif'
    with rio.open(ch_map_tif, 'r+') as src:
        mask_ds0 = src.dataset_mask()
        mask_b1 = src.read_masks(1)
        data = src.read(indexes=1)
        #  print(f"{mask_ds0=}")
        #  print(f"{mask_b1=}")
        profile = src.profile.copy()
    profile['count'] = 2
    masked_band1 = ma.masked_array(data, np.zeros(shape=data.shape))
    masked_band2 = ma.masked_array(data, np.ones(shape=data.shape))
    # print(f"{masked_band1.mask=}")
    # print(f"{masked_band2.mask=}")
    with rio.open(test_file, 'w', **profile) as src:
        src.write(masked_band1, indexes=1, masked=True)  # 0s first
        src.write(masked_band2, indexes=2, masked=True)  # then 1s
    with rio.open(test_file, 'r') as src:
        dsmask0 = src.dataset_mask()
        mask1 = src.read_masks(indexes=1)
        mask2 = src.read_masks(indexes=2)
    # are both masks equal
    np.testing.assert_equal(mask1, mask2)
    # both correspond to the last mask written
    np.testing.assert_equal(mask1, np.zeros(shape=mask1.shape))
    with rio.open(test_file, 'w', **profile) as src:
        src.write(masked_band1, indexes=1, masked=True)  # 0s first
    # check if the other band changed as well
    with rio.open(test_file, 'r') as src:
        dsmask1 = src.dataset_mask()
        mask1 = src.read_masks(indexes=1)
        mask2 = src.read_masks(indexes=2)
    np.testing.assert_equal(mask1, mask2)
    # check if the dataset mask changed as well
    np.testing.assert_equal(dsmask0, np.where(dsmask1==255, 0, 255))
    # No error means changing a band mask changes the dataset mask

@ALL_MAPS
def test_masking(datafiles):
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    test_file = datafiles / 'test.tif'
    source = Source(path=ch_map_tif)
    source.import_profile()  # load profile from file
    # we are going to duplicate the file
    out_source = Source(path=test_file)  # create a new object
    out_source.profile.update(source.profile)  # use the profile
    out_source.profile.update({'count':2})  # updated it
    out_source.init_source()  # create the file (and export the profile)
    # duplicate into two bands
    band1 = source.get_band(bidx=1)
    band2 = source.get_band(bidx=1)
    band2.bidx=2
    band1.source=out_source
    band2.source=out_source
    b1_data = band1.get_data()
    b2_data = band2.get_data()
    # define two oposite masks
    masked_band1 = ma.masked_array(b1_data, np.zeros(shape=b1_data.shape))
    masked_band2 = ma.masked_array(b2_data, np.ones(shape=b2_data.shape))
    # try to write "per band" masks
    with band1.data_writer() as write:
        write(masked_band1, masked=True)
    with band2.data_writer() as write:
        write(masked_band2, masked=True)
    # use the "band mask" for both bands
    band1.set_mask_reader(use='self')
    band2.set_mask_reader(use='self')
    # get the "band-" masks
    b1_mask_reader = band1.get_mask_reader()
    b2_mask_reader = band2.get_mask_reader()
    with b1_mask_reader() as read_mask:
        b1_mask = read_mask()
    with b2_mask_reader() as read_mask:
        b2_mask = read_mask()
    # the masks should be to opposite of each other:
    np.testing.assert_equal(masked_band1.mask, ~masked_band2.mask)
    # also when reading from the file - but it isn't
    np.testing.assert_equal(b1_mask, np.where(b2_mask!=255, 0, 255))

@ALL_MAPS
def test_masking_all_none(datafiles):
    """make sure the not masking and the masking all works as expected"""
    from riogrande.helper import aggregated_selector
    ch_map_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    test_file = datafiles / 'test.tif'
    source = Source(path=ch_map_tif)
    source.import_profile()  # load profile from file
    # we are going to duplicate the file
    out_source = Source(path=test_file)  # create a new object
    out_source.profile.update(source.profile)  # use the profile
    out_source.profile.update({'count':3})  # updated it
    out_source.init_source()  # create the file (and export the profile)
    # duplicate into two bands
    band1 = source.get_band(bidx=1)
    band2 = source.get_band(bidx=1)
    band3 = source.get_band(bidx=1)
    band2.bidx=2
    band3.bidx=3
    band1.source=out_source
    band2.source=out_source
    band3.source=out_source
    b1_data = band1.get_data()
    b2_data = band2.get_data()
    b3_data = band3.get_data()
    # define two oposite masks and a random one
    masked_band1 = ma.masked_array(b1_data, np.zeros(shape=b1_data.shape))
    masked_band2 = ma.masked_array(b2_data, np.ones(shape=b2_data.shape))
    masked_band3 = ma.masked_array(b2_data,
                                   np.random.randint(0, 2,
                                                     size=b2_data.shape,
                                                     dtype=np.bool_))
    with band3.data_writer() as write:
        write(masked_band3, masked=True)
    # use the "band mask" for both bands
    band1.set_mask_reader(use='mask_none')
    band2.set_mask_reader(use='mask_all')
    band3.set_mask_reader(use='source')
    # get the "band-" masks
    b1_mask_reader = band1.get_mask_reader()
    b2_mask_reader = band2.get_mask_reader()
    b3_mask_reader = band3.get_mask_reader()
    with b1_mask_reader() as read_mask:
        b1_mask = read_mask()
    assert set(np.unique(b1_mask)) == {1}
    with b2_mask_reader() as read_mask:
        b2_mask = read_mask()
    assert set(np.unique(b2_mask)) == {0}
    with b3_mask_reader() as read_mask:
        b3_mask = read_mask()
    assert set(np.unique(b3_mask)) == {0, 255}
    # the masks should be to opposite of each other:
    np.testing.assert_equal(masked_band1.mask, ~masked_band2.mask)
    # also when reading from the file - but it isn't
    np.testing.assert_(not np.array_equal(
        b1_mask, b3_mask
        ))
    np.testing.assert_(not np.array_equal(
        b2_mask, b3_mask
        ))
    # make sure `mask_none` gives all Ture selector
    selector_mask_none = aggregated_selector([b1_mask, ], logic='all')
    assert set(np.unique(selector_mask_none)) == {True,}
    selector_mask_all = aggregated_selector([b2_mask, ], logic='all')
    assert set(np.unique(selector_mask_all)) == {False,}
