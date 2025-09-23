import pytest
import os
import rasterio as rio
import numpy as np

from .conftest import ALL_MAPS, get_example_data, get_file

from riogrande.exceptions import (
    BandSelectionAmbiguousError,
    BandSelectionNoMatchError
)
from riogrande.helper import check_compatibility
from riogrande import io as rgio
from riogrande import io_ as rgio_


def test_load_block():
    """This is just a smoketest"""
    with pytest.raises(rio.RasterioIOError):
        rgio.load_block(source='non-existing',
                        view=(0, 0, 10, 10))


@ALL_MAPS
def test_resampling(datafiles):
    # TODO: We really should consider whether we want to keep this in the RioGrande or not.
    #  It is a nice feature to be able to resample and coregister rasters - but actually,
    #  it is cryptic the way it is now, (resampling method etc not specifyable).
    """Make sure our re-sampling method works as expected.
    """
    landcover_map = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ndvi_map = get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    # make sure the compatibility check fails
    with pytest.raises(TypeError):
        check_compatibility(ndvi_map, landcover_map)
    # re-sample the landcover_map to match the resolution of the ndvi_map
    rgio._coregister_raster(landcover_map, ndvi_map, output=str(landcover_map))
    # now check that the shape of the data actually matches
    with rio.open(ndvi_map, 'r') as src:
        # get the shape and the projection
        ndvi_profile = src.profile.copy()
        ndvi_data = src.read(indexes=1)
    with rio.open(landcover_map, 'r') as src:
        # get the shape and the projection
        lc_profile = src.profile.copy()
        lc_data = src.read(indexes=1)
    # assert ndvi_profile == lc_profile
    assert ndvi_data.shape == lc_data.shape
    # finally, check again
    check_compatibility(ndvi_map, landcover_map)

@ALL_MAPS
def test_band_tagging(datafiles):
    """Test how reading and writing of tags work
    """
    our_namespace = "LANDIV"
    gen_tag1 = dict(
        info = 'This is a test map'
    )
    gen_tag2 = dict(
        some_value = 2.333
    )
    band_tags = {
        1: dict(category=0, some='value'),
        2: dict(category=100, extra={"bla": 2, "blu": [1,2,4]}),
        3: dict(category=5, beta=2.3, some='value')
    }
    # create some tif with 3 bands:
    outfile = datafiles / 'out.tif'
    bands, profile = get_example_data(bands=3)
    with rio.open(outfile, 'w', **profile) as src:
        for i, band in enumerate(bands, start=1):
            src.write(band, indexes=1)
    def print_tag_details(src, ns=None):
        print(f"{src.tags(ns=ns)=}")
        print(f"{src.tag_namespaces()=}")
        print(f"{src.indexes=}")
        for idx in src.indexes:
            band = src.read(indexes=idx)
            print(f"band {idx}:")
            print(f"\t{src.tags(bidx=idx, ns=ns)=}")
            print(f"\t{src.tag_namespaces(bidx=idx)=}")
    # read the tif
    with rio.open(outfile, 'r') as src:
        default_ns_tags = src.tags()
        # print('\nWithout any tags set:')
        # print_tag_details(src)
    # now we add a namespace and some tags
    # print(f'\nNow we set a new namespace "{our_namespace}" and ')
    # print('in there the tags:\n')
    # print(f'- General:\n\t{gen_tag1}\n\t{gen_tag2}')
    # print('- Per band:')
    # for bidx, b_tags in band_tags.items():
    #     print(f'\t{bidx=}:\n\t\t{b_tags}')
    with rio.open(outfile, 'r+') as src:
        # file wide tags
        # src.update_tags(ns=our_namespace, **gen_tag1)
        rgio._set_tags(src=src, **gen_tag1)
        # src.update_tags(ns=our_namespace, **gen_tag2)
        rgio._set_tags(src=src, **gen_tag2)
        # now we add tags per band
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            rgio._set_tags(src=src, bidx=idx, **band_tags[idx])
    # read the tif
    # print('\nAnd now we read the tags again form the tif:')
    with rio.open(outfile, 'r') as src:
        assert src.tags() == default_ns_tags
        assert src.tags(ns=our_namespace) != default_ns_tags
        # print('First without specifying the namespace:')
        # print_tag_details(src)
        # print(f'\nAnd now in NAMESPACE "{our_namespace}":')
        # print_tag_details(src, ns=our_namespace)

@ALL_MAPS
def test_tag_matching(datafiles):
    """Test how reading and writing of tags work
    """
    our_namespace = "LANDIV"
    gen_tag1 = dict(
        info = 'file1',
        some_value = 4.3
    )
    gen_tag2 = dict(
        info = 'file2',
        some_value = 2.333,
        some = 'value'
    )
    band_tags = {
        1: dict(category=0, some='value'),
        2: dict(category=100, extra={"bla": 2, "blu": [1,2,4]}),
        3: dict(category=5, beta=2.3, some='value')
    }
    # create some tif with 3 bands:
    outfile1 = datafiles / 'out1.tif'
    outfile2 = datafiles / 'out2.tif'
    bands, profile = get_example_data(bands=3)
    with rio.open(outfile1, 'w', **profile) as src:
        for i, band in enumerate(bands, start=1):
            src.write(band, indexes=1)
    with rio.open(outfile2, 'w', **profile) as src:
        for i, band in enumerate(bands, start=1):
            src.write(band, indexes=1)
    # now we add a namespace and some tags to the first file
    with rio.open(outfile1, 'r+') as src:
        # file wide tags
        # src.update_tags(ns=our_namespace, **gen_tag1)
        rgio._set_tags(src=src, **gen_tag1)
        # now we add tags per band
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            rgio._set_tags(src=src, bidx=idx, **band_tags[idx])
    # now to the second file
    band_tags[2]['category'] = 101  # change one category to not match
    with rio.open(outfile2, 'r+') as src:
        # file wide tags
        rgio._set_tags(src=src, **gen_tag2)
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            rgio._set_tags(src=src, bidx=idx, **band_tags[idx])
    # now we check the matching:
    with rio.open(outfile1, 'r') as src:
        # category matches to band 1:
        _bidx = 1
        _category = band_tags[_bidx]['category']
        assert rgio._get_bidx(src=src,
                             category=_category) == _bidx
        # category 222 does not exist
        with pytest.raises(BandSelectionNoMatchError):
            rgio._get_bidx(src=src,
                          category=222)
        with pytest.raises(BandSelectionAmbiguousError):
            rgio._get_bidx(src=src, some='value')
        # check that serialization works
        _bidx = 2
        _tag = 'extra'
        _extra = band_tags[_bidx][_tag]
        _bla = _extra['bla']  # this is an int
        _blu = _extra['blu']  # this is a list
        # we read the serialized tags from the tif and convert them
        tags = rgio._get_tags(src=src, bidx=_bidx)
        assert tags[_tag]['bla'] == _bla
        assert tags[_tag]['blu'] == _blu
    # now test the get_bands
    sources = str(datafiles / 'out*.tif')
    # find all some: 'value' tags:
    some_value = rgio.get_bands(source=sources, some='value')
    assert (str(outfile2), None) in some_value  # the tag in the dataset
    # the bands
    assert (str(outfile1), 1) in some_value
    assert (str(outfile1), 3) in some_value
    assert (str(outfile2), 1) in some_value
    assert (str(outfile2), 3) in some_value

@ALL_MAPS
def test_tif_compression(datafiles):
    """Test whether compression produces correct ouput and transfers tags
    """
    test_data = (
        get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles),
        get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    )
    for file in test_data:
        # decompress it
        file_decompressed = rgio.compress_tif(file, compression=None)
        decompressed_size = os.path.getsize(file_decompressed)
        # compress it
        file_compressed = rgio.compress_tif(file_decompressed)
        compressed_size = os.path.getsize(file_compressed)
        assert decompressed_size > compressed_size
        # decompress it again
        file_re_decompressed = rgio.compress_tif(file_compressed, compression=None)
        re_decompressed_size = os.path.getsize(file_re_decompressed)
        assert decompressed_size == re_decompressed_size
        # replace file:
        _ = rgio.compress_tif(file_re_decompressed, output=file_re_decompressed)
        replaced_size = os.path.getsize(file_re_decompressed)
        # make sure the original file changed
        assert replaced_size < re_decompressed_size

@ALL_MAPS
def test_compression_tagging(datafiles):
    """Test whether compression transfers all tags corerctly
    """
    test_data = (
        get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles),
        get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    )
    dataset_tags = dict(
        ds_tag='test'
    )
    for file in test_data:
        orig_file_tagged = rgio.output_filename(file, "orig_tagged")
        # create file copy with tags
        target = {}
        with rio.open(file) as src:
            profile = src.profile
            with rio.open(orig_file_tagged, 'w', **profile) as dst:
                rgio._set_tags(dst, **dataset_tags)
                for bidx in range(1, src.count + 1):
                    dst.write(src.read(bidx), bidx)
                    rgio._set_tags(dst, bidx, category=np.random.randint(low=0, high=255))
                    target[bidx] = rgio._get_tags(src=dst, bidx=bidx)
        file_tagged = rgio.output_filename(file, "tagged")
        # uncompress it
        file_tagged = rgio.compress_tif(orig_file_tagged, compression=None)

        # compress file and check if tags match
        file_compress = rgio.compress_tif(file_tagged)
        test = {}
        with rio.open(file_compress) as src:
            dataset_tags_copied = rgio._get_tags(src=src)
            for bidx in range(1, src.count + 1):
                tags = rgio._get_tags(src=src, bidx=bidx)
                test[bidx] = tags

        assert dataset_tags_copied == dataset_tags
        assert len(target) == len(test)
        assert target == test

@ALL_MAPS
def test_band_count_contrib(datafiles):
    """Check the count of valid pixels for a band
    """
    test_data = (
        get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles),
        get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles)
    )
    for test_file in test_data:
        band = rgio_.Band(source=rgio_.Source(path=test_file), bidx=1)
        valids = band.count_valid_pixels(selector=None, no_data=0)
        print(f"{valids=}")
        assert isinstance(valids, int)
        limit_count = int(0.5*valids)
        valid = band.count_valid_pixels(selector=None, no_data=0,
                                         limit_count=limit_count)
        assert isinstance(valid, bool)
        assert valid
        limit_count = int(1.5*valids)
        valid = band.count_valid_pixels(selector=None, no_data=0,
                                         limit_count=limit_count)
        assert isinstance(valid, bool)
        assert ~valid
