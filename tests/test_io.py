import pytest
import rasterio as rio
import numpy as np
from skimage.filters import gaussian

from .config import ALL_MAPS, get_example_data
from landiv_blur.exceptions import (
    BandSelectionAmbiguousError,
    BandSelectionNoMatchError
)
from landiv_blur.helper import check_compatibility
from landiv_blur import io as lbio
from landiv_blur import processing as lbproc


def test_load_block():
    """This is just a smoketest"""
    with pytest.raises(rio.RasterioIOError):
        lbio.load_block('non-existing', start=(0, 0), size=(10, 10))


@ALL_MAPS
def test_import_export(datafiles):
    """Export per-cell entropy map after layered blur, load it and compare.
    """
    start = (1020, 1020)
    size = (700, 700)
    ch_map_tif = list(datafiles.iterdir())[0]
    block = lbio.load_block(ch_map_tif, start=start, size=size, indexes=1)
    entropy_layer = lbproc.get_entropy(block['data'], range(8),
                                       normed=True,
                                       img_filter=gaussian)
    outfile = datafiles / 'out.tif'
    lbio.export_to_tif(
        destination=str(outfile),
        data=entropy_layer,
        orig_profile=block['orig_profile'],
        # we need the transform from the window from block 1
        transform=block['transform']
    )
    block_2 = lbio.load_block(outfile, start=(0, 0), size=size)
    # NOTE: if the arrays contain np.nan then np.all will always be False
    assert np.all(np.nan_to_num(entropy_layer,
                  nan=-1) == np.nan_to_num(block_2['data'], nan=-1))
    assert block['transform'] == block_2['transform']


@ALL_MAPS
def test_resampling(datafiles):
    """Make sure our re-sampling method works as expected.
    """
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    # make sure the compatibility check fails
    with pytest.raises(TypeError):
        check_compatibility(ndvi_map, landcover_map)
    # re-sample the landcover_map to match the resolution of the ndvi_map
    lbio.coregister_raster(landcover_map, ndvi_map, output=str(landcover_map))
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
        print('\nWithout any tags set:')
        print_tag_details(src)
    # now we add a namespace and some tags
    print(f'\nNow we set a new namespace "{our_namespace}" and ')
    print('in there the tags:\n')
    print(f'- General:\n\t{gen_tag1}\n\t{gen_tag2}')
    print('- Per band:')
    for bidx, b_tags in band_tags.items():
        print(f'\t{bidx=}:\n\t\t{b_tags}')
    with rio.open(outfile, 'r+') as src:
        # file wide tags
        # src.update_tags(ns=our_namespace, **gen_tag1)
        lbio.set_tags(src=src, **gen_tag1)
        # src.update_tags(ns=our_namespace, **gen_tag2)
        lbio.set_tags(src=src, **gen_tag2)
        # now we add tags per band
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            lbio.set_tags(src=src, bidx=idx, **band_tags[idx])
    # read the tif
    print('\nAnd now we read the tags again form the tif:')
    with rio.open(outfile, 'r') as src:
        assert src.tags() == default_ns_tags
        assert src.tags(ns=our_namespace) != default_ns_tags
        print('First without specifying the namespace:')
        print_tag_details(src)
        print(f'\nAnd now in NAMESPACE "{our_namespace}":')
        print_tag_details(src, ns=our_namespace)


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
        lbio.set_tags(src=src, **gen_tag1)
        # now we add tags per band
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            lbio.set_tags(src=src, bidx=idx, **band_tags[idx])
    # now to the second file
    band_tags[2]['category'] = 101  # change one category to not match
    with rio.open(outfile2, 'r+') as src:
        # file wide tags
        lbio.set_tags(src=src, **gen_tag2)
        for idx in src.indexes:
            # src.update_tags(ns=our_namespace, bidx=idx, **band_tags[idx])
            lbio.set_tags(src=src, bidx=idx, **band_tags[idx])
    # now we check the matching:
    with rio.open(outfile1, 'r') as src:
        # category matches to band 1:
        _bidx = 1
        _category = band_tags[_bidx]['category']
        assert lbio.get_bidx(src=src,
                             category=_category) == _bidx
        # category 222 does not exist
        with pytest.raises(BandSelectionNoMatchError):
            lbio.get_bidx(src=src,
                          category=222)
        with pytest.raises(BandSelectionAmbiguousError):
            lbio.get_bidx(src=src, some='value')
        # check that serialization works
        _bidx = 2
        _tag = 'extra'
        _extra = band_tags[_bidx][_tag]
        _bla = _extra['bla']  # this is an int
        _blu = _extra['blu']  # this is a list
        # we read the serialized tags from the tif and convert them
        tags = lbio.get_tags(src=src, bidx=_bidx)
        assert tags[_tag]['bla'] == _bla
        assert tags[_tag]['blu'] == _blu
    # now test the get_bands
    sources = str(datafiles / 'out*.tif')
    # find all some: 'value' tags:
    some_value = lbio.get_bands(source=sources, some='value')
    assert (str(outfile2), None) in some_value  # the tag in the dataset
    # the bands
    assert (str(outfile1), 1) in some_value
    assert (str(outfile1), 3) in some_value
    assert (str(outfile2), 1) in some_value
    assert (str(outfile2), 3) in some_value
