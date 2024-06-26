import pytest
import rasterio as rio
import numpy as np
from skimage.filters import gaussian

from .config import ALL_MAPS
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
