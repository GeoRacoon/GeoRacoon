import pytest
import rasterio
import numpy as np
from skimage.filters import gaussian

from .config import ALL_MAPS
from landiv_blur import io as lbio
from landiv_blur import processing as lbproc


def test_load_block():
    """This is just a smoketest"""
    with pytest.raises(rasterio.RasterioIOError):
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
