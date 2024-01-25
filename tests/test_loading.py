import pytest
import rasterio
from landiv_blur import loading


def test_load_block():
    """This is just a smoketest"""
    with pytest.raises(rasterio.RasterioIOError):
        loading.load_block('non-existing', start=(0, 0), size=(10, 10))
