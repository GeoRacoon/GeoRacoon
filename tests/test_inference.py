from landiv_blur import io as lbio
from landiv_blur import inference as lbinf

from .config import ALL_MAPS


@ALL_MAPS
def test_preparation(datafiles):
    """Test the preparation of predictors based on a response matrix
    """
    test_data = list(datafiles.iterdir())
    landcover_map = test_data[0]
    ndvi_map = test_data[1]
    lbio.coregister_raster(landcover_map, ndvi_map, output=str(landcover_map))
    lbinf.prepare_predictors(ndvi_map,
                             (landcover_map, 1, (2,3,4)),
                              with_intercept=True,)
    print(test_data)
    # ch_data = lbio.load_map(ch_map_tif)['data']
