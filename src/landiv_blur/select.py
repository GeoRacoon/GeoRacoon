import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds

from .prepare import check_compatibility
from .io import load_block


def read_clip(source, clipping):
    """Read out the `clipping` map area from `source`

    ..note::
      This method will be moved to the io submodule
    """
    # make sure we use the same projection
    check_compatibility(source, clipping)
    with rio.open(clipping) as ref:
        bounds = ref.bounds
    with rio.open(source) as src:

        riow = from_bounds(*bounds, src.transform)
        transform = src.window_transform(riow)
        clipped_data = src.read(window=riow)
        # TODO: allow for re-sampling to gthe the same shape
        #       see: https://gis.stackexchange.com/questions/434441/specifying-target-resolution-when-resampling-with-rasterio
        return {
            'data': clipped_data,
            'transform': transform,
            'orig_profile': src.profile.copy()
        }


def mask_relative(source, masker, masking, no_value=np.nan):
    """Mask the source map where the masker map matches the masking value
    """
    # we only need the mask where the source file is
    clipper_data = read_clip(masker, source)['data']
    print(f"{clipper_data.shape=}")
    source_map = load_block(source=source, start=None, size=None, indexes=None)
    source_data = source_map['data']
    print(f"{source_data.shape=}")
    # TODO: we still need to make sure masker and source have the same
    #       resolution after clipping
    masked = np.where(clipper_data == masking, source_data, no_value)
    return {
        'data': masked,
        'transform': source_map['transform'],
        'orig_profile': source_map['profile'].copy()
    }
