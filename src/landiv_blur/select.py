# not_needed (though might be helpful) 
# needs_work (docstring missing)
# not_tested
import numpy as np
import rasterio as rio
from rasterio.windows import from_bounds

from typing import Any

from .helper import check_compatibility
from .io import load_block


def read_clip(source:str, clipping, **tags):
    # TODO: not_needed (I guess)
    """Read out the `clipping` map area from `source`

    ..note::
      This method will be moved to the io submodule

    Parameters
    ----------
    source: str
      The path to the tif file to load
    clipping: str
      The path to the tif file to use for clipping
    **tags:
      Arbitrary number of keyword arguments to describe the band to select.    

      See `io.get_bidx` for further details

    """
    # not_needed (though might be helpful) 
    # needs_work (better doc - address TODO or remove)
    # not_tested
    # -
    # make sure we use the same projection
    check_compatibility(source, clipping)
    with rio.open(clipping) as ref:
        bounds = ref.bounds
    with rio.open(source) as src:
        bidx = get_bidx(src=src, **tags)
        riow = from_bounds(*bounds, src.transform)
        transform = src.window_transform(riow)
        clipped_data = src.read(indexes=bidx, window=riow)
        # TODO: allow for re-sampling to gthe the same shape
        #       see: https://gis.stackexchange.com/questions/434441/specifying-target-resolution-when-resampling-with-rasterio
        return {
            'data': clipped_data,
            'transform': transform,
            'orig_profile': src.profile.copy()
        }


def mask_relative(source:str,
                  masker:str,
                  masking:Any,
                  no_value:Any=np.nan,
                  **tags):
    # TODO: not_needed (I guess)
    """Mask source map where a band from the masker map has a specific value

    Parameters
    ----------
    source: str
      The path to the tif file to load
    masker: str
      The path to the tif file to use create the mask from
    masking:
      Value to find in the band of the masker map
    no_value:
      Value to set in the source map where the band of the masker map equates
      to the `masking` value
    **tags:
      Select the band from the masker map by proving an arbitrary number of
      keyword arguments.

      See `io.get_bidx` for further details

    """
    # not_needed (though might be helpful) 
    # needs_work (better doc)
    # not_tested
    # -

    # we only need the mask where the source file is
    clipper_data = read_clip(source=masker, clipping=source, **tags)['data']
    # print(f"{clipper_data.shape=}")
    source_map = load_block(source=source,
                            view=None,
                            indexes=None)
    source_data = source_map['data']
    # print(f"{source_data.shape=}")
    # TODO: we still need to make sure masker and source have the same
    #       resolution after clipping
    masked = np.where(clipper_data == masking, source_data, no_value)
    return {
        'data': masked,
        'transform': source_map['transform'],
        'orig_profile': source_map['profile'].copy()
    }
