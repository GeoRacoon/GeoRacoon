import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


def load_map(source, indexes=None):
    """Load a map from a tif

    Return
    ------
    dict:
       Returns the callback of
       `load_block(source=source, start=None, size=None, indexes=indexes)`
    """
    return load_block(source=source, start=None, size=None, indexes=indexes)


def load_block(source, start=None, size=None, indexes=None):
    """Get a block from a *.tif file along with the transform

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate

      If not provided (or set to `None`) then the coordinate (0,0) is used
    size: tuple
      width and height of the block to extract

      If not provided the entire map is loaded.
    indexes: list of int, int or None
      If a list is provided a 3D array is returned, if not a 2D array.

    Return
    ------
    dict:
       data: holding a numpy array with the actual data
       transform: an ???.Affine object that encodes the transformation used
       orig_meta: The meta information of the original .tif file
       orig_profile: The profile information of the original .tif file
    """
    with rasterio.open(source) as img:
        # TODO: rasterio Window allows using slices. In doing so we could
        #       harmonize what we call blocks and views and just work with
        #       slices.
        # Lookup table for the color space in the source file
        colorspace = dict(zip(img.colorinterp, img.indexes))

        if len(colorspace.keys()) == 3:
            # Read the image in the proper order so the numpy array is RGB
            rgb_idxs = [
                colorspace[ci]
                for ci in (ColorInterp.red,
                           ColorInterp.green,
                           ColorInterp.blue)
            ]
        else:
            rgb_idxs = 1
        if any((start, size)):
            assert all((start, size)), \
                   f"{start=} and {size=} both need to be set or both None"
            riow = Window(*start, *size)
            transform = img.window_transform(riow)
        else:
            riow = None
            transform = img.profile.copy()
        return {
            'data': img.read(rgb_idxs, window=riow),
            'transform': transform,
            'orig_profile': img.profile.copy()
        }


def export_to_tif(destination, data, orig_profile, start=(0, 0),  **pparams):
    """Export a np.array to tif, only updating a window if data is smaller

    .. note::
      This function will overwrite the dtype of the destination tif with the
      value provided in pparams or the data type of `data`.

    Parameters
    ----------
    destination: str
        location to export save the .tif file
    data: np.array
        The map to export
    start: tuple
      horizontal and vertical starting coordinate
    orig_profile: dict
        the profile of the original map
        (see https://rasterio.readthedocs.io/en/stable/topics/profiles.html)
    **pparams:
        further parameter to be added to the profile
    """
    profile = orig_profile.copy()
    # Note: we no longer update the size automatically as for Windows this is
    # not correct, pass height and width explicitly to update via pparams
    # # update for the correct dimensions
    # profile['height'] = data.shape[1]
    # profile['width'] = data.shape[0]
    # set the dtype explicitly of get it from the data
    profile['dtype'] = pparams.pop('dtype', str(data.dtype))
    profile.update(pparams)
    # write it:
    size = data.shape[::-1]  # since positions are inverted in numpy
    with rasterio.open(destination, "w", **profile) as dest:
        dest.write(data, window=Window(*start, *size), indexes=1)
