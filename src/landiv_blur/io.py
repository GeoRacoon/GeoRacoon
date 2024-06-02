import os

import rasterio
from math import floor

import rasterio as rio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from rasterio.enums import Resampling

from rasterio.warp import calculate_default_transform, reproject, Resampling

from .helper import (
    check_crs,
    get_scale_factor,
)

def load_map(source, indexes=None):
    """Load a map from a tif

    Return
    ------
    dict:
       Returns the callback of
       `load_block(source=source, start=None, size=None, indexes=indexes)`
    """
    return load_block(source=source, start=None, size=None, indexes=indexes)


def load_block(source, start=None, size=None, indexes=None, scaling=None,
               **params):
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

      ..note::
        The index of the first band is 1 not 0!

    scaling: tuple[float] | None
      Factors to rescale the number of pixels. Values >1 will upscale.

      ..note::
        If scaling is provided, the keyword argument `scaling_method` should
        also be given and identify a method from `rasterio.enums.Resampling`
        to apply for the scaling

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
            idxs = [
                colorspace[ci]
                for ci in (ColorInterp.red,
                           ColorInterp.green,
                           ColorInterp.blue)
            ]
        elif indexes is not None:
            idxs = indexes
        else:
            idxs = img.indexes
        if any((start, size)):
            assert all((start, size)), \
                   f"{start=} and {size=} both need to be set or both None"
            riow = Window(*start, *size)
            transform = img.window_transform(riow)
            width = size[0]
            height = size[1]
        else:
            width = img.width
            height = img.height
            riow = None
            transform = img.transform

        # perform a re-scaling if needed
        if scaling:
            out_shape = (
                img.count,
                floor(img.height * scaling[0]),
                floor(img.width * scaling[1])
            )
            print(out_shape)
            print(width)
            resampling = params.get('scaling_method', Resampling.bilinear)
        else:
            out_shape = None
            resampling = Resampling.nearest
        # read out the desired part
        data = img.read(idxs,
                        window=riow,
                        out_shape=out_shape,
                        resampling=resampling)
        if scaling:
            # scale image transform
            transform = transform * transform.scale(
                (width / data.shape[-1]),
                (height / data.shape[-2])
            )
        return {
            'data': data,
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



def resample_to(source, reference, output=None, **params):
    """Re-sample the source map so to match the resolution of the reference map

    Parameters
    ----------
    source: str
      The path to the tif file you want to re-sample
    reference: str
      The path to the tif file with the desired resolution
    output: str (optional)
      The path to write the re-projected map to.

      ..note::
        If not provided, the output file will take the name of the input file
        and get a _<linear units>_<width>.<height> attached.

    Return
    ------
    str:
      The name of the file that hold the re-projected map
    """
    resampling = params.get('scaling_method', Resampling.bilinear)
    # here it is also checked that the units match:
    scale_factor = get_scale_factor(source=source, target=reference)
    with rio.open(source) as src:
        profile = src.profile.copy()
        src_unit = src.profile['crs'].linear_units.lower()
        data = src.read(
            out_shape=(
                src.count,
                floor(src.height * scale_factor[0]),
                floor(src.widht * scale_factor[1])
            ),
            resampling=resampling
        )
        transform = src.transform * src.transform.scale(
            scale_factor[0],
            scale_factor[1]
        )
        height = data.shape[0]
        width = data.shape[1]
        profile.update(
            dict(height=height,
                 width=width,
                 transform=transform)
        )
    if output is None:
        _name, _ext = os.path.splitext(source)
        output_file = f"{_name}_{src_unit}_{width}.{height}{_ext}"
    else:
        output_file = output
    with rio.open(output_file, "w", **profile) as dst:
        dst.write(data)


def project_to(source, reference, output=None):
    """Re-projects the source map into the coordinate system of a reference map

    Parameters
    ----------
    source: str
      The path to the tif file you want to change projection
    reference: str
      The path to the tif file with the projection to apply
    output: str (optional)
      The path to write the re-projected map to.

      ..note::
        If not provided, the output file will take the name of the input file
        and add the CRS of the new projection at the end of the name.

    Return
    ------
    str:
      The name of the file that hold the re-projected map
    """
    with rio.open(reference) as ref:
        dst_crs = str(ref.profile['crs'])
    with rio.open(source) as src:
        src_crs = str(src.crs)
        if src_crs == dst_crs:
            print(f"There is nothing to project! {src_crs=} to {dst_crs=}")
            return None

        transform, width, height = calculate_default_transform(src.crs,
                                                               dst_crs,
                                                               src.width,
                                                               src.height,
                                                               *src.bounds)
        kwargs = src.meta.copy()
        # prepare the resulting profile
        kwargs.update({
          'crs': dst_crs,
          'transform': transform,
          'width': width,
          'height': height
        })

        if output is None:
            _base_name, _ext = os.path.splitext(source)
            output = f"{_base_name}_{dst_crs}{_ext}"
        with rio.open(output, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
        return output
