import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


def load_block(source, start, size):
    """Return block from a *.tif file along with the transform

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    """
    with rasterio.open(source) as img:
        riow = Window(*start, *size)
        # Lookup table for the color space in the source file
        colorspace = dict(zip(img.colorinterp, img.indexes))

        if len(colorspace.keys()) == 3:
            # Read the image in the proper order so the numpy array is RGB
            rgb_idxs = [
                colorspace[ci]
                for ci in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            ]
        else:
            rgb_idxs = 1
        return img.read(rgb_idxs, window=riow), img.window_transform(riow)
