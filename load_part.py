import rasterio
from rasterio.enums import ColorInterp
from rasterio.plot import show
from rasterio.windows import Window
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

FOREST = 'darkgreen'
OUT = 'lightgray'
WATER = 'lightblue'
URBAN = 'red'
GRASSLAND = 'lightgreen'


def _get_class_colormap():
    """Create a custom colormap of the 8 classes we use.


    """
    return ListedColormap([
        OUT,
        URBAN,
        FOREST,
        GRASSLAND,
        'blue',
        'purple',
        WATER,
        'orange'
    ])

def show_slice(source, size, hstart, vstart, output):
    with rasterio.open(source) as img:
        riowindow = Window(hstart, vstart, size, size)

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
        data = img.read(rgb_idxs, window=riowindow)

        cmap = _get_class_colormap()
        fig, ax = plt.subplots(figsize = (16, 16))
        # pass affine transform corresponding to the window
        to_display = show(data,
                          ax=ax,
                          transform=img.window_transform(riowindow),
                          cmap=cmap)
        im = to_display.get_images()[0]
        fig.colorbar(im, ax=ax)
        fig.savefig(output)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--source",
                   type=str,
                   help="Path to the file to load")
    parser.add_argument("--output",
                   type=str,
                   help="Where to save the image")
    parser.add_argument("--size",
                   type=int,
                   default=10000,
                   help="Number of pixels for with & height")
    parser.add_argument("--hstart",
                   type=int,
                   default=0,
                   help="Where to start horizontally")
    parser.add_argument("--vstart",
                   type=int,
                   default=0,
                   help="Where to start vertically")
    # parse the arguments
    args = parser.parse_args()
    show_slice(
        args.source,
        args.size,
        hstart=args.hstart,
        vstart=args.vstart,
        output=args.output,
    )
    


