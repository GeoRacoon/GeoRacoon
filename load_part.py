import numpy as np
import rasterio
from skimage.filters import gaussian
from rasterio.enums import ColorInterp
from rasterio.plot import show
from rasterio.windows import Window
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import entropy

OUT = 'black'
FOREST = 'darkgreen'
WATER = 'lightblue'
URBAN = 'red'
GRASSLAND = 'lightgreen'
COLORS = [
    OUT,
    URBAN,
    FOREST,
    GRASSLAND,
    'blue',
    'purple',
    WATER,
    'orange'
]

DPI = 200

def _get_class_colormap(colors = COLORS):
    """Create a custom colormap of the 8 classes we use.


    """
    return ListedColormap(colors)

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

def filter_for_layer(data, layer: int, is_value=1, not_value=0):
    """Filter for only the particular layer
    """
    return np.where(data==layer, is_value, not_value)

def get_layer_data(data, layer, img_filter, params):
    """Return the data of a single layer after filtering
    """
    _data = filter_for_layer(data, layer)
    print(np.unique(_data))
    if img_filter:
        _data = apply_filter(_data, img_filter, **params)
    print(np.unique(_data))
    return _data

def show_layer(data, layer, transform, ax):
    """
    """
    colors=[COLORS[0], COLORS[layer-1]]
    if len(np.unique(data)) == 2:
        cmap=_get_class_colormap(colors=colors)
    else:
        cmap=LinearSegmentedColormap.from_list("Custom", colors, N=20)
    return show(
        data,
        ax=ax,
        transform=transform,
        cmap=cmap
    )


def show_block(source, start, size, output):
    """Show only a specific block in a tif with all layers

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    """
    data, transform = load_block(source, start, size)

    cmap = _get_class_colormap()

    plot_landtypes(data, transform, output, cmap)


def plot_block(source, start, size, ax):
    """Plot the landtypes data and save it to a file.
    """
    data, transform = load_block(source, start, size)
    cmap = _get_class_colormap()
    # pass affine transform corresponding to the window
    to_display = show(data,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    return to_display.get_images()[0]


def plot_landtypes(source, start, size, output):
    """Plot the landtypes data and save it to a file.
    
    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    output: str
      Where to store the image
    """
    fig, ax = plt.subplots(figsize = (16, 16))
    im = plot_block(source, start, size, ax)
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)

def plot_layers(source, start, size, output, img_filter=None, params=None):
    """Plot each layer in isolation
    
    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    output: str
      Where to store the image
    """
    data, transform = load_block(source, start, size)

    fig, axs = plt.subplots(2, 4, figsize=(128, 64))

    for row in range(2):
        for col in range(4):
            print((col, row))
            layer = 1 + col + row * 4
            print(layer)
            _data = get_layer_data(data, layer, img_filter, params)
            show_layer(_data, layer, transform, axs[row, col])
    fig.savefig(output, dpi=DPI)

def apply_filter(data, img_filter, **params):
    """Apply a filter to the provided data

    Parameters
    ----------
    data: np.array
    img_filter: callback
    params: dict
      keyword parameter passed as is to the callback function
    """
    return img_filter(data, **params)

def plot_entropy(source, start, size, output, img_filter=None, params=None):
    """Plot the entropy in each pixel after layer diffusion
    
    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    output: str
      Where to store the image
    """
    data, transform = load_block(source, start, size)

    all_layers = list()
    for row in range(2):
        for col in range(4):
            layer = 1 + col + row * 4
            _data = get_layer_data(data, layer, img_filter, params)
            all_layers.append(_data)

    # calculate the entropy
    stacked_layers = np.stack(all_layers, axis=2)
    entropy_layer = entropy(stacked_layers, axis=2)

    fig, ax = plt.subplots(figsize = (16, 16))

    cmap = LinearSegmentedColormap.from_list("Custom", ['black', 'white'], N=20)
    # pass affine transform corresponding to the window
    to_display = show(entropy_layer,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    im = to_display.get_images()[0]
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)


def main(args):
    """Generate all plots
    """
    plot_landtypes(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }.{ args.format }",
    )
    plot_layers(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_layers.{ args.format }",
    )
    # now with filter
    plot_layers(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_layers_filtered_{args.sigma}.{ args.format }",
        img_filter=gaussian,
        params=dict(sigma=args.sigma,)
    )
    plot_entropy(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_layers_entropy_{args.sigma}.{ args.format }",
        img_filter=gaussian,
        params=dict(sigma=args.sigma,)
    )

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
    parser.add_argument("--format",
                   type=str,
                   default='png',
                   help="What format to use")
    parser.add_argument("--sigma",
                   type=float,
                   default=1,
                   help="standard deviation for gaussian kernel")
    # parse the arguments
    args = parser.parse_args()
    main(args)
