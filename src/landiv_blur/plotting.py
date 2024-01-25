import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rasterio.plot import show

from .io import load_block
from .processing import (
    get_layer_data,
    get_entropy
)

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


def _get_class_colormap(colors=COLORS):
    """Create a custom colormap of the 8 classes we use.
    """
    return ListedColormap(colors)


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
    block = load_block(source, start, size)
    data, transform = block['data'], block['transform']

    cmap = _get_class_colormap()

    plot_landtypes(data, transform, output, cmap)


def plot_block(source, start, size, ax):
    """Plot the landtypes data and save it to a file.
    """
    block = load_block(source, start, size)
    data, transform = block['data'], block['transform']
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
    fig, ax = plt.subplots(figsize=(16, 16))
    im = plot_block(source, start, size, ax)
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)


def show_layer(data, layer, transform, ax):
    """
    """
    colors = [COLORS[0], COLORS[layer-1]]
    if len(np.unique(data)) == 2:
        cmap = _get_class_colormap(colors=colors)
    else:
        cmap = LinearSegmentedColormap.from_list("Custom", colors, N=20)
    return show(
        data,
        ax=ax,
        transform=transform,
        cmap=cmap
    )


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
    block = load_block(source, start, size)
    data, transform = block['data'], block['transform']

    fig, axs = plt.subplots(2, 4, figsize=(128, 64))

    for row in range(2):
        for col in range(4):
            print((col, row))
            layer = 1 + col + row * 4
            print(layer)
            _data = get_layer_data(data, layer, img_filter, params)
            show_layer(_data, layer, transform, axs[row, col])
    fig.savefig(output, dpi=DPI)


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
    block = load_block(source, start, size)
    data, transform = block['data'], block['transform']
    entropy_layer = get_entropy(data, layers=range(8))

    fig, ax = plt.subplots(figsize=(16, 16))

    cmap = LinearSegmentedColormap.from_list(
        "Custom", ['black', 'white'], N=20)
    # pass affine transform corresponding to the window
    to_display = show(entropy_layer,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    im = to_display.get_images()[0]
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)
