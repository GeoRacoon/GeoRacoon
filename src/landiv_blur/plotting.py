import math
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
WATER = 'blue'
URBAN = 'red'
GRASSLAND = 'lightgreen'
BLACK = 'black'
COLORS = [
    GRASSLAND,
    FOREST,
    WATER,  # 'teal',
    'orange',
    'yellow',
    'purple',
    "brown",  # 'purple',
    URBAN,
    BLACK,
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


def plot_block(source, start, size, ax, scaling=None,
               scaling_params=dict(), fig_params=dict()):
    """Plot the landtypes data and save it to a file.
    """
    block = load_block(source=source, start=start, size=size,
                       scaling=scaling, **scaling_params)
    data, transform = block['data'], block['transform']
    cmap = fig_params.get('cmap', _get_class_colormap())
    
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
    """Handles the coloring of the layer and calls rio.show
    """
    colors = [OUT, COLORS[layer-1]]
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


def plot_layers(source, start, size, img_filter=None, params=None,
                scaling=None, layers=None,
                fig_params=dict(), scaling_params=dict()):
    """Plot each layer in isolation

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract

    keywords:
      axs:
        Axes to draw on
      gs:
        GridSpec
      output: str
        Where to store the image
    """
    block = load_block(source, start, size, indexes=None, scaling=scaling,
                       **scaling_params)
    data, transform = block['data'], block['transform']

    axs = fig_params.get('axs', None)
    gs = fig_params.get('gs', None)
    row_limit = 4  # number of plots in a row
    if axs is None and gs is None:
        fig, axs = plt.subplots(2, 4, figsize=(128, 64))

        def _get_axis(row, col):
            return axs[row, col]
    elif axs is None:
        fig = fig_params.get('fig')
        gsr = fig_params.get('gsr', 0)
        gsc = fig_params.get('gsc', 0)
        row_limit = fig_params.get('rl', 4)
        rstep = fig_params.get('rstep', 1)
        cstep = fig_params.get('cstep', 1)

        def _get_axis(row, col):
            return fig.add_subplot(gs[gsr+(row*rstep):gsr+(row*rstep+rstep),
                                      gsc+col: gsc+col+cstep])
    else:

        def _get_axis(row, col):
            return axs[row, col]

    if not layers:
        # TODO: use get_lct to get the number of layers
        rows = range(2)
        cols = range(4)
    if len(layers) <= row_limit:
        rows = range(1)
        cols = range(len(layers))
    else:
        # TODO: this is just hard-coded structure
        rows = range(3)
        cols = range(1)
    for row in rows:
        for col in cols:
            print((col, row))
            _layer = col + row * row_limit
            if layers:
                try:
                    layer = layers[_layer]
                except IndexError:
                    # we plotted all layers
                    continue
            else:
                layer = _layer
            _data = get_layer_data(data, layer, img_filter, params)
            show_layer(_data, layer, transform, _get_axis(row,  col))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    output = fig_params.get('output', None)
    if output:
        fig.savefig(output, dpi=DPI)
    return axs


def plot_entropy(source, start, size, output=None, scaling=None,
                 fig_params=dict(), scale_params=dict()):
    """Plot the entropy in each pixel from a tif file

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate
    size: tuple
      width and height of the block to extract
    """
    block = load_block(source=source, start=start, size=size,
                       scaling=scaling, **scale_params)
    data, transform = block['data'], block['transform']
    entropy_layer = data
    ax = fig_params.get('ax', None)
    fig = fig_params.get('fig', None)
    do_print = False
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 16))
        do_print = True

    cmap = LinearSegmentedColormap.from_list(
        "Custom", ['black', 'white'], N=40)
    # pass affine transform corresponding to the window
    to_display = show(entropy_layer,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    im = to_display.get_images()[0]
    if do_print:
        fig.colorbar(im, ax=ax)
        fig.savefig(output, dpi=DPI)
    return ax, (im, )


def plot_entropy_full(source, start, size, output, img_filter=None,
                      params=None):
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
    entropy_layer = get_entropy(data, layers=range(8), normed=True)

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
