from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rasterio.plot import show

from .io import load_block
from .processing import (
    get_category_data,
    get_entropy,
    get_categories
)

# TODO: The user should be ablre to provide a category: color mapping
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


def show_block(source:str, output, start=None, size=None):
    """Show only a specific block in a tif with all categories

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
    data = block['data']
    # cmap = _get_class_colormap()
    # plot_categories(data, transform, output, cmap)
    plot_categories(data, source, size, output)


def plot_block(source:str, start, size, ax, scaling=None,
               scaling_params=dict(), fig_params=dict()):
    """Plot categorical data and save it to a file.
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


def plot_categories(source, output, start=None, size=None):
    """Plot categorical data and save it to a file.

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


def show_category(data, category, transform, ax):
    """Handles the coloring of a category and calls rio.show
    """
    colors = [OUT, COLORS[category-1]]
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


def figure_categories(source, start, size, img_filter=None, params=None,
                      scaling=None, categories:list|None=None,
                      fig_params=dict(), scaling_params=dict()):
    """Plot each category on a separate axes

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

    if not categories:
        # TODO: use get_categories to get the number of categories
        rows = range(2)
        cols = range(4)
    if categories is not None and len(categories) <= row_limit:
        rows = range(1)
        cols = range(len(categories))
    else:
        # TODO: this is just hard-coded structure
        rows = range(3)
        cols = range(1)
    for row in rows:
        for col in cols:
            print((col, row))
            _category = col + row * row_limit
            if categories:
                try:
                    category = categories[_category]
                except IndexError:
                    # we plotted all categories
                    continue
            else:
                category = _category
            _data = get_category_data(data, category=category,
                                      img_filter=img_filter,
                                      filter_params=params)
            show_category(_data, category, transform, _get_axis(row,  col))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    output = fig_params.get('output', None)
    if output:
        fig.savefig(output, dpi=DPI)
    return axs


def plot_entropy(source:str, start:tuple[int,int], size:tuple[int,int],
                 output:str, scaling=None,
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
    entropy_array = data
    ax = fig_params.get('ax', None)
    fig = fig_params.get('fig', None)
    do_print = False
    if not ax:
        fig, ax = plt.subplots(figsize=(16, 16))
        do_print = True

    cmap = LinearSegmentedColormap.from_list(
        "Custom", ['black', 'white'], N=40)
    # pass affine transform corresponding to the window
    to_display = show(entropy_array,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    im = to_display.get_images()[0]
    if do_print:
        fig.colorbar(im, ax=ax)
        fig.savefig(output, dpi=DPI)
    return ax, (im, )


def plot_entropy_full(source, start, size, output, img_filter=None,
                      filter_params:dict|None=None,
                      entropy_params:dict|None=None
                      ):
    """Plot the entropy in each pixel after category diffusion

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
    cats = get_categories(data)
    entropy_array = get_entropy(data, categories=cats,
                                filter_params=filter_params,
                                entropy_params=entropy_params,
                                normed=True)

    fig, ax = plt.subplots(figsize=(16, 16))

    cmap = LinearSegmentedColormap.from_list(
        "Custom", ['black', 'white'], N=20)
    # pass affine transform corresponding to the window
    to_display = show(entropy_array,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    im = to_display.get_images()[0]
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)
