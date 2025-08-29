from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from rasterio.plot import show

from riogrande.io import load_block
from .processing import (
    get_category_data,
    get_entropy,
    get_categories
)

# TODO: We should think whether we want to include this afterall in the package. It could be helpfull actully.
#  --> for now all here is "is_needed"

# not_needed (we need to decide if we want to keep this around)
# needs_work (doc; define what it should contain; where it should reside)
# not_tested
# usedin_both (most likely the io module)

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
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Create a custom colormap of the 8 classes we use.
    """
    return ListedColormap(colors)


def plot_block(source:str,
               ax,
               view:None|tuple[int,int,int,int]=None,
               scaling_params=dict(),
               fig_params=dict(),
               **tags):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Plot categorical data and save it to a file.

    Parameters
    ----------
    source: str
      The path to the tif file to load
    view:
      An optional tuple (x, y, width, height) defining the area to load.

      If `None` is provided (the default) then the entire file is loaded.

    scaling_params:
      Optional dictionary to set a rescaling of the data.
      If provided, the following keywords are accepted:

      scaling: tuple[float,float]
        Factors to rescale the number of pixels. Values >1 will upscale.
      method: rasterio.enums.Resampling
        The resampling method. If not provided then the bilinear resampling
        is used.

    fig_params:
      Parametrization of the figure:

      axs:
        Axes to draw on
      gs:
        GridSpec
      output: str
        Where to store the image

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `get_bidx` for further details

    """
    block = load_block(source=source, view=view,
                       scaling_params=scaling_params,
                       **tags)
    data, transform = block['data'], block['transform']
    cmap = fig_params.get('cmap', _get_class_colormap())

    # pass affine transform corresponding to the window
    to_display = show(data,
                      ax=ax,
                      transform=transform,
                      cmap=cmap)
    return to_display.get_images()[0]


def plot_categories(source:str,
                    output:str,
                    view:None|tuple[int,int,int,int]=None,
                    **tags):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Plot categorical data and save it to a file.

    Parameters
    ----------
    source:
      The path to the tif file to load
    output:
      Where to store the image
    view:
      An optional tuple (x, y, width, height) defining the view to show

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `get_bidx` for further details
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    im = plot_block(source, ax=ax, view=view, **tags)
    fig.colorbar(im, ax=ax)
    fig.savefig(output, dpi=DPI)


def show_category(data, category, transform, ax):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
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


def figure_categories(source:str,
                      view:None|tuple[int,int,int,int]=None,
                      img_filter=None, params=None,
                      categories:list|None=None,
                      fig_params=dict(), scaling_params=dict(),
                      **tags):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Plot each category on a separate axes

    Parameters
    ----------
    source: str
      The path to the tif file to load
    view:
      An optional tuple (x, y, width, height) defining the view to show
    scaling_params:
      Optional dictionary to set a rescaling of the data.
      If provided, the following keywords are accepted:

      scaling: tuple[float,float]
        Factors to rescale the number of pixels. Values >1 will upscale.
      method: rasterio.enums.Resampling
        The resampling method. If not provided then the bilinear resampling
        is used.

    fig_params:
      Parametrization of the figure:

      axs:
        Axes to draw on
      gs:
        GridSpec
      output: str
        Where to store the image

    **tags:
      Arbitrary number of keyword arguments to describe the band to select
      from source

      See `get_bidx` for further details
    """
    block = load_block(source, view=view,
                       scaling_params=scaling_params, **tags)
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


def plot_entropy(source:str,
                 view:None|tuple[int,int,int,int]=None,
                 fig_params=dict(),
                 scaling_params=dict(),
                 **tags):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Plot the entropy in each pixel from a tif file

    Parameters
    ----------
    source: str
      The path to the tif file to load
    view:
      An optional tuple (x, y, width, height) defining the view to show
    scaling_params:
      Optional dictionary to set a rescaling of the data.
      If provided, the following keywords are accepted:

      scaling: tuple[float,float]
        Factors to rescale the number of pixels. Values >1 will upscale.
      method: rasterio.enums.Resampling
        The resampling method. If not provided then the bilinear resampling
        is used.
    fig_params:
      Parametrization of the figure:

      axs:
        Axes to draw on
      gs:
        GridSpec
      output: str
        Where to store the image

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `get_bidx` for further details
    """
    block = load_block(source=source, view=view,
                       scaling_params=scaling_params,
                       **tags)
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
        fig.savefig(fig_params['output'], dpi=fig_params.get('dpi', DPI))
    return ax, (im, )


def plot_entropy_full(source:str,
                      output:str,
                      view:None|tuple[int,int,int,int]=None,
                      img_filter=None,
                      filter_params:dict|None=None,
                      entropy_params:dict|None=None,
                      **tags
                      ):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Plot the entropy in each pixel after category diffusion

    Parameters
    ----------
    source: str
      The path to the tif file to load
    output: str
      Where to store the image
    view:
      An optional tuple (x, y, width, height) defining the view to show

    **tags:
      Arbitrary number of keyword arguments to describe the band to select from
      the source map (i.e. the band that contains the categorical values).

      See `get_bidx` for further details
    """
    block = load_block(source=source,
                       view=view,
                       **tags)
    data, transform = block['data'], block['transform']
    cats = get_categories(data)
    entropy_array = get_entropy(data, categories=cats,
                                img_filter=img_filter,
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
