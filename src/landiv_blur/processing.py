from __future__ import annotations
from typing import Callable

from copy import copy

import numpy as np
from scipy.stats import entropy

from .helper import dtype_range
from .io import load_block
from .prepare import get_view, relative_view


def select_layer(data, layer: int | list[int],
                 as_dtype: type = np.uint8, limits: tuple | None = None):
    """Filter for only the particular layer


    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the per-pixel land-cover type
    layer:
      The specific land-cover type (or land-cover types) to select
    as_dtype:
      The data-type of the output matrix

      ..note::
        The output matrix will contain the maximal value possible for this data
        in pixels of the layer to select for and the minimal value in pixels
        associated to other layers.

    limits:
        Optonal custom limits to use. If provided `limits` must contain two
        values (is, isnot) that will be used to indicate if a pixel is of that
        land-cover type or not.

    Returns
    -------
    np.array:
      Matrix of type `as_dtype` in the same shape of `data`
    """
    if limits:
        _is, _is_not = limits
    else:
        _is, _is_not = dtype_range(as_dtype)

    # NOTE: we might want to scale to [0, 1] if the as_dtype is float
    if isinstance(layer, int):
        _selected = [layer,]
    else:
        _selected = layer
    return np.where(np.isin(data, _selected), _is, _is_not)


def apply_filter(data, img_filter:Callable, **params):
    """Apply a filter to the provided data

    Parameters
    ----------
    data: np.array
    img_filter: callback
    params: dict
      keyword parameter passed as is to the callback function
    """
    return img_filter(data, **params)


def get_max_entropy(nbr_lct):
    """Maximal entropy value possible for a given number of land-cover types

    Parameters
    ----------
    nbr_lct: int
      The number of different land-cover types considered

    Return
    ------
    float:
      The maximal entropy defined be a uniform distribution
    """
    return entropy(np.ones(nbr_lct))


def get_lct(data, ):
    """Return the list of land-cover types present in the data.
    """
    lctypes = np.unique(data)
    lctypes.sort()
    print(f"Inferring the layers from the provided data. Got:\n\n\t{lctypes}")
    return list(map(int, lctypes))

def get_layer_data(data, layer:int | list[int], img_filter=None,
                   filter_params:dict|None=None, output_dtype:type|None=None,
                   as_dtype:type=np.uint8):
    """Return the data of a single layer optionaly after applying a filter

    ..Note::
      If no image filter is provided then setting either `as_type` or `output_dtype`
      to the data type the output array should be in.

      If an image filter is provided then `as_type` will convert the data prior to
      applying the filter.

    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the per-pixel land-cover type
    layer:
      The specific land-cover type (or land-cover types) to select
    img_filter: callable
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian
    filter_params:
      Parameter to pass to the filter callable
    output_dtype:
      Set the data-type of the resulting image

      ..Note::
        This can be particularly usefull if you apply an image filter.

        For example, a Gaussian filter returns a map of type `np.float64` with
        values in $[0, 1]$.
        With `outptu_dtype=np.uint8` these values are mapped to the range [0, 255]
        and an array of type `np.uint8` is returend with a considerably smaller
        memory footprint.
    as_dtype:
      The data-type that should be used to represent the single layers in before
      applying the (optional) filter.

      ..Note::
        This should only be changed if the data contains more than 255 different
        land-cover types
    """
    # strip the layer
    _data = select_layer(data, layer, as_dtype=as_dtype)
    filter_params = filter_params or dict()
    # apply the image filter if provided
    if img_filter:
        _data = apply_filter(_data, img_filter, **filter_params)
        # convert to the desired otput type
        if output_dtype:
            n_max, _ = dtype_range(output_dtype)
            _data = _data * n_max
            _data = _data.astype(output_dtype, copy=False)
    return _data


def get_filtered_layers(data, layers=None, img_filter=None,
                        output_dtype:type|None=np.uint8,
                        filter_params:dict|None=None):
    """Apply an image filter to all layers in `data` and return each layer

    Parameters
    ----------
    data: np.array
      A land-cover type matrix or any other matrix with integers
    layers: iterable or None
      A list or other iterable with the land-cover types to include.
      If not provided then all land-cover types are included
    img_filter: callable
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian
    output_dtype:
      Specify what data type to use for the returned data.
      See `get_layer_data` for more details.
    filter_params:
      Parameter to pass to the filter callable

    Returns
    -------
    dict:
      For each layer (key) the filtered data
    """
    if layers is None:
        layers = get_lct(data)
    all_layers = dict()
    for layer in layers:
        _data = get_layer_data(data=data, layer=layer,
                               img_filter=img_filter,
                               filter_params=filter_params,
                               output_dtype=output_dtype)
        all_layers[layer] = _data
    return all_layers


def compute_entropy(filtered_data_layers: dict, normed:bool=True,
                    output_dtype:type|None=np.uint8,
                    **entropy_params):
    """

    Parameters
    ----------
    filtered_data_layers:
      A dictionary holding for each layer a data array
    normed:
      Determines if the values in the provided arrays should be normed or not.
    output_dtype:
      Set the data-type of the resulting image

      ..Note::
        This argument is ignored if `normed=False`.

        scipy.stats.entropy returns data of type `np.float64` which might consume
        a considerable amount of memory, when applied to larger maps.

        With `output_dtype=np.uint8` entropy values are mapped to the range [0, 255]
        and an array of type `np.uint8` is returend with a considerably smaller
        memory footprint.
    """
    all_layers = list(filtered_data_layers.values())
    # calculate the entropy
    stacked_layers = np.stack(all_layers, axis=2)
    entropy_layer = entropy(stacked_layers, axis=2, **entropy_params)
    if normed:
        max_entropy = get_max_entropy(len(all_layers))
        entropy_layer = entropy_layer / max_entropy
        if output_dtype:
            _max, _min = dtype_range(output_dtype)
            entropy_layer = (entropy_layer * _max).astype(output_dtype)
    return entropy_layer


def get_entropy(data, layers=None, normed=False, img_filter=None,
                output_dtype:type|None=None,
                filter_params:dict|None=None,
                entropy_params:dict|None=None):
    """Compute the Shannon entropy per cell

    Parameters
    ----------
    data: np.array
      A land-cover type matrix or any other matrix with integers
    layers: iterable or None
      A list or other iterable with the land-cover types to include.
      If not provided then all land-cover types are included
    normed: bool
      Determine whether or not the entropy should be normalized.
      If set to `True` each cell is normed by the maximal entropy value
      possible, i.e. `entropy(np.ones(len(layers)))`.
    output_dtype: np.dtype
      The data-type to use for the returned array.

      ..note::

        This argument is only taken into account if `normed=True`.
    img_filter: callable
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian

    filter_params:
      Parameter to pass to the filter callable
    entropy_params:
      Parameter to pass to scipy.stats.entropy
    """
    filter_params = filter_params or dict()
    blur_layers = get_filtered_layers(data=data, layers=layers,
                                      img_filter=img_filter,
                                      filter_params=filter_params)
    entropy_params = entropy_params or dict()
    return compute_entropy(filtered_data_layers=blur_layers, normed=normed,
                           output_dtype=output_dtype, **entropy_params)

def view_blurred(source,
                 view,
                 inner_view,
                 layers: list,
                 img_filter,
                 filter_params: dict = dict(),
                 blur_as_int: bool = False):
    """Return blurred layers of a view

    ..Note::
      This method will move to the `parallel` sub-module

    Parameters
    ----------
    source: str
      The path to the tif file to load
    """
    # view = params.pop('view')
    # inner_view = params.pop('inner_view')
    # blur_as_int = params.pop('blur_as_int', False)
    # layers = copy(params.pop('layers'))
    # read out block from original file
    start = (view[0], view[1])
    size = (view[2], view[3])
    result = load_block(
        source,
        start=start,
        size=size,
        indexes=1,
    )
    data = result.pop('data')
    print(f"{data.shape=}")
    # transform = result.pop('transform')
    # orig_profile = result.pop('orig_profile')
    # perform blur
    if blur_as_int:
        output_dtype=np.uint8
    else:
        output_dtype=None
    blur_layers = get_filtered_layers(
        data=data,
        layers=layers,
        img_filter=img_filter,
        filter_params=filter_params,
        output_dtype=output_dtype
    )
    # get the relative view
    for layer, data in blur_layers.items():
        blur_layers[layer] = np.copy(
            get_view(data,
                     relative_view(view, inner_view))
        )
    return dict(data=blur_layers, view=inner_view)


def  view_entropy(blur_layers, view, entropy_as_ubyte:bool=False):
    """Return entropy-based landscape type heterogeneity measure

    ..Note::
      This method will move to the `parallel` sub-module
    """
    if entropy_as_ubyte:
        normed = True
        dtype = np.uint8
    else:
        normed = False
        dtype = None
    # calculate entropy
    entropy_layer = compute_entropy(
        filtered_data_layers=blur_layers,
        normed=normed,
        output_dtype=dtype,
    )
    return dict(data=entropy_layer, view=view)


def get_entropy_view(source,
                     view,
                     inner_view,
                     layers: list,
                     img_filter,
                     filter_params:dict=dict(),
                     entropy_as_ubyte: bool = False,
                     blur_as_int: bool = False):
    """
    """
    blurred_data = view_blurred(
        source=source,
        view=view,
        inner_view=inner_view,
        layers=layers,
        img_filter=img_filter,
        filter_params=filter_params,
        blur_as_int=blur_as_int
    )
    assert blurred_data['view'] == inner_view
    entropy_view = view_entropy(blur_layers=blurred_data['data'],
                                view=blurred_data['view'],
                                entropy_as_ubyte=entropy_as_ubyte)
    entropy_view['view'] = blurred_data['view']
    return entropy_view
