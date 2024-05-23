from __future__ import annotations

from copy import copy

import numpy as np
from scipy.stats import entropy

from .io import load_block
from .prepare import get_view, relative_view


def dtype_range(dtype):
    """Get the range of the specified dtype
    """
    try:
        _max = dtype(np.iinfo(dtype).max)
        _min = dtype(np.iinfo(dtype).min)
    except ValueError:
        try:
            _max = dtype(np.finfo(dtype).max)
            _min = dtype(np.finfo(dtype).min)
        except ValueError:
            raise ValueError(f"{dtype=} is not a valid dtype of "
                             "`output_data`")
    return _max, _min

def filter_for_layer(data, layer: int | list[int], as_dtype=np.uint8):
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

    Returns
    -------
    np.array:
      Matrix of type `as_dtype` in the same shape of `data`
    """
    _is, _is_not = dtype_range(as_dtype)

    # NOTE: we might want to scale to [0, 1] if the as_dtype is float
    if isinstance(layer, int):
        _selected = [layer,]
    else:
        _selected = layer
    return np.where(np.isin(data, _selected), _is, _is_not)


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
    return list(map(int, lctypes))

def convert_to_type():
    pass

def get_layer_data(data, layer, img_filter=None, params=None,
                   as_dtype=np.uint8, output_dtype=None):
    """Return the data of a single layer after filtering
    """
    _data = filter_for_layer(data, layer, as_dtype=as_dtype)
    params = params or dict()
    if img_filter:
        _data = apply_filter(_data, img_filter, **params)
        # convert to the desired otput type
        if output_dtype:
            n_max, _ = dtype_range(output_dtype)
            _data = _data * n_max
            _data = _data.astype(output_dtype, copy=False)
    return _data


def get_filtered_layers(data, layers=None, img_filter=None, **params):
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
    params:
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
                               params=params)
        all_layers[layer] = _data
    return all_layers


def compute_entropy(filtered_data_layers: dict, normed, dtype):
    """
    """
    all_layers = list(filtered_data_layers.values())
    # calculate the entropy
    stacked_layers = np.stack(all_layers, axis=2)
    entropy_layer = entropy(stacked_layers, axis=2)
    if normed:
        max_entropy = get_max_entropy(len(all_layers))
        entropy_layer = entropy_layer / max_entropy
        print(f"{np.nanmax(entropy_layer)=}")
        if dtype:

            try:
                _limit_value = dtype(np.iinfo(dtype).max)
            except ValueError:
                try:
                    _limit_value = dtype(np.finfo(dtype).max)
                except ValueError:
                    raise ValueError(f"{dtype=} is not a valid dtype")
            entropy_layer = (entropy_layer * _limit_value).astype(dtype)
    return entropy_layer


def get_entropy(data, layers=None, normed=False, img_filter=None,
                dtype=None, **params):
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
    dtype: np.dtype
      The data-type to use for the returned array.

      ..note::

        This argument is only taken into account if `normed=True`.
    img_filter: callable
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian

    params:
      Parameter to pass to the filter callable
    """
    filtered_layers = get_filtered_layers(data=data, layers=layers,
                                          img_filter=img_filter, **params)
    return compute_entropy(filtered_data_layers=filtered_layers, normed=normed,
                           dtype=dtype)

def view_blurred(source,
                 view,
                 inner_view,
                 layers: list,
                 img_filter,
                 filter_params: dict = dict(),
                 blur_as_int: bool = False):
    """Return blurred layers of a view

    Parameters
    ----------
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
        size=size
    )
    data = result.pop('data')
    # transform = result.pop('transform')
    # orig_profile = result.pop('orig_profile')
    # perform blur
    blur_layers = get_filtered_layers(
        data=data,
        layers=layers,
        img_filter=img_filter,
        **filter_params
    )
    if blur_as_int:
        _maxint, _ = dtype_range(np.uint8)
        for k, data in blur_layers.items():
            blur_layers[k] = blur_layers[k] * _maxint
            blur_layers[k] = blur_layers[k].astype(np.uint8, copy=False)
    return dict(data=blur_layers, view=inner_view)

def  view_entropy(blur_layers, view, inner_view, entropy_as_ubyte: bool = False):
    """Return entropy-based landscape type heterogeneity measure for a view
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
        dtype=dtype,
    )
    usable_block = np.copy(
        get_view(entropy_layer,
                        relative_view(view, inner_view))
    )
    return dict(data=usable_block, view=inner_view)


def get_entropy_view(source,
                     view,
                     inner_view,
                     layers: list,
                     img_filter,
                     filter_params,
                     entropy_as_ubyte: bool = False,
                     blur_as_int: bool = False):
    """
    """
    blur_layers, _inner_view = view_blurred(
        source=source,
        view=view,
        inner_view=inner_view,
        layers=layers,
        img_filter=img_filter,
        filter_params=filter_params,
        blur_as_int=blur_as_int
    )
    assert _inner_view == inner_view
    return view_entropy(blur_layers=blur_layers,
                        view=view,
                        inner_view=inner_view,
                        entropy_as_ubyte=entropy_as_ubyte)
