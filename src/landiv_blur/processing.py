from __future__ import annotations

import numpy as np
from scipy.stats import entropy


def filter_for_layer(data, layer: int | list[int], output_dtype=np.uint8):
    """Filter for only the particular layer


    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the per-pixel land-cover type
    layer:
      The specific land-cover type (or land-cover types) to select
    output_dtype:
      The data-type of the output matrix

      ..note::
        The output matrix will contain the maximal value possible for this data
        in pixels of the layer to select for and the minimal value in pixels
        associated to other layers.

    Returns
    -------
    np.array:
      Matrix of type `output_dtype` in the same shape of `data`
    """
    try:
        _is = output_dtype(np.iinfo(output_dtype).max)
        _is_not = output_dtype(np.iinfo(output_dtype).min)
    except ValueError:
        try:
            _is = output_dtype(np.finfo(output_dtype).max)
            _is_not = output_dtype(np.finfo(output_dtype).min)
        except ValueError:
            raise ValueError(f"{output_dtype=} is not a valid dtype of "
                             "`output_data`")
    # NOTE: we might want to scale to [0, 1] if the output_dtype is float
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
    return lctypes


def get_layer_data(data, layer, img_filter=None, params=None,
                   output_dtype=np.uint8):
    """Return the data of a single layer after filtering
    """
    _data = filter_for_layer(data, layer, output_dtype=output_dtype)
    params = params or dict()
    if img_filter:
        _data = apply_filter(_data, img_filter, **params)
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
