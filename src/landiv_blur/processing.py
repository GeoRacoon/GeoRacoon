import numpy as np
from scipy.stats import entropy


def filter_for_layer(data, layer: int, is_value=1, not_value=0):
    """Filter for only the particular layer
    """
    return np.where(data == layer, is_value, not_value)


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


def get_layer_data(data, layer, img_filter, params=None):
    """Return the data of a single layer after filtering
    """
    _data = filter_for_layer(data, layer)
    params = params or dict()
    if img_filter:
        _data = apply_filter(_data, img_filter, **params)
    return _data


def get_entropy(data, layers=range(8), img_filter=None, **params):
    """Compute the Shannon entropy per cell

    Parameters
    ----------
    data: np.array
      A land-cover type matrix or any other matrix with integers
    layers: iterable
      A list or other iterable with the land-cover types to include
    img_filter: callable
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian
    params:
      Parameter to pass to the filter callable
    """
    all_layers = list()
    for layer in layers:
        _data = get_layer_data(data, layer, img_filter, params)
        all_layers.append(_data)

    # calculate the entropy
    stacked_layers = np.stack(all_layers, axis=2)
    entropy_layer = entropy(stacked_layers, axis=2)
    return entropy_layer
