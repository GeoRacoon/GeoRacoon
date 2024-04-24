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


def get_layer_data(data, layer, img_filter, params=None):
    """Return the data of a single layer after filtering
    """
    _data = filter_for_layer(data, layer)
    params = params or dict()
    if img_filter:
        _data = apply_filter(_data, img_filter, **params)
    return _data


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
    if layers is None:
        layers = get_lct(data)
    all_layers = list()
    for layer in layers:
        _data = get_layer_data(data, layer, img_filter, params)
        all_layers.append(_data)

    # calculate the entropy
    stacked_layers = np.stack(all_layers, axis=2)
    entropy_layer = entropy(stacked_layers, axis=2)
    if normed:
        max_entropy = get_max_entropy(len(layers))
        entropy_layer = entropy_layer / max_entropy
        if dtype:
            _limit_value = np.iinfo(dtype).max
            entropy_layer = (entropy_layer * _limit_value).astype(dtype)
    return entropy_layer
