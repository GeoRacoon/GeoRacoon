from __future__ import annotations
from typing import Callable
from collections.abc import Callable, Collection, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy

from .helper import dtype_range
from .io import load_block
from .prepare import get_view, relative_view


def select_category(data:NDArray,
                    category: int | list[int],
                    as_dtype: type = np.uint8,
                    limits: tuple | None = None):
    """Filter for particular category or categories


    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the category of each pixel
    category:
      The specific category or list of categories to select
    as_dtype:
      The data-type of the output matrix

      ..note::
        The output matrix will contain the maximal value possible for this data
        type in all the cells in `data` that match the category and the
        minimal value in the cells associated to any other category.

    limits:
        Optional custom limits to use. If provided `limits` must contain two
        values (`is`, `isnot`) that will be used to indicate if a cell is of that
        category or not.

    Returns
    -------
    np.array:
      Matrix of type `as_dtype` in the same shape of `data`
    """
    if limits:
        _is, _is_not = limits
    else:
        _is, _is_not = dtype_range(as_dtype)

    if isinstance(category, int):
        _selected = [category,]
    else:
        _selected = category
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


def get_max_entropy(nbr:int):
    """Maximal entropy value possible for a given number of categories

    Parameters
    ----------
    nbr: int
      The number of different categories considered

    Returns
    -------
    float
      The maximal entropy defined be a uniform distribution
    """
    return entropy(np.ones(nbr))


def get_categories(data:NDArray, )->list[int]:
    """Return the list of categories present in the data.

    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the category of each pixel

    Returns
    -------
    list
      List of unique categories present in the data
    """
    categories = np.unique(data)
    categories.sort()
    print("Inferring the number of categories from the provided data."
          f"\nGot:\n\t{categories}")
    return list(map(int, categories))


def get_category_data(data:NDArray,
                      category:int | list[int],
                      img_filter:Callable|None=None,
                      filter_params:dict|None=None,
                      output_dtype:type|None=None,
                      as_dtype:type=np.uint8)->NDArray:
    """Return the data of a single category, optionally after applying a filter

    ..Note::
      If no image filter is provided then either `as_type` or `output_dtype`
      must be set to the data type the output array should be in.

      If an image filter is provided then `as_type` will convert the data prior to
      applying the filter.

    Parameters
    ----------
    data: np.array
      Matrix of type `int` indicating the per-cell category
    category:
      The specific category (or categories) to select
    img_filter:
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian
    filter_params:
      Parameter to pass to the filter callable
    output_dtype:
      Set the data-type of the resulting image

      ..Note::
        This can be particularly useful if you apply an image filter.

        For example, a Gaussian filter returns a map of type `np.float64` with
        values in $[0, 1]$.
        With `output_dtype=np.uint8` these values are mapped to the range [0, 255]
        and an array of type `np.uint8` is returned with a considerably smaller
        memory footprint.

    as_dtype:
      The data-type of the `np.array` that will encode the category prior to
      applying the (optional) filter.

      ..Note::
        The default value of `np.uint8` should be the best choice in most cases.
        Only if the data contains more than 255 different categories it would make
        sense to change it to `np.uint16`.
    """
    # strip the category/categories
    _data = select_category(data, category, as_dtype=as_dtype)
    filter_params = filter_params or dict()
    # apply the image filter if provided
    if img_filter:
        _data = apply_filter(_data, img_filter, **filter_params)
        # convert to the desired output type
        if output_dtype:
            n_max, _ = dtype_range(output_dtype)
            _data = _data * n_max
            _data = _data.astype(output_dtype, copy=False)
    return _data


def get_filtered_categories(data:NDArray,
                            categories: None|Collection=None,
                            img_filter=None,
                            output_dtype:type|None=np.uint8,
                            filter_params:dict|None=None)->dict[int, NDArray]:
    """Extract each category into a separate `np.array` and apply an image filter

    ..Note::

      See `get_category_data` for further details about the category extraction
      from `data`.

    Parameters
    ----------
    data: np.array
      A land-cover type matrix or any other matrix with integers
    categories: Collection or None
      A list or other collection with the categories to extract.
      If not provided then all categories found in `data` are included.
    img_filter: callable
      a filter function to apply on the np.array of each category.
      See e.g. skimage.filter.gaussian
    output_dtype:
      Specify what data type to use for the returned data.
    filter_params:
      Parameter to pass to the filter callable

    Returns
    -------
    dict:
      For each category (key) the filtered data
    """
    if categories is None:
        categories = get_categories(data)
    all_categories = dict()
    for category in categories:
        _data = get_category_data(data=data, category=category,
                                  img_filter=img_filter,
                                  filter_params=filter_params,
                                  output_dtype=output_dtype)
        all_categories[category] = _data
    return all_categories


def compute_entropy(data_arrays: Sequence[NDArray],
                    normed:bool=True,
                    output_dtype:type|None=np.uint8,
                    **entropy_params)->NDArray:
    """Per cell entropy computed over a series of data arrays

    Parameters
    ----------
    data_arrays:
      A series of data arrays to stack and compute the per-cell entropy for
    normed:
      Determines if the values in the provided arrays should be normed or not.
    output_dtype:
      Set the data-type of the resulting `np.array`

      ..Note::
        This argument is ignored if `normed=False`.

        scipy.stats.entropy returns data of type `np.float64` which might consume
        a considerable amount of memory, when applied to larger arrays.

        With `output_dtype=np.uint8` entropy values are mapped to the range [0, 255]
        and an array of type `np.uint8` is returned with a considerably smaller
        memory footprint.

    **entropy_params:
      You might add further keyword arguments that will be passed to
      `scipy.stats.entropy`

    Returns
    -------
    entropy:
      A `np.array` with identical shape as the elements in `data_arrays` holding the
      per-cell entropy
    """
    # calculate the entropy
    _stacked = np.stack(data_arrays, axis=2)
    entropy_array = entropy(_stacked, axis=2, **entropy_params)
    if normed:
        max_entropy = get_max_entropy(len(data_arrays))
        entropy_array = entropy_array / max_entropy
        if output_dtype:
            _max, _ = dtype_range(output_dtype)
            entropy_array = (entropy_array * _max).astype(output_dtype)
    return entropy_array


def get_entropy(data:NDArray,
                categories:Collection|None=None,
                normed:bool=False,
                img_filter:Callable|None=None,
                output_dtype:type|None=None,
                filter_params:dict|None=None,
                entropy_params:dict|None=None)->NDArray:
    """Compute the Shannon entropy per cell directly from a 2D array of categorical data


    ..Note::
      This function relies on `get_filtered_categories` to create per category indicator
      arrays on which then a filter is applied, i.e. `img_filter` if provided.
      The resulting sequence of arrays is then passed to `compute_entropy` to calculate
      the per-cell entropy.

    Parameters
    ----------
    data:
      A categorical matrix (i.e. with integer values)
    categories:
      A collection of categories to extract individual bands for.
      If not provided then all categories are included
    normed:
      Determine whether or not the entropy should be normalized.
      If set to `True` each cell is normed by the maximal entropy value
      possible, i.e. `entropy(np.ones(len(categories)))`.
    output_dtype:
      The data-type to use for the returned array.

      ..note::

        This argument is only taken into account if `normed=True`.

    img_filter:
      A filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian

    filter_params:
      Optional parameter to pass to the filter callable, `img_filter`
    entropy_params:
      Optional parameter to pass to scipy.stats.entropy

    Returns
    -------
    entropy:
      A `np.array` with identical shape as the elements in `data_arrays` holding the
      per-cell entropy
    """
    filter_params = filter_params or dict()
    blurred_categories = get_filtered_categories(data=data,
                                                 categories=categories,
                                                 img_filter=img_filter,
                                                 filter_params=filter_params)
    entropy_params = entropy_params or dict()
    return compute_entropy(data_arrays=tuple(blurred_categories.values()),
                           normed=normed,
                           output_dtype=output_dtype, **entropy_params)

def view_blurred(source:str,
                 view:tuple[int,int,int,int],
                 inner_view:tuple[int,int,int,int],
                 categories:Collection|None,
                 img_filter:Callable,
                 filter_params:dict = dict(),
                 output_dtype:type|None = np.uint8,
                 **tags):

    """Uses a tif file with categorical data to compute blurred binary arrays

    The provided tif file must contain a band with categorical data (i.e. of type `uint`).
    You may use the `**tags` to specify which band to read, by default only the
    first band is read.
    The method then generates for each of the category first an indicator array
    (i.e. a dichotomous array indicating the presence/absence of a category)
    and then applies the filter method to each of these arrays.

    ..Note::
      This method will move to the `parallel` sub-module

    Parameters
    ----------
    source:
      The path to the tif file to load
    view:
      defines the view of the data array to update
    inner_view:
      defines the inner part of the view without border effects
    categories:
      A collection of categories to extract and create individual arrays for
    img_filter:
      a filter function that will be applied to each category indicator array
    filter_params:
      A dictionary with the keyword arguments to pass to the filter function
    output_dtype:
      Set the data type of the arrays that are returned.

      ..Note::
        If provided, the output of the filter function will be rescaled to the range of
        this data type.

        See `get_category_data` for further details.

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.    

      See `io.load_block` for further details.
    """
    # read out block from original file
    result = load_block(source, view=view, scaling_params=None, **tags)
    data = result.pop('data')
    print(f"{data.shape=}")
    # transform = result.pop('transform')
    # orig_profile = result.pop('orig_profile')
    # perform blur
    blurred_categories = get_filtered_categories(
        data=data,
        categories=categories,
        img_filter=img_filter,
        filter_params=filter_params,
        output_dtype=output_dtype
    )
    # get the relative view
    for category, data in blurred_categories.items():
        blurred_categories[category] = np.copy(
            get_view(data,
                     relative_view(view, inner_view))
        )
    return dict(data=blurred_categories, view=inner_view)


def  view_entropy(category_arrays:dict[int, NDArray],
                  view:tuple[int,int,int,int],
                  normed:bool = True,
                  output_dtype:type|None = np.uint8):
    """Return a per-cell entropy computed from the per category arrays.

    ..Note::
      The `view` parameter is simply passed along and returned with in order to 
      keep track of where the `category_arrays` data belongs.

      This method will move to the `parallel` sub-module

    Parameters
    ----------
    data_arrays:
      A series of data arrays to stack and compute the per-cell entropy for
    view:
      defining the view of the data arrays to consider
    output_dtype:
      Set the data-type of the returned array.

      See `compute_entropy` for further details
    """
    entropy_array = compute_entropy(
        data_arrays=tuple(category_arrays.values()),
        normed=normed,
        output_dtype=output_dtype,
    )
    return dict(data=entropy_array, view=view)


def get_entropy_view(source:str,
                     view:tuple[int,int,int,int],
                     inner_view:tuple[int,int,int,int],
                     categories: Collection,
                     img_filter,
                     filter_params:dict=dict(),
                     entropy_as_ubyte: bool = False,
                     blur_as_int: bool = False,
                     normed:bool=True,
                     **tags):
    """Returns the entropy for some categories over a view from a tif file

    ..Warning::
      This function is deprecated and should not be used

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.    

      See `io.load_block` for further details.
    """
    if blur_as_int:
        blur_output_dtype = np.uint8
    else:
        blur_output_dtype = np.float64
    if entropy_as_ubyte:
        entropy_output_dtype = np.uint8
    else:
        entropy_output_dtype = np.float64

    blurred_data = view_blurred(
        source=source,
        view=view,
        inner_view=inner_view,
        categories=categories,
        img_filter=img_filter,
        filter_params=filter_params,
        output_dtype=blur_output_dtype,
        **tags
    )
    assert blurred_data['view'] == inner_view

    entropy_view = view_entropy(category_arrays=blurred_data['data'],
                                view=blurred_data['view'],
                                output_dtype=entropy_output_dtype,
                                normed=normed)
    entropy_view['view'] = blurred_data['view']
    return entropy_view
