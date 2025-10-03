from __future__ import annotations
from typing import Callable
from collections.abc import Callable, Collection, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from skimage.filters import gaussian

from riogrande.helper import dtype_range, convert_to_dtype
from riogrande.io import Source, Band, load_block
from riogrande.prepare import get_view, relative_view

from .filters import bpgaussian


def select_category(data:NDArray,
                    category: int | list[int],
                    as_dtype: type|str = "uint8",
                    limits: tuple | None = None):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
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
    # is_needed (only internally)
    # needs_work (docs)
    # is_tested
    # usedin_processing (though might be part of the io module)
    if isinstance(as_dtype, str):
        _as_dtype = np.dtype(as_dtype)
    else:
        _as_dtype = as_dtype
    if limits:
        _is, _is_not = limits
    else:
        _is, _is_not = map(lambda x: np.array(x).astype(_as_dtype), dtype_range(_as_dtype))

    if isinstance(category, int):
        _selected = [category,]
    else:
        _selected = category
    return np.where(np.isin(data, _selected), _is, _is_not)


def _apply_filter(data, img_filter:Callable, **params):
    # TODO: is_needed - no_work - is_tested - usedin_processing
    """Apply a filter to the provided data

    Parameters
    ----------
    data: np.array
    img_filter: callback
    params: dict
      keyword parameter passed as is to the callback function
    """
    # is_needed (internally and for tests)
    # needs_work (docs)
    # not_tested (directly)
    # usedin_processing (part of io)
    return img_filter(data, **params)


def get_max_entropy(nbr:int):
    # TODO: is_needed - no_work - is_tested - usedin_processing
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
    # is_needed (internally)
    # needs_work (make internal; docs)
    # not_tested (but used in tests)
    # usedin_processing
    return entropy(np.ones(nbr))


def get_categories(data:NDArray, )->list[int]:
    # TODO: is_needed - no_work - is_tested - usedin_processing
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
    # is_needed (tests; motly internal - only plotting otherwise)
    # no_work
    # not_tested (used in tests)
    # usedin_processing (primarily, but ideally part of a io module)
    categories = np.unique(data)
    categories.sort()
    print("Inferring the number of categories from the provided data."
          f"\nGot:\n\t{categories}")
    return list(map(int, categories))


def get_category_data(data:NDArray,
                      category:int | list[int],
                      img_filter:Callable|None=None,
                      filter_params:dict|None=None,
                      filter_output_range:tuple|None=None,
                      as_dtype:type|str|None=None,
                      output_range:tuple|None=None,
                      data_as_dtype:type|str="uint8")->NDArray:
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """Return the data of a single category, optionally after applying a filter

    .. note::

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
    filter_output_range:
      Optional tuple to pass the output range of the applied filter
    output_dtype:
      Set the data-type of the resulting image
    output_range:
      Optional tuple to overwrite the [min,max] range of the output dtype.

      .. note::

        This can be particularly useful if you apply an image filter.

        For example, a Gaussian filter returns a map of type `np.float64` with
        values in $[0, 1]$.
        With `output_dtype="uint8"` these values are mapped to the range [0, 255]
        and an array of type `np.uint8` is returned with a considerably smaller
        memory footprint.

    data_as_dtype:
      The data-type of the `np.array` that will encode the category prior to
      applying the (optional) filter.

      ..Note::
        The default value of `np.uint8` should be the best choice in most cases.
        Only if the data contains more than 255 different categories it would make
        sense to change it to `np.uint16`.
    """
    # is_needed (mostly for tests)
    # no_work
    # not_tested
    # usedin_processing (but could go to io)

    # strip the category/categories
    _data = select_category(data, category, as_dtype=data_as_dtype)
    filter_params = filter_params or dict()
    # apply the image filter if provided
    _data = filter_data(data=_data,
                        img_filter=img_filter,
                        filter_params=filter_params,
                        filter_output_range=filter_output_range,
                        as_dtype=as_dtype,
                        output_range=output_range)
    return _data


def view_data(source:Source|str,
              bands: list[Band|int]|None,
              view:tuple[int,int,int,int],
              in_range:None|NDArray|Collection,
              as_dtype:type|str|None,
              output_range:None|NDArray|Collection,
              ):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """Get a data view from tif file + optionally convert and rescale the values

    You may use the `**tags` to specify which band to read, by default only the
    first band is read.

    If tags for multiple bands are provided then all matching bands are returned

    Parameters
    ----------
    source:
      The path to the tif file to load
    bands:
        A collection of strings or `io.Band` object the specify which bands to use
    view:
      defines the view of the data array to update
    in_range:
        an array or list from which min and max will be used as input range
    as_dtype:
      Set the data type of the arrays that are returned.

      ..note::
        If provided, the loaded data will be rescaled to the range of
        this data type or `out_range` (if provided).

    output_range:
      Optional tuple to overwrite the [min,max] range of the output

      See `io.load_block` for further details.
    """
    # is_needed (internal only)
    # no_work
    # not_tested
    # usedin_processing (if at all; but might become part of io module)

    # read out block from original file

    if not isinstance(source, Source):
        source = Source(path=source)
    if bands is None:
        # print('No specific bands selected, using all')
        bands = source.get_bands()
    elif any(isinstance(band, int) for band in bands):
        bands = [band if isinstance(band, Band) else source.get_band(bidx=band)
                 for band in bands]
    # print(f"{bands=}")
    data_views = dict()
    for band in bands:
        result = band.load_block(view=view, scaling_params=None)
        data = result.pop('data')
        if as_dtype is not None:
            data = convert_to_dtype(data=data,
                                    as_dtype=as_dtype,
                                    in_range=in_range,
                                    out_range=output_range)
        data_views[band] = dict(data=data, view=view)
    return data_views


def filter_data(data:NDArray,
                replace_nan_with: int|float|None = None,
                img_filter=None,
                filter_output_range:tuple|None=None,
                filter_params:dict|None=None,
                as_dtype:type|str|None=None,
                output_range:tuple|None=None,
                )->NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_processing
    # TODO: remove all default arguments!
    """Applies a filter to an `np.array`

    Parameters
    ----------
    data: np.array
      A 2D numpy array

    replace_nan_with:
      Replace nan values in `data` with the provided value

    img_filter: callable
      a filter function to apply on the np.array

      See e.g. skimage.filter.gaussian

    as_dtype:
      Specify what data type to use for the returned data.

    filter_output_range:
      Optional tuple to specify the data range returned by the filter

    output_range:
      Optional tuple to specify the data range into which the data returend
      by the filter should be rescaled into.

      ..note::
        This can be used to map to the range `[0,1]` for `floats`:

        ```python
            `as_dtype="float64"`
            `output_range=(0.0, 1.0)`
        ```
    filter_params:
      Parameter to pass to the filter callable

    Returns
    -------
    np.array:
      Resulting data array
    """
    # is_needed (internally only )
    # needs_work (see TODO)
    # is_tested
    # usedin_processing (part of io)

    # check if nan exists
    # TODO: move this out of this function
    if np.isnan(np.sum(data)) and replace_nan_with is None:
        if img_filter == gaussian:  # only warn for gaussian
            warnings.warn(
                f"Raster array has NaN - this will crop areas where the given "
                "function encounters NaNs. If needed: Set a replacement value "
                f"for NaNs ({replace_nan_with=})")
    # create a mask for NaN values (for restoring later)
    nan_mask = np.isnan(data)
    # replace nan with a provided value
    if replace_nan_with is not None:
        data = np.nan_to_num(data, nan=replace_nan_with)
    # apply the filter if one is chosen
    if img_filter is not None:
        if filter_output_range is None and as_dtype is not None:
            warnings.warn(
                f"We are applying the filter {img_filter} and convert the "
                f"resulting output to {as_dtype} without knowing the range of "
                "the data produced by the filter. Rescaling to another data "
                "type is likely to produce unexpected results if the input "
                "range is unknown (e.g. if the filter outputs floats then the "
                "entire range of float is used as input range, which is not "
                "what you want if the filter produces values only in the range "
                "[0, 1], for example. Please set the filter output range with "
                "`filter_output_range` to avoid unpleasant surprises."
            )
        filter_params = filter_params or dict()
        if img_filter in (gaussian, bpgaussian):
            if np.issubdtype(data.dtype, np.integer) and \
            filter_params.get('preserve_range', None) is None:
                # the gaussian filter will rescale the input
                warnings.warn(
                    "A gaussian filter will be applied to data of type int. "
                    "The gaussian filter will first rescale the data to "
                    "floats ([0,1] for uints and [-1,1] for ints), thus the "
                    "output will be of a different data type and scale. "
                    "To avoid this set `filter_params['preserve_range']=True`"
                )
        _data = _apply_filter(data, img_filter, **filter_params)
    else:
        _data = data
    # now we convert and optionally rescale
    _data = convert_to_dtype(data=_data,
                             as_dtype=as_dtype,
                             in_range=filter_output_range,
                             out_range=output_range)
    # restore NaNs to original positions (if possible)
    if np.issubdtype(_data.dtype, np.floating):
        _data[nan_mask] = np.nan
    return _data


def view_filtered(source: Source | str,
                  view: tuple[int, int, int, int],
                  inner_view: tuple[int, int, int, int],
                  data_in_range: None | NDArray | Collection = None,
                  data_as_dtype: type | str | None = None,
                  data_output_range: None | NDArray | Collection = None,
                  replace_nan_with: int | float | None = None,
                  img_filter: Callable | None = None,
                  filter_params: dict | None = None,
                  filter_output_range: tuple | None = None,
                  as_dtype: type | str | None = None,
                  output_range: tuple | None = None,
                  bands: list[Band | int] | None = None,
                  selector_band: Band | None = None,
                  ):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    """Extracts and possibly converts a view from the source file

    """
    # not_needed (used in a unneeded function in parallel)
    # needs_work (docs)
    # not_tested
    # usedin_both (probably not used at all!)
    data_views = view_data(source=source,
                           bands=bands,
                           view=view,
                           in_range=data_in_range,
                           as_dtype=data_as_dtype,
                           output_range=data_output_range)

    if selector_band is not None:
        selector_data = selector_band.load_block(view=view)['data']
        selectors = np.unique(selector_data, ).tolist()
        if np.nan in selectors:
            selectors.remove(np.nan)
    else:
        selectors = [0, ]
        selector_data = np.zeros(shape=(view[3], view[2]), dtype=np.uint8)

    # get output data type for final output - to use for filtered aggregation later
    if isinstance(as_dtype, str):
        _as_dtype = np.dtype(as_dtype)
    else:
        _as_dtype = as_dtype

    filtered_datas = {}
    for band, data_view in data_views.items():
        # We need to use the same no data for the filtering as defined in the views
        # TODO: we could also get this nodata from checking data_as_dtype
        _nodata = 0
        dv_dtype = data_view['data'].dtype
        if np.issubdtype(dv_dtype, np.floating):
            _nodata = np.nan

        _filtered_data = np.zeros(shape=(view[3], view[2]), dtype=_as_dtype)

        for select in selectors:
            _selector = np.where(selector_data == select, True, False)
            select_data_view = np.where(_selector, data_view['data'], _nodata)

            _part_filtered_data = filter_data(
                data=select_data_view,
                replace_nan_with=replace_nan_with,
                img_filter=img_filter,
                filter_params=filter_params,
                filter_output_range=filter_output_range,
                as_dtype=as_dtype,
                output_range=output_range)
            _filtered_data += np.where(_selector, _part_filtered_data, 0)
        # only keep the inner view
        filtered_datas[band.get_bidx()] = np.copy(
            get_view(_filtered_data,
                     relative_view(view, inner_view))
        )
    return dict(data=filtered_datas, view=inner_view)


def get_filtered_categories(data: NDArray,
                            categories: None | Collection = None,
                            img_filter: None | Callable = None,
                            output_dtype: type | str | None = "uint8",
                            output_range: tuple | None = None,
                            filter_output_range: tuple | None = None,
                            filter_params: dict | None = None) -> dict[int, NDArray]:
    # TODO: is_needed - needs_work - is_tested - usedin_processing
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
    # is_needed (only in tests)
    # needs_work (docs)
    # not_tested (used in tests thought)
    # usedin_processing

    if categories is None:
        categories = get_categories(data)
    all_categories = dict()
    for category in categories:
        _data = get_category_data(data=data, category=category,
                                  img_filter=img_filter,
                                  filter_params=filter_params,
                                  filter_output_range=filter_output_range,
                                  as_dtype=output_dtype,
                                  output_range=output_range,
                                  )
        all_categories[category] = _data
    return all_categories


def compute_entropy(data_arrays: Sequence[NDArray],
                    normed: bool = True,
                    max_entropy_categories: int | None = None,
                    as_dtype: type | str | None = None,
                    output_range: tuple | None = None,
                    **entropy_params) -> NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_processing
    """Per cell entropy computed over a series of data arrays

    Parameters
    ----------
    data_arrays:
      A series of data arrays to stack and compute the per-cell entropy for
    normed:
      Determines if the values in the provided arrays should be normed or not.

      .. note::

        If `normed=False` then the entropy is not rescaled before it is
        returned.

    max_entropy_categories:
      If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
      Same as the `as_dtype`, this argument is ignored if `normed=False`.

    as_dtype:

      Set the data-type of the resulting `np.array`

      .. note::

        In most cases you want to use this argument only if `normed=True`
        (the default).

        scipy.stats.entropy returns data of type `np.float64` which might consume
        a considerable amount of memory, when applied to larger arrays.

        With `as_dtype="uint8"` and `normed=True` entropy values are mapped to
        the range [0, 255] and an array of type `np.uint8` is returned with a
        considerably smaller memory footprint.

    output_range:
      Set the output range for the resulting `np.array`

      ..Note::
        This argument is ignored if `normed=False` as the entropy is not
        bounded.

    **entropy_params:
      You might add further keyword arguments that will be passed to
      `scipy.stats.entropy`

    Returns
    -------
    entropy:
      A `np.array` with identical shape as the elements in `data_arrays` holding the
      per-cell entropy
    """
    # is_needed (internally and in tests)
    # no_work
    # not_tested
    # usedin_processing

    # calculate the entropy
    _stacked = np.stack(data_arrays, axis=2)
    entropy_array = entropy(_stacked, axis=2, **entropy_params)
    in_dtype = entropy_array.dtype

    if isinstance(as_dtype, str):
        _as_dtype = np.dtype(as_dtype)
    else:
        _as_dtype = as_dtype

    if normed:
        if max_entropy_categories is None:
            max_entropy = get_max_entropy(len(data_arrays))
        else:
            max_entropy = get_max_entropy(max_entropy_categories)
        # We normalize the entropy by setting the in_range accordingly
        if np.issubdtype(_as_dtype, np.floating) and output_range is None:
            # use the normalization range [0, 1] for float output by default
            output_range = (0.0, 1.0)
        elif np.issubdtype(_as_dtype, np.integer) and output_range is None:
            # TODO: we should decide whether we want this for entropy - but if normed I feel it helps with usability
            #       useres will not want to sset this if they use uint8 I guess
            # use the general possible range for Integers
            _intmax, _intmin = dtype_range(_as_dtype)
            output_range = (_intmin, _intmax)
        input_range = [0.0, max_entropy],
    else:
        if output_range is not None:
            warnings.warn(
                f"Calculating the entropy with {normed=} ignores "
                f"{output_range=} as a non-normalized entropy value is not "
                "bounded and can thus not be mapped to a data range."
            )
        if as_dtype is not None and str(in_dtype) != str(as_dtype):
            # we do not normalize but convert
            warnings.warn(
                f"The computed entropy will be converted from {str(in_dtype)} "
                f"to {as_dtype} without rescaling. If this is not what you "
                "want set the `output_range` parameter."
            )
        output_range = None
        input_range = None

    # convert (an rescale if normed)
    entropy_array = convert_to_dtype(data=entropy_array,
                                     as_dtype=_as_dtype,
                                     in_range=input_range,
                                     out_range=output_range)
    return entropy_array


def compute_interaction(data_arrays: Sequence[NDArray],
                        input_dtype: type|str|None=None,
                        standardize:bool=False,
                        normed:bool=True,
                        output_dtype:type|str|None=None,
                        output_range: tuple | None = None,
                        **interaction_params)->NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_processing
    r"""Per cell interaction computed over a series of data arrays
    For 'float' inputs:
        .. math::
            LC_i \times LC_j

    For 'uint8' inputs:
        .. math::
            \frac{\left(\frac{LC_i}{255} \times \frac{LC_j}{255}\right)}{\frac{1}{n^2}} \times 255

    Parameters
    ----------
    data_arrays:
      A series of data arrays to stack and compute the per-cell interaction for
    standardize:
      Whether the interaction (strength) should be standardized by the fraction of area used (A*B)/(A+B)
    normed:
      Determines if the values in the provided arrays should be normed or not.
    output_dtype:
      Set the data-type of the resulting `np.array`

    Returns
    -------
    interaction:
      A `np.array` with identical shape as the elements in `data_arrays` holding the
      per-cell interaction between given layers
    """
    # is_needed (internally; as non-parallelized version)
    # no_work
    # is_tested
    # usedin_processing

    # TODO: all of this does not work with floats yet --> implement

    array_dtype = data_arrays[0].dtype
    if array_dtype != input_dtype:
        raise ValueError(f"Array data type {array_dtype} does not match provided input data type {input_dtype}")

    # define rescaling based on input type
    if input_dtype:
        if isinstance(input_dtype, str):
            input_dtype = np.dtype(input_dtype)
        _max_scale, _ = dtype_range(input_dtype)
        if np.issubdtype(input_dtype, np.floating):
            _max_scale = 1

    # calculate the interaction (A * B)
    interaction_array = np.ones_like(data_arrays[0], dtype=float)
    for arr in data_arrays:
        interaction_array *= (arr / _max_scale)

    # standardize by the sum (A + B) -> result is A*B/(A+B)
    if standardize:
        standardize_array = np.zeros_like(data_arrays[0], dtype=float)
        for arr in data_arrays:
            standardize_array += (arr / _max_scale)
        interaction_array = np.divide(interaction_array, standardize_array,
                                      out=np.zeros_like(interaction_array, dtype=float),
                                      where=standardize_array != 0)
    if normed:
        max_interaction = 1 / len(data_arrays)**len(data_arrays)
        interaction_array = interaction_array / max_interaction
        if output_dtype:
            if isinstance(output_dtype, str):
                output_dtype = np.dtype(output_dtype)
            _max, _ = dtype_range(output_dtype)
            if np.issubdtype(output_dtype, np.floating):
                _max = 1
                interaction_array = (interaction_array * _max).astype(output_dtype)
            else:
                # when rescaling to uint8 it is important to not have values > 1 before rescaling (else 256 would be 0)
                interaction_array[interaction_array > 1] = 1
                # np.ceil relevant to avoid artefacts from rounding (when using unit8)
                interaction_array = np.ceil((interaction_array * _max)).astype(output_dtype)
    return interaction_array


def get_entropy(data:NDArray,
                categories:Collection|None=None,
                normed:bool=False,
                max_entropy_categories:int|None=None,
                img_filter:Callable|None=None,
                as_dtype:type|str|None=None,
                output_range:tuple|None=None,
                filter_params:dict|None=None,
                entropy_params:dict|None=None,
                filter_output_range:tuple|None=None,
                **params)->NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_processing
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
    max_entropy_categories:
      If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
      This argument is ignored if `normed=False`.
    as_dtype:
      The data-type to use for the returned array.

      ..note::

        This argument is only taken into account if `normed=True`.
    output_range:
      The data-range to use for the returned array.

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
    # is_needed
    # needs_work (docs; see TODO)
    # is_tested
    # usedin_processing

    # TODO: This function does not allow the set `as_dtype` of
    # `get_filtered_categories`
    # > Change call signature to
    #   `get_entropy(blur_params:dict, entropy_params:dict,...)`
    filter_params = filter_params or dict()
    blur_output_dtype = params.pop('blur_output_dtype', None)
    blurred_categories = get_filtered_categories(data=data,
                                                 categories=categories,
                                                 img_filter=img_filter,
                                                 filter_output_range=filter_output_range,
                                                 output_dtype=blur_output_dtype,
                                                 filter_params=filter_params)
    entropy_params = entropy_params or dict()
    return compute_entropy(data_arrays=tuple(blurred_categories.values()),
                           normed=normed,
                           max_entropy_categories=max_entropy_categories,
                           as_dtype=as_dtype,
                           output_range=output_range,
                           **entropy_params)


def view_blurred(source:str,
                 view:tuple[int,int,int,int],
                 inner_view:tuple[int,int,int,int],
                 categories:Collection|None,
                 img_filter:Callable,
                 filter_params:dict = dict(),
                 filter_output_range:tuple|None=None,
                 output_dtype:type|str|None = "uint8",
                 output_range:tuple|None=None,
                 **tags):
    # TODO: is_needed - needs_work - is_tested - usedin_processing
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
    output_range:
      Optional argument to set an output range of the filter method.

      ..Note::
        If not provided and the filter method produces data of any float type,
        hen the output range is assumed to be `[0, 1]` and values outside this
        range will be set to the closer limit (e.g. `1.2` will become `1` and
        becomes `0.0`)).

    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `io.load_block` for further details.
    """
    # is_needed
    # needs_work (docs - minor)
    # is_tested
    # usedin_processing

    # read out block from original file
    result = load_block(source, view=view, scaling_params=None, **tags)
    data = result.pop('data')
    # print(f"{data.shape=}")
    # transform = result.pop('transform')
    # orig_profile = result.pop('orig_profile')
    # perform blur
    blurred_categories = get_filtered_categories(
        data=data,
        categories=categories,
        img_filter=img_filter,
        filter_params=filter_params,
        filter_output_range=filter_output_range,
        output_dtype=output_dtype,
        output_range=output_range,
    )
    # get the relative view
    for category, data in blurred_categories.items():
        blurred_categories[category] = np.copy(
            get_view(data,
                     relative_view(view, inner_view))
        )
    return dict(data=blurred_categories, view=inner_view)


def view_entropy(category_arrays:dict[int, NDArray],
                  view:tuple[int,int,int,int],
                  normed:bool = True,
                  max_entropy_categories: int|None = None,
                  output_dtype:type|str|None = None,
                  output_range:tuple|None = None):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
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
    normed:
    max_entropy_categories:
    output_dtype:
      Set the data-type of the returned array.

      See `compute_entropy` for further details
    max_entropy_categories:
      If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
      This argument is ignored if `normed=False`.
    """
    # is_needed
    # no_work
    # not_tested
    # usedin_processing
    entropy_array = compute_entropy(
        data_arrays=tuple(category_arrays.values()),
        normed=normed,
        max_entropy_categories=max_entropy_categories,
        as_dtype=output_dtype,
        output_range=output_range,
    )
    return dict(data=entropy_array, view=view)


def view_interaction(category_arrays:dict[int, NDArray],
                     view:tuple[int,int,int,int],
                     input_dtype: type|str|None = np.uint8,
                     standardize: bool = False,
                     normed:bool = True,
                     output_dtype: type | str | None = None,
                     output_range: tuple | None = None):
    # TODO: is_needed - needs_work - not_tested - usedin_processing
    """Return a per-cell interaction computed from the per category arrays.

    Parameters
    ----------
    data_arrays:
      A series of data arrays to stack and compute the per-cell interaction for
    view:
      defining the view of the data arrays to consider
    output_dtype:
      Set the data-type of the returned array.

      See `comptue_interaction` for further details
    """
    # is_needed
    # needs_work (docs)
    # not_tested
    # usedin_processing
    interaction_array = compute_interaction(
        data_arrays=tuple(category_arrays.values()),
        input_dtype=input_dtype,
        standardize=standardize,
        normed=normed,
        output_dtype=output_dtype,
        output_range=output_range,
    )
    return dict(data=interaction_array, view=view)


def get_entropy_view(source:str,
                     view:tuple[int,int,int,int],
                     inner_view:tuple[int,int,int,int],
                     categories: Collection,
                     img_filter,
                     filter_params:dict=dict(),
                     max_entropy_categories:int|None=None,
                     blur_as_int:bool|None=None,
                     filter_output_range:tuple|None=None,
                     blur_output_dtype:type|str|None=None,
                     output_dtype:type|str|None=None,
                     output_range:tuple|None=None,
                     normed:bool=True,
                     **tags):
    # TODO: not_needed
    """Returns the entropy for some categories over a view from a tif file

    ..Warning::
      This function is deprecated and should not be used

    Parameters
    ----------
    max_entropy_categories:
      If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
      This argument is ignored if `normed=False`.

    output_range:
      The data-range to use for the returned array.

      ..note::
        This argument is only taken into account if `normed=True`.
    **tags:
      Arbitrary number of keyword arguments to describe the band to select.

      See `io.load_block` for further details.
    """
    # is_needed (only in tests for parallel)
    # no_work
    # not_tested
    # usedin_processing
    warnings.warn("This function is deprecated and will be removed",
                  category=DeprecationWarning)
    if blur_as_int is None:
        assert blur_output_dtype is not None
    else:
        if blur_as_int:
            blur_output_dtype = "uint8"
        else:
            blur_output_dtype = "float64"

    blurred_data = view_blurred(
        source=source,
        view=view,
        inner_view=inner_view,
        categories=categories,
        img_filter=img_filter,
        filter_params=filter_params,
        filter_output_range=filter_output_range,
        output_dtype=blur_output_dtype,
        **tags
    )
    assert blurred_data['view'] == inner_view

    entropy_view = view_entropy(category_arrays=blurred_data['data'],
                                view=blurred_data['view'],
                                output_dtype=output_dtype,
                                output_range=output_range,
                                normed=normed,
                                max_entropy_categories=max_entropy_categories,
                                )
    entropy_view['view'] = blurred_data['view']
    return entropy_view
