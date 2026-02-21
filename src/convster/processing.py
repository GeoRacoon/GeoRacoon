"""
Preparing and processing data
"""

from __future__ import annotations
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


def select_category(data: NDArray, category: int | list[int],
                    as_dtype: type | str = "uint8", limits: tuple | None = None) -> NDArray:
    """
    Filter an array for particular category or categories.

    Parameters
    ----------
    data : NDArray
        Input matrix of integers indicating the category of each pixel.
    category : int or list[int]
        The category (or list of categories) to select.
    as_dtype : type or str
        Data type of the output matrix.

        .. note::
           The output matrix will contain the maximal value possible for this
           data type in cells that match `category`, and the minimal value
           in all other cells.
    limits : tuple or None
        Custom limits for output values. Must be a pair `(is_value, is_not_value)`.
        If provided, these override the default min/max values inferred from
        `as_dtype`.

    Returns
    -------
    NDArray
       Matrix of type `as_dtype` with the same shape as `data`.

    See Also
    --------
    :func:`get_categories` : Infer the list of categories from an array.
    :func:`get_category_data` : Extract and optionally filter data for one or more categories.

    Examples
    --------
    >>> data = np.array([
    ...     [0, 1, 2],
    ...     [2, 1, 0],
    ...     [1, 0, 2]
    ... ])
    >>> select_category(data, category=1)
    array([[  0, 255,   0],
          [  0, 255,   0],
          [255,   0,   0]], dtype=uint8)

    >>> select_category(data, category=[1, 2], as_dtype="int16", limits=(1000, -1000))
    array([[-1000,  1000,  1000],
          [ 1000,  1000, -1000],
          [ 1000, -1000,  1000]], dtype=int16)
    """
    if isinstance(as_dtype, str):
        _as_dtype = np.dtype(as_dtype)
    else:
        _as_dtype = as_dtype
    if limits:
        _is, _is_not = limits
    else:
        _is, _is_not = map(lambda x: np.array(x).astype(
            _as_dtype), dtype_range(_as_dtype))

    if isinstance(category, int):
        _selected = [category, ]
    else:
        _selected = category
    return np.where(np.isin(data, _selected), _is, _is_not)


def get_categories(data: NDArray) -> list[int]:
    """
    Return the sorted list of categories present in the data.

    Parameters
    ----------
    data : NDArray
        Array of integers indicating the category of each pixel.

    Returns
    -------
    list of int
        Sorted list of unique categories present in the data.
        Uses :func:`numpy.unique` to determine unique values.

    See Also
    --------
    :func:`select_category` : Filter an array for a specific category.
    :func:`get_filtered_categories` : Extract all categories into separate arrays.

    Examples
    --------
    >>> a = np.array([[0, 1, 2],
    ...               [2, 1, 0],
    ...               [1, 0, 2]])
    >>> get_categories(a)
    [0, 1, 2]
    """
    categories = np.unique(data)
    categories.sort()
    print("Inferring the number of categories from the provided data."
          f"\nGot:\t{categories}")
    return list(map(int, categories))


def get_category_data(data: NDArray,
                      category: int | list[int],
                      img_filter: Callable | None = None,
                      filter_params: dict | None = None,
                      filter_output_range: tuple | None = None,
                      as_dtype: type | str | None = None,
                      output_range: tuple | None = None,
                      data_as_dtype: type | str = "uint8") -> NDArray:
    """
    Return the data of one or more categories, optionally after applying a filter.

    Parameters
    ----------
    data : NDArray
        Matrix indicating the per-cell category.
    category : int or list[int]
        The category (or categories) to extract.
    img_filter : Callable or None
        Filter function applied to the selected category data (e.g.,
        ``skimage.filters.gaussian``).
    filter_params : dict or None
        Parameters passed to ``img_filter``.
    filter_output_range : tuple or None
        Output value range to apply after filtering, if applicable.
    as_dtype : type or str or None
        Desired data type of the output array.
    output_range : tuple or None
        Custom value range ``(min, max)`` to which the output will be scaled.
        Useful when filters produce floating-point values.

        For example, a Gaussian filter returns a ``float64`` array with
        values in ``[0, 1]``. With ``as_dtype="uint8"``, these values are
        mapped to ``[0, 255]``, reducing memory usage.
    data_as_dtype : type or str
        Data type of the array used to encode the category mask before filtering.
        Default is ``"uint8"``. For datasets with more than 255 categories,
        ``"uint16"`` may be more appropriate.

    Returns
    -------
    Filtered or unfiltered array of the selected category, converted and
    scaled according to the specified options.

    Notes
    -----
    - If no image filter is provided, either ``as_dtype`` or ``output_range``
      must be set to define the data type or range of the output array.
    - If an image filter is provided, ``as_dtype`` converts the data before
      the filter is applied.

    See Also
    --------
    :func:`select_category` : Create a binary indicator array for a category.
    :func:`get_filtered_categories` : Apply this function across all categories.
    :func:`_filter_data` : Apply a filter and rescale the resulting data.
    """
    # strip the category/categories
    _data = select_category(data, category, as_dtype=data_as_dtype)
    filter_params = filter_params or dict()
    # apply the image filter if provided
    _data = _filter_data(data=_data,
                         img_filter=img_filter,
                         filter_params=filter_params,
                         filter_output_range=filter_output_range,
                         as_dtype=as_dtype,
                         output_range=output_range)
    return _data


def get_filtered_categories(data: NDArray,
                            categories: None | Collection = None,
                            img_filter: None | Callable = None,
                            output_dtype: type | str | None = "uint8",
                            output_range: tuple | None = None,
                            filter_output_range: tuple | None = None,
                            filter_params: dict | None = None) -> dict[int, NDArray]:
    """
    Extract each category from a data array into separate arrays and optionally apply a filter.

    Parameters
    ----------
    data : NDArray
        Array containing integer categories, e.g., a land-cover type matrix.
    categories : Collection or None
        Collection of categories to extract. If None, all categories in `data` are extracted.
    img_filter : Callable or None
        Callable to apply as a filter to each category array (e.g., `skimage.filters.gaussian`).
    output_dtype : type or str or None
        Data type for the returned arrays (default: "uint8").
    output_range : tuple or None
        Range to rescale the filtered arrays.
    filter_output_range : tuple or None
        Expected output range of the filter for proper scaling.
    filter_params : dict or None
        Dictionary of parameters to pass to the filter function.

    Returns
    -------
    dict
        A dictionary mapping each category to its filtered and optionally rescaled array.

    Notes
    -----
    - See :func:`get_category_data` for details on extracting category-specific data.

    See Also
    --------
    :func:`get_category_data` : Extract and optionally filter data for one category.
    :func:`get_categories` : Infer the list of categories from an array.
    """
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


def get_max_entropy(nbr: int) -> float:
    """
    Maximum entropy value for a given number of categories.

    The maximum Shannon entropy occurs when the distribution is uniform
    across `nbr` categories, i.e. all categories have equal probability.

    Parameters
    ----------
    nbr : int
        The number of categories.

    Returns
    -------
    float
        The maximal entropy for a uniform distribution with `nbr` categories.
        Computed using :func:`scipy.stats.entropy` with a uniform distribution.

    See Also
    --------
    :func:`compute_entropy` : Compute per-cell entropy over a series of data arrays.

    Examples
    --------
    >>> get_max_entropy(2)
    0.6931471805599453
    >>> get_max_entropy(10)
    2.302585092994046
    """
    return entropy(np.ones(nbr))


def compute_entropy(data_arrays: Sequence[NDArray],
                    normed: bool = True,
                    max_entropy_categories: int | None = None,
                    as_dtype: type | str | None = None,
                    output_range: tuple | None = None,
                    **entropy_params) -> NDArray:
    """
    Compute per-cell entropy over a series of data arrays.

    The input arrays are stacked along a new axis, and entropy is calculated for each cell.
    The resulting array can be normalized, converted to a different dtype, and rescaled
    to a specified output range.

    Parameters
    ----------
    data_arrays : Sequence[NDArray]
        Sequence of arrays to compute per-cell entropy over. All arrays must have the same shape.
    normed : bool
        If True (default), entropy values are normalized according to the maximum possible entropy.
        If False, the raw entropy is returned without rescaling.
    max_entropy_categories : int or None
        Maximum number of categories to use for normalization when `normed=True`.
        Ignored if `normed=False`.
    as_dtype : type or str or None
        Data type for the output array. Useful to reduce memory usage when `normed=True`.
    output_range : tuple or None
        Range to rescale normalized entropy values. Ignored if `normed=False`.
    **entropy_params : dict
        Additional keyword arguments passed to :func:`scipy.stats.entropy`.

    Returns
    -------
    np.ndarray
        Array of the same shape as the input arrays, containing the per-cell entropy.

    Notes
    -----
    - When `normed=True`, the entropy is mapped to [0, 1] for float outputs by default,
      or to the full range of the specified integer type if `as_dtype` is integer.
    - Converting to a different dtype without normalization may produce unbounded results.
    - For large arrays, using a smaller `as_dtype` (e.g., 'uint8') can save memory.
    - Normalization uses :func:`get_max_entropy` to determine the maximum entropy given
      the number of input arrays, and :func:`~riogrande.helper.convert_to_dtype` for
      rescaling.

    See Also
    --------
    :func:`get_max_entropy` : Compute the maximum entropy for a given number of categories.
    :func:`_get_entropy` : Internal wrapper combining blurring and entropy computation.

    Examples
    --------
    >>> data1 = np.array([[10, 5],
    ...                   [4, 1]])
    >>> data2 = np.array([[1, 5],
    ...                   [2, 9]])
    >>> compute_entropy([data1, data2], normed=True, as_dtype='float32')
    array([[0.439497  , 1.        ],
           [0.91829586, 0.4689956 ]], dtype=float32)
    """
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

    # convert (and rescale if normed)
    entropy_array = convert_to_dtype(data=entropy_array,
                                     as_dtype=_as_dtype,
                                     in_range=input_range,
                                     out_range=output_range)
    return entropy_array


def _get_entropy(data: NDArray,
                 categories: Collection | None = None,
                 normed: bool = False,
                 max_entropy_categories: int | None = None,
                 img_filter: Callable | None = None,
                 as_dtype: type | str | None = None,
                 output_range: tuple | None = None,
                 filter_params: dict | None = None,
                 entropy_params: dict | None = None,
                 filter_output_range: tuple | None = None,
                 **params) -> NDArray:
    # NOTE: This function is only used for testing purposes.
    """
    Compute the Shannon entropy per cell from a 2D categorical array.

    This method performs first a gaussian blurring, followed by a per-cell
    entropy calculation.

    Parameters
    ----------
    data : NDArray
        2D array of integer categorical values.
    categories : Collection or None
        Collection of categories to extract. If None, all categories in `data` are used.
    normed : bool
        If True, entropy values are normalized by the maximum possible entropy
        given the number of categories.
    max_entropy_categories : int or None
        Maximum number of categories to use for normalization. Ignored if `normed=False`.
    img_filter : Callable or None
        Filter function applied to the per-category arrays, e.g., `skimage.filters.gaussian`.
    filter_params : dict or None
        Parameters to pass to `img_filter`.
    filter_output_range : tuple or None
        Expected output range of the filter for proper scaling.
    as_dtype : type or str or None
        Output data type for the entropy array. Only applied if `normed=True`.
    output_range : tuple or None
        Range to rescale normalized entropy values. Only applied if `normed=True`.
    entropy_params : dict or None
        Additional keyword arguments passed to `compute_entropy`.
    **params : dict
        Additional arguments, e.g., `blur_output_dtype` for intermediate filtered arrays.

    Returns
    -------
    np.ndarray
        Array of the same shape as `data`, containing the per-cell Shannon entropy.

    Notes
    -----
    - This function relies on :func:`get_filtered_categories` to create per-category arrays.
    - The filtered arrays are passed to :func:`compute_entropy` to calculate entropy.
    - Normalization scales values to [0,1] (or to the range of `as_dtype` if integer).
    - The `as_dtype` and `output_range` parameters only affect the normalized output.

    See Also
    --------
    :func:`get_filtered_categories` : Extract and optionally filter all category arrays.
    :func:`compute_entropy` : Compute per-cell entropy over a series of data arrays.
    """
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


def compute_interaction(data_arrays: Sequence[NDArray],
                        input_dtype: type | str | None = None,
                        standardize: bool = False,
                        normed: bool = True,
                        output_dtype: type | str | None = None,
                        output_range: tuple | None = None
                        ) -> NDArray:
    r"""
    Compute per-cell interaction (inspired by the Simpson Index) across a series of data arrays.

    The interaction is calculated as the element-wise product of the input arrays.
    Optionally, the interaction can be standardized, normalized, and converted to
    a specified output data type.

    For float inputs:
       .. math::
           interaction = LC_1 \times LC_2 \times \dots \times LC_n

    For integer (e.g., uint8) inputs:
       .. math::
           interaction = \frac{\left(\frac{LC_1}{\text{max}} \times \frac{LC_2}{\text{max}} \times \dots \right)}{(1/n^n)} \times \text{max}

    Parameters
    ----------
    data_arrays : Sequence[np.ndarray]
        Sequence of arrays to compute per-cell interaction over. All arrays must have the same shape.
    input_dtype : type | str | None, default=None
        Expected data type of the input arrays. Raises an error if actual dtype does not match.
    standardize : bool, default=False
        If True, interaction is standardized by the sum of the layers:
        :math:`interaction = \frac{A \cdot B \cdot ...}{A + B + ...}`.
    normed : bool, default=True
        If True, interaction values are normalized to the theoretical maximum interaction:
        :math:`(1/n)^n` for n arrays.
    output_dtype : type | str | None, default=None
        Data type for the output array. Values are rescaled appropriately for integer outputs.
    output_range : tuple | None, default=None
        Target range for output values (currently used only for integer outputs; reserved for future use).

    Returns
    -------
    np.ndarray
        Array of the same shape as the input arrays, containing the per-cell interaction.

    Notes
    -----
    - Standardization (`standardize=True`) scales the interaction by the sum of input layers.
    - Conversion to integer types uses scaling and :func:`numpy.ceil` to avoid rounding artifacts.
    - `normed=True` ensures the maximum possible interaction corresponds to 1 (float) or the maximum of the integer type.
    - Input/output range detection uses :func:`~riogrande.helper.dtype_range`.

    See Also
    --------
    :func:`compute_entropy` : Compute per-cell entropy over a series of data arrays.

    Examples
    --------
    # Example 1: float inputs, 2 arrays
    >>> a = np.array([[0.5, 0.25], [0.0, 0.05]])
    >>> b = np.array([[0.5, 0.25], [0.0, 0.3]])
    >>> compute_interaction([a, b], standardize=True, normed=True)
    array([[1.        , 0.5       ],
           [0.        , 0.17142857]])
    >>> compute_interaction([a, b], standardize=True, normed=False)
    array([[0.25      , 0.125     ],
           [0.        , 0.04285714]])
    # Example 2: integer inputs (uint8), 3 arrays, float output
    >>> a = np.array([[85, 100], [50, 60]], dtype=np.uint8)
    >>> b = np.array([[85, 50], [100, 80]], dtype=np.uint8)
    >>> c = np.array([[85, 105], [100, 10]], dtype=np.uint8)
    >>> compute_interaction([a, b, c], standardize=True, normed=True, output_dtype=np.float64)
    array([[1.        , 0.85487482],
           [0.83044983, 0.13287197]])
    """
    array_dtype = data_arrays[0].dtype

    # define rescaling based on input type
    if input_dtype is None:
        input_dtype = array_dtype
    if isinstance(input_dtype, str):
        input_dtype = np.dtype(input_dtype)
    if array_dtype != input_dtype:
        raise ValueError(f"Array data type {array_dtype} does not match "
                         f"provided input data type {input_dtype}")

    # determine scaling for input
    _max_scale, _ = dtype_range(input_dtype)
    if np.issubdtype(input_dtype, np.floating):
        _max_scale = 1.0

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
                                      out=np.zeros_like(
                                          interaction_array, dtype=float),
                                      where=standardize_array != 0)
    if normed:
        n = len(data_arrays)
        max_interaction = 1 / (n ** n)
        interaction_array /= max_interaction

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
            interaction_array = np.ceil(
                (interaction_array * _max)).astype(output_dtype)
    return interaction_array


def _view_data(source: Source | str,
               bands: list[Band | int] | None,
               view: tuple[int, int, int, int],
               in_range: None | NDArray | Collection,
               as_dtype: type | str | None,
               output_range: None | NDArray | Collection,
               ) -> dict:
    """
    Load a view from a TIFF file and optionally convert and rescale the data.

    You may use the `**tags` to specify which band to read, by default only the
    first band is read. If tags for multiple bands are provided then all
    matching bands are returned.

    Parameters
    ----------
    source : Source or str
        The path to the TIFF file or a `Source` object to load.
    bands : list[Band or int] or None
        Collection of bands to read. Can be `Band` objects or band indices. If None,
        all bands are loaded.
    view : tuple[int, int, int, int]
        A tuple defining the subset of the data to read (e.g., (x_start, y_start, x_end, y_end)).
    in_range : NDArray or Collection or None
        Optional input range to use for rescaling.
    as_dtype : type or str or None
        Optional data type to convert the returned data to. The data will be rescaled
        to match this type if provided.
    output_range : NDArray or Collection or None
        Optional tuple to overwrite the [min,max] range of the output.
        See `io.load_block` for further details.

    Returns
    -------
    dict
        A dictionary mapping each :class:`~riogrande.io.models.Band` object to a dictionary
        containing:

        - ``'data'``: the loaded and optionally rescaled data array
        - ``'view'``: the view tuple used for loading the data

    See Also
    --------
    :func:`_view_filtered` : Load, filter, and rescale a view from a source.
    :func:`_filter_data` : Apply a filter and rescale the resulting data.
    """
    # read out block from original file
    if not isinstance(source, Source):
        source = Source(path=source)
    if bands is None:
        print('No specific bands selected, using all')
        bands = source.get_bands()
    elif any(isinstance(band, int) for band in bands):
        bands = [band if isinstance(band, Band) else source.get_band(bidx=band)
                 for band in bands]

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


def _apply_filter(data: NDArray, img_filter: Callable, **params) -> NDArray:
    """
    Apply a filter function to an array.

    Parameters
    ----------
    data : NDArray
        Input array to be filtered.
    img_filter : Callable
        A callable that accepts `data` as the first argument, along with any
        keyword arguments, and returns a filtered array.
    **params : dict
        Additional keyword arguments passed directly to `img_filter`.

    Returns
    -------
    NDArray
        The filtered result from calling `img_filter(data, **params)`.

    See Also
    --------
    :func:`_filter_data` : Full pipeline: NaN handling, filtering, and dtype conversion.
    """
    return img_filter(data, **params)


def _filter_data(data: NDArray,
                 replace_nan_with: int | float | None = None,
                 img_filter=None,
                 filter_output_range: tuple | None = None,
                 filter_params: dict | None = None,
                 as_dtype: type | str | None = None,
                 output_range: tuple | None = None,
                 ) -> NDArray:
    """
    Apply a filter to a 2D NumPy array with optional NaN handling, type conversion,
    and output rescaling.

    Parameters
    ----------
    data : NDArray
        A 2D numpy array to be filtered.
    replace_nan_with : int or float or None
        Value to replace NaNs in `data`. If None, NaNs remain unchanged.
    img_filter : Callable or None
        Filter function to apply to the array (e.g., `skimage.filters.gaussian`).
    filter_output_range : tuple or None
        Range of values produced by the filter. Used when converting or rescaling the output.
    filter_params : dict or None
        Additional parameters to pass to the filter callable.
    as_dtype : type or str or None
        Data type for the returned array. For example, `np.float64` or `'float32'`.
    output_range : tuple or None
        Target range to rescale the filtered data into. Useful to map the data to [0, 1]
        or any custom range.

    Returns
    -------
    np.ndarray
        The filtered (and optionally rescaled) array with NaNs restored if applicable.

    Notes
    -----
    - NaNs in the original array are restored after filtering if the output dtype supports
      floating-point NaNs.
    - `filter_output_range` should be set when `as_dtype` or `output_range` is used to
      avoid unexpected scaling.
    - Uses :func:`~riogrande.helper.convert_to_dtype` for dtype conversion and rescaling.

    See Also
    --------
    :func:`_apply_filter` : Apply a filter function to an array.
    :func:`_view_filtered` : Load a view and apply filtering end-to-end.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.filters import gaussian
    >>> data = np.array([[1, 2, 1], [np.nan, 4, 5], [2, 4, 6]] )
    >>> data
    array([[ 1.,  2.,  1.],
           [nan,  4.,  5.],
           [ 2.,  4.,  6.]])
    >>> _filter_data(data, replace_nan_with=0, img_filter=gaussian, filter_params={'sigma': 1})
    array([[1.33293744, 1.9624767 , 2.25847273],
           [       nan, 2.85862763, 3.7419668 ],
           [2.27905522, 3.62953263, 4.84767823]])
    """
    # check if nan exists
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


def _view_filtered(source: Source | str,
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
                   ) -> dict:
    """
    Extract a view from a source file and apply filtering and rescaling.

    Parameters
    ----------
    source : Source or str
        Path to the source file or a `Source` object.
    view : tuple[int, int, int, int]
        Region to load from the source (x_start, y_start, x_end, y_end).
    inner_view : tuple[int, int, int, int]
        Subregion to extract from the filtered result.
    data_in_range : NDArray or Collection or None
        Input range used for loading the raw data.
    data_as_dtype : type or str or None
        Data type to convert the loaded data to before filtering.
    data_output_range : NDArray or Collection or None
        Range to rescale the data after loading.
    replace_nan_with : int or float or None
        Value to replace NaNs during filtering.
    img_filter : Callable or None
        Filter function to apply to the data.
    filter_params : dict or None
        Parameters to pass to the filter function.
    filter_output_range : tuple or None
        Expected output range of the filter for proper scaling.
    as_dtype : type or str or None
        Data type for the final filtered output.
    output_range : tuple or None
        Range to rescale the final filtered output.
    bands : list[Band or int] or None
        Bands to load from the source. Defaults to all available bands.
    selector_band : Band or None
        Band used to select regions for filtering and aggregation.

    Returns
    -------
    dict
        A dictionary with keys:

        - ``'data'``: maps band indices to filtered arrays corresponding to `inner_view`
        - ``'view'``: the `inner_view` tuple representing the returned region

    See Also
    --------
    :func:`_view_data` : Load a view from a source without filtering.
    :func:`_filter_data` : Apply a filter and rescale the resulting data.
    """
    data_views = _view_data(source=source,
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
        _nodata = 0
        dv_dtype = data_view['data'].dtype
        if np.issubdtype(dv_dtype, np.floating):
            _nodata = np.nan

        _filtered_data = np.zeros(shape=(view[3], view[2]), dtype=_as_dtype)

        for select in selectors:
            _selector = np.where(selector_data == select, True, False)
            select_data_view = np.where(_selector, data_view['data'], _nodata)

            _part_filtered_data = _filter_data(
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
                     relative_view(view, inner_view)
                     )
        )
    return dict(data=filtered_datas, view=inner_view)


def view_blurred(source: str,
                 view: tuple[int, int, int, int],
                 inner_view: tuple[int, int, int, int],
                 categories: Collection | None,
                 img_filter: Callable,
                 filter_params: dict = dict(),
                 filter_output_range: tuple | None = None,
                 output_dtype: type | str | None = "uint8",
                 output_range: tuple | None = None,
                 **tags):
    """
    Compute blurred binary arrays for each category in a categorical TIFF file.

    The provided TIFF file must contain at least one band with categorical data
    (e.g., of type `uint`). For each specified category, an indicator array is
    created (dichotomous array marking presence/absence of that category), which
    is then filtered using the provided `img_filter` function.

    .. note::
       This method will be moved to the `parallel` sub-module in a future release.

    Parameters
    ----------
    source : str
        Path to the TIFF file to load.
    view : tuple of int
        A 4-tuple defining the view of the data array to update: `(start_row, end_row, start_col, end_col)`.
    inner_view : tuple of int
        A 4-tuple defining the inner part of the view, excluding border effects.
    categories : Collection, optional
        A collection of category values to extract. If `None`, all categories are processed.
    img_filter : Callable
        A function that will be applied to each category indicator array.
    filter_params : dict, optional
        Keyword arguments to pass to `img_filter`. Default is an empty dictionary.
    filter_output_range : tuple, optional
        Output range for the filtered arrays. If `None`, no explicit rescaling is applied.
    output_dtype : type or str, optional
        Data type for the returned arrays. Default is `"uint8"`.

        .. note::
            If provided, the output of the filter function will be rescaled to the
            range of this data type. See `get_category_data` for details.
    output_range : tuple, optional
        Explicit output range for the filtered arrays. If not provided and the
        filter produces float-type data, the range `[0, 1]` is assumed, with
        values clipped to this range.
    **tags : keyword arguments
        Arbitrary keyword arguments to specify which band to read from the TIFF
        file. See :func:`~riogrande.io.load_block` for further details.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'data'``: a dictionary mapping each category to its blurred array
        - ``'view'``: the `inner_view` defining the effective area of the returned arrays

    See Also
    --------
    :func:`view_entropy` : Compute per-cell entropy for a set of category arrays.
    :func:`view_interaction` : Compute per-cell interaction for a set of category arrays.
    :func:`get_filtered_categories` : Extract all categories with optional filtering.
    """
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


def view_entropy(category_arrays: dict[int, NDArray],
                 view: tuple[int, int, int, int],
                 normed: bool = True,
                 max_entropy_categories: int | None = None,
                 output_dtype: type | str | None = None,
                 output_range: tuple | None = None) -> dict:
    """
    Compute the per-cell entropy for a set of category arrays within a specified view.


    Parameters
    ----------
    category_arrays : dict[int, NDArray]
        Dictionary mapping category indices to their corresponding arrays.
    view : tuple[int, int, int, int]
        A tuple defining the subregion of the arrays to process (e.g., (x_start, x_end, y_start, y_end)).
    normed : bool
        If True, normalize the entropy values to the range [0, 1] using the maximum
        possible entropy determined by `max_entropy_categories`.
    max_entropy_categories : int or None
        The maximum number of categories used for normalization. Ignored if `normed=False`.
    output_dtype : type or str or None
        Data type for the returned entropy array. If None, the dtype is inferred.
    output_range : tuple or None
        Range to scale the output values to, e.g., (0, 1).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'data'``: NDArray of computed entropy values for the specified view.
        - ``'view'``: tuple defining the original view of the data arrays.

    See Also
    --------
    :func:`view_blurred` : Compute blurred binary arrays per category.
    :func:`view_interaction` : Compute per-cell interaction for a set of category arrays.
    :func:`compute_entropy` : Underlying entropy computation function.
    """
    entropy_array = compute_entropy(
        data_arrays=tuple(category_arrays.values()),
        normed=normed,
        max_entropy_categories=max_entropy_categories,
        as_dtype=output_dtype,
        output_range=output_range,
    )
    return dict(data=entropy_array, view=view)


def view_interaction(category_arrays: dict[int, NDArray],
                     view: tuple[int, int, int, int],
                     input_dtype: type | str | None = np.uint8,
                     standardize: bool = False,
                     normed: bool = True,
                     output_dtype: type | str | None = None,
                     output_range: tuple | None = None) -> dict:
    """
    Compute the per-cell interaction metric for a set of category arrays within a specified view.

    The function returns a dictionary containing the computed interaction array and the
    original view. Interaction values can be standardized, normalized, and returned
    in a specific data type or range.

    Parameters
    ----------
    category_arrays : dict[int, NDArray]
        Dictionary mapping category indices to their corresponding arrays.
    view : tuple[int, int, int, int]
        A tuple defining the subregion of the arrays to process (e.g., (x_start, x_end, y_start, y_end)).
    input_dtype : type or str or None
        Data type for input conversion before computing interactions.
    standardize : bool
        If True, standardize the input arrays before computing interaction.
    normed : bool
        If True, normalize the computed interaction values.
    output_dtype : type or str or None
        Data type for the returned interaction array. If None, the dtype is inferred.
    output_range : tuple or None
        Range to scale the output values to, e.g., (0, 1).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``'data'``: NDArray of computed interaction values for the specified view.
        - ``'view'``: tuple defining the original view of the data arrays.

    See Also
    --------
    :func:`view_blurred` : Compute blurred binary arrays per category.
    :func:`view_entropy` : Compute per-cell entropy for a set of category arrays.
    :func:`compute_interaction` : Underlying interaction computation function.
    """
    interaction_array = compute_interaction(
        data_arrays=tuple(category_arrays.values()),
        input_dtype=input_dtype,
        standardize=standardize,
        normed=normed,
        output_dtype=output_dtype,
        output_range=output_range,
    )
    return dict(data=interaction_array, view=view)


def get_entropy_view(source: str,
                     view: tuple[int, int, int, int],
                     inner_view: tuple[int, int, int, int],
                     categories: Collection,
                     img_filter,
                     filter_params: dict = dict(),
                     max_entropy_categories: int | None = None,
                     blur_as_int: bool | None = None,
                     filter_output_range: tuple | None = None,
                     blur_output_dtype: type | str | None = None,
                     output_dtype: type | str | None = None,
                     output_range: tuple | None = None,
                     normed: bool = True,
                     **tags):
    """Returns the entropy for some categories over a view from a tif file

    .. warning::
      This function is deprecated and should not be used

    Parameters
    ----------
    max_entropy_categories : int or None
        If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
        This argument is ignored if `normed=False`.

    output_range : tuple or None
        The data-range to use for the returned array.

        .. note::
            This argument is only taken into account if `normed=True`.

    **tags : dict
        Arbitrary number of keyword arguments to describe the band to select.

        See :func:`~riogrande.io.load_block` for further details.

    See Also
    --------
    :func:`view_blurred` : Compute blurred binary arrays per category.
    :func:`view_entropy` : Compute per-cell entropy for a set of category arrays.
    """
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
