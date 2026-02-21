"""
This module contains functions to facilitate carrying out various inference
methods.

In particular, it implements a multiple linear regression approach that allows
to use categorical and any other type of maps as predictors for some response
variable that is also provided as a map.

An exemplary use case is the usage of land-cover types and various derivatives
thereof as predictors for NDVI or the temperature based response maps.

"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from rasterio.windows import Window

from operator import mul
from collections.abc import Collection

from sklearn.linear_model import LinearRegression

from riogrande.helper import (
    check_compatibility,
    view_to_window,
    convert_to_dtype
)
from riogrande.io import Source, Band

from .helper import (
    usable_pixels_info,
    usable_pixels_count,
)


def _to_numpy_selector(rasterio_mask: NDArray) -> NDArray:
    """
    Convert rasterio mask to a boolean selector array.

    Converts a rasterio mask (e.g., from `read_masks(band)`) into a boolean
    numpy array suitable for indexing. The returned array is an inverted mask
    where `True` indicates usable cells and `False` indicates masked cells.

    Parameters
    ----------
    rasterio_mask : NDArray
        Mask array as returned by :meth:`rasterio.io.DatasetReader.read_masks`.
        Non-zero values indicate valid data; zero values indicate masked data.

    Returns
    -------
    NDArray of bool
        Boolean array with the same shape as `rasterio_mask`. `True` values
        indicate cells that can be used; `False` values indicate cells that
        should be excluded.

    See Also
    --------
    :func:`_enrich_selector` : Refine a selector by combining predictor masks.
    :func:`prepare_selector` : Build a combined selector from response and predictor masks.
    """
    return np.where(rasterio_mask != 0, True, False)


def _enrich_selector(selector: NDArray, *predictors: Band, verbose: bool = False) -> NDArray:
    """Refine selector array by combining masks from predictor bands.

    Updates the input selector by applying logical AND operations with masks
    extracted from each predictor band. This ensures only cells that are valid
    across all predictors are selected.

    Parameters
    ----------
    selector : NDArray of bool
        Boolean array indicating usable cells in a 2D raster. `True` values
        indicate cells that can be used; `False` values indicate cells that
        should be excluded.
    *predictors : Band
        Variable number of :class:`~riogrande.io.models.Band` objects, each specifying one or more
        predictor variables. Bands sharing the same mask reader are grouped
        together for efficiency. See :func:`prepare_predictors` for more details.
    verbose : bool, optional
        If True, print runtime information including predictor names, mask
        readers, and usable pixel counts. Default is False.

    Returns
    -------
    NDArray of bool
        Boolean array with the same shape as `selector`. Contains the logical
        AND of the input selector and all predictor masks, indicating cells
        that are valid across all inputs.

    See Also
    --------
    :func:`_to_numpy_selector` : Convert a rasterio mask to a boolean selector.
    :func:`prepare_selector` : Build a combined selector from response and predictor masks.
    """
    aggr_selector = np.copy(selector)
    pred_mask_readers = dict()
    for predictor in predictors:
        pred_mask_reader = predictor.get_mask_reader()
        if pred_mask_reader in pred_mask_readers:
            pred_mask_readers[pred_mask_reader].append(predictor)
        else:
            pred_mask_readers[pred_mask_reader] = [predictor, ]
    # print(f"{pred_mask_readers=}")

    for mask_reader in pred_mask_readers:
        with mask_reader() as read_mask:
            _pred_mask = read_mask()
            all_pixels = mul(*_pred_mask.shape)
        pred_selector = _to_numpy_selector(_pred_mask)
        if verbose:
            data_pixels = usable_pixels_count(pred_selector)
            print("##########")
            _pred_string = '- ' + '\n\t- '.join(map(str, pred_mask_readers[mask_reader]))
            print(f"Predictor(s):\n\t{_pred_string}")
            print(f"\tUse mask: {mask_reader}")
            usable_pixels_info(all_pixels, data_pixels)
        np.logical_and(aggr_selector, pred_selector, out=aggr_selector)
    return aggr_selector


def prepare_selector(response: str | Band, *predictors: Band, extra_masking_band: Band | None = None,
                     verbose=False) -> NDArray:
    """
    Create a boolean selector based on the masks of response and predictors.

    The selector is a boolean array indicating which pixels can be used (True)
    and which cannot (False) based on the combined masks of all input bands.

    Parameters
    ----------
    response : str or Band
        A Band object or path string describing the response data. If a string
        is provided, it will be converted to a Band object with bidx=1.
    *predictors : Band
        Variable number of Band objects, each specifying one or more predictor
        variables. Their masks will be combined with the response mask.
    extra_masking_band : Band, optional
        Additional Band object treated as a rasterio mask, where values equal
        to 0 will be masked out. Default is None.
    verbose : bool, optional
        If True, prints runtime information about usable pixels at each masking
        stage. Default is False.

    Returns
    -------
    NDArray of bool
        Boolean array of the same shape as the response band, where True
        indicates usable pixels and False indicates masked pixels.

    Notes
    -----
    The selector combines masks using logical AND operations, meaning a pixel
    is only usable (True) if it is valid across all input bands.

    The mask reader can be configured using
    :meth:`~riogrande.io.models.Band.set_mask_reader` with ``use='band'``
    or ``use='source'``.

    See Also
    --------
    :func:`_enrich_selector` : Refine a selector using predictor masks.
    :func:`init_X` : Initialize the predictor matrix using the selector.
    :func:`prepare_predictors` : High-level function that calls this internally.

    Examples
    --------
    >>> selector = prepare_selector(response_band, predictor1, predictor2, verbose=True)
    >>> valid_data = response_data[selector]
    """
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    # Note: we can set which reader to use with responze.set_mask_reader(use='band'/'source')
    response_mask_reader = response.get_mask_reader()
    src_profile = response.source.import_profile()
    with response_mask_reader() as read_mask:
        mask = read_mask()
    src_width = src_profile["width"]
    src_height = src_profile["height"]
    all_pixels = src_width * src_height
    selector = _to_numpy_selector(mask)
    if verbose:
        print("\nResponse data:")
        data_pixels = usable_pixels_count(selector)
        usable_pixels_info(all_pixels, data_pixels)
    # now handle the extra mask band
    if extra_masking_band is not None:
        extra_selector = _to_numpy_selector(extra_masking_band.get_data())
        # combine with logical and, as only both True should lea to True
        selector = np.logical_and(selector, extra_selector)
        if verbose:
            print("\nResponse data after extra masking:")
            data_pixels = usable_pixels_count(selector)
            usable_pixels_info(all_pixels, data_pixels)

    # first we get all the masks to compute an overall mask
    aggr_selector = _enrich_selector(selector, *predictors, verbose=verbose)

    data_pixels = usable_pixels_count(aggr_selector)
    if verbose:
        print("\nFinal selector:\n")
        usable_pixels_info(all_pixels, data_pixels)

    return aggr_selector


def init_X(predictors: Collection[Band], selector: NDArray, window: Window | None,
           include_intercept: bool, as_dtype: type | str) -> NDArray:
    """
    Initialize the predictor matrix X with appropriate dimensions.

    Creates an empty predictor matrix with rows corresponding to usable pixels
    (as determined by the selector) and columns for each predictor band, plus
    an optional intercept column.

    Parameters
    ----------
    predictors : Collection of Band
        Collection of Band objects, each specifying one or more predictor
        variables. The number of predictors determines the number of columns
        in the output matrix (excluding the intercept).
    selector : NDArray of bool
        Boolean array to select usable cells. Only pixels where selector is
        True will be included in the matrix rows.
    window : Window or None
        Limits the data array to a specific spatial window. If provided, the
        window is converted to slices using :meth:`rasterio.windows.Window.toslices`.
        If None, the entire selector array is used.
    include_intercept : bool
        If True, adds an extra column of ones at the end of the matrix for
        fitting intercept terms in regression models.
    as_dtype : type or str
        Data type for the output array. Can be a numpy dtype or string
        specification (e.g., 'float64', np.float32).

    Returns
    -------
    NDArray
        Zero-initialized array of shape (n_rows, n_cols) where:

        - n_rows is the count of usable pixels in the (windowed) selector
        - n_cols is ``len(predictors) + 1`` if include_intercept else ``len(predictors)``

    Notes
    -----
    The `window` parameter is intended for parallelized processing where
    different processes handle different spatial subsets of the data.

    If the window slicing results in an IndexError (e.g., window is completely
    outside the selector bounds), the function returns an array with 0 rows.

    See Also
    --------
    :func:`populate_X` : Fill the initialized predictor matrix with data.
    :func:`partial_X` : Initialize and populate a predictor matrix in one step.
    """
    if window is not None:
        partial = window.toslices()
    else:
        partial = slice(None)
    try:
        nbr_rows = usable_pixels_count(selector[partial])
    except IndexError:
        nbr_rows = 0
    nbr_cols = len(predictors)
    if include_intercept:
        nbr_cols += 1
    # create emtpy predictor array
    return np.zeros((nbr_rows, nbr_cols), dtype=np.dtype(as_dtype))


def populate_X(X: NDArray, predictors: Collection[Band], as_dtype: type | str,
               window: Window | None, selector: NDArray, include_intercept: bool):
    """
    Populate predictor matrix X with data from predictor bands.

    Reads data from each predictor band, applies the selector mask within the
    specified window, and fills the columns of matrix X. Optionally adds an intercept
    column of ones.

    Parameters
    ----------
    X : NDArray
        Pre-initialized array to be populated with predictor data. Modified
        in-place. Should have shape (n_usable_pixels, n_predictors) or
        (n_usable_pixels, n_predictors + 1) if include_intercept is True.
    predictors : Collection of Band
        Collection of Band objects, each specifying one predictor variable.
        Data from each band will be read and placed in the corresponding
        column of X.
    as_dtype : type or str
        Target data type for predictor values. Converted using
        :func:`~riogrande.helper.convert_to_dtype` without rescaling
        (``in_range`` and ``out_range`` are ``None``).
    window : Window or None
        Limits data reading to a specific spatial window. If provided, the
        window is converted to slices using :meth:`rasterio.windows.Window.toslices`.
        If None, the entire data array is used.
    selector : NDArray of bool
        Boolean array to select usable pixels. Applied after windowing to
        extract only valid data points for the predictor matrix.
    include_intercept : bool
        If True, the last column of X is filled with ones to represent the
        intercept term in regression models.

    Returns
    -------
    None
        X is modified in-place.

    See Also
    --------
    :func:`init_X` : Allocate the empty predictor matrix.
    :func:`partial_X` : Initialize and populate in one step.
    """
    # Create a window
    if window is not None:
        _selector = selector[window.toslices()]
    else:
        _selector = selector

    for i, predictor in enumerate(predictors):
        with predictor.data_reader() as read:
            pred_data = read(window=window)
            # perform type conversion without any rescaling

            pred_data_converted = convert_to_dtype(pred_data, as_dtype=as_dtype,
                                                   in_range=None, out_range=None)

            X[:, i] = pred_data_converted[_selector].reshape(1, -1)

        # apply the mask and populate X
    # attach intercept column (of 1s) if chosen
    if include_intercept:
        X[:, -1] = 1.0


def prepare_predictors(response: str | Band, *predictors: Band | str, view: tuple[int, int, int, int] | None = None,
                       include_intercept=True, verbose: bool = False):
    """
    Generate the predictor matrix and response vector for multiple linear regression.

    This function constructs the predictor matrix ``X`` and the response vector
    ``y`` used in a multiple linear regression model of the form

    .. math::

       \\mathbf{y} = X\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}

    where ``y`` represents the response data and ``X`` the predictor matrix.

    The response data are used as a reference to determine valid pixels. A mask
    is extracted from the response (e.g., nodata values or an 8-bit mask) and
    applied to all predictors. Predictor-specific missing values are also added
    to the mask, ensuring that only pixels with complete information across all
    variables are included in the regression.

    This masking procedure reduces the number of pixels used in the analysis,
    yielding a smaller but denser predictor matrix.

    Notes
    -----
    An :exc:`~coonfit.exceptions.InferenceError` is raised if an invalid
    predictor is detected. A predictor is considered invalid in the following cases:

    * After masking, the predictor contains only zeros.
    * The predictor represents categorical data where all selected pixels
     belong to a category and ``include_intercept`` is ``True``.

    No general test for linear dependence between predictors is performed.

    Parameters
    ----------
    response : str or Band
        Response variable. Either a ``Band`` object or a string specifying the
        path to a raster file (``.tif``) containing the response data.

        If a string is provided, the raster must contain exactly one band.
    *predictors : Band or str
        One or more predictors specified as ``Band`` objects or file paths.

        If a string is provided, it is interpreted as the path to a raster file
        and **all bands** in that file are added as individual predictors.

        Predictor data are cast to the same data type as the response. No
        rescaling is performed.
    view : tuple of int, optional
        Spatial subset of the data specified as ``(x, y, width, height)``.
        If not provided, the entire response raster is used.
    include_intercept : bool, default=True
        If ``True``, an additional column of ones is appended to the predictor
        matrix to model an intercept term.
    verbose : bool, default=False
        If ``True``, print processing information.

    Returns
    -------
    X : NDArray of shape (n_samples, n_features)
        Predictor matrix. The number of features corresponds to the total number
        of predictors, plus one if ``include_intercept`` is ``True``.
    y : NDArray of shape (n_samples,)
        Response vector containing the response values corresponding to the
        selected pixels.

    See Also
    --------
    :func:`prepare_selector` : Build the boolean selector used internally.
    :func:`init_X` : Allocate the predictor matrix.
    :func:`transposed_product` : Compute X.T @ X for large spatial datasets.
    """
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    # make sure the Source profile is updated
    response_profile = response.source.import_profile()
    response_dtype = response_profile['dtype']
    # make sure the predictors are bands
    _predictors = []
    for pred in predictors:
        if not isinstance(pred, Band):
            _source = Source(path=pred)
            _bands = _source.get_bands()
            _predictors.extend(_bands)
        else:
            _predictors.append(pred)
    # get all paths and check the compatibility
    _sources = [response.source.path, ]
    _sources.extend(
        [pred.source.path for pred in _predictors]
    )
    # make sure used tif files are compatible (i.e. check crs and units)
    check_compatibility(*set(_sources))
    # extract the mask from response and enrich it with masks from predictors
    aggr_selector = prepare_selector(response, *_predictors, verbose=verbose)

    riow = view_to_window(view)
    # create empty predictor array
    X = init_X(_predictors,
               selector=aggr_selector,
               window=riow,
               include_intercept=include_intercept,
               as_dtype=response_dtype)

    populate_X(X=X, predictors=_predictors, window=riow, selector=aggr_selector,
               include_intercept=include_intercept, as_dtype=response_dtype)
    # return predictor matrix and the response vector
    y = partial_response(response=response,
                         window=riow,
                         selector=aggr_selector)
    return X, y


def transposed_product(predictors: Collection[Band], view: tuple[int, int, int, int] | None, selector: NDArray,
                       include_intercept: bool = False, as_dtype: str | type = "float64"):
    """
    Compute the transposed product ``X.T @ X`` for a set of predictors.

    This function extracts predictor values within a specified spatial view,
    applies a boolean selector to filter valid pixels, constructs the predictor
    matrix ``X``, and returns its transposed product. The result is commonly
    used in linear regression for computing normal equations.

    Parameters
    ----------
    predictors : Collection[Band]
        Collection of ``Band`` objects defining the predictor variables.
    view : tuple of int or None
        Spatial subset specified as ``(x, y, width, height)``.
        If ``None``, the full spatial extent is used.
    selector : NDArray of bool
        Boolean array indicating which pixels are valid and should be included
        in the computation.
    include_intercept : bool, default=False
        If ``True``, include an additional column of ones in the predictor
        matrix to model an intercept term.
    as_dtype : str or type, default="float64"
        Data type of the resulting array.

    Returns
    -------
    transprodX : NDArray of shape (n_features, n_features)
        The transposed product of the predictor matrix, ``X.T @ X``.
        The number of features corresponds to the number of predictors,
        plus one if ``include_intercept`` is ``True``.

    See Also
    --------
    :func:`get_optimal_weights` : Compute regression weights from X and y.
    :func:`get_optimal_weights_source` : Compute weights using precomputed inverse.
    :func:`partial_X` : Generate the predictor matrix ``X``.
    """
    riow = view_to_window(view)
    X = partial_X(predictors=predictors,
                  window=riow,
                  selector=selector,
                  include_intercept=include_intercept,
                  as_dtype=as_dtype)
    # create transprodX matrix:
    # ###
    # this is equivalent
    transprodX = X.T @ X
    # to this:
    # nbr_pred = X.shape[1]
    # transprodX = np.zeros((nbr_pred, nbr_pred), dtype=as_dtype)
    # for i in range(nbr_pred):
    #     col1 = X[:,i]
    #     for j in range(i, nbr_pred):
    #         col2 = X[:,j]
    #         transprodX[i,j] = np.sum(np.multiply(col1, col2))
    # ###
    return transprodX


def get_optimal_weights(X, y):
    """
    Compute the optimal regression weights for a multiple linear regression.

    The multiple linear regression model is defined as

    .. math::

        \\mathbf{y} = X\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}

    where ``X`` is the predictor matrix, ``y`` the response vector,
    ``\\boldsymbol{\\beta}`` the vector of regression weights, and
    ``\\boldsymbol{\\epsilon}`` a random error term.

    The optimal least-squares solution for ``\\boldsymbol{\\beta}`` is given by

    .. math::

        \\boldsymbol{\\beta} = (X^T X)^{-1} X^T \\mathbf{y}

    which is computed directly using NumPy linear algebra routines.

    Notes
    -----
    * The matrix ``X^T X`` may be singular or ill-conditioned, in which case
      the inverse does not exist or the solution may be numerically unstable.
    * In such cases, alternative approaches such as singular value
      decomposition (SVD) or :func:`numpy.linalg.lstsq` are recommended.

    Parameters
    ----------
    X : NDArray of shape (n_samples, n_features)
        Predictor matrix where each row corresponds to an observation and each
        column to a predictor variable.
    y : NDArray of shape (n_samples,)
        Response vector.

    Returns
    -------
    beta : NDArray of shape (n_features,)
        Optimal regression weights minimizing the least-squares error.

    See Also
    --------
    :func:`get_approx_weights` : Estimate weights via scikit-learn's LinearRegression.
    :func:`transposed_product` : Compute X.T @ X from spatial band data.
    :func:`prepare_predictors` : Build X and y from raster bands.
    """
    return (np.linalg.inv(X.T @ X) @ X.T) @ y


def partial_response(response: str | Band, window: Window | None, selector: NDArray):
    """
    Extract and return the response values within a window after applying a selector.

    This function reads the response raster data, optionally restricts it to a
    spatial window, and applies a boolean selector to return only the valid
    response values. The resulting array is suitable for use as the response
    vector in regression analyses.

    Parameters
    ----------
    response : str or Band
        Response variable specified either as a ``Band`` object or as the path
        to a raster file (``.tif``). If a string is provided, the raster must
        contain a single band.
    window : rasterio.windows.Window or None
        Spatial window defining the subset of the response raster to read.
        If ``None``, the full raster extent is used.
    selector : NDArray of bool
        Boolean array used to select valid pixels. If ``window`` is provided,
        the selector is sliced accordingly before being applied.

    Returns
    -------
    y : NDArray of shape (n_samples,)
        One-dimensional array containing the selected response values.

    See Also
    --------
    :func:`partial_X` : Generate the corresponding predictor matrix.
    :func:`prepare_predictors` : Build X and y together from raster bands.
    """
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    if window is not None:
        _selector = selector[window.toslices()]
    else:
        _selector = selector
    with response.data_reader(mode='r') as read:
        response_data = read(window=window)
    return response_data[_selector]


def partial_X(predictors: Collection[Band], window: Window | None, selector: NDArray,
              include_intercept: bool, as_dtype: type | str):
    """
    Generate a (partial) predictor matrix ``X``.

    This function constructs the predictor matrix from a collection of raster
    bands, optionally restricted to a spatial window. A boolean selector is
    applied to include only valid pixels. The function is intended to support
    partial or chunked processing of predictors, for example in parallelized
    workflows.

    Parameters
    ----------
    predictors : Collection[Band]
        Collection of ``Band`` objects defining the predictor variables.
        Individual bands may represent continuous or categorical predictors.
    window : rasterio.windows.Window or None
        Spatial window defining the subset of predictor data to read.
        If ``None``, the full spatial extent is used.

        Notes
        -----
        This argument is primarily intended to enable partial processing of
        predictors, e.g. in parallel or tiled computations.
    selector : NDArray of bool
        Boolean array used to select valid pixels from the predictor data.
    include_intercept : bool
        If ``True``, an additional column of ones is appended to the predictor
        matrix to model an intercept term.
    as_dtype : str or type
        Data type used for the resulting predictor matrix. This is also the
        type used to represent categorical predictors, if present.

    Returns
    -------
    X : NDArray of shape (n_samples, n_features)
        Predictor matrix containing the selected predictor values. The number
        of features corresponds to the number of predictors, plus one if
        ``include_intercept`` is ``True``.

    See Also
    --------
    :func:`init_X` : Allocate the empty predictor matrix.
    :func:`populate_X` : Fill the predictor matrix with data.
    :func:`transposed_product` : Compute X.T @ X using this function internally.
    """
    X = init_X(predictors,
               selector=selector,
               window=window,
               include_intercept=include_intercept,
               as_dtype=as_dtype,
               )
    populate_X(X=X, predictors=predictors,
               window=window,
               selector=selector,
               include_intercept=include_intercept,
               as_dtype=as_dtype)
    return X


def get_optimal_weights_source(Y: NDArray, response: str | Band, predictors: Collection[Band],
                               view: tuple[int, int, int, int] | None, selector,
                               include_intercept: bool = False, as_dtype="float64") -> dict[Band, float]:
    """
    Compute optimal regression weights directly from predictors and a precomputed
    inverse transposed product.

    The multiple linear regression model is defined as

    .. math::

        \\mathbf{y} = X\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}

    The least-squares solution for the regression weights is

    .. math::

        \\hat{\\boldsymbol{\\beta}} = (X^T X)^{-1} X^T \\mathbf{y}

    Defining

    .. math::

        Y = (X^T X)^{-1}

    this function computes

    .. math::

        \\hat{\\boldsymbol{\\beta}} = Y X^T \\mathbf{y}

    directly from the predictor data and the response values, without explicitly
    recomputing ``X.T @ X``.

    Parameters
    ----------
    Y : NDArray of shape (n_features, n_features)
        Inverse of the transposed product of the predictor matrix,
        ``(X.T @ X)^{-1}``.

        Notes
        -----
        Typically, ``Y`` is obtained by inverting the output of
        :func:`transposed_product` via :func:`numpy.linalg.inv`.
    response : str or Band
        Response variable specified either as a ``Band`` object or as the path
        to a raster file (``.tif``). If a string is provided, the raster must
        contain a single band.
    predictors : Collection[Band]
        Collection of ``Band`` objects defining the predictor variables.
    view : tuple of int or None
        Spatial subset specified as ``(x, y, width, height)``.
        If ``None``, the full spatial extent is used.
    selector : NDArray of bool
        Boolean array indicating which pixels are valid and should be included
        in the regression.
    include_intercept : bool, default=False
        If ``True``, include an intercept term in the regression. The intercept
        weight is returned under the key ``'intercept'``.
    as_dtype : str or type, default="float64"
        Data type used for constructing the predictor matrix.

    Returns
    -------
    weights : dict
        Dictionary mapping each predictor to its fitted regression weight.
        If ``include_intercept`` is ``True``, an additional key ``'intercept'``
        is included.

    See Also
    --------
    :func:`transposed_product` : Compute X.T @ X, whose inverse is ``Y``.
    :func:`get_optimal_weights` : Direct computation from X and y.
    :func:`partial_X` : Generate the partial predictor matrix used here.
    """
    riow = view_to_window(view)
    # First
    part_X = partial_X(predictors=predictors,
                       window=riow,
                       selector=selector,
                       include_intercept=include_intercept,
                       as_dtype=as_dtype)
    part_y = partial_response(response=response,
                              window=riow,
                              selector=selector)
    betas = Y @ part_X.T @ part_y

    if include_intercept:
        pred_list = list(predictors)
        pred_list.append('intercept')
        predictors = tuple(pred_list)
    if len(betas) != len(predictors):
        raise ValueError(f"Number of predictors {len(predictors)} not equal with number of fitted values {len(betas)}")
    return {pred: beta for pred, beta in zip(predictors, betas)}


def get_approx_weights(X: NDArray, y: NDArray,
                       fit_intercept: bool = False) -> LinearRegression:
    """
    Estimate regression weights using numerical optimization.

    This function fits a multiple linear regression model of the form

    .. math::

        \\mathbf{y} = X\\boldsymbol{\\beta} + \\boldsymbol{\\epsilon}

    where ``X`` is the predictor matrix, ``y`` the response vector,
    ``\\boldsymbol{\\beta}`` the regression weights, and
    ``\\boldsymbol{\\epsilon}`` a random error term.

    The weights are estimated using scikit-learn’s
    :class:`sklearn.linear_model.LinearRegression` estimator, which computes a
    least-squares solution using numerical linear algebra routines.

    Parameters
    ----------
    X : NDArray of shape (n_samples, n_features)
        Predictor matrix where each row corresponds to an observation and each
        column to a predictor variable.
    y : NDArray of shape (n_samples,)
        Response vector.
    fit_intercept : bool, default=False
        Whether to fit an intercept term.

        Notes
        -----
        If ``X`` was constructed using :func:`prepare_predictors` with
        ``include_intercept=True``, then ``fit_intercept`` should be set to
        ``False`` to avoid fitting the intercept twice.

    Returns
    -------
    regression : :class:`sklearn.linear_model.LinearRegression`
        Fitted linear regression model.

    See Also
    --------
    :func:`get_optimal_weights` : Analytic least-squares solution (normal equations).
    :func:`prepare_predictors` : Build X and y from raster bands.
    """
    regression = LinearRegression(fit_intercept=fit_intercept)
    regression.fit(X, y)
    return regression
