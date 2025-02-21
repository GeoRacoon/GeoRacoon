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
import rasterio as rio

from operator import mul
from collections.abc import Collection

from rasterio.windows import Window
from sklearn.linear_model import LinearRegression

from numpy.typing import NDArray

from .exceptions import InferenceError
from .helper import (check_compatibility,
                     usable_pixels_info,
                     usable_pixels_count,
                     view_to_window,
                     convert_to_dtype)
from .processing import select_category
from .io_ import Source, Band


def to_numpy_selector(rasterio_mask: NDArray) -> NDArray:
    """Converts rasterio mask (e.g. `read_masks(band)`) into a `numpy.bool_'

    ..Note::
        The returned array is an inverted mask in the sense that a cell that
        should be used will have the value `True` and a cell that should not
        be used is `False`.

    Parameters
    ----------
    rasterio_mask:
      Output of a mask as returned by `rasterio.io.DatasetReader.read_masks`

    Returns
    -------
    np.array:
        A 2D array of np.bool_ values indicating what cell can be used.
    """
    return np.where(rasterio_mask != 0, True, False)


def enrich_selector(selector: NDArray,
                    *predictors: Band,
                    verbose: bool = False) -> NDArray:
    """Complete the selector with the masks extracted from the pridictors

    Parameters
    ----------
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    *predictors:
      An arbitrary number of `.io_.Band` objects each specifying one or
      several predictors.

      See `prepare_predictors` for more details.

    verbose: Default: False
      If the method should print runtime info

    Returns
    -------
    np.array:
        A 2D array of np.bool_ values indicating what cell can be used.
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
        pred_selector = to_numpy_selector(_pred_mask)
        if verbose:
            data_pixels = usable_pixels_count(pred_selector)
            print("##########")
            _pred_string = '- ' + '\n\t- '.join(map(str, pred_mask_readers[mask_reader]))
            print(f"Predictor(s):\n\t{_pred_string}")
            print(f"\tUse mask: {mask_reader}")
            usable_pixels_info(all_pixels, data_pixels)
        np.logical_and(aggr_selector, pred_selector, out=aggr_selector)
    return aggr_selector


def prepare_selector(response: str | Band,
                     *predictors: Band,
                     extra_masking_band: Band|None=None,
                     verbose=False) -> NDArray:
    """Creates a boolean selector based on the masks of response and predictors

    The selector is a np.array of type np.bool_ indicating which well can be
    used (`True`) and which cannot (`False`)

    Parameters
    ----------
    _See `prepare_predictors` for a detailed description description._

    response:
      A `.io_.Band` object describing the response data.
    *predictors:
      An arbitrary number of `io_.Band` objects each specifying one or several
      predictors.
    extra_masking_band: Optional `io_.Band` object that is treated as a rasterio mask, i.e. values equal to 0
      will be masked.
    verbose: Default: False
      If the method should print runtime info

    Returns
    -------
    np.array
      Boolean array of the same shape as `response`

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
    selector = to_numpy_selector(mask)
    if verbose:
        print("\nResponse data:")
        data_pixels = usable_pixels_count(selector)
        usable_pixels_info(all_pixels, data_pixels)
    # now handle the extra mask band
    if extra_masking_band is not None:
        extra_selector = to_numpy_selector(extra_masking_band.get_data())
        # combine with logical and, as only both True should lea to True
        selector = np.logical_and(selector, extra_selector)
        if verbose:
            print("\nResponse data after extra masking:")
            data_pixels = usable_pixels_count(selector)
            usable_pixels_info(all_pixels, data_pixels)

    # first we get all the masks to compute an overall mask
    aggr_selector = enrich_selector(selector, *predictors, verbose=verbose)

    data_pixels = usable_pixels_count(aggr_selector)
    if verbose:
        print("\nFinal selector:\n")
        usable_pixels_info(all_pixels, data_pixels)

    return aggr_selector


def init_X(predictors: Collection[Band],
           selector: NDArray,
           window: Window | None,
           include_intercept: bool,
           as_dtype:type|str) -> NDArray:
    """Initiates the matrix X with the appropriate width and height

    Parameters
    ----------
    predictors:
      Collection of `io_.Band` objects each specifying one or several predictors.
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    window:
      Limits the data array to a specific window. The window is converted to a
      `slice` with `window.toslices()`.

      ..Note::
        The intended purpose of this argument is to allow to process only parts
        of the predictors in the case of a parallelized approach.

        See `.parallel.partial_optimal_betas` and `get_optimal_weights_source`
        for further details
      
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
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


def populate_X(X: NDArray,
               predictors: Collection[Band],
               as_dtype: type|str,
               window: Window | None,
               selector: NDArray,
               include_intercept: bool):
    """Adds column per predictor with selector applied in the window view

    ...Note::
      $X$ is updated in place.
    
    Parameters
    ----------
    X:
      Initiated np.NDArray which will hold the data of each predictor in a
      separate column.
    predictors:
      An arbitrary number of `io_.Band` objects each specifying a predictor
    predictor_datas:
      List of arrays each being used as a predictor
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    window:
      Limits the data array to a specific window. The window is converted to a
      `slice` with `window.toslices()`.

      ..Note::
        The intended purpose of this argument is to allow to process only parts
        of the predictors in the case of a parallelized approach.

        See `.parallel.partial_optimal_betas` and `get_optimal_weights_source`
        for further details
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
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


def prepare_predictors(response: str | Band,
                       *predictors: Band | str,
                       view: tuple[int, int, int, int] | None = None,
                       include_intercept=True,
                       verbose: bool = False):
    r"""Generates and returns the parameters for a multiple linear regression

    The parameters returned are $X$ and $\vec{y}$ from the multiple linear
    regression:

        $$\vec{y} = X\vec{\beta} + \vec{\epsilon}$$

    Where $\vec{y}$ represents the response data and $X$ the predictor matrix.

    This method uses the response data as reference to filter the predictors.
    It does so by extracting the mask (i.e. `nodata` value or an 8bit mask)
    from the response tif and applying it to each of the predictors.

    Since some of the predictor data might also have missing values, we
    iterate once over all predictor data and add pixels with missing values
    to the mask.
    In doing so we avoid including pixels with incomplete information into the
    regression analysis.

    In doing so we can reduce the amount of pixel to include in the multiple
    linear regression analysis, therefore, reducing the size of the response
    vector, leading to a denser but smaller predictor matrix.
    
    .. note::

      This function returns an InferenceError if some predictor is invalid.
      
      An invalid predictor can happen if it is a linear combination of the
      other predictors.

      We do not perform a systematic test for that but check the following
      cases:

      - If a predictor, after applying the final selector, contains only 0's
      - If a predictor consists of categorical data (i.e. a 2D array that
        categorizes each pixel) with each selected pixel always belonging
        to a category (i.e. no unselected categories or unselected are masked
        - see details for `predictors` argument) and `include_intercept` is
        set to True.

    Parameters
    ----------
    response:
      Either a `.io_.Band` object or a string that specifies the path to a map
      (.tif file) that holds the response data.

      .. note::

        If a string is provided then the response must be stored in a single
        band.

    *predictors:
      An arbitrary number of either `io_.Band` objects or strings specifying one
      or several predictors.

      If a string is provided then it is treated as the path to a `tif` file
      and **all** bands in the file are added as individual predictor each.

      .. warning::

        Predictor data is converted to the same data type as the response
        before fitting.
        However, **no rescaling is performed**.

        Thus with a response of type `float` and a predictor of type 'uint8'
        the `uint8` values are simple converted to floats (e.g. 255 > 255.0)

        **Rescaling the predictor data must be done separately beforehand!**
        
    view:
      An optional tuple (x, y, width, height) defining the view of the predictors
      and response data to consider.
      If not provided then the entire response map is used.
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    verbose:
        Print out processing steps

    Return
    ------
    np.array:
        2D array with the width of the total number of predictors (+1 if the
        intercept is fitted as well) and the height being equal to the number
        of data pixels with valid data.
    """
    # make sure response is a band
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

    # NOTE: check_predictors is useless if we do not compute the bands for
    #       categorical data on the fly. This is because the check simply
    #       relies on the selection criterion of the bands to use and not
    #       on the actual data.
    #       TODO: we should introduce a check of linear dependency between
    #             the columns if we want a reliable check. Or then do not
    #             check ant let the matrix inversion fail

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


def transposed_product(predictors: Collection[Band],
                       view: tuple[int, int, int, int] | None,
                       selector: NDArray,
                       include_intercept: bool = False,
                       as_dtype:str|type="float64"
                       ):
    """Extracts the selector of the predictor data in the provided view.

    Parameters
    ----------
    predictors:
      An arbitrary number of of `io_.Band` objects each specifying a predictor
    view:
      An optional tuple (x, y, width, height) defining the view of the predictors
      and response data to consider.
      If not provided then the entire response map is used.
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    as_dtype:
      The data type to use for the resulting array
    """
    riow = view_to_window(view)
    X = partial_X(predictors=predictors,
                  window=riow,
                  selector=selector,
                  include_intercept=include_intercept,
                  as_dtype=as_dtype)
    # create transprodX matrix:
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
    r"""Compute the optimal weight of a multiple linear regression.

    The multiple linear regression is defined by the equation:

        $$\vec{y} = X\vec{\beta} + \vec{\epsilon}$$

    Where $\vec{beta}$ holds the weigts of the different predictors,
    $\vec{\epsilon}$ is a random variable and $X$ a matrix with each line
    corresponding to an observation (or a pixel in the case of a raster map).
    Each column of $X$ stacks the values of one predictors.

    The optimal solution for $\beta$ is given by
    $(X^T \dot X)^{-1} \dot X^T \dot \vec{y}$ which we can simply compute with
    numpy.

    ..Note::

        - It is not guaranteed that $X^T \dot X$ has an inverse.
        - We could also use SVD decompositin of X. An illustrative example
          can be found here: https://sthalles.github.io/svd-for-regression/
    """
    return (np.linalg.inv(X.T @ X) @ X.T) @ y


def partial_response(response: str | Band,
                     window: Window | None,
                     selector: NDArray):
    """Returns the window view of the response data after applying the selector

    Parameters
    ----------
    response:
      Path to a map (.tif file) that holds the response data
    window:
      ...
    selector:
      ...
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


def partial_X(predictors: Collection[Band],
              window: Window | None,
              selector: NDArray,
              include_intercept: bool,
              as_dtype:type|str):
    """Generate (a partial) predictor matrix, $`X`$.

    If `window` is provided then only the specified selection will
    be read out from the predictors which allows to use this method
    when partially processing the predictors.

    Parameters
    ----------
    predictors:
      Collection of `io_.Band` objects each specifying one or several predictors.
    window:
      Limits the data array to a specific window. The window is converted to a
      `slice` with `window.toslices()`.

      .. note::

        The intended purpose of this argument is to allow to process only parts
        of the predictors in the case of a parallelized approach.

    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    as_dtype:
      The data type to represent each category in if a categorical predictor
      is present

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


def get_optimal_weights_source(Y: NDArray,
                               response: str | Band,
                               predictors: Collection[Band],
                               view: tuple[int, int, int, int] | None,
                               selector,
                               include_intercept: bool = False,
                               as_dtype="float64"
                               ) -> dict[Band, float]:
    r"""Calculate the optimal weights directly from predictors and the inverse of
    the transposed product, Y.

    $$\hat{\vec{\beta}} = (X^T @ X)^{-1} X^T \vec{y}$$
    
    And we define $Y = (X^T @ X)^{-1}$, thus:
    
    $$\hat{\vec{\beta}} = Y @ X^T \vec{y}$$
    
    Which leads to:
    
    $$\hat{\beta}_j = \Sigma_{n=1}^N y_n \Sigma_{m=1}^M{Y_{j,m}l_{n}^{m}}$$

    Parameters
    ----------
    Y:
      An $`MxM`$ matrix that is the result of the transposed product of the
      predictor data.

      ..Note::
        Typically you would use the output of `transposed_product` to compute
        `Y`.
    response:
      A `.io_.Band` object describing the response data.
    predictors:
      Collection of `io_.Band` objects each specifying one or several predictors.
    view:
      An optional tuple (x, y, width, height) defining the view of the predictors
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    as_dtype:
      The data type to represent each category in if a categorical predictor
      is present
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


def get_approx_weights(X: NDArray,
                       y: NDArray,
                       fit_intercept: bool = False) -> LinearRegression:
    r"""Numerical optimization to determine weights in a mlt. lin. regression.

    The multiple linear regression is defined by the equation:

        $$\vec{y} = X\vec{\beta} + \vec{\epsilon}$$

    Where $\vec{beta}$ holds the weigts of the different predictors,
    $\vec{\epsilon}$ is a random variable and $X$ a matrix with each line
    corresponding to an observation (or a pixel in the case of a raster map).
    Each column of $X$ stacks the values of one predictors.

    The approxymation is perforemd with scikit-learn's linear regression
    estimator:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    Parameters
    ----------
    X: np.array
        MxN 2D array where each of the M columns holds the N sample data for
        a specify predictor.
    y: np.array
        Nx1 response values
    fit_intercept: bool
        Determines if the intercept should be fitted explicitely, or if the
        data is expected to be centered.

        ..Note::
            If X was created with the `prepare_predicotrs` function with
            'include_intercept=True` then `fit_intercept` sould be set to
            `False`.

    Returns
    -------
    sklearn.linear_model.LinearRegression:
        Fitted linear regression model
    """
    regression = LinearRegression(fit_intercept=fit_intercept)
    regression.fit(X, y)
    return regression
