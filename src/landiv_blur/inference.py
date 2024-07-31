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

from rasterio.windows import Window
from sklearn.linear_model import LinearRegression

from numpy.typing import NDArray

from .exceptions import InferenceError
from .helper import (check_compatibility,
                     usable_pixels_info,
                     usable_pixels_count,
                     view_to_window,)
from .processing import select_category

def to_numpy_selector(rasterio_mask:NDArray)->NDArray:
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
    return np.where(rasterio_mask == 255, True, False)


# TODO: this should change to a dict base approach using tags
def check_predictors(selector:NDArray,
                     *predictors: tuple[str,
                                        int,
                                        tuple[int, ...] | None,
                                        bool | None],
                     include_intercept:bool,
                     verbose:bool=False):
    """Checks if with these predictors analytical solution can be calculated

    Parameters
    ----------
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.

      See `prepare_predictors` for more details.
    verbose: Default: False
      If the method should print runtime info

    """
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        with rio.open(pred_file_path, 'r') as psrc:
            _pred_mask = psrc.read_masks(band)
            all_pixels = psrc.width * psrc.height
        pred_selector = to_numpy_selector(_pred_mask)
        if verbose:
            data_pixels = usable_pixels_count(pred_selector)
            print("##########")
            print(f"- predictor:\n\t{pred_file_path=}")
            usable_pixels_info(all_pixels, data_pixels)
        # now handle the case of a category
        if len(predictor) >= 3:
            extract_values = predictor[2]
            if len(predictor) >= 4:
                mask_unselected = predictor[3]
            else:
                mask_unselected = False
            if mask_unselected and include_intercept:
                raise InferenceError(
                    f"Invalid:\n\t{predictor=}\n"
                    f"Using {include_intercept=} and "
                    "masking the unselected categories"
                    f" (i.e. {predictor[3]=}) leads to"
                    " a non-invertible (X^T X) matrix."
                )
        else:
            extract_values = None
        if extract_values is not None:

            all_types = set()
            with rio.open(pred_file_path, mode='r') as psrc:
                for _, window in psrc.block_windows(band):
                    wdata = psrc.read(band, window=window)
                    all_types = all_types.union(
                        np.unique(wdata[selector[window.toslices()]])
                    )
            for e_value in extract_values:
                if e_value not in all_types:
                    raise InferenceError(f"Invalid\n\t{predictor=}\n"
                                         f"the category {e_value} is no longer"
                                         " present when aggregated filter is"
                                         " applied.\nFOR THIS TO WORK YOU"
                                         f" NEED OT REMOVE BAND {e_value}!")
        if verbose:
            print("  passed!")
            print("##########")


# TODO: needs adaptation for predictors as dict using tags
def enrich_selector(selector:NDArray,
                    *predictors: tuple[str,
                                       int,
                                       tuple[int, ...] | None,
                                       bool | None],
                    verbose:bool=False)->NDArray:
    """Complete the selector with the masks extracted from the pridictors

    Parameters
    ----------
    selector: np.array
        a `np.bool_` array to select usable cells in a numpy 2D array
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.

      See `prepare_predictors` for more details.

    verbose: Default: False
      If the method should print runtime info

    Returns
    -------
    np.array:
        A 2D array of np.bool_ values indicating what cell can be used.
    """
    aggr_selector = np.copy(selector)
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        with rio.open(pred_file_path, 'r') as psrc:
            _pred_mask = psrc.read_masks(band)
            all_pixels = psrc.width * psrc.height
        pred_selector = to_numpy_selector(_pred_mask)
        if verbose:
            data_pixels = usable_pixels_count(pred_selector)
            print("##########")
            print(f"- predictor:\n\t{pred_file_path=}")
            usable_pixels_info(all_pixels, data_pixels)
        np.logical_and(aggr_selector, pred_selector, out=aggr_selector)
        # now handle the case of a categorical band
        if len(predictor) >= 3:
            extract_values = predictor[2]
        else:
            extract_values = None
        if len(predictor) >= 4:
            mask_unselected = predictor[3]
        else:
            mask_unselected = False
        if extract_values is not None and mask_unselected:
            # get all categories
            all_cats = set()
            with rio.open(pred_file_path, mode='r') as psrc:
                for _, window in psrc.block_windows(band):
                    wdata = psrc.read(band, window=window)
                    wdw_slices = window.toslices()
                    np.logical_and(
                        aggr_selector[wdw_slices],
                        np.isin(wdata, extract_values),
                        out=aggr_selector[wdw_slices]
                    )
                    all_cats = all_cats.union(np.unique(wdata))
            if verbose:
                unselected_types = all_cats.difference(extract_values)
                data_pixels = usable_pixels_count(aggr_selector)
                print(f"After masking {unselected_types=}:")
                usable_pixels_info(all_pixels, data_pixels)
                print("##########")
    return aggr_selector

# TODO: needs adaptation to use dict for predictors with tags
def prepare_selector(response: str,
                     *predictors: tuple[str,
                                        int,
                                        tuple[int, ...] | None,
                                        bool | None],
                     verbose=False)->NDArray:
    """Creates a boolean selector based on the masks of response and predictors

    The selector is a np.array of type np.bool_ indicating which well can be
    used (`True`) and which cannot (`False`)

    Parameters
    ----------
    _See `prepare_predictors` for a detailed description description._

    response:
      Path to a map (.tif file) that holds the response data
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
    verbose: Default: False
      If the method should print runtime info

    Returns
    -------
    np.array
      Boolean array of the same shape as `response`

    """
    # read out the response
    with rio.open(response, 'r') as src:
        # get the shape and the projection
        src_profile = src.profile.copy()
        # get the nodata values or try to read the mask
        mask = src.read_masks(1)
    src_width = src_profile["width"]
    src_height = src_profile["height"]
    all_pixels = src_width * src_height
    # determine the number of rows (height*width - # of masked pixels)
    # we want a numpy boolean mask
    selector = to_numpy_selector(mask)
    if verbose:
        print("\nResponse data:")
        data_pixels = usable_pixels_count(selector)
        usable_pixels_info(all_pixels, data_pixels)

    # first we get all the masks to compute an overall mask
    aggr_selector = enrich_selector(selector, *predictors, verbose=verbose)

    data_pixels = usable_pixels_count(aggr_selector)
    if verbose:
        print("\nFinal selector:\n")
        usable_pixels_info(all_pixels, data_pixels)

    return aggr_selector


def init_X(predictors: list[tuple[str,
              int,
              tuple[int, ...] | None,
              bool | None]],
           selector:NDArray,
           window:Window|None,
           include_intercept:bool)->NDArray:
    """Initiates the matrix X with the appropriate width and height

    Parameters
    ----------
    predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
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
    nbr_cols = 0
    for pred in predictors:
        if len(pred) >= 3 and pred[2] is not None:
            nbr_cols += len(pred[2])
        else:
            nbr_cols += 1
    if include_intercept:
        nbr_cols += 1
    # create emtpy predictor array
    return np.zeros((nbr_rows, nbr_cols), np.float64)


def populate_X(X:NDArray,
               predictor_datas:list[NDArray],
               window:Window|None,
               selector:NDArray,
               include_intercept:bool):
    """Adds column per predictor_datas with selector applied in the window view

    ..Note::
      $X$ is updated in place.
    
    Parameters
    ----------
    predictors_datas:
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
    if window is not None:
        _selector = selector[window.toslices()]
    else:
        _selector = selector 
    # apply the mask and populate X
    for i, data in enumerate(predictor_datas):
        # reshape to (-1,1)
        # hstack to predictor array
        X[:, i] = data[_selector].reshape(1, -1)
    # attach intercept column (of 1s) if chosen
    if include_intercept:
        X[:, -1] = 1.0


# TODO: here we use band indexes, but we should switch to using tags
#       both for predictors and for response
def prepare_predictors(response: str,
                       *predictors: tuple[str,
                                          int,
                                          tuple[int, ...] | None,
                                          bool | None],
                       view: tuple[int, int, int, int] | None = None,
                       include_intercept=True,
                       verbose: bool = False):
    r"""Generates and returns the parameters for a multiple linear regressio

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
    In doing so we avoid including pixels with incomplete informaiton into the
    regression analysis.

    In doing so we can reduce the amount of pixel to include in the multiple
    linear regression analysis, therefore, reducing the size of the response
    vector, leading ot a denser but smaller predictor matrix.
    
    ..Note::
      This function returns an InferenceError if some predictor is invalid.
      
      An invalid predictor can happen if it is a linear combination of the
      other predictors.

      We do not perform a systematic test for that but check the following
      cases:

      - If a predictor, after applying the final selector, contains only 0's
      - If a predictor consists of categorical data (i.e. a 2D array that
        categorizes each pixel) with each selected pixel alwasy belonging
        to a category (i.e. no unselected categories or unselected are masked
        - see details for `prepictors` argument) and `include_intercept` is
        set to True.

    Parameters
    ----------
    response:
      Path to a map (.tif file) that holds the response data

      ..Note::
        The response must be stored in a single band.
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.

      Each tuple must contain:

      - **first element** the path to a .tif file from which to load the data.
      - **second element** is an int providing the band to load form the tif
        file.
      Optional are:

      - **third element** provide the values to extract from a
        _categorical band_. The values are extracted from the band and the
        routine will create individual predictors for each.
      - **fourth element** is a boolean to indicate if the un-selected
        categories from the band should be masked.
        
        **default=`False`**

        ..Warning::
            If you selected a _categorical band_ and masked the un-selected
            categories then `include_intercept` must be `False` as otherwise
            we end up with a singular matrix (i.e. not invertible) since the
            last column (i.e. the 1's) is a linear combination of all the
            columns with the selected categories.

      ..Note::
        If the **third element** is providing a `tuple` of values, then the
        routine tries to extract these values from the band that that was
        provided in the **second element**.

        E.g. the following would extract band 1 and use and create 3
        predictor maps each one a binary map indicating the presence of the
        values 0, 2 and 3:

        ```
        ('path/to/tif.tif', 1, (0,2,3))
        ```
        This is equivalent to 
        ```
        ('path/to/tif.tif', 1, (0,2,3), False)
        ```
        Setting the **fourth** element to `True`,
        ```
        ('path/to/tif.tif', 1, (0,2,3), True)
        ```
        will extract all values from band `1` and mask all cells with categories
        other than `0`, `2`, or `3`.
        
    view:
      An optional tuple (x, y, width, height) defining the view of the predictors
      and response data to consider.
      If not provided then the entire response map is used.
    include_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.

    Return
    ------
    np.array:
        2D array with the width of the total number of predictors (+1 if the
        intercept is fitted as well) and the height being equal to the number
        of data pixels with valid data.
    """
    # make sure used tif files are compatible (i.e. check crs and units)
    check_compatibility(response, *(pred[0] for pred in predictors))
    # extract the mask from response and enrich it with masks from predictors
    aggr_selector = prepare_selector(response, *predictors, verbose=verbose)
    # check that we have valid predictor columns
    check_predictors(aggr_selector,
                     *predictors,
                     include_intercept=include_intercept,
                     verbose=verbose)
    riow = view_to_window(view)
    # create empty predictor array
    X = init_X(list(predictors),
               selector=aggr_selector,
               window=riow,
               include_intercept=include_intercept)
    # get response data
    with rio.open(response, 'r') as src:
        # TODO: use tags to get the band index 
        response_dtype = src.dtypes[0]
    # loop again over the predictors to extract data
    pred_datas = extract_predictor_data(*predictors,
                                        window=riow,
                                        as_dtype=response_dtype)
    populate_X(X=X, predictor_datas=pred_datas, window=riow,
               selector=aggr_selector, include_intercept=include_intercept)
    # return predictor matrix and the response vector
    y = partial_response(response, riow, aggr_selector)
    return X, y


def extract_predictor_data(*predictors: tuple[str,
                                              int,
                                              tuple[int, ...] | None,
                                              bool | None],
                           window: Window | None,
                           as_dtype):
    """Extract the data form the predictors

    Parameters
    ----------
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
    window:
      Limits the data array to a specific window
    as_dtype:
      The data type to represent each category in if a categorical predictor
      is present
    """
    # Create a window
    pred_datas = []
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        if len(predictor) >= 3:
            extract_values = predictor[2]
        else:
            extract_values = None
        with rio.open(pred_file_path, 'r') as psrc:
            pred_data = psrc.read(indexes=band, window=window)
        if extract_values is not None:
            for value in extract_values:
                pred_datas.append(
                    select_category(pred_data, category=value,
                                    as_dtype=as_dtype,
                                    # TODO: limits should only be used for float
                                    limits=(1.0, 0.0))
                )
        else:
            pred_datas.append(pred_data.astype(as_dtype))
    return pred_datas


def transposed_product(predictors:list[tuple[str,
                                             int,
                                             tuple[int, ...] | None,
                                             bool | None]],
                       view:tuple[int,int,int,int]|None,
                       selector:NDArray,
                       include_intercept: bool = False,
                       as_dtype=np.float64
                       ):
    """Extracts the selector of the predictor data in the provided view.

    Parameters
    ----------
    predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
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


# TODO: we use band index here, switch to using tags
def partial_response(response:str,
                     window:Window|None,
                     selector:NDArray):
    """Returns the window view of the response data after applying the selector

    Parameters
    ----------
    response:
      Path to a map (.tif file) that holds the response data
    """
    if window is not None:
        _selector = selector[window.toslices()]
    else:
        _selector = selector 
    with rio.open(response, 'r') as src:
        # NOTE: for now we assume the response has just one band
        response_data = src.read(indexes=1, window=window)
        # response_dtype = src.dtypes[0]
    return response_data[_selector]

def partial_X(predictors: list[tuple[str,
              int,
              tuple[int, ...] | None,
              bool | None]],
              window:Window|None,
              selector:NDArray,
              include_intercept:bool,
              as_dtype):
    """Generate (a partial) predictor matrix, $`X`$.

    If `window` is provided then only the specified selection will
    be read out from the predicotrs which allows to use this method
    when partially processing the prodictors.

    Parameters
    ----------
    predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
    window:
      Limits the data array to a specific window. The window is converted to a
      `slice` with `window.toslices()`.

      ..Note::
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
               include_intercept=include_intercept)
    # read out the data
    pred_datas = extract_predictor_data(*predictors,
                                        window=window,
                                        as_dtype=as_dtype)
    populate_X(X=X,
               predictor_datas=pred_datas,
               window=window,
               selector=selector,
               include_intercept=include_intercept)
    return X


def get_optimal_weights_source(Y:NDArray,
                               response:str,
                               predictors: list[tuple[str,
                                             int,
                                             tuple[int, ...] | None,
                                             bool | None]],
                               view:tuple[int,int,int,int]|None,
                               selector,
                               include_intercept: bool = False,
                               as_dtype=np.float64
                               )->NDArray:
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
      Path to a map (.tif file) that holds the response data
    predictors:
      An arbitrary number of tuples, each specifying one or several predictors.
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
    part_y = partial_response(response, riow, selector)
    betas = Y @ part_X.T @ part_y
    return betas


def get_approx_weights(X:NDArray,
                       y:NDArray,
                       fit_intercept:bool=False)->LinearRegression:
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
