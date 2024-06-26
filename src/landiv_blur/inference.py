"""
This module contains functions to facilitate carrying out various inference
methods.

In particular, it implements a multiple linear regression approach that uses
land-cover types and various derivatives thereof as predictors for different
directly or indirectly
measured variables, like the NDVI or the temperature

"""
from __future__ import annotations

import numpy as np
import rasterio as rio

from scipy import sparse as ssp

from .helper import (check_compatibility,
                     usable_pixels_info,
                     get_scale_factor)
from .processing import select_layer


def prepare_predictors(
        response: str, *predictors: tuple[str, int,
                                          tuple[int] | None],
        view: tuple[int, int, int, int] | None = None, with_intercept=True,
        **params):
    """Generates and returns the parameters for a multiple linear regression

    The parameters returned are $X$ and $\vec{y}$ from the multiple linear
    regression:

        $$\vec{y} = X\vec{\beta} + \vec{\epsilon}$$

    Where $\vec{y}$ represents the response data and $X$ the predictor matrix.

    This method uses the response data as reference to filter the predictors.
    It does so by extracting the mask (i.e. `nodata` value or an 8bit mask
    layer) from the response tif and applying it to each of the predictors.

    Since some of the predictor data might also have missing values, we
    iterate once over all predictor data and add pixels with missing values
    to the mask.
    In doing so we avoid including pixels with incomplete informaiton into the
    regression analysis.

    In doing so we can reduce the amount of pixel to include in the multiple
    linear regression analysis, therefore, reducing the size of the response
    vector, leading ot a denser but smaller predictor matrix.

    Parameters
    ----------
    response:
      Path to a map (.tif file) that holds the response data

      ..Note::
        The response must be stored in a single pand.
    *predictors:
      An arbitrary number of tuples, each specifying one or several predictors.

      Each tuple must contain as **first element** the path to a .tif file from
      which to load the data.
      The **second element** is an int providing the band to load form the tif
      file.
      The **third element** is optonal and can be used to provide a list of
      values to extract from the band and create individual predictor for each

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
    view:
      An optional tuple (x, y, width, height) defining the view to consider.
      If not provided then the entire response map is used.
    with_intercept:
      Determine if the predictor matrix should also contain an extra column of
      1s at the end, which is needed if also the intercepts should be fitted.

    Return
    ------
    np.array:
        2D array with the width of the total number of predictors (+1 if the
        intercept is fitted as well) and the height being equal to the number
        of data pixels with valid data.
    """
    # first make sure all used tif files are compatible (i.e. check crs and
    # units)
    check_compatibility(response, *(pred[0] for pred in predictors))
    print(get_scale_factor(response, predictors[0][0]))
    # read out the response
    # Q: Should we rely on rasterio directly or use our own interface, i.e.
    #    io.load_block?
    with rio.open(response, 'r') as src:
        # get the shape and the projection
        src_profile = src.profile.copy()
        # get the nodata values or try to read the mask
        mask = src.read_masks(1)
        # NOTE: for now we assume the response has just one band
        response_data = src.read(indexes=1)
        response_dtype = src.dtypes[0]
    src_width = src_profile["width"]
    src_height = src_profile["height"]
    nodata = src_profile["nodata"]
    src_type = src_profile["dtype"]
    # determine the number of rows (height*widht - # of masked pixels)
    # we want a numpy boolean mask
    npmask = np.where(mask == 255, True, False)
    vals, counts = np.unique(npmask, return_counts=True)
    data_pixels = counts[vals]  # vals: [True, False] or inv. in any case ok
    all_pixels = src_width * src_height
    print(f"{response=}")
    usable_pixels_info(all_pixels, data_pixels)
    no_data_pixels = all_pixels - data_pixels

    # for each entry in predictors
    # first we get all the masks to compute an overall mask
    aggregated_mask = np.copy(npmask)
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        if len(predictor) == 3:
            extract_values = predictor[2]
        else:
            extract_values = None
        with rio.open(pred_file_path, 'r') as psrc:
            psrc_profile = psrc.profile.copy()
            _mask = psrc.read_masks(band)
        _npmask = np.where(_mask == 255, True, False)
        vals, counts = np.unique(_npmask, return_counts=True)
        data_pixels = counts[vals]
        print(f"{pred_file_path}")
        usable_pixels_info(all_pixels, data_pixels)
        np.logical_and(aggregated_mask, _npmask, out=aggregated_mask)
    vals, counts = np.unique(aggregated_mask, return_counts=True)
    data_pixels = counts[vals]
    print("Final mask:")
    usable_pixels_info(all_pixels, data_pixels)

    no_data_pixels = all_pixels - data_pixels

    # determine the number of columns and rows for X
    nbr_rows = int(src_width * src_height - no_data_pixels)
    nbr_cols = 0
    for pred in predictors:
        if len(pred) == 3:
            nbr_cols += len(pred[2])
        else:
            nbr_cols += 1
    if with_intercept:
        nbr_cols += 1

    # create emtpy predictor array
    X = np.zeros((nbr_rows, nbr_cols))

    # now loop again over the predictors to extract data and then populate X
    pred_datas = []
    for predictor in predictors:
        pred_file_path, band = predictor[:2]
        if len(predictor) == 3:
            extract_values = predictor[2]
        else:
            extract_values = None
        with rio.open(pred_file_path, 'r') as psrc:
            psrc_profile = psrc.profile.copy()
            pred_data = psrc.read(indexes=band)
        if extract_values:
            for value in extract_values:
                pred_datas.append(
                    select_layer(pred_data, layer=value,
                                 as_dtype=response_dtype,
                                 limits=(1.0, 0.0))
                )
        else:
            pred_datas.append(pred_data.astype(response_dtype))
    # Now apply the mask and populate X
    for i, data in enumerate(pred_datas):
        # reshape to (-1,1)
        # hstack to predictor array
        X[:,i] = data[aggregated_mask].reshape(1, -1)
    # attach intercept column (of 1s) if chosen
    if with_intercept:
        X[:,-1] = 1.0
    # return predictor matrix and the response vector
    y = response_data[aggregated_mask]
    return X, y
