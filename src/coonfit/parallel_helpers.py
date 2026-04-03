"""
Internal worker functions for parallelized regression computations.

This module contains the individual job functions executed by worker processes
in the ``coonfit`` parallel pipeline. These functions are not part of the
public API and are called exclusively by :mod:`~coonfit.parallel`.

Worker functions cover:

- **Matrix aggregation**: Combining partial ``X.T @ X`` or beta matrices
  received from a multiprocessing queue (:func:`_combine_matrices`).
- **Partial products**: Computing ``X.T @ X`` and beta coefficients for a
  single spatial block (:func:`_partial_transposed_product`,
  :func:`_partial_optimal_betas`).
- **Predictor validation**: Counting valid pixels per predictor band and
  checking that each band meets the minimum contribution threshold
  (:func:`_process_band_count_valid`, :func:`_check_predictor_consistency`).
- **Model prediction**: Applying fitted regression weights to a spatial block
  to produce model output values (:func:`_block_model_prediction`).
- **Goodness-of-fit**: Partially computing the sum of squared residuals (SSR)
  and total sum of squares (SST) for RMSE and R² evaluation
  (:func:`_block_ssr`, :func:`_block_sst`).
"""

from __future__ import annotations

import math
from collections.abc import Collection

from typing import Union

import numpy as np

from multiprocessing import Queue
from numpy.typing import NDArray

from riogrande.io import Source, Band
from riogrande.helper import (
    view_to_window,
    convert_to_dtype,
    get_or_set_context,
    get_nbr_workers,
)
from riogrande.timing import TimedTask
from riogrande import parallel as rgpara

from .inference import (
    transposed_product,
    get_optimal_weights_source
)
from .exceptions import InvalidPredictorError


def _combine_matrices(output_q: Queue) -> tuple[NDArray | None, tuple]:
    """Aggregate matrices containing partial sums from a queue.

    This function listens to a queue and combines partial matrix results
    into a single aggregated matrix. It replaces NaN values with zeros
    during aggregation and continues until receiving a "kill" signal.

    Parameters
    ----------
    output_q : Queue
        Queue containing dictionaries with partial matrices under the 'X' key
        and optional control signals under the 'signal' key. The function
        terminates when it receives a dictionary with signal="kill".

    Returns
    -------
    out_matrix : NDArray or None
        The aggregated matrix with all partial sums combined. NaN values
        are replaced with zeros. Returns None if no matrices were processed.
    timer : tuple of TimedTask
        A single-element tuple containing a TimedTask object with timing
        information for the aggregation process.

    Notes
    -----
    - NaN values in partial matrices are automatically replaced with zeros
     before aggregation.
    - The function blocks until a "kill" signal is received.
    - Each partial matrix extraction is timed with a new lap marker
      via :meth:`~riogrande.timing.TimedTask.new_lab`.

    See Also
    --------
    :func:`_partial_transposed_product` : Produces X.T @ X blocks for this queue.
    :func:`_partial_optimal_betas` : Produces beta blocks for this queue.
    """
    out_matrix = None
    with TimedTask() as timer:
        while True:
            output = output_q.get()
            signal = output.get('signal', None)
            if signal:
                if signal == "kill":
                    # print(f"\n\nDone with the matrix aggregation.\n\n")
                    break
            partial = output.pop('X')
            if out_matrix is not None:
                part_array = np.array(partial)
                part_array[np.isnan(part_array)] = 0
                out_matrix += part_array
            else:
                out_matrix = np.array(partial)
                # replace nan's with 0's
                out_matrix[np.isnan(out_matrix)] = 0
            timer.new_lab()
    return out_matrix, (timer,)


def _partial_transposed_product(params: dict, output_q: Queue):
    """Execute transposed product computation in parallel and send results to queue.

    This function wraps `transposed_product` for parallel execution by calling
    it with the provided parameters and placing the result in the output queue.
    Results are wrapped in a dictionary with key 'X'.

    Parameters
    ----------
    params : dict
        Keyword arguments to pass to :func:`~coonfit.inference.transposed_product`.
    output_q : Queue
        Queue where the computed result will be placed. The result is wrapped
        in a dictionary as ``{'X': result}``.

    See Also
    --------
    :func:`_combine_matrices` : Aggregates the results from this function.
    :func:`~coonfit.inference.transposed_product` : The underlying computation.
    """

    def _wrap(tpX):
        return dict(X=tpX)

    rgpara.runner_call(
        output_q,
        transposed_product,
        params,
        wrapper=_wrap
    )


def _partial_optimal_betas(params: dict, output_q: Queue):
    """Execute optimal weight computation in parallel and send results to queue.

       This function wraps `get_optimal_weights_source` for parallel execution by
       calling it with the provided parameters and placing the result in the output
       queue. The resulting dictionary of beta weights is converted to a list of
       values before being placed in the queue.

       Parameters
       ----------
       params : dict
           Keyword arguments to pass to :func:`~coonfit.inference.get_optimal_weights_source`.
       output_q : Queue
           Queue where the computed result will be placed. The beta weights
           dictionary is converted to a list and wrapped as ``{'X': list_of_values}``.

       See Also
       --------
       :func:`_combine_matrices` : Aggregates the results from this function.
       :func:`~coonfit.inference.get_optimal_weights_source` : The underlying computation.
       """

    def _wrap(beta_dict):
        return dict(X=list(beta_dict.values()))  # ok for python >= 3.6 (dict keeps order)

    rgpara.runner_call(
        output_q,
        get_optimal_weights_source,
        params,
        wrapper=_wrap
    )


def _process_band_count_valid(band: Band, selector: NDArray[np.bool_], no_data: Union[int, float],
                              limit_count: int) -> tuple[dict, tuple]:
    """
    Count valid pixels in a band and return timing information.

    This function wraps the Band.count_valid_pixels method to add timing
    capabilities for performance monitoring in parallel processing workflows.

    Parameters
    ----------
    band : Band
        The band object on which to count valid pixels.
    selector : NDArray of bool
        Boolean mask array indicating which pixels to consider in the count.
        True values indicate pixels to include.
    no_data : int or float
        Value representing no-data or missing pixels. Pixels with this value
        are excluded from the valid pixel count.
    limit_count : int
        Minimum number of valid pixels required. The interpretation of this
        parameter depends on the Band.count_valid_pixels implementation.

    Returns
    -------
    valid_counts : dict
        Dictionary mapping the band to its valid pixel count, formatted as
        ``{band: count}``. Uses :meth:`~riogrande.io.models.Band.count_valid_pixels`.
    timer : tuple of TimedTask
        Single-element tuple containing a :class:`~riogrande.timing.TimedTask`
        object with timing information for the counting operation.

    See Also
    --------
    :func:`_check_predictor_consistency` : Uses this function in parallel to validate predictors.
    """
    with TimedTask() as timer:
        valid = band.count_valid_pixels(selector=selector,
                                        no_data=no_data,
                                        limit_count=limit_count)
    return {band: valid}, (timer,)



def _check_predictor_consistency(predictors: Collection[Band],
                                 selector: NDArray[np.bool_],
                                 tolerance: float = 0.0,
                                 no_data=0.0,
                                 sanitize: bool = False,
                                 **params) -> Collection[Band]:
    """Check if all predictors contain sufficient valid data after applying selector.

    This function validates that each predictor band has a minimum fraction
    of valid (non-masked, non-no_data) pixels within the selected region.
    Predictors failing to meet the threshold can either raise an exception
    or be automatically removed from the collection.

    Parameters
    ----------
    predictors : Collection of Band
        A collection with arbitrary many predictor bands to validate.
        See :func:`~coonfit.inference.prepare_predictors` for further details
        on how to specify predictors.
    selector : NDArray of bool
        Boolean array with the same shape as the predictors that indicates
        which cells are usable. True values indicate pixels to consider.
    tolerance : float, optional
        Minimum fraction of valid pixels required for a predictor to be
        considered valid. Default is 0.0.

        The fraction is computed as (number of valid pixels) / (total number
        of True values in selector). A predictor is invalid if its fraction
        of valid pixels is at or below this threshold.

        For example, tolerance=0.01 requires at least 1% valid pixels.
    no_data : float, optional
        Value indicating invalid or missing data. Pixels with this value
        are not counted as valid. Default is 0.0.
    sanitize : bool, optional
        If True, automatically remove predictors that fail the validity check.
        If False, raise an InvalidPredictorError when invalid predictors are
        found. Default is False.
    **params : dict
        Optional arguments for multiprocessing:

        - nbrcpu : int, optional
            Number of CPUs to use. If not set, uses (available threads - 1).
        - start_method : str, optional
            Starting method for multiprocessing jobs ('fork', 'spawn', or
            'forkserver').

    Returns
    -------
    valid_predictors : Collection of Band
        The remaining valid predictors. If `sanitize=False` (default), this
        corresponds to the input `predictors` collection if all pass validation.
        If `sanitize=True`, returns only predictors that meet the minimum
        valid pixel threshold.

    Raises
    ------
    :exc:`~coonfit.exceptions.InvalidPredictorError`
        If `sanitize=False` and one or more predictors fail to meet the
        minimum valid pixel threshold.

    Notes
    -----
    The function uses multiprocessing (via :func:`~riogrande.helper.get_or_set_context`
    and :func:`~riogrande.helper.get_nbr_workers`) to validate predictors in parallel
    for improved performance with large collections.

    The minimum number of valid pixels required is calculated as:
    ``max(1, ceil(tolerance * total_selected_pixels))``
    ensuring at least 1 valid pixel is required even when tolerance=0.0.

    See Also
    --------
    :func:`_process_band_count_valid` : Worker function counting valid pixels per band.
    :func:`_block_model_prediction` : Uses the validated predictors for model prediction.
    """
    print(f'check_predictor_consistency - {predictors=}')
    _vals, _counts = np.unique(selector, return_counts=True)
    total_selected = int(_counts[_vals][0])
    # convert the tolerance fraction to an actual number of pixels
    limit_count = max(1, math.ceil(tolerance * total_selected))
    # for each predictor
    # - apply the selector
    # - count the number of 'valid' cells > valid
    # - if valid/total_selecte is <= tolerance > predictor is problematic
    #   - either kick it out (if 'sanitize') or raise exception
    # set the per job parameter
    job_params = []
    for predictor in predictors:
        jparams = dict(
            band=predictor,
            selector=selector,
            limit_count=limit_count,
            no_data=no_data
        )
        job_params.append(jparams)
    start_method = params.get('start_method', None)
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
        all_jobs = []
        for jparams in job_params:
            all_jobs.append(pool.apply_async(_process_band_count_valid,
                                             kwds=jparams))
        band_validity = dict()
        durations = []
        for job in all_jobs:
            valid_band, (time,) = job.get()
            band_validity.update(valid_band)
            durations.append(time.get_duration())
    if not all(band_validity.values()):
        valid_predictors = []
        invalid_predictors = []
        invalid_str = ''
        for band, valid in band_validity.items():
            if not valid:
                invalid_str += f"- {band}\n"
                invalid_predictors.append(band)
            else:
                valid_predictors.append(band)
        if sanitize:
            print("WARNING: Some predictors do not satify the minimal "
                  f"contribution condition:\n{invalid_str}\n"
                  "They will be removed from the list of predictors!")
            return valid_predictors
        else:
            raise InvalidPredictorError(f"Invalid predictors:\n{invalid_str}")
    else:
        return predictors


def _block_model_prediction(params: dict, job_out_q: Queue) -> TimedTask:
    """
    Compute model predictions for a single block in a parallel processing workflow.

    This function applies a fitted linear regression model to a spatial block
    (view) of predictor data, handling optional selectors and categorical
    stratification. The computed predictions are placed in a queue for
    aggregation.

    Parameters
    ----------
    params : dict
        Dictionary containing all parameters for processing a single block.
        Required keys:

        - view : tuple of int
            (x, y, width, height) defining the spatial block to process.
        - predictors : Collection of Band
            Band objects used as predictors in the regression model.
        - optimal_weights : dict or dict of dict
            Optimal weights for each predictor. If `selector_band` is not
            provided, this is a simple dict mapping predictors to weights.
            If `selector_band` is provided, this maps category values to
            dicts of predictor weights.

            May include an 'intercept' key for the model intercept (beta0).
        - as_dtype : str or type
            Data type for the output prediction array.

        Optional keys:

        - selector : NDArray of bool, optional
            Boolean mask to selectively compute predictions. Pixels where
            selector is False are set to NaN in the output.
        - selector_band : Band, optional
            Band containing categorical data for stratified predictions.
            If provided, `optimal_weights` must map each category to its
            own weight dictionary.
        - predictors_as_dtype : str or type, optional
            Data type to convert predictor data to before applying weights.

    job_out_q : Queue
        Multiprocessing queue where the computed block result is placed.
        Results are dictionaries with keys 'data' (prediction array) and
        'view' (spatial location).

    Returns
    -------
    timer : :class:`~riogrande.timing.TimedTask`
        TimedTask object containing timing information for the block
        processing operation.

    See Also
    --------
    :func:`_check_predictor_consistency` : Validates predictors before this step.
    """
    with TimedTask() as timer:
        predictors = params.pop('predictors')
        optimal_weights = params.pop('optimal_weights')
        # for the interaction only the inner view is needed
        view = params.get('view')
        window = view_to_window(view)
        selector = params.get('selector')
        selector_band = params.get('selector_band')
        predictors_as_dtype = params.get('predictors_as_dtype')
        as_dtype = params.get('as_dtype')
        width = view[2]
        height = view[3]
        # start with an all zero map and in the correct data type
        model_data = np.zeros(shape=(height, width), dtype=as_dtype)

        if selector is not None:
            _selector = selector[window.toslices()]
            model_data[~_selector] = np.nan

        if selector_band is not None:
            selector_data = selector_band.load_block(view=view)['data']
            selectors = np.unique(selector_data, ).tolist()
            if np.nan in selectors:
                selectors.remove(np.nan)
        else:
            # just pretend that optimal weights is expressed for some dummy selector
            selectors = [0, ]
            selector_data = np.zeros_like(model_data, dtype=np.uint8)
            optimal_weights = {0: optimal_weights}

        for select in selectors:
            _opt_weights = optimal_weights[select]
            _selector = np.where(selector_data == select, True, False)
            if 'intercept' in _opt_weights:
                model_data += np.where(_selector, _opt_weights['intercept'], 0)
            for pred in predictors:
                block_data = pred.load_block(view=view)['data']
                if predictors_as_dtype is not None:
                    block_data = convert_to_dtype(
                        block_data,
                        as_dtype=predictors_as_dtype,
                        in_range=None, out_range=None
                    )
                # add each predictor data layer multiplied by its weight
                if pred in _opt_weights:
                    # add the contributions of each predictor in the final output dtype
                    model_data += np.where(_selector, (_opt_weights[pred] * block_data).astype(as_dtype), 0)
        output = dict(
            data=model_data,
            view=view
        )
        job_out_q.put(output)
    return timer


def _block_ssr(params: dict, ssr_parts: list):
    """
    Partially calculate the Sum of Squares of the Residuals (SSR) for a given data window.

    This function computes the residuals between the observed response and model predictions,
    squares them, and accumulates their sum along with the count of valid (non-NaN) entries
    into `ssr_parts`. It is intended for internal, partial computations of SSR in blocks.

    Parameters
    ----------
    params : dict
        Dictionary containing required inputs:
        - 'response' : object
            An object with a `get_data(window)` method returning the observed data.
        - 'model' : object
            An object with a `get_data(window)` method returning model predictions.
        - 'selector' : array-like
            Boolean mask indicating which data points to include.
        - 'view' : optional
            A view object used to determine the window of data to process.
    ssr_parts : list
        A list to which a tuple `(sum_of_squares, count)` will be appended. Each tuple
        contains the sum of squared residuals and the number of valid residuals.

    Returns
    -------
    None
      The function appends results directly to `ssr_parts` and does not return anything.
      Uses :func:`~riogrande.helper.view_to_window` to obtain the spatial window.

    See Also
    --------
    :func:`_block_sst` : Compute partial Total Sum of Squares (SST) for a window.
    """
    response = params.pop("response")
    model = params.pop("model")
    selector = params.pop("selector")
    view = params.get('view')
    window = view_to_window(view)

    # Get data fromte window
    response_data = response.get_data(window=window)
    model_data = model.get_data(window=window)

    # Selector application
    _selector = selector[window.toslices()]
    response_data[~_selector] = np.nan
    model_data[~_selector] = np.nan

    # Calculate Residuals and aggregate them
    residuals = np.subtract(response_data, model_data)
    residuals_pwr = np.power(residuals, 2)
    return ssr_parts.append((np.nansum(residuals_pwr), np.count_nonzero(~np.isnan(residuals_pwr))))


def _block_sst(params: dict, sst_parts: list):
    """
    Partially calculate the Total Sum of Squares (SST) for a given data window.

    This function computes the squared differences between observed responses and the
    mean response, accumulating their sum along with the count of valid (non-NaN) entries
    into `sst_parts`. It is intended for internal, partial computations of SST in blocks.

    Parameters
    ----------
    params : dict
        Dictionary containing required inputs:
        - 'response' : object
            An object with a `get_data(window)` method returning the observed data.
        - 'y_mean' : float or array-like
            Mean of the response variable used to compute deviations.
        - 'selector' : array-like
            Boolean mask indicating which data points to include.
        - 'view' : optional
            A view object used to determine the window of data to process.
    sst_parts : list
        A list to which a tuple `(sum_of_squares, count)` will be appended. Each tuple
        contains the sum of squared deviations from the mean and the number of valid entries.

    Returns
    -------
    None
        The function appends results directly to `sst_parts` and does not return anything.
        Uses :func:`~riogrande.helper.view_to_window` to obtain the spatial window.

    See Also
    --------
    :func:`_block_ssr` : Compute partial Sum of Squares of Residuals (SSR) for a window.
    """
    response = params.pop("response")
    y_mean = params.pop("y_mean")
    selector = params.pop("selector")
    view = params.get('view')
    window = view_to_window(view)

    # Get data from window
    response_data = response.get_data(window=window)

    # Selector application
    _selector = selector[window.toslices()]
    response_data[~_selector] = np.nan

    # Calculate Residuals and aggregate them
    diff_mean = np.subtract(response_data, y_mean)
    diff_pwr = np.power(diff_mean, 2)
    return sst_parts.append((np.nansum(diff_pwr), np.count_nonzero(~np.isnan(diff_pwr))))
