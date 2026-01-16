"""
This module contains functions to parallellize various inference methods.
"""
# is_needed
# needs_work (the module is too big!)
# not_tested (partially)
# usedin_both (should be split up!)
from __future__ import annotations

from collections.abc import Collection

from typing import Union

import numpy as np

from multiprocessing import (Queue, Manager)
from numpy.typing import NDArray

from riogrande.io import Source, Band
from riogrande.helper import (
    view_to_window,
    check_compatibility,
    get_or_set_context,
    get_nbr_workers,
)
from riogrande.prepare import (
    create_views,
)
from riogrande import parallel as rgpara

from .helper import check_rank_deficiency
from . import parallel_helpers as lph


def compute_model(predictors: Collection[Band],
                  optimal_weights: dict[dict] | dict | None,
                  output_file: str,
                  block_size: tuple[int, int],
                  predictors_as_dtype: str | type | None = None,
                  profile: dict | None = None,
                  selector: NDArray[np.bool_] | None = None,
                  selector_band: Band | None = None,
                  verbose: bool = False,
                  **params) -> str:
    """
    Create a tif file with the model prediction values from a fitted model.

    Parameters
    ----------
    predictors : Collection of Band
        Collection of predictor bands used in the multiple linear regression.
    optimal_weights : dict of dict, dict, or None
        Holding for each predictor the optimal weight.
        If weights include a key named "intercept",
        this will be used as the intercept (beta0) for the model prediction.
        If a `selector_band` is provided, then it must hold for each
        categorical value (key) a dictionary with the optimal weights per
        predictor.

        See `selector_band` parameter for more details.

    output_file : str
        Path to where the model result should be written to.
    block_size : tuple of int
        Size (width, height) in pixels of the block that a single job
        processes.
    predictors_as_dtype : str, type, or None, optional
        Datatype to convert predictor input to (e.g. np.float32) prior to
        computing their contribution.

        .. note::

          Only a type conversion is supported prior to computing the predictors
          contribution. Rescaling of a predictor needs to happen in a separate
          step beforehand.

    profile : dict or None, optional
        The profile to use for the newly created output tif.
        By default the profile is copied from the first source of the
        predictor bands, updating the count to 1.
    selector : ndarray of bool or None, optional
        A selector array to use to selectively calculate the model prediction.

        If a boolean array is provided then it is applied as an (inverted) mask:
        only pixels that result to `True` are calculated.

        If a categorical array (np.uint8) is provided instead, then it is assumed
        that the `optimal_weights` use it as a selector for (masking) the
        processing of the model.
    selector_band : Band or None, optional
        A band object with categorical data.

        If provided, the `optimal_weights` needs to hold for each of the category
        a dictionary with predictor specific weights.
    verbose : bool, optional
        Print out processing steps. Default is False.
    **params
        Optional arguments for the multiprocessing:

        - nbrcpu : int, optional
            Number of CPUs to use for parallel processing.
        - start_method : str, optional
            Multiprocessing start method ('fork', 'spawn', or 'forkserver').
        - compress : bool, optional
            If True, apply LZW compression to the final output file.

    Returns
    -------
    output_file : str
       Path to the newly created tif file holding the model prediction data.
       If compression is enabled, this will be the path to the compressed file.

    Notes
    -----
    The function uses multiprocessing to process the image in blocks for
    improved performance. Each block is processed independently and results
    are aggregated into the final output file.

    Timing information for each job and the total duration is printed to
    standard output upon completion.
    """
    # get all source files
    sources = tuple(set(pred.source.path for pred in predictors))
    check_compatibility(*sources)

    # Handle the output profile
    if profile is None:
        profile = Source(path=sources[0]).import_profile()
        profile['count'] = 1

    # make sure to get the data output type from the profile
    as_dtype = profile.get('dtype')

    # Note: with profile one could provide invalid dimensions
    height = profile['height']
    width = profile['width']
    out_band = None  # cerate a new band bidx=1
    tags = dict(category='model')

    # Parameter for the aggregation task
    combine_params = dict(
        profile=profile,
        output_file=output_file,
        band=out_band,
        tags=tags
    )

    # Parameter for the individual jobs
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))
    block_params = []
    for view in inner_views:
        bparams = dict(view=view,
                       predictors=predictors,
                       predictors_as_dtype=predictors_as_dtype,
                       selector=selector,
                       selector_band=selector_band,
                       optimal_weights=optimal_weights,
                       as_dtype=as_dtype,
                       )
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    job_out_q = manager.Queue()
    start_method = params.get('start_method', None)
    # get number of workers
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
        # start the aggregator task
        combiner_job = pool.apply_async(rgpara.combine_views,
                                        (combine_params, job_out_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(lph._block_model_prediction,
                                             (bparams, job_out_q)))
        # collect results
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get().get_duration())

        # once we have all the blocks, add a last element to the queue to stop
        # the combination process
        job_out_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()

    # lzw-compress final output
    compress = params.pop('compress', False)
    if compress:
        out_source = Source(output_file)
        out_source.compress(output=None)
        output_file = str(out_source.path)
        print("Files compressed successfully")

    total_duration = combiner_job.get().get_duration()
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")
    return output_file


# TODO: mpc_params should become first class parameter if mandatory
def get_XT_X(response: str | Band,
             *predictors: Band | str,
             selector: NDArray,
             include_intercept: bool = True,
             verbose: bool = False,
             **mpc_params
             ) -> np.ndarray:
    # is_needed
    # needw_work (doc; see TODO's)
    # is_tested
    """Calculate X.T @ X in parallel directly from view of the predictor data

    .. note::
      `response` is only used to get the correct dimension of the data

    Parameters
    ----------
    response:
      Path to a tif file that contains the response data. The file is only used
      to get the dimensions.
    *predictors:
      A collection with arbitrary many predictors to use.
      See inference.prepare_predictors for further details on how to specify
      predictors.
    selector:
        a `np.bool_` array to select usable cells in a numpy 2D array
    include_intercept: _optional_
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    verbose: _optional_
      If the method should print runtime info
    **mpc_params:
      Parametrization of the multiporcessing approach.
      Needed are:

      view_size:
        The size (width, height) in pixels of a single view (excluding borders)
      
      Optional:

      nbrcpu: int
        The number of cpu's to use. If not set then then the available number
        of threads -1 are used.
      start_method: str
        Starting method for multiprocessing jobs

    """

    print(f'get_XT_X - {response=}, {predictors=}')
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    start_method = mpc_params.get('start_method', None)
    view_size = mpc_params.get('view_size')
    nbr_workers = get_nbr_workers(number=mpc_params.get('nbrcpu', None))
    src_profile = response.source.import_profile()
    src_width = int(src_profile.get('width'))
    src_height = int(src_profile.get('height'))
    # create a list of views and put it into runner_params
    size = (src_width, src_height)
    border = (0, 0)
    #  _ is for the inner_views which we do not need
    views, _ = create_views(view_size=view_size,
                            border=border,
                            size=size)
    part_params = []
    for view in views:
        pparams = dict(predictors=predictors,
                       view=view,
                       selector=selector,
                       include_intercept=include_intercept)
        part_params.append(pparams)
    # create the arguments for the aggregation script
    # start the processes 
    manager = Manager()
    output_q = manager.Queue()
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
        # start the aggregation step
        matrix_aggregator = pool.apply_async(
            lph._combine_matrices,
            (output_q,)
        )
        all_jobs = []
        for pparams in part_params:
            all_jobs.append(pool.apply_async(
                lph._partial_transposed_product,
                (pparams, output_q)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        output_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        recombined_tpX, _ = matrix_aggregator.get()
    return recombined_tpX


def get_optimal_betas(*predictors: Band | str,
                      Y: np.ndarray,
                      response: str | Band,
                      selector,
                      include_intercept=True,
                      verbose: bool = False,
                      as_dtype=np.float64,
                      **mpc_params
                      ):
    # is_needed
    # needw_work (doc)
    # is_tested
    """
    """
    print(f'get_optimal_betas - {response=}, {predictors=}')
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    start_method = mpc_params.get('start_method', None)
    view_size = mpc_params.get('view_size')
    nbr_workers = get_nbr_workers(number=mpc_params.get('nbrcpu', None))
    src_profile = response.source.import_profile()
    src_width = int(src_profile.get('width'))
    src_height = int(src_profile.get('height'))
    size = (src_width, src_height)
    border = (0, 0)
    #  _ is for the inner_views which we do not need
    views, _ = create_views(view_size=view_size,
                            border=border,
                            size=size)
    # define the parameters for each job
    part_params = []
    for view in views:
        pparams = dict(Y=Y,
                       response=response,
                       predictors=predictors,
                       view=view,
                       selector=selector,
                       include_intercept=include_intercept,
                       as_dtype=as_dtype
                       )
        part_params.append(pparams)
    # create the arguments for the aggregation script
    # start the processes 
    manager = Manager()
    output_q = manager.Queue()
    if verbose:
        print(f"using {nbr_workers=}")

    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
        # start the aggregation step
        matrix_aggregator = pool.apply_async(
            lph._combine_matrices,
            (output_q,)
        )
        all_jobs = []
        for pparams in part_params:
            all_jobs.append(pool.apply_async(
                lph._partial_optimal_betas,
                (pparams, output_q)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        output_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        betas, _ = matrix_aggregator.get()

    # append predictor if intercept set
    if include_intercept:
        pred_list = list(predictors)
        pred_list.append('intercept')
        predictors = tuple(pred_list)

    if len(betas) != len(predictors):
        raise ValueError(f"Number of predictors {len(predictors)} not equal with number of fitted values {len(betas)}")

    if verbose:
        print(f"{predictors=}")
        print(f"{betas=}")
    return {pred: beta for pred, beta in zip(predictors, betas)}


def get_XT_X_dependency(response: str | Band,
                        predictors: Collection[Band],
                        block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                        include_intercept: bool = True,
                        limit_contribution: float = 0.0,
                        no_data: Union[int, float] = 0.0,
                        sanitize_predictors: bool = False,
                        verbose: bool = False,
                        **params
                        ) -> dict[Band, str]:
    # is_needed (in tests only)
    # needw_work (doc)
    # is_tested
    """Test for linear dependency of columsn (before using other functions to fit the MLR)

    Parameters
    ----------
    response:
      Path to a tif file that contains the response data. The file is only used
      to get the dimensions.
    *predictors:
      A collection with arbitrary many predictors to use.
      See inference.prepare_predictors for further details on how to specify
      predictors.
    block_size:
        Block sizes for specific functions or a default block size for all functions.
        If a dictionary is provided, it should map function names to block sizes
        ('prepare_selector': tuple, 'get_XT_X': tuple, 'get_optimal_betas': tuple).
        If a single tuple is provided, it will be used for all functions.
    include_intercept:
        Whether to fit the intercept when computing weights
    verbose:
        Print out processing step infos
    limit_contribution:
        The fraction of cells, among all valid cells, each predictor must
        contribute for it to be considered a valid predictors.
        By default (i.e. `limit_contribution=0.0`) a single value is enough.
    no_data:
        Each cell with this value is considered to be invalid.
        Most likely you will never have to change this!
    sanitize_predictors:
        Determines if predictors that end up
        contributing not a single data-point should be removed automatically.
        By default this values is set to `False` which raises an exception
        if a predictor ends up contributing nothing.
    **params:
        Optional arguments:

        - `nbr_cpus` (int): how many CPUs should be used (by default the number
          of available CPUs minus one will be used.
        - `start_method` (str): Determines how the workers should start a
          process. Accepted are 'spawn', 'fork' or 'forkserver'.
    """

    # if block sizes are provided as dictionary - some pre-check on input is desired - else
    block_size_params = dict(prepare_selector=None, get_XT_X=None)
    if isinstance(block_size, tuple):
        for key in block_size_params:
            block_size_params[key] = block_size
    if isinstance(block_size, dict):
        if block_size_params.keys() != block_size.keys() or not all(isinstance(v, tuple) for v in block_size.values()):
            raise ValueError(f"Block size dict does not conform with all necessary keys and value-types: {block_size=}")
        block_size_params.update(block_size)

    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)

    print("Creating selector...")
    selector = rgpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size_params["prepare_selector"],
        verbose=verbose,
        **params
    )

    # If selector is empty (meaning all is FALSE) - no need to proceed
    _vals = np.unique(selector)
    if len(_vals) == 1:
        if _vals[0] == False:  # if not _vals[0] is less explicit
            print(f"WARNING: no pixels to fit, selector masks all pixels")
            return None

    print("Check consistency of remaining predictor data...")
    nbr_predictors = len(predictors)
    predictors = lph._check_predictor_consistency(predictors,
                                              selector=selector,
                                              tolerance=limit_contribution,
                                              sanitize=sanitize_predictors,
                                              no_data=no_data,
                                              verbose=verbose,
                                              **params)

    if len(predictors) != nbr_predictors:
        # for details here: see compute_weights
        selector = rgpara.prepare_selector(
            response,
            *predictors,
            block_size=block_size_params["prepare_selector"],
            verbose=verbose,
            **params
        )

    print("Calculate X.T @ X...")
    tpX = get_XT_X(response,
                   *predictors,
                   selector=selector,
                   include_intercept=include_intercept,
                   verbose=verbose,
                   view_size=block_size_params["get_XT_X"],
                   **params)

    print("Check linear dependency...")
    rank_def_cols = check_rank_deficiency(tpX)
    return {predictors[k]: v for k, v in rank_def_cols.items()}


def compute_weights(response: str | Band,
                    predictors: Collection[Band],
                    block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                    include_intercept: bool = True,
                    as_dtype=np.float64,
                    limit_contribution: float = 0.0,
                    no_data: Union[int, float] = 0.0,
                    sanitize_predictors: bool = False,
                    return_linear_dependent_predictors: bool = False,
                    verbose: bool = False,
                    **params
                    ) -> dict[Band, float] | dict[Band, str] | None:
    # is_needed
    # needs_work (see TODO's; doc)
    # is_tested
    """Compute the optimal weight in a multiple linear regression

    Parameters
    ----------
    response:
      Path to a tif file that contains the response data. The file is only used
      to get the dimensions.
    *predictors:
      A collection with arbitrary many predictors to use.
      See inference.prepare_predictors for further details on how to specify
      predictors.
    block_size:
        Block sizes for specific functions or a default block size for all functions.
        If a dictionary is provided, it should map function names to block sizes
        ('prepare_selector': tuple, 'get_XT_X': tuple, 'get_optimal_betas': tuple).
        If a single tuple is provided, it will be used for all functions.
    include_intercept:
        Whether to fit the intercept when computing weights
    as_dtype:
       ...
    verbose:
        Print out processing step infos
    limit_contribution:
        The fraction of cells, among all valid cells, each predictor must
        contribute for it to be considered a valid predictors.
        By default (i.e. `limit_contribution=0.0`) a single value is enough.
    no_data:
        Each cell with this value is considered to be invalid.
        Most likely you will never have to change this!
    sanitize_predictors:
        Determines if predictors that end up
        contributing not a single data-point should be removed automatically.
        By default this values is set to `False` which raises an exception
        if a predictor ends up contributing nothing.
    return_linear_dependent_predictors:
        Determines if predictors that end up being linearly dependent from each other,
        should be returned as a dictionary ({Band: "type of dependency"}
        This will stop the fitting process - therefore no fit is performed.
        If this parameter is not set, the funciton return None

    **params:
        Optional arguments:
        - `extra_masking_band` (NDArray|None): An additional Band object to be used
          directly as a mask (i.e. all cells of value 0 are masked).
        - `nbr_cpus` (int): how many CPUs should be used (by default the number
          of available CPUs minus one will be used.
        - `start_method` (str): Determines how the workers should start a
          process. Accepted are 'spawn', 'fork' or 'forkserver'.

    """

    # if block sizes are provided as dictionary - some pre-check on input is desired - else
    block_size_params = dict(prepare_selector=None, get_XT_X=None, get_optimal_betas=None)
    if isinstance(block_size, tuple):
        for key in block_size_params:
            block_size_params[key] = block_size
    if isinstance(block_size, dict):
        if block_size_params.keys() != block_size.keys() or not all(isinstance(v, tuple) for v in block_size.values()):
            raise ValueError(f"Block size dict does not conform with all necessary keys and value-types: {block_size=}")
        block_size_params.update(block_size)

    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)

    print("Creating selector...")
    extra_masking_band = params.pop("extra_masking_band", None)
    selector = rgpara.prepare_selector(
        response,
        *predictors,
        block_size=block_size_params["prepare_selector"],
        verbose=verbose,
        extra_masking_band=extra_masking_band,
        **params
    )

    # If selector is empty (meaning all is FALSE) - no need to proceed
    _vals = np.unique(selector)
    if len(_vals) == 1:
        if _vals[0] == False:  # if not _vals[0] is less explicit
            # TODO: remove None as output and raise error or warning
            print(f"WARNING: no pixels to fit, selector masks all pixels")
            return None

    print("Check consistency of remaining predictor data...")
    nbr_predictors = len(predictors)
    predictors = lph._check_predictor_consistency(predictors,
                                              selector=selector,
                                              tolerance=limit_contribution,
                                              sanitize=sanitize_predictors,
                                              no_data=no_data,
                                              verbose=verbose,
                                              **params)
    # TODO: I think this is irrelevant
    #  (tested if selector above & below are equal - they are if no Source has been removed (think of mask)
    if len(predictors) != nbr_predictors:
        # TODO: implement a check for
        #   (a) all predictors use source as mask_reader
        #   (b) len(sources for predictors) is same as before
        #   --> then we can skip this recalculation
        # the consistency check removed some predictors
        # we re-create the selector in this case since the dropped out
        # predictor(s) might have masked some cells
        selector = rgpara.prepare_selector(
            response,
            *predictors,
            block_size=block_size_params["prepare_selector"],
            verbose=verbose,
            extra_masking_band=extra_masking_band,
            **params
        )
        # NOTE: We do not need to _check_predictor_consistency again sine
        #       removing the predictor leads to at least the same valid
        #       pixels, if not more.

    print("Calculate X.T @ X...")
    tpX = get_XT_X(response,
                   *predictors,
                   selector=selector,
                   include_intercept=include_intercept,
                   verbose=verbose,
                   view_size=block_size_params["get_XT_X"],
                   **params)

    print("Check linear dependency...")
    # Check rank deficiency of matrix
    _check_tpX = tpX.copy()
    if include_intercept:
        # if intercept fitted
        # we don't want to check it for lin dependency (its the last column always - see XT X)
        _check_tpX = _check_tpX[:, :-1]
    rank_def = check_rank_deficiency(_check_tpX)

    if rank_def:
        linear_dependent_predictors = {predictors[k]: v for k, v in rank_def.items()}
        if return_linear_dependent_predictors:
            print(f"WARNING: Rank deficiency detected returning affected predictors")
            return linear_dependent_predictors
        else:
            # TODO: remove None as output and raise error or warning
            print(f"WARNING: matrix not invertible - Rank deficiency detected",
                  f"{linear_dependent_predictors=}")
            return None

    print("Inverting X.T @ X...")
    Y = np.linalg.inv(tpX)
    # print(f"{tpX=}\n{Y=}")
    # print("#####\n#####\n#####")

    print("Calculate Y @ X.T @ y (optimal weights)...")
    betas_dict = get_optimal_betas(*predictors,
                                   Y=Y,
                                   response=response,
                                   selector=selector,
                                   include_intercept=include_intercept,
                                   verbose=verbose,
                                   as_dtype=as_dtype,
                                   view_size=block_size_params["get_optimal_betas"],
                                   **params)
    return betas_dict


def block_ssr(params: dict, ssr_parts: list):
    # is_needed (internally only)
    # needs_work (docs; create a test)
    # not_tested
    """Partialy calculate the Sum of Squares for the Residuals (SSR)
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


def block_sst(params: dict, sst_parts: list):
    # is_needed (internally only)
    # needs_work (docs; create a test; see TODO's)
    # not_tested
    """Partialy calculate the Sum of Squares Total (SST)
    """

    #TODO: maybe reduce redundancy between this function an the one above (ssr)

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


def calculate_rmse(response: str | Band,
                   model: str | Band,
                   selector: NDArray[np.bool_],
                   block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                   verbose: bool = False,
                   **params):
    # is_needed
    # needs_work (see TODO's; docs)
    # is_tested
    """Compute the Root mean square error (RSME) based on a predicted model and original response data.
     The formula for RMSE is:

        RMSE = sqrt(Σ((prediction_i - actual_i)²) / n)

    Note: Prepare masks accordingly for model and response file, a selector will be calculated based on those.
       Parameters
    ----------
    response:
        Band object or path to tif file of response data used for computing optimal weights.
    model:
        Band object or path to tif file of model prediction data derived from optimal weights.
    selector:
        Boolean numpy array to mask response and model by. This is key to only use areas of interest where goodness of
        fit wants to be estimated. The user is responsible for choosing such accordingly.
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    verbose:
        Trigger verbose output.
    **params:
        Optional arguments:
        - `nbr_cpus` (int): how many CPUs should be used (by default the number
          of available CPUs minus one will be used.
        - `start_method` (str): Determines how the workers should start a
          process. Accepted are 'spawn', 'fork' or 'forkserver'.
    """

    if not isinstance(response, Band):
        response = Band(source=Source(path=response), bidx=1)
    if not isinstance(model, Band):
        model = Band(source=Source(path=model), bidx=1)

    # Check compatibility
    response.source.check_compatibility(model.source)

    src_profile = response.source.import_profile()
    width = int(src_profile.get('width'))
    height = int(src_profile.get('height'))

    # Parameter for the individual jobs
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))
    block_params = []
    for view in inner_views:
        bparams = dict(view=view,
                       response=response,
                       selector=selector,
                       model=model, )
        block_params.append(bparams)

    manager = Manager()
    ssr_parts = manager.list()
    start_method = params.get('start_method', None)

    # get number of workers
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:

        # start the block calculation processing
        all_jobs = []
        for pparams in block_params:
            all_jobs.append(pool.apply_async(block_ssr,
                                             (pparams, ssr_parts)))
        # collect results
        job_timers = []
        for job in all_jobs:
            job_timers.append(job.get())
        pool.close()
        pool.join()

    # Aggregate results
    # TODO: this can be calcualted nicer (maybe use sth else than
    total_ssr = sum(res[0] for res in ssr_parts)
    total_n = sum(res[1] for res in ssr_parts)
    rmse = np.sqrt(total_ssr / total_n)
    return rmse


def calculate_r2(response: str | Band,
                 model: str | Band,
                 selector: NDArray[np.bool_],
                 block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                 verbose: bool = False,
                 **params):
    # not_needed (but should be useful)
    # needs_work (see TODO's; docs)
    # is_tested
    """Compute the Coefficient of Determination (R2) based on a predicted model and original response data.
    The formula for R2 is:
        R^2 = 1 - (SS_res / SS_tot)
    Where:
        - SS_res is the sum of squares of residuals:
            SS_res = Σ(y_i - f_i)^2
            where y_i are the observed values, and f_i are the predicted values.
        - SS_tot is the total sum of squares:
            SS_tot = Σ(y_i - ȳ)^2
            where ȳ is the mean of the observed values.

    Note: Prepare masks accordingly for model and response file, a selector will be calculated based on those.
    ----------
    response:
        Band object or path to tif file of response data used for computing optimal weights.
    model:
        Band object or path to tif file of model prediction data derived from optimal weights.
    selector:
        Boolean numpy array to mask response and model by. This is key to only use areas of interest where goodness of
        fit wants to be estimated. The user is responsible for choosing such accordingly.
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    verbose:
        Trigger verbose output.
    **params:
        Optional arguments:
        - `nbr_cpus` (int): how many CPUs should be used (by default the number
          of available CPUs minus one will be used.
        - `start_method` (str): Determines how the workers should start a
          process. Accepted are 'spawn', 'fork' or 'forkserver'.
    """

    if not isinstance(response, Band):
        response = Band(source=Source(path=response), bidx=1)
    if not isinstance(model, Band):
        model = Band(source=Source(path=model), bidx=1)

    # Check compatibility
    response.source.check_compatibility(model.source)

    src_profile = response.source.import_profile()
    width = int(src_profile.get('width'))
    height = int(src_profile.get('height'))

    # Parameter for the individual jobs
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))

    # Calculate overall mean of response (needed for SST)
    # TODO: this has not been parallelized (but we can as well of course)
    total_sum = 0
    total_n = 0
    for view in inner_views:
        window = view_to_window(view)
        response_data = response.get_data(window=window)
        _selector = selector[window.toslices()]
        response_data[~_selector] = np.nan
        total_sum += np.nansum(response_data)
        total_n += np.count_nonzero(~np.isnan(response_data))
    y_mean = total_sum / total_n

    # Block parameters
    block_params = []
    for view in inner_views:
        bparams = dict(view=view,
                       response=response,
                       model=model,
                       selector=selector,
                       y_mean=y_mean)
        block_params.append(bparams)

    manager = Manager()
    ssr_parts = manager.list()
    sst_parts = manager.list()
    start_method = params.get('start_method', None)

    # get number of workers
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:

        # start the block calculation processing
        all_jobs = []
        for pparams in block_params:
            all_jobs.append(pool.apply_async(block_ssr,
                                             (pparams, ssr_parts)))
            all_jobs.append(pool.apply_async(block_sst,
                                             (pparams, sst_parts)))
        # collect results
        job_timers = []
        for job in all_jobs:
            job_timers.append(job.get())
        pool.close()
        pool.join()

    # Aggregate results
    total_ssr = sum(res[0] for res in ssr_parts)
    total_sst = sum(res[0] for res in sst_parts)
    r2 = 1 - (total_ssr / total_sst)
    return r2
