"""
This module contains functions to parallellize various inference methods.
"""
from __future__ import annotations

from collections.abc import Collection
import warnings
from typing import Union

import numpy as np

from multiprocessing import Manager
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


def get_XT_X(response: str | Band,
             *predictors: Band | str,
             selector: NDArray,
             include_intercept: bool = True,
             verbose: bool = False,
             **mpc_params
             ) -> np.ndarray:
    """
    Calculate X.T @ X matrix in parallel from predictor data blocks.

    This function computes the transpose-product matrix (X.T @ X) used in
    linear regression by processing the predictor data in parallel blocks.
    The response parameter is only used to determine the spatial dimensions
    of the computation.

    Parameters
    ----------
    response : str or Band
        Path to a tif file or Band object containing the response data.
        Only the spatial dimensions (width, height) are used; the actual
        response values are not required for this computation.
    *predictors : Band or str
        Variable number of predictor bands to include in the design matrix X.
        Can be Band objects or paths to source files.
        See `inference.prepare_predictors` for details on predictor
        specification.
    selector : ndarray
        Boolean array (np.bool_) to select usable cells. Must have the same
        spatial dimensions as the response. True values indicate pixels to
        include in the computation.
    include_intercept : bool, optional
        If True, append a column of ones to the design matrix X to fit an
        intercept term. Default is True.
    verbose : bool, optional
        If True, print runtime information including number of workers used.
        Default is False.
    **mpc_params
        Multiprocessing configuration parameters.

        Required:

        - view_size : tuple of int
            Size (width, height) in pixels of a single view/block to process.

        Optional:

        - nbrcpu : int, optional
            Number of CPUs to use. If not set, uses (available threads - 1).
        - start_method : str, optional
            Starting method for multiprocessing ('fork', 'spawn', or
            'forkserver').

    Returns
    -------
    XT_X : ndarray
        The transpose-product matrix (X.T @ X) of shape (n_predictors, n_predictors).
        If `include_intercept=True`, the shape is (n_predictors+1, n_predictors+1)
        with the intercept column included as the last row and column.

    Notes
    -----
    The function implements parallel computation by:

    1. Dividing the spatial domain into non-overlapping blocks (views)
    2. Computing partial X.T @ X matrices for each block in parallel
    3. Aggregating the partial results into the final matrix

    This approach is memory-efficient for large spatial datasets as it avoids
    loading the entire design matrix into memory at once.

    The selector mask is applied consistently across all blocks to ensure
    only valid pixels contribute to the computation.
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
                      ) -> dict[Band | str, float]:
    """
    Calculate optimal regression coefficients (betas) in parallel from spatial data.

    This function computes the optimal weights (beta coefficients) for a
    multiple linear regression by processing predictor data in parallel blocks.
    The computation solves for beta in the normal equation: beta = (X.T @ X)^-1 @ X.T @ y.

    Parameters
    ----------
    *predictors : Band or str
       Variable number of predictor bands to include in the regression.
       Can be Band objects or paths to source files.
       See `inference.prepare_predictors` for details on predictor
       specification.
    Y : ndarray
       The pre-computed X.T @ y vector, where y is the response vector.
       This should be a 1D array with length equal to the number of predictors
       (or number of predictors + 1 if `include_intercept=True`).
    response : str or Band
       Path to a tif file or Band object containing the response data.
       Used to determine spatial dimensions for block processing.
    selector : ndarray
       Boolean array (np.bool_) to select usable cells. Must have the same
       spatial dimensions as the response. True values indicate pixels to
       include in the regression.
    include_intercept : bool, optional
       If True, fit an intercept term by appending a column of ones to the
       design matrix X. The intercept will be included in the returned
       dictionary with key 'intercept'. Default is True.
    verbose : bool, optional
       If True, print runtime information including number of workers,
       predictors, and computed beta values. Default is False.
    as_dtype : dtype, optional
       Data type to use for internal computations. Default is np.float64.
    **mpc_params
       Multiprocessing configuration parameters.

       Required:

       - view_size : tuple of int
           Size (width, height) in pixels of a single view/block to process.

       Optional:

       - nbrcpu : int, optional
           Number of CPUs to use. If not set, uses (available threads - 1).
       - start_method : str, optional
           Starting method for multiprocessing ('fork', 'spawn', or
           'forkserver').

    Returns
    -------
    optimal_weights : dict of {Band or str: float}
       Dictionary mapping each predictor to its optimal regression coefficient.
       If `include_intercept=True`, includes an additional entry with key
       'intercept' for the intercept term (beta_0).

    Raises
    ------
    ValueError
       If the number of computed beta values does not match the number of
       predictors (plus intercept if applicable).


    Notes
    -----
    The function implements parallel computation by:

    1. Dividing the spatial domain into non-overlapping blocks (views)
    2. Computing partial contributions for each block in parallel
    3. Aggregating the partial results to obtain final beta coefficients

    The optimal weights solve the ordinary least squares problem:

    .. math::
       \\beta = (X^T X)^{-1} X^T y

    where X is the design matrix of predictors and y is the response vector.

    This approach is memory-efficient for large spatial datasets as it avoids
    loading the entire design matrix into memory at once.

    Examples
    --------
    >>> # Compute optimal betas with intercept
    >>> selector = np.ones((1000, 1000), dtype=bool)
    >>> Y = np.array([10.5, 20.3, 15.7, 5.2])  # Pre-computed X.T @ y
    >>> weights = get_optimal_betas(
    ...     band1, band2, band3,
    ...     Y=Y,
    ...     response='response.tif',
    ...     selector=selector,
    ...     include_intercept=True,
    ...     view_size=(512, 512),
    ...     nbrcpu=4
    ... )
    >>> weights
    {<Band: band1>: 0.523, <Band: band2>: 1.245, <Band: band3>: -0.334, 'intercept': 5.12}

    >>> # Without intercept
    >>> Y = np.array([10.5, 20.3, 15.7])
    >>> weights = get_optimal_betas(
    ...     band1, band2, band3,
    ...     Y=Y,
    ...     response='response.tif',
    ...     selector=selector,
    ...     include_intercept=False,
    ...     view_size=(512, 512)
    ... )
    >>> weights
    {<Band: band1>: 0.523, <Band: band2>: 1.245, <Band: band3>: -0.334}
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
    """Test predictors for linear dependency before fitting multiple linear regression.

    This function checks whether predictor columns are linearly dependent by
    computing the X.T @ X matrix and analyzing its rank. Linear dependencies
    can cause numerical instability or singularity in regression fitting and
    should be resolved before proceeding with model estimation.

    Parameters
    ----------
    response : str or Band
        Path to a tif file or Band object containing the response data.
        Used to determine spatial dimensions and create the selector mask.
    predictors : Collection of Band
        Collection of predictor bands to test for linear dependency.
        See `inference.prepare_predictors` for details on predictor
        specification.
    block_size : tuple of int or dict of {str: tuple of int}
        Block sizes for parallel processing functions.

        - If tuple: (width, height) in pixels, applied to all processing steps.
        - If dict: Maps function names to specific block sizes with keys:
          'prepare_selector', 'get_XT_X'. Each value should be a tuple
          (width, height).
    include_intercept : bool, optional
        If True, include an intercept term when computing X.T @ X.
        Default is True.
    limit_contribution : float, optional
        Minimum fraction of valid cells each predictor must contribute to
        be considered valid. Value between 0.0 and 1.0.
        Default is 0.0 (a single valid value is sufficient).
    no_data : int or float, optional
        Value representing invalid or missing data. Cells with this value
        are excluded from analysis. Default is 0.0.
    sanitize_predictors : bool, optional
        If True, automatically remove predictors that fail the contribution
        threshold. If False, raise an exception when invalid predictors are
        found. Default is False.
    verbose : bool, optional
        If True, print processing step information. Default is False.
    **params
        Optional multiprocessing arguments:

        - nbrcpu : int, optional
            Number of CPUs to use. If not set, uses (available CPUs - 1).
        - start_method : str, optional
            Process start method: 'spawn', 'fork', or 'forkserver'.

    Returns
    -------
    dependencies : dict of {Band: str} or None
        Dictionary mapping each rank-deficient predictor to a description
        of its linear dependency. Returns None if the selector masks all
        pixels (no data to fit).

        If no linear dependencies are found, returns an empty dictionary.
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
    """Compute optimal regression coefficients for multiple linear regression.

    This function fits a multiple linear regression model by computing the
    optimal weights (beta coefficients) for each predictor. The computation
    involves creating a selector mask, validating predictor consistency,
    calculating the X.T @ X matrix, checking for linear dependencies, and
    solving the normal equations.

    Parameters
    ----------
    response : str or Band
        Path to a tif file or Band object containing the response data.
        Used to determine spatial dimensions and create the selector mask.
    predictors : Collection of Band
        Collection of predictor bands to use in the regression.
        See `inference.prepare_predictors` for details on predictor
        specification.
    block_size : tuple of int or dict of {str: tuple of int}
        Block sizes for parallel processing functions.

        - If tuple: (width, height) in pixels, applied to all processing steps.
        - If dict: Maps function names to specific block sizes with keys:
          'prepare_selector', 'get_XT_X', 'get_optimal_betas'. Each value
          should be a tuple (width, height).
    include_intercept : bool, optional
        If True, fit an intercept term in the regression model.
        Default is True.
    as_dtype : dtype, optional
        Data type for internal computations and output weights.
        Default is np.float64.
    limit_contribution : float, optional
        Minimum fraction of valid cells each predictor must contribute to
        be considered valid. Value between 0.0 and 1.0.
        Default is 0.0 (a single valid value is sufficient).
    no_data : int or float, optional
        Value representing invalid or missing data. Cells with this value
        are excluded from analysis. Default is 0.0.
    sanitize_predictors : bool, optional
        If True, automatically remove predictors that fail the contribution
        threshold. If False, raise an exception when invalid predictors are
        found. Default is False.
    return_linear_dependent_predictors : bool, optional
        If True and linear dependencies are detected, return a dictionary
        mapping linearly dependent predictors to their dependency type instead
        of computing weights. This stops the fitting process.
        If False and dependencies are detected, raise an error.
        Default is False.
    verbose : bool, optional
        If True, print processing step information. Default is False.
    **params
        Optional arguments:

        - extra_masking_band : Band or None, optional
            Additional Band object to use directly as a mask. All cells with
            value 0 are masked.
        - nbrcpu : int, optional
            Number of CPUs to use. If not set, uses (available CPUs - 1).
        - start_method : str, optional
            Process start method: 'spawn', 'fork', or 'forkserver'.

    Returns
    -------
    optimal_weights : dict of {Band or str: float} or None
        Dictionary mapping each predictor (and 'intercept' if
        `include_intercept=True`) to its optimal regression coefficient.
        Returns None if the selector masks all pixels or if linear
        dependencies are detected and `return_linear_dependent_predictors=False`.
    linear_dependencies : dict of {Band: str}
        Only returned if `return_linear_dependent_predictors=True` and
        linear dependencies are detected. Maps each linearly dependent
        predictor to a description of its dependency type.

    Raises
    ------
    ValueError
        If `block_size` is a dictionary but doesn't contain the required keys
        or has invalid value types.
    InvalidPredictorError
        If `sanitize_predictors=False` and predictors fail the contribution
        threshold.
    LinAlgError
        If the X.T @ X matrix is singular and cannot be inverted (when
        `return_linear_dependent_predictors=False`).

    Warnings
    --------
    UserWarning
        Issued when the selector masks all pixels (no data to fit).
    UserWarning
        Issued when linear dependencies are detected.

    Notes
    -----
    The function performs the following workflow:

    1. Creates a selector mask identifying valid pixels
    2. Validates predictor consistency and removes invalid predictors if
       `sanitize_predictors=True`
    3. Recalculates selector if predictors were removed
    4. Computes X.T @ X matrix in parallel
    5. Checks for rank deficiency (linear dependencies)
    6. Inverts X.T @ X matrix: (X.T @ X)^-1
    7. Computes optimal weights: beta = (X.T @ X)^-1 @ X.T @ y

    The optimal weights solve the ordinary least squares problem:

    .. math::
        \\beta = (X^T X)^{-1} X^T y

    where X is the design matrix of predictors and y is the response vector.

    Linear dependency detection excludes the intercept column from the rank
    check, as the intercept is always included as the last column when
    `include_intercept=True`.

    If predictors are removed during sanitization, the selector is recomputed
    to ensure consistency, as removed predictors may have contributed to
    masking certain pixels.

    Examples
    --------
    >>> # Basic usage with uniform block size
    >>> predictors = [band1, band2, band3]
    >>> weights = compute_weights(
    ...     response='response.tif',
    ...     predictors=predictors,
    ...     block_size=(512, 512),
    ...     include_intercept=True,
    ...     verbose=True
    ... )
    >>> weights
    {<Band: band1>: 0.523, <Band: band2>: 1.245, <Band: band3>: -0.334, 'intercept': 5.12}

    >>> # Use different block sizes for different steps
    >>> block_sizes = {
    ...     'prepare_selector': (1024, 1024),
    ...     'get_XT_X': (512, 512),
    ...     'get_optimal_betas': (256, 256)
    ... }
    >>> weights = compute_weights(
    ...     response='response.tif',
    ...     predictors=predictors,
    ...     block_size=block_sizes,
    ...     limit_contribution=0.05,
    ...     sanitize_predictors=True,
    ...     nbrcpu=4
    ... )

    >>> # Detect and return linear dependencies
    >>> deps = compute_weights(
    ...     response='response.tif',
    ...     predictors=predictors,
    ...     block_size=(512, 512),
    ...     return_linear_dependent_predictors=True
    ... )
    >>> if isinstance(deps, dict) and all(isinstance(v, str) for v in deps.values()):
    ...     print(f"Linear dependencies found: {deps}")
    ... else:
    ...     print(f"Optimal weights: {deps}")
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
            warnings.warn("No pixels to fit - selector masks all pixels", UserWarning)
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
        # NOTE: We do not need to _check_predictor_consistency again since
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
            warnings.warn(f"Matrix not invertible - rank deficiency detected. "
                          f"Linear dependent predictors: {linear_dependent_predictors}", UserWarning)
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


def calculate_rmse(response: str | Band,
                   model: str | Band,
                   selector: NDArray[np.bool_],
                   block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                   verbose: bool = False,
                   **params) -> float:
    """
    Compute the Root Mean Square Error (RMSE) for a predicted model and observed response data.

    RMSE measures the average magnitude of the residuals (differences between
    predicted and observed values) and is defined as:

        RMSE = sqrt(Σ((prediction_i - actual_i)²) / n)

    where:
        - prediction_i are model-predicted values
        - actual_i are observed response values
        - n is the number of valid observations

    The function processes data in blocks for memory efficiency and parallelization.

    Parameters
    ----------
    response : Band or str
        A `Band` object or a path to a raster/tif file containing the observed response data.
    model : Band or str
        A `Band` object or a path to a raster/tif file containing predicted values from the model.
    selector : NDArray[np.bool_]
        Boolean mask specifying which data points should be included in the RMSE calculation.
        Points where `selector` is False are ignored.
    block_size : tuple[int, int] or dict[str, tuple[int, int]]
        Size of the block (width, height) in pixels for processing data chunks.
        If a dictionary, it should contain named blocks.
    verbose : bool, default=False
        If True, prints status information during computation.
    **params : optional
        Additional parameters for parallel processing:
        - `nbr_cpus` (int): Number of CPUs to use (default: all available minus one).
        - `start_method` (str): Multiprocessing start method ('spawn', 'fork', or 'forkserver').

    Returns
    -------
    float
        The RMSE value. Lower values indicate better model fit.

    Examples
    --------
    >>> response_band = Band(source=Source(path="response.tif"), bidx=1)
    >>> model_band = Band(source=Source(path="model.tif"), bidx=1)
    >>> selector_mask = np.ones((height, width), dtype=bool)
    >>> rmse = calculate_rmse(response_band, model_band, selector_mask, block_size=(512, 512))
    >>> print(f"RMSE = {rmse:.3f}")
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
            all_jobs.append(pool.apply_async(lph._block_ssr,
                                             (pparams, ssr_parts)))
        # collect results
        job_timers = []
        for job in all_jobs:
            job_timers.append(job.get())
        pool.close()
        pool.join()

    # Aggregate results
    total_ssr = sum(res[0] for res in ssr_parts)
    total_n = sum(res[1] for res in ssr_parts)
    rmse = np.sqrt(total_ssr / total_n)
    return rmse


def calculate_r2(response: str | Band,
                 model: str | Band,
                 selector: NDArray[np.bool_],
                 block_size: Union[dict[str, tuple[int, int]], tuple[int, int]],
                 verbose: bool = False,
                 **params) -> float:
    """
    Compute the Coefficient of Determination (R²) for a predicted model and observed response data.

    The coefficient of determination, R², quantifies the proportion of variance in the response
    variable that is predictable from the model. It is calculated as:

        R² = 1 - (SS_res / SS_tot)

    where:
        - SS_res is the sum of squared residuals: Σ(y_i - f_i)²
        - SS_tot is the total sum of squares: Σ(y_i - ȳ)²
        - y_i are observed values, f_i are predicted values, and ȳ is the mean of observed values.

    The function processes data in blocks for memory efficiency and parallelization.

    Parameters
    ----------
    response : Band or str
        A `Band` object or a path to a raster/tif file containing the observed response data.
    model : Band or str
        A `Band` object or a path to a raster/tif file containing predicted values from the model.
    selector : NDArray[np.bool_]
        Boolean mask to specify which data points should be included in the R² calculation.
        Points where `selector` is False are ignored. The user must prepare the mask appropriately.
    block_size : tuple[int, int] or dict[str, tuple[int, int]]
        Size of the block (width, height) in pixels for processing data chunks.
        If a dictionary, it should contain named blocks.
    verbose : bool, default=False
        If True, prints status information during computation.
    **params : optional
        Additional parameters for parallel processing:
        - `nbr_cpus` (int): Number of CPUs to use (default: all available minus one).
        - `start_method` (str): Multiprocessing start method ('spawn', 'fork', or 'forkserver').

    Returns
    -------
    float
        The R² coefficient ranging from -∞ to 1, where 1 indicates perfect prediction.

    Examples
    --------
    >>> response_band = Band(source=Source(path="response.tif"), bidx=1)
    >>> model_band = Band(source=Source(path="model.tif"), bidx=1)
    >>> selector_mask = np.ones((height, width), dtype=bool)
    >>> r2 = calculate_r2(response_band, model_band, selector_mask, block_size=(512, 512))
    >>> print(f"R² = {r2:.3f}")
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
            all_jobs.append(pool.apply_async(lph._block_ssr,
                                             (pparams, ssr_parts)))
            all_jobs.append(pool.apply_async(lph._block_sst,
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
