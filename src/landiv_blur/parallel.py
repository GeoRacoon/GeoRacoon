from __future__ import annotations
"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
from typing import Any
from collections.abc import Callable

from copy import copy

import numpy as np
import rasterio as rio

from multiprocessing import Pool, Queue, Manager, cpu_count
from numpy.typing import NDArray

from .helper import (view_to_window,)
from .timing import TimedTask
from .plotting import plot_entropy
from .processing import view_blurred, view_entropy
from .prepare import create_views
from .inference import (
        transposed_product,
        get_optimal_weights_source,
    )
from .io import set_tags

# TODO: this needs adaptation once !36 is merged (using tags instead of indexes)
def combine_blurred_categories(output_params: dict, blur_q:Queue):
    """Listen to queue (blur_q) and write blurred blocks to a single file
    """
    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        output_file = output_params.pop('output_file')
        print(f"{output_file=}")
        print(f"{output_dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = output_dtype
        profile['count'] = output_params.get('count', profile['count'])
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                output = blur_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                categories_data = output.pop('data')
                inner_view = copy(output.pop('view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = view_to_window(inner_view)
                for idx, (band, data) in enumerate(categories_data.items(),
                                                   start=1):
                    dst.write(data, window=w, indexes=idx)
                    dst.set_band_description(idx, f'LC_{band}')
                    set_tags(dst, bidx=idx, category=band)
                print(f"Wrote out bands for blurred block {inner_view=}")
                timer.new_lab()
    return timer


def combine_matrices(output_q:Queue):
    """Adding up matrices that hold partial sums
    """
    out_matrix = None
    with TimedTask() as timer:
        while True:
            output = output_q.get()
            signal = output.get('signal', None)
            if signal:
                if signal == "kill":
                    print(f"\n\nDone with the matrix aggregation.\n\n")
                    break
            partial = output.pop('X')
            if out_matrix is not None:
                out_matrix += partial
            else:
                out_matrix = partial
            timer.new_lab()
    return out_matrix, (timer,) 


def partial_transposed_product(params:dict, output_q:Queue):
    """Run .inference.transposed_product in parallel
    """
    def _wrap(tpX):
        return dict(X=tpX)
    runner_call(
        output_q,
        transposed_product,
        params,
        wrapper=_wrap
        )

def partial_optimal_betas(params:dict, output_q:Queue):
    """Runs .inference.get_optimal_weights_source in parallel
    """
    def _wrap(betas):
        return dict(X=betas)
    runner_call(
        output_q,
        get_optimal_weights_source,
        params,
        wrapper=_wrap
    )


# TODO: make single function for this and combine_blurred_categories
def combine_entropy_blocks(output_params: dict, entropy_q:Queue):
    """Listen to queue (entropy_q) and write computed block to single file
    """
    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        output_file = output_params.pop('output_file')
        print(f"{output_file=}")
        print(f"{output_dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = output_dtype
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                # load the entropy_q
                output = entropy_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                data = output.pop('data')
                inner_view = copy(output.pop('view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = view_to_window(inner_view)
                dst.write(data,
                          window=w, indexes=1)
                dst.set_band_description(1, f'Entropy')
                set_tags(dst, bidx=1, category="entropy")
                # lbio.export_to_tif(destination=output_file, data=data,
                #                    start=start, orig_profile=profile)
                # delete partial block tif
                print(f"Wrote out entropy block {inner_view=}")
                timer.new_lab()
    plot_entropy(output_file, start=(0, 0),
                 size=(profile['width'], profile['height']),
                 output=f"{output_file}.preview.pdf")
    return timer


def runner_call(queue: Queue[Any],
                callback:Callable,
                params:dict,
                wrapper:Callable|None=None):
    """Put the results of callback using parameter into the queue

    If provided `wrapper(callback(**params))` is put into the queue.

    """
    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output


def block_heterogeneity(params:dict, entropy_q:Queue, blur_q:Queue)->TimedTask:
    """Per block (i.e. view) heterogeneity measure based on entropy

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      view: tuple
        (x, y, width, height) defining the outer border of the view or block
        to process
      inner_view: tuple
        (x, y, width, height) defining the usable part of the block, i.e.
        without the borders
      blur_as_int: bool
        If the blurred category arrays should be converted to `np.uint8` before
        computing the entropy.
      img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
      filter_params:
        Parameter to pass to the filter callable, `img_filter`
      
      Optionally the following parameters can be set:

      entropy_as_ubyte: bool, Default=False
        Should the entropy be normalized and returned as ubyte?

    entropy_q: multiprocessing.Queue
      The queue to push the entropy maps through
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    with TimedTask() as timer:
        # this is only needed for the entropy part below
        view = params.get('view')
        blur_as_int = params.pop('blur_as_int')
        blur_params = dict(
            view=view,
            inner_view=params.get('inner_view'),
            img_filter=params.get('img_filter'),
            filter_params=params.get('filter_params'),
            output_dtype=np.uint8 if blur_as_int else np.float64,
        )
        blurred_view = runner_call(
            blur_q,
            view_blurred,
            blur_params
        )
        blured_data = blurred_view['data']
        view = blurred_view['view']
        entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
        entropy_params = dict(
            category_arrays=blured_data,
            view=view,
            output_dtype=np.uint8 if entropy_as_ubyte else None,
        )
        # This would return the entropy data
        _ = runner_call(
            entropy_q,
            view_entropy,
            entropy_params
        )
    return timer


# TODO: mpc_params should become first class parameter if mandatory
def get_XT_X(response: str,
             *predictors: tuple[str,
                                int,
                                tuple[int, ...] | None,
                                bool | None],
             selector:NDArray,
             include_intercept:bool=True,
             verbose:bool=False,
             **mpc_params
             )->np.ndarray:
    """Calculate X.T @ X in parallel directly from view of the predictor data

    ..Note::
      `response` is only used to get the correct dimension of the data

    Parameters
    ----------
    response:
      Path to a tif file that contains the reponse data. The file is only used
      to get the dimensions.
    *predictors:
      A collection with arbitrary many predictors to use.
      See inference.prepare_predictors for further details on how to specify
      predicotrs.
    selector:
        a `np.bool_` array to select usable cells in a numpy 2D array
    include_intercept: _optional_
      Determine if the predictor matrix should also contain an extra column of
      1's at the end, which is needed if also the intercepts should be fitted.
    verbose: _optional_
      If the method should print runtime info
    **mpc_arams:
      Parametrization of the multiporcessing approach.
      Needed are:

      view_size:
        The size (width, height) in pixels of a single view (excluding borders)
      
      Optional:

      nbr_cpus: int
        The number of cpu's to use. If not set then then the available number
        of threads -1 are used.

    """
    view_size = mpc_params.get('view_size')
    nbr_cpus = mpc_params.get('nbr_cpus', cpu_count() - 1)
    with rio.open(response, 'r') as src:
        src_width = src.width
        src_height = src.height
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
        print(f"using {nbr_cpus=}")
    pool = Pool(nbr_cpus)
    # start the aggregation step
    matrix_aggregator = pool.apply_async(
        combine_matrices,
        (output_q, )
    )
    all_jobs = []
    for pparams in part_params:
        all_jobs.append(pool.apply_async(
            partial_transposed_product,
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


def get_optimal_betas(*predictors: tuple[str,
                                         int,
                                         tuple[int, ...] | None,
                                         bool | None],
                      Y: np.ndarray,
                      response: str,
                      selector,
                      include_intercept=True,
                      verbose: bool = False,
                      as_dtype=np.float64,
                      **mpc_params
                      ):
    """
    """
    view_size = mpc_params.get('view_size')
    nbr_cpus = mpc_params.get('nbr_cpus', cpu_count() - 1)
    with rio.open(response, 'r') as src:
        src_width = src.width
        src_height = src.height
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
        print(f"using {nbr_cpus=}")
    pool = Pool(nbr_cpus)
    # start the aggregation step
    matrix_aggregator = pool.apply_async(
        combine_matrices,
        (output_q, )
    )
    all_jobs = []
    for pparams in part_params:
        all_jobs.append(pool.apply_async(
            partial_optimal_betas,
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
    return betas 
