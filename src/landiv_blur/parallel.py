"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
from __future__ import annotations

from copy import copy

import numpy as np
import rasterio as rio
import multiprocessing as mproc

from .helper import (view_to_window,)
from .timing import TimedTask
from .plotting import plot_entropy
from .processing import view_blurred, view_entropy
from .prepare import create_views
from .inference import (
        prepare_selector,
        transposed_product,
        get_optimal_weights_source,
    )



def combine_blurred_land_cover_types(output_params: dict, blur_q):
    """Listen to queue (blur_q) and write blurred blocks to a single file
    """
    with TimedTask() as timer:
        as_int = output_params.pop('as_int', False)
        output_file = output_params.pop('output_file')
        print(f"{output_file=}")
        print(f"{as_int=}")
        profile = output_params.pop('profile')
        profile['dtype'] = rio.float64
        if as_int:
            profile['dtype'] = rio.uint8
        profile['count'] = output_params.get('count', profile['count'])
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                output = blur_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                layer_data = output.pop('data')
                inner_view = copy(output.pop('view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = view_to_window(inner_view)
                for band, data in layer_data.items():
                    dst.write(data,
                              window=w, indexes=band+1)
                print(f"Wrote out bands for blurred block {inner_view=}")
                timer.new_lab()
    return timer


def combine_matrices(output_q):
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


def partial_transposed_product(params, output_q):
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

def partial_optimal_betas(params, output_q):
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


def combine_entropy_blocks(output_params: dict, entropy_q):
    """Listen to queue (entropy_q) and write computed block to single file
    """
    with TimedTask() as timer:
        output_file = output_params.pop('output_file')
        print(f"{output_file=}")
        as_ubyte = output_params.pop('as_ubyte')
        profile = output_params.pop('profile')
        profile['dtype'] = rio.float64
        if as_ubyte:
            profile['dtype'] = rio.uint8
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
                # lbio.export_to_tif(destination=output_file, data=data,
                #                    start=start, orig_profile=profile)
                # delete partial block tif
                print(f"Wrote out entropy block {inner_view=}")
                timer.new_lab()
    plot_entropy(output_file, start=(0, 0),
                 size=(profile['width'], profile['height']),
                 output=f"{output_file}.preview.pdf")
    return timer

def runner_call(queue, callback, params, wrapper=None):
    """Put the results of callback using parameter into the queue

    If provided `wrapper(callback(**params))` is put into the queue.

    """
    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output

def block_heterogeneity(params, entropy_q, blur_q):
    """Block entropy-based landscape type heterogeneity measure

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for the single worker
    entropy_q: multiprocessing.Queue
      The queue to push the entropy maps through
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    with TimedTask() as timer:
        # this is only needed for the entropy part below
        view = params.get('view')
        entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
        blurred_view = runner_call(
            blur_q,
            view_blurred,
            params
        )
        blur_layers = blurred_view['data']
        view = blurred_view['view']
        entropy_layer = runner_call(
            entropy_q,
            view_entropy,
            dict(
                blur_layers=blur_layers,
                view=view,
                entropy_as_ubyte=entropy_as_ubyte
            )
        )
    return timer


def get_XT_X(response: str,
             *predictors: tuple[str,
                                int,
                                tuple[int, ...] | None,
                                bool | None],
             selector,
             include_intercept=True,
             verbose: bool = False,
             **mpc_params
             )->np.ndarray:
    """Calculate X.T @ X in parallel directly from view of the predictor data

    ..Note::
      The response is only used to generate a mask for unused/unusable pixels.
    """
    view_size = mpc_params.get('view_size')
    nbr_cpus = mpc_params.get('nbr_cpus', mproc.cpu_count() - 1)
    with rio.open(response, 'r') as src:
        src_widht = src.width
        src_height = src.height
    # create a list of views and put it into runner_params
    size = (src_widht, src_height)
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
    manager = mproc.Manager()
    output_q = manager.Queue()
    if verbose:
        print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
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
    nbr_cpus = mpc_params.get('nbr_cpus', mproc.cpu_count() - 1)
    with rio.open(response, 'r') as src:
        src_widht = src.width
        src_height = src.height
    size = (src_widht, src_height)
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
    manager = mproc.Manager()
    output_q = manager.Queue()
    if verbose:
        print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
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
