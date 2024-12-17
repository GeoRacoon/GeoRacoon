"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
from __future__ import annotations

import os
import math
from typing import Any, Dict
from collections.abc import Callable, Collection

from copy import copy

from typing import Union

import numpy as np
import rasterio as rio

from multiprocessing import (
    Queue,
    Manager,
    cpu_count,
    get_context)
from numpy.typing import NDArray

from .io_ import Source, Band
from .helper import (view_to_window,
                     output_filename,
                     reduced_mask,
                     aggregated_selector,
                     check_compatibility,
                     check_rank_deficiency, convert_to_dtype)
from .timing import TimedTask
from .plotting import plot_entropy
from .processing import view_blurred, view_entropy, view_filtered, view_interaction
from .prepare import create_views, get_blur_params, update_view
from .filters.gaussian import (img_filter,
                               gaussian,
                               compatible_border_size)
from .inference import (
    transposed_product,
    get_optimal_weights_source)
from .io import set_tags, write_band, compress_tif
from .exceptions import InvalidPredictorError


# NOTE: The first element will be picked by default
MPC_STARTER_METHODS = ['spawn', 'fork', 'forkserver']


def combine_views(output_params: dict,
                  job_out_q: Queue):
    """Listens to a queue and writes provided view into a file
    """

    with TimedTask() as timer:
        output_file = output_params.pop('output_file')
        profile = output_params.pop('profile')
        out_band = output_params.pop('band')
        out_tag = output_params.pop('tags')
        if out_band is None:
            out_band = Band(source=Source(path=output_file),
                            bidx=1,
                            tags=out_tag)
        # create the file
        out_band.init_source(profile=profile)
        # write out the tags
        out_band.export_tags()
        with out_band.data_writer(mode='r+') as write:
            while True:
                # get the next output from a block job
                output = job_out_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {out_band.source.path}\n\n")
                        break
                data = output.pop('data')
                view = copy(output.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                print(f"Wrote out block {view=}")
                timer.new_lab()
    return timer


# TODO: this needs adaptation once !36 is merged (using tags instead of indexes)
def combine_blurred_categories(output_params: dict, blur_q: Queue) -> TimedTask:
    """Listen to queue (blur_q) and write blurred blocks to a single file

    Parameters
    ----------
    output_params:
      dtype:
          Data type to use for the output data
      nodata:
        Value to be used as nodata value (if not provided `None` is used)

    Returns
    -------
    TimedTask:
        Can report the duration of the task
    """
    with TimedTask() as timer:
        dtype = output_params.pop('dtype')
        output_file = output_params.pop('output_file')
        # print(f"{output_file=}")
        # print(f"{dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = dtype
        # overwrite the profile if explicitely provided:
        profile['nodata'] = output_params.pop('nodata', profile.get('nodata', None))
        profile['count'] = output_params.get('count', profile['count'])
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                output = blur_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        # TODO: FINALIZE_TASK:
                        # print(f"\n\nClosing: {output_file}\n\n")
                        break
                categories_data = output.pop('data')
                inner_view = copy(output.pop('view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = view_to_window(inner_view)
                for bidx, (band, data) in enumerate(categories_data.items(),
                                                    start=1):
                    # TODO: UPDATE_TASK
                    # NOTE: downside of this is that we set the tags
                    #       every time, unfortunately, in the FINALIZE_TASK
                    #       we do not have the bidx 
                    write_band(src=dst, bidx=bidx, data=data, window=w,
                               category=band)
                    # NOTE: we might want keep the description unchanged:
                    dst.set_band_description(bidx, f'LC_{band}')
                # print(f"Wrote out bands for blurred block {inner_view=}")
                timer.new_lab()
        # print(f"\n\n########\n\nProfile")

    return timer


def combine_matrices(output_q: Queue) -> tuple[NDArray | None, tuple]:
    """Adding up matrices that hold partial sums

    Parameters
    ----------
    output_q:
        The queue this job listens to.

    Returns
    -------
    matrice, (TimedTask, ):
        The first object is the aggregated matrix, the second holds a
        `TimedTask` object that holds information on the duration of this task
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


def fill_matrix(matrix: NDArray, aggr_q: Queue) -> tuple[NDArray | None, tuple]:
    """Filling up a matrix

    Parameters
    ----------
    matrix:
      ...
    aggr_q:
        The queue this job listens to.
        Each element in the queue must be a `dict` containing either:
        - "view" + "data" specifying where to write what
        - "signal" with value:
          - "kill": will terminate the process and return the filled matrix

    Returns
    -------
    matrice, (TimedTask, ):
        The first object is the filled matrix, the second holds a
        `TimedTask` object that holds information on the duration of this task
    """
    with TimedTask() as timer:
        while True:
            output = aggr_q.get()
            signal = output.get('signal', None)
            if signal:
                if signal == "kill":
                    # print(f"\n\nDone with the matrix aggregation.\n\n")
                    break
            view = output.pop('view')
            block_data = output.pop('data')
            update_view(data=matrix, view=view, block=block_data)
            timer.new_lab()
    return matrix, (timer,)


def partial_transposed_product(params: dict, output_q: Queue):
    """Run `.inference.transposed_product` in parallel

    Parameters
    ----------
    params:
        The keywords arguments passed to `.inference.transposed_product`
    output_q:
        The queue this job listens to.
    """

    def _wrap(tpX):
        return dict(X=tpX)

    runner_call(
        output_q,
        transposed_product,
        params,
        wrapper=_wrap
    )


def partial_optimal_betas(params: dict, output_q: Queue):
    """Runs .inference.get_optimal_weights_source in parallel

    Parameters
    ----------
    params:
        The keywords arguments passed to
        `.inference.get_optimal_weights_source`
    output_q:
        The queue this job listens to.
    """

    def _wrap(beta_dict):
        return dict(X=list(beta_dict.values()))  # ok for python >= 3.6 (dict keeps order)

    runner_call(
        output_q,
        get_optimal_weights_source,
        params,
        wrapper=_wrap
    )


def data_writer(writer: Callable, writer_params: dict, aggr_q: Queue) -> TimedTask:
    """Write out data using the context manager `writer`

    This function can be used with the various context managers defined
    in the `io_.Source` and `io_.Band` classes.

    Parameters
    ----------
    writer:
        A `io_.Source` or `io_.Band` `data_write` (or `mask_writer`)
    writer_params:
        Keyword arguments that will be passed to the `writer` method
    aggr_q:
        The queue this job listens to.

    Returns
    -------
    TimedTask:
        Can report the duration of the task

    """
    with TimedTask() as timer:
        with writer(**writer_params) as write:
            while True:
                # load the entropy_q
                job_out = aggr_q.get()
                signal = job_out.get('signal', None)
                if signal:
                    if signal == "kill":
                        # print(f"\n\nTerminating data writer.\n\n")
                        break
                data = job_out.pop('data')
                view = copy(job_out.pop('view'))
                w = view_to_window(view)
                # 
                write(data, window=w)
                # print(f"Wrote out block {view=}")
                timer.new_lab()
    return timer


def process_band_count_valid(band: Band,
                             selector:NDArray[np.bool_],
                             no_data:Union[int,float],
                             limit_count:int):
    with TimedTask() as timer:
        valid = band.count_valid_pixels(selector=selector,
                                        no_data=no_data,
                                        limit_count=limit_count)
    return {band: valid}, (timer,)


def process_block(task: Callable,
                  source: str | Source,
                  bands: Collection[Band] | None,
                  view: tuple[int, int, int, int],
                  task_params: dict,
                  read_params: dict,
                  open_params: dict,
                  out_q: Queue) -> TimedTask:
    """Processes a section of the data in the source file.

    This is a general purpose function that can be used to process a large .tif
    in a parallelized manner.

    Parameters
    ----------
    task:
        Function that will be called on the data from the specified band.
        The first argument of the function must be `data`, a `numpy.array`
        that holds the data from this section.
    source:
        Either a string or an `io_.Source` object
    bands:
        A collection of strings or `io_.Band` object the specify which bands to use
    view:
      A tuple (x, y, width, height) defining the view of data to extract and
      process
    task_params:
        Keyword arguments that will be passed to the callable `task`
    read_params:
        Keyword arguments that are passed to the open method of the `source` object
    open_params:
        Keyword arguments that are passed to the reader method of the `source` object
    out_q: 
        The queue this job will put the output of the callable `task` into
        

    Returns
    -------
    TimedTask:
        Can report the duration of the task
    """
    with TimedTask() as timer:
        if not isinstance(source, Source):
            source = Source(path=source)
        if bands is None:
            # print('No specific bands selected, using all')
            bands = source.get_bands()
        assert all(band.source == source for band in bands), "Not all bands point to the correct source!"
        window = view_to_window(view)
        with source.data_reader(bands=bands, **open_params) as read:
            data = read(window=window, **read_params)
        _ = runner_call(callback=task,
                        params=dict(array=data, **task_params),
                        queue=out_q,
                        wrapper=lambda x: dict(data=x, view=view))
        # print(f"{view=}\n{data=}\nmask={_}")
    return timer


def process_masks(task: Callable,
                  bands: Collection[Band],
                  view: tuple[int, int, int, int],
                  task_params: dict,
                  read_params: dict,
                  open_params: dict,
                  aggr_q: Queue) -> TimedTask:
    """Processes a section of the mask for each band

    This is a general purpose function that can be used to process a large .tif
    in a parallelized manner.

    Parameters
    ----------
    task:
        Function that will be called on the data from the specified band.
        The first argument of the function must be `data`, a `numpy.array`
        that holds the data from this section.
    bands:
        A collection of strings or `io_.Band` object the specify which bands to use
    view:
      A tuple (x, y, width, height) defining the view of data to extract and
      process
    task_params:
        Keyword arguments that will be passed to the callable `task`
    read_params:
        Keyword arguments that are passed to the open method of the `source` object
    open_params:
        Keyword arguments that are passed to the reader method of the `source` object
    aggr_q: 
        The queue this job will put the output of the callable `task` into
        

    Returns
    -------
    TimedTask:
        Can report the duration of the task
    """
    window = view_to_window(view)
    with TimedTask() as timer:
        masks = []
        for band in bands:
            mask_reader = band.get_mask_reader()
            with mask_reader(**open_params) as read_mask:
                _mask = read_mask(window=window, **read_params)
                masks.append(_mask)
        _ = runner_call(callback=task,
                        params=dict(masks=masks, **task_params),
                        queue=aggr_q,
                        wrapper=lambda x: dict(data=x, view=view))
        # print(f"{view=}\n{data=}\nmask={_}")
    return timer


def combine_entropy_blocks(output_params: dict,
                           entropy_q: Queue):
    """Listen to queue (entropy_q) and write computed block to single file
    """

    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        output_file = output_params.pop('output_file')
        # print(f"{output_file=}")
        # print(f"{output_dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = output_dtype
        out_band = output_params.pop('out_band', None)
        if out_band is None:
            out_band = Band(source=Source(path=output_file),
                            bidx=1,
                            tags=dict(category='entropy'))
        # create the file
        out_band.init_source(profile=profile)
        # write out the tags
        out_band.export_tags()
        # print(f'INIT:\n{out_band.source=}\n{out_band=}')

        # with out_band.source.open(mode='r+', **profile) as dst:
        with out_band.data_writer(mode='r+', **profile) as write:
            while True:
                # load the entropy_q
                output = entropy_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {out_band.source.path}\n\n")
                        break
                data = output.pop('data')
                view = copy(output.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                print(f"Wrote out entropy block {view=}")
                timer.new_lab()
    return timer


def combine_interaction_blocks(output_params: dict,
                               interaction_q: Queue):
    """Listen to queue (interaction_q) and write computed block to single file
    """

    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        output_file = output_params.pop('output_file')
        # print(f"{output_file=}")
        # print(f"{output_dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = output_dtype
        out_band = output_params.pop('out_band', None)
        out_tag = output_params.pop('output_tag', dict(category='interaction'))
        if out_band is None:
            out_band = Band(source=Source(path=output_file),
                            bidx=1,
                            tags=out_tag)
        # create the file
        out_band.init_source(profile=profile)
        # write out the tags
        out_band.export_tags()
        # print(f'INIT:\n{out_band.source=}\n{out_band=}')

        # with out_band.source.open(mode='r+', **profile) as dst:
        with out_band.data_writer(mode='r+', **profile) as write:
            while True:
                # load the interaction_q
                output = interaction_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {out_band.source.path}\n\n")
                        break
                data = output.pop('data')
                view = copy(output.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                print(f"Wrote out interaction block {view=}")
                timer.new_lab()
    return timer


def runner_call(queue: Queue[Any],
                callback: Callable,
                params: dict,
                wrapper: Callable | None = None):
    """Put the results of callback using parameter into the queue

    If provided `wrapper(callback(**params))` is put into the queue.

    """
    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output


def extract_categories(source: str | Source,
                       categories: list,
                       output_file: str,
                       img_filter: None|Callable,
                       filter_params: dict,
                       block_size: tuple[int, int],
                       blur_as_int: bool = True,
                       output_params:None|dict = None,
                       verbose: bool = False,
                       **params):
    """Load per-category maps from resource, apply a filter and export.

    Parameters
    ----------
    source:
      Either the path to, or a `io_.Source` object representing a the `.tif`
      file to load the data from.
    categories:
      The categorical values to separate into single bands
    output_file:
      The path to write the resulting data to
    img_filter:
      a filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian
    filter_params:
      Parameter to pass to the filter callable
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    blur_as_int:
        If the blurred category arrays should be converted to `np.uint8`
    output_params:
        Keyword arguments for the output file:
        nodata:
          Value to be used as nodata value (if not provided `None` is used)
        dtype:
          Data type into which the output of the filer function will be converted

          .. note::
            This overwrites `output_dtype`, which will me deprecated in the future
    verbose:
        Print out processing step infos
    **params:
        Optional arguments for the multiprocessing:

        nbrcpu: int
          The number of cpu's to use. If not set then then the available number
          of threads -1 are used.
        start_method: str
          Starting method for multiprocessing jobs

    Returns
    -------
    output_file:
       Path to the resulting tif file
    """
    print(f'extract_categories - {source=}, {categories=}')
    if isinstance(source, str):
        source = Source(path=source)
    with source.open() as src:
        width = src.width
        height = src.height
        profile = copy(src.profile)
    # the border size of a block should be at least as large as the kernel size
    # TODO: this should be a computed term, rather than simply set
    # set the block size in pixels
    if verbose:
        print("The chosen source tif has a dimension of:"
            f"\n\t{width=}\n\t{height=}")
        print(f"The block size without border is {block_size=} pixels")
    border = compatible_border_size(**filter_params)
    if verbose:
        print(f"The resulting border size is {border=} pixels")

    # now let's prepare the output parameters:
    count = len(categories)

    # prepare output params
    if output_params is None:
        output_params = dict()
    if blur_as_int:
        output_params['dtype'] = np.uint8
    else:
        output_params['dtype'] = np.float64

    blur_output_params = dict(
        profile=profile,
        count=count,
        output_file=output_file,
    )
    # use directly the `output_params` arg:
    blur_output_params.update(output_params)

    views, inner_views = create_views(view_size=block_size,
                                      border=border,
                                      size=(width, height))

    block_params = []
    # The parameter for the filter we want to apply:
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=source.path,
                       view=view,
                       inner_view=inner_view,
                       categories=categories,
                       img_filter=img_filter,
                       filter_params=filter_params,
                       blur_as_int=blur_as_int, )
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    blur_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the blurred category writer task
        blur_combiner = pool.apply_async(combine_blurred_categories,
                                         (blur_output_params, blur_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(block_filter,
                                             (bparams, blur_q)))
        # collect results
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get().get_duration())

        # once we have all the blocks, add a last element to the queue to stop
        # the combination process
        blur_q.put(dict(signal='kill'))
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

    total_duration = blur_combiner.get().get_duration()
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")
    return output_file


# TODO: This could be called in extract_categories
def apply_filter(source: str | Source,
                 output_file: str,
                 block_size: tuple[int, int],
                 bands: list[Band] | None = None,
                 data_in_range:None|NDArray|Collection=None,
                 data_output_dtype:type|None=np.uint8,
                 data_output_range:None|NDArray|Collection=None,
                 img_filter=None,
                 filter_params:dict|None=None,
                 filter_output_range:Collection|None=(0.,1.),
                 output_dtype:type|None=np.uint8,
                 output_range:tuple|None=None,
                 verbose: bool = False,
                 output_params:None|dict = None,
                 **params
                 )->str:
    """
    Parameters
    ----------
    source : str
        Path to the file with the bands
    output_file:
      The path to write the resulting data to
    bands:
        An optional selection of bands to apply the filter to.
        If not provided all bands are used.
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    data_in_range:
        an array or list from which min and max will be used as range of the
        input data
    data_output_dtype:
      Set the data type that the input data should be converted to before
      applying the the filter

      ..note::
        If provided, the loaded data will be rescaled to the range of
        this data type or `out_range` (if provided).
    data_output_range:
      an array or list from which min and max will be used as limits loaded
      data if its data type is changed
    img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
    filter_params:
      Parameter to pass to the filter callable
    filter_output_range:
      The range of values the applied filter function can return
    output_dtype:
        Data type into which the output of the filer function will be converted
    output_nodata:
        Set what value should be used as nodata value in the output file
        Default: None
    output_range:
      an array or list from which min and max will be used as limits for the
      returned output, if `output_dtype` is provided
    output_params:
        Keyword arguments for the output file:

        nodata:
          Value to be used as nodata value (if not provided `None` is used)
        dtype:
          Data type into which the output of the filer function will be converted

          .. note::
            This overwrites `output_dtype`, which will me deprecated in the future

    verbose:
        Print out processing step infos
    **params:
        Optional arguments for the multiprocessing (e.g. nbr_cpus)
        start_method: str
          Starting method for multiprocessing jobs
    """
    if isinstance(source, str):
        source = Source(path=source)
    with source.open() as src:
        width = src.width
        height = src.height
    if verbose:
        print("The chosen source tif has a dimension of:"
            f"\n\t{width=}\n\t{height=}")
        print(f"The block size without border is {block_size=} pixels")
    if bands is None:
        bands = source.get_bands()
    else:
        sources = set()
        sources.add(source)
        for band in bands:
            sources.add(band.source)
        assert len(sources) == 1, "Only bands with the same source are "\
                                  f"allowed!\nWe have\n\t{sources=}"
        source = sources.pop()

    # prepare output params
    if output_params is None:
        output_params = dict()
    if output_dtype is not None:
        output_params['dtype'] = output_dtype

    # we pass indexes
    indexes = [band.get_bidx() for band in bands]
    profile = source.import_profile()
    border = compatible_border_size(**filter_params)
    if verbose:
        print(f"The resulting border size is {border=} pixels")

    # set the parameter for the resulting file
    blur_output_params = dict(
        profile=profile,
        count=len(indexes),
        output_file=output_file,
    )
    # use directly the `output_params` arg:
    blur_output_params.update(output_params)

    views, inner_views = create_views(view_size=block_size,
                                      border=border,
                                      size=(width, height))

    block_params = []
    # The parameter for the filter we want to apply:
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=str(source.path),
                       bands=bands,
                       view=view,
                       inner_view=inner_view,
                       data_in_range=data_in_range,
                       data_output_dtype=data_output_dtype,
                       data_output_range=data_output_range,
                       img_filter=img_filter,
                       filter_params=filter_params,
                       filter_output_range=filter_output_range,
                       output_dtype=output_dtype,
                       output_range=output_range,
                       )
        block_params.append(bparams)
    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    blur_q = manager.Queue()
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS

    with get_context(start_method).Pool(nbr_cpus) as pool:

        # start the blurred category writer task
        blur_combiner = pool.apply_async(
            func=combine_blurred_categories,
            kwds=dict(output_params=blur_output_params, blur_q=blur_q)
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:

            all_jobs.append(pool.apply_async(
                func=runner_call,
                kwds=dict(queue=blur_q,
                        callback=view_filtered,
                        params=bparams)
            ))
        # collect results
        job_outputs = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_outputs.append(job.get())

        # once we have all the blocks, add a last element to the queue to stop
        # the combination process
        blur_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()
    total_duration = blur_combiner.get().get_duration()
    print(f"{total_duration=}")

    # lzw-compress final output
    compress = params.pop('compress', False)
    if compress:
        out_source = Source(output_file)
        out_source.compress(output=None)
        output_file = str(out_source.path)
        print("Files compressed successfully")

    return output_file



def compute_entropy(source: str | Source,
                    output_file: str,
                    block_size: tuple[int, int],
                    blur_params: dict,  # TODO: is only used to format output_file
                    categories: list | None = None,
                    entropy_as_ubyte: bool = True,
                    normed: bool = True,
                    max_entropy_categories: int | None = None,
                    plot_pdf_preview: bool = True,
                    verbose: bool = False,
                    **params):
    """Compute the entropy-based heterogeneity from several category bands

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    output_file : str
        Path to where the heterogeneity tif should be saved
    categories: list
        Specify which of the land-cover types to use as categories.
        If not provided then all the land-cover types are used.
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    blur_params : dict
        Parameters for the Gaussian blur. It must contain at least either
        `diameter` or `sigma` in a in meters or any other measure of distance.
    entropy_as_ubyte:
        Should the entropy be normalized and returned as ubyte?
    normed:
        If the entropy should be normalized
    max_entropy_categories:
      If normed is true, this determines the maximum n for Entropy to be used to caluclate the maximum to norm by.
      Same as the output_dtype, this argument is ignored if `normed=False`.
    plot_pdf_preview:
        Whether a preview (.pdf) plot of the result should be generated
    verbose:
        Print out processing steps
    **params:
        Optional arguments for the multiprocessing:

        nbrcpu: int
          The number of cpu's to use. If not set then then the available number
          of threads -1 are used.
        start_method: str
          Starting method for multiprocessing jobs
    

    Returns
    -------
    output_file:
       Path to the resulting tif file

    """
    print(f'compute_entropy - {source=}, {categories=}')
    if isinstance(source, str):
        source = Source(path=source)
    with source.open(mode='r') as src:
        width = src.width
        height = src.height
        profile = copy(src.profile)

    if verbose:
        print("The chosen source tif has a dimension of:"
              f"\n\t{width=}\n\t{height=}\n")

    # adapt the profile:
    profile['count'] = 1  # single band for entropy

    # now let's prepare the output parameters:
    if categories is None:
        categories = list(source.get_tag_values(tag='category').values())
        print('WARNING: Inferring the number of categories to use from the\n'
              '         source file.')

    # TODO: provide the categories we want to include
    input_bands = [Band(source=source, tags=dict(category=category))
                   for category in categories]
    if verbose:
        band_choice_str = '\n\t'.join((f'{band.get_bidx()}:{band.tags}' for band in input_bands))
        print("Chosen bands for the entropy calculation:\n"
              f"\t{band_choice_str} \n")

    # TODO: We should not change the output file
    entropy_output_file = output_filename(
        base_name=output_file,
        out_type='entropy',
        blur_params=blur_params
    )

    if entropy_as_ubyte:
        entropy_output_dtype = np.uint8
    else:
        entropy_output_dtype = rio.float64
    entropy_output_params = dict(
        input_bands=input_bands,
        # blur_params=blur_params,
        profile=profile,
        output_dtype=entropy_output_dtype,
        output_file=entropy_output_file,
        output_tags=dict(category='entropy'),
    )
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))

    block_params = []
    for inner_view in inner_views:
        bparams = dict(input_bands=input_bands,
                       categories=categories,
                       inner_view=inner_view,
                       normed=normed,
                       max_entropy_categories=max_entropy_categories,
                       entropy_as_ubyte=entropy_as_ubyte, )
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    entropy_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the entropy writer task
        entropy_combiner = pool.apply_async(combine_entropy_blocks,
                                            (entropy_output_params, entropy_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(block_entropy,
                                             (bparams, entropy_q)))
        # collect results
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get().get_duration())

        # once we have all the blocks, add a last element to the queue to stop
        # the combination process
        entropy_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()

    # Plot preview as pdf
    if plot_pdf_preview:
        plot_entropy(source=entropy_output_file,
                     view=(0, 0, profile['width'], profile['height']),
                     category='entropy',  # select the layer by tag
                     fig_params=dict(output=f"{entropy_output_file}.preview.pdf"))

    # lzw-compress final output
    compress = params.pop('compress', False)
    if compress:
        out_source = Source(entropy_output_file)
        out_source.compress(output=None)
        entropy_output_file = str(out_source.path)
        print("Files compressed successfully")

    total_duration = entropy_combiner.get().get_duration()
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")
    return entropy_output_file


def compute_interaction(source: str | Source,
                        output_file: str,
                        block_size: tuple[int, int],
                        blur_params: dict,  # TODO: is only used to format output_file
                        categories: list | None = None,
                        interaction_as_ubyte: bool = True,
                        standardize: bool = False,
                        normed: bool = True,
                        verbose: bool = False,
                        **params):
    """Compute the interaction-strength from heterogeneity from several category bands in a pairwise,
    tree-way-interaction, four-way-interaction.... manner

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    output_file : str
        Path to where the heterogeneity tif should be saved
    categories: list
        Specify which of the land-cover types to use as categories.
        If not provided then all the land-cover types are used.
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    blur_params : dict
        Parameters for the Gaussian blur. It must contain at least either
        `diameter` or `sigma` in a in meters or any other measure of distance.
    interaction_as_ubyte:
        Should the interaction be normalized and returned as ubyte?
    standardize:
    normed:
    verbose:
        Print out processing steps
    **params:
        Optional arguments for the multiprocessing (e.g. nbr_cpus)


    Returns
    -------
    output_file:
       Path to the resulting tif file

    """
    if isinstance(source, str):
        source = Source(path=source)
    with source.open(mode='r') as src:
        width = src.width
        height = src.height
        profile = copy(src.profile)
        input_dtype = np.dtype(profile['dtype'])

    if verbose:
        print("The chosen source tif has a dimension of:"
              f"\n\t{width=}\n\t{height=}\n")

    # adapt the profile:
    profile['count'] = 1  # single band for interaction

    # now let's prepare the output parameters:
    if categories is None:
        categories = list(source.get_tag_values(tag='category').values())
        print('WARNING: Inferring the number of categories to use from the\n'
              '         source file.')

    input_bands = [Band(source=source, tags=dict(category=category))
                   for category in categories]
    if verbose:
        band_choice_str = '\n\t'.join((f'{band.get_bidx()}:{band.tags}' for band in input_bands))
        print("Chosen bands for the interaction calculation:\n"
              f"\t{band_choice_str} \n")

    interaction_output_file = output_filename(
        base_name=output_file,
        out_type='interaction',
        blur_params=blur_params
    )

    if interaction_as_ubyte:
        interaction_output_dtype = np.uint8
    else:
        interaction_output_dtype = rio.float64
    interaction_output_params = dict(
        input_bands=input_bands,
        profile=profile,
        output_dtype=interaction_output_dtype,
        output_file=interaction_output_file,
        output_tags=dict(category='interaction'),
    )
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))

    block_params = []
    for inner_view in inner_views:
        bparams = dict(input_bands=input_bands,
                       categories=categories,
                       input_dtype=input_dtype,
                       inner_view=inner_view,
                       standardize=standardize,
                       normed=normed,
                       interaction_as_ubyte=interaction_as_ubyte, )
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    interaction_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the interaction writer task
        interaction_combiner = pool.apply_async(combine_interaction_blocks,
                                                (interaction_output_params, interaction_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(block_interaction,
                                            (bparams, interaction_q)))
        # collect results
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get().get_duration())

        # once we have all the blocks, add a last element to the queue to stop
        # the combination process
        interaction_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()

    # lzw-compress final output
    compress = params.pop('compress', False)
    if compress:
        out_source = Source(interaction_output_file)
        out_source.compress(output=None)
        interaction_output_file = str(out_source.path)
        print("Files compressed successfully")

    total_duration = interaction_combiner.get().get_duration()
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")
    return interaction_output_file


def compute_model(predictors: Collection[Band],
                  optimal_weights: dict | None,
                  output_file: str,
                  block_size: tuple[int, int],
                  predictors_as_dtype: None=None,
                  profile: dict | None = None,
                  selector: NDArray[np.bool_] | None = None,
                  verbose: bool = False,
                  **params):
    """Create a tif file with the model prediction values from a fitted model.

    Parameters
    ----------
    predictors:
        Collection predictors used in the multiple linar regression.
    optimal_weights:
        Holding for each predictor the optimal weight
    output_file:
        Path to where the model result should be written to
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    predictors_as_dtype:
        Datatype to convert predictor input to (e.g. np.float32) this will rescale them to [0, 1],
        which is used in the compute weights function.
    profile:
        The profile to use for the newly created output tif.
        By default the profile is copied from the first source of the
        bredictor bands, updating the count to 1.
    selector:
        A numpy boolean array to use as a selector for (masking) the processing of the model.
    verbose:
        Print out processing steps
    **params:
        Optional arguments for the multiprocessing (e.g. nbr_cpus)

    Returns
    -------
    output_tif:
       Path to the newly created tif file holding the model prediction data

    """
    # get all source files
    sources = tuple(set(pred.source.path for pred in predictors))
    check_compatibility(*sources)

    # Handle the output profile
    if profile is None:
        profile = Source(path=sources[0]).import_profile()
        profile['count'] = 1

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
                       optimal_weights=optimal_weights,)
        block_params.append(bparams)


    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    job_out_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the aggregator task
        combiner_job = pool.apply_async(combine_views,
                                        (combine_params, job_out_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(block_model_prediction,
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


def compute_mask(source: str | Source,
                 block_size: tuple[int, int],
                 nodata=0,
                 logic: str = 'all',
                 bands: list[Band] | None = None,
                 verbose: bool = False,
                 **params):
    """Compute the mask of a dataset in parallel and write it it out to the file

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    nodata:
        Supply nodata value to use for mask computation
    logic:
        Either a string or a callable.
        Allowed strings are:
        - `"any"`: Masekd will be each cell for which any of the bands matches the nodata value
        - `"all"`: Masked will be each cell for which all of the bands match the nodata value

        ..note::

            We might, at some point in the future, allow callables here.

            If a callable is provided it takes the data of a window (3D array if multiple
            bands are present, 2D otherwise) and must return a 2D array of
            np.uint8 with 0 for invalid pixels and any value > 0 for valid ones

    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    bands:
        An optional selection of bands to use. If not provided all bands are
        used.
    verbose:
        Print out processing step infos
    **params:
        Optional arguments for the multiprocessing:

        nbrcpu: int
          The number of cpu's to use. If not set then then the available number
          of threads -1 are used.
        start_method: str
          Starting method for multiprocessing jobs
    
    """
    print(f'compute_mask - {source=}')
    if isinstance(source, str):
        source = Source(path=source)
    # make sure the profile is up to date
    source.import_profile()

    width = source.profile.get('width')
    height = source.profile.get('height')

    if bands is None:
        bands = source.get_bands()
    else:
        # make sure the profile is up to date for Bands (if not taken from source but provided as parameter)
        for band in bands:
            band.source.import_profile()

    # set the per-block parameter
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))

    block_params = []
    for view in inner_views:
        task_params = dict(
            nodata=nodata,
            logic=logic,
            #  inner_view=inner_view,
        )
        read_params = dict()
        open_params = dict(mode='r', )
        bparams = dict(task=reduced_mask,
                       source=source,
                       bands=bands,
                       view=view,
                       task_params=task_params,
                       open_params=open_params,
                       read_params=read_params,
                       )
        # inner_view=inner_view,
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    aggr_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the aggregator task
        aggr_params = dict(mode='r+')  # nothing else to pass
        aggregator_job = pool.apply_async(
            func=data_writer,  # the callable
            kwds=dict(  # its arguments:
                writer=source.mask_writer,
                writer_params=aggr_params,
                aggr_q=aggr_q,
            ),
        )
        # start the block jobs
        block_jobs = []
        for bparams in block_params:
            block_jobs.append(pool.apply_async(
                func=process_block,
                kwds=dict(**bparams, out_q=aggr_q, )
            ))
        # collect results
        job_timers = []
        for job in block_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get().get_duration())

        aggr_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()
        total_duration = aggregator_job.get().get_duration()


def prepare_selector(*bands: Band,
                     block_size: tuple[int, int],
                     verbose=False,
                     **params) -> NDArray:
    """Compute a boolean selector from masks of the provided `io_.Band` objects


    Parameters
    ----------
    bands:
        A collection of strings or `io_.Band` object the specify which bands to use
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    verbose:
        Print out processing step infos
    **params:
        Optional arguments for the multiprocessing:

        nbrcpu: int
          The number of cpu's to use. If not set then then the available number
          of threads -1 are used.
        start_method: str
          Starting method for multiprocessing jobs

    Returns
    -------
    NDArray:
       A boolean array that can be used as selector
    """
    print(f'prepare_selector - {bands=}')
    # make sure the bands are compatible
    _source0 = bands[0].source
    if len(bands) > 1:
        _source0.check_compatibility(*(b.source for b in bands[1:]))
    # make sure the profile is up to date
    source0_profile = bands[0].source.import_profile()

    width = int(source0_profile['width'])
    height = int(source0_profile['height'])

    # set the per-block parameter
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height))
    # set the per job parameter
    block_params = []
    for view in inner_views:
        bparams = dict(
            task=aggregated_selector,
            bands=bands,
            view=view,
            task_params=dict(logic='all'),
            open_params=dict(mode='r'),
            read_params=dict(),
        )
        block_params.append(bparams)
    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    aggr_q = manager.Queue()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    # get number of cpu's
    nbr_cpus = params.get('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the aggregator job
        # set the aggregator parameter - just an all-False selector
        aggr_params = dict(
            matrix=np.full((height, width), False),
            aggr_q=aggr_q
        )
        aggregator_job = pool.apply_async(
            func=fill_matrix,  # the callable
            kwds=aggr_params,  # and its arguments
        )
        # start the block jobs
        block_jobs = []
        for bparams in block_params:
            block_jobs.append(pool.apply_async(
                func=process_masks,
                kwds=dict(**bparams, aggr_q=aggr_q, )
            ))
        # collect results
        job_timers = []
        for job in block_jobs:
            job_timers.append(job.get().get_duration())
        # now initiate shutdown of aggregator
        aggr_q.put(dict(signal='kill'))
        pool.close()
        # wait for the *_combiner tasks to finish
        pool.join()
        selector, (timer,) = aggregator_job.get()
        if selector is None:
            print("WARNING: The selector creation retunred no selector: "
                  "All pixel are used!")
            selector = np.full(shape=(height, width), fill_value=True)
        total_duration = timer.get_duration()
    return selector


def check_predictor_consistency(predictors: Collection[Band],
                                selector:NDArray[np.bool_],
                                tolerance:float=0.0,
                                no_data=0.0,
                                sanitize:bool=False,
                                verbose:bool=False,
                                **params)->Collection[Band]:
    """Check if with the selector all the predictors still contain data

    Parameters
    ----------
    predictors:
      A collection with arbitrary many predictors to use.
      See inference.prepare_predictors for further details on how to specify
      predictors.
    selector:
      A boolean array in the same shape of the predictors that indicates
      which cells are usable
    tolerance:
      Determines the limit fraction below which a predictor is considered to
      be completely masked.
      By default (i.e. `tolerance=0.0`) a single cell with a valid value is
      enough to consider the predictor to be valid.

      The fraction of valid cells is computed as the number of valid-cells
      the predictor has divided by the total number of considered cells
      (i.e. the count of `True` in `selector`).
    no_data:
      Value of a cell considered as invalid data.
    sanitize:
      Determines if predictors that end up contributing not a single data-point
      (after applying the `selector`) should be removed automatically.

      By defautl this values is set to `False` which raises an exception
      if a predictor ends up contriuting nothing.
    **params:
        Optional arguments for the multiprocessing:

        nbrcpu: int
          The number of cpu's to use. If not set then then the available number
          of threads -1 are used.
        start_method: str
          Starting method for multiprocessing jobs

    Returns
    -------
    Collection[Band]:
      The remaining predictors. By default (i.e. `sanitize=False`) this simply
      corresponds to the argument that was provided in `predictors`.
      If `sanitize=True` then the colleciton will no longer contain predictors
      that get completely masked when the `selector` is applied.
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
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    nbr_cpus = params.get('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"Predictor consistency check using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        all_jobs = []
        for jparams in job_params:
            all_jobs.append(pool.apply_async(process_band_count_valid,
                                             kwds=jparams))
        band_validity = dict()
        durations = []
        for job in all_jobs:
            valid_band, (time, ) = job.get()
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


def block_model_prediction(params: dict, job_out_q: Queue) -> TimedTask:
    """Per block (i.e. view) model prediction for a fitted regression

    The created view is alwasy returned as np.float64

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      view: tuple
        (x, y, width, height) defining the view or block to process
      predictors: Collection
        A collection of _io.Band objects that were used as predictors
      optimal_weights: dict
        Provides the optimal weight for each predictor
    job_out_q: multiprocessing.Queue
      The queue to push the block data to
    """
    with TimedTask() as timer:
        predictors = params.pop('predictors')
        optimal_weights = params.pop('optimal_weights')
        # for the interaction only the inner view is needed
        view = params.get('view')
        window = view_to_window(view)
        selector = params.get('selector')
        predictors_as_dtype = params.get('predictors_as_dtype')
        width = view[2]
        height = view[3]
        # start with an all zero map
        model_data = np.zeros(shape=(height, width), dtype=np.float32)

        if selector is not None:
            _selector = selector[window.toslices()]
            model_data[~_selector] = np.nan

        for pred in predictors:
            block_data = pred.load_block(view=view)['data']
            if predictors_as_dtype is not None:
                block_data = convert_to_dtype(block_data, as_dtype=predictors_as_dtype)
            # add each predictor data layer multiplied by its weight
            model_data += optimal_weights[pred] * block_data
        output = dict(
            data=model_data,
            view=view
        )
        job_out_q.put(output)
    return timer


def block_entropy(params: dict, entropy_q: Queue) -> TimedTask:
    """Per block (i.e. view) heterogeneity measure based on entropy

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      source: str
        Path to the tif file to use
      view: tuple
        (x, y, width, height) defining the outer border of the view or block
        to process
      inner_view: tuple
        (x, y, width, height) defining the usable part of the block, i.e.
        without the borders
      
      Optionally the following parameters can be set:

      entropy_as_ubyte: bool, Default=False
        Should the entropy be normalized and returned as ubyte?
      normed: bool, Default=True
        Determines if the values in the provided arrays should be normed or not.

    entropy_q: multiprocessing.Queue
      The queue to push the entropy maps through
    """
    with TimedTask() as timer:
        input_bands = params.pop('input_bands')
        blurred_data = dict()
        # for the entropy only the inner view is needed
        view = params.get('inner_view')
        window = view_to_window(view)
        for band in input_bands:
            bidx = band.get_bidx(match='category')
            blurred_data[bidx] = band.get_data(window=window)

        entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
        normed = params.pop('normed', True)
        max_entropy_categories = params.pop('max_entropy_categories', None)
        entropy_params = dict(
            category_arrays=blurred_data,
            view=view,
            normed=normed,
            max_entropy_categories=max_entropy_categories,
            output_dtype=np.uint8 if entropy_as_ubyte else None,
        )
        # This would return the entropy data
        _ = runner_call(
            entropy_q,
            view_entropy,
            entropy_params
        )
    return timer


def block_interaction(params: dict, interaction_q: Queue) -> TimedTask:
    """Per block (i.e. view) interaction measure based on given categories

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      source: str
        Path to the tif file to use
      view: tuple
        (x, y, width, height) defining the outer border of the view or block
        to process
      inner_view: tuple
        (x, y, width, height) defining the usable part of the block, i.e.
        without the borders

      Optionally the following parameters can be set:

      interaction_as_ubyte: bool, Default=False
        Should the interaction be normalized and returned as ubyte?
      standardize: bool, Default=False
      normed: bool, Default=True

    interaction_q: multiprocessing.Queue
      The queue to push the interaction maps through
    """
    with TimedTask() as timer:
        input_bands = params.pop('input_bands')
        blurred_data = dict()
        # for the interaction only the inner view is needed
        view = params.get('inner_view')
        window = view_to_window(view)
        for band in input_bands:
            bidx = band.get_bidx(match='category')
            blurred_data[bidx] = band.get_data(window=window)

        input_dtype = params.pop('input_dtype', None)
        standardize = params.pop('standardize', False)
        normed = params.pop('normed', True)
        interaction_as_ubyte = params.pop('interaction_as_ubyte', False)
        interaction_params = dict(
            view=view,
            input_dtype=input_dtype,
            standardize=standardize,
            normed=normed,
            category_arrays=blurred_data,
            output_dtype=np.uint8 if interaction_as_ubyte else None,
        )
        # This would return the interaction data
        _ = runner_call(
            interaction_q,
            view_interaction,
            interaction_params
        )
    return timer


def block_filter(params: dict, blur_q: Queue) -> TimedTask:
    """Per block (i.e. view) heterogeneity measure based on entropy

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      source: str
        Path to the tif file to use
      view: tuple
        (x, y, width, height) defining the outer border of the view or block
        to process
      inner_view: tuple
        (x, y, width, height) defining the usable part of the block, i.e.
        without the borders
      blur_as_int: bool
        If the blurred category arrays should be converted to `np.uint8`.
      img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
      filter_params:
        Parameter to pass to the filter callable, `img_filter`
      
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    with TimedTask() as timer:
        # this is only needed for the entropy part below
        blur_as_int = params.pop('blur_as_int')
        blur_params = dict(
            source=params.get('source'),
            view=params.get('view'),
            inner_view=params.get('inner_view'),
            categories=params.get('categories'),
            img_filter=params.get('img_filter'),
            filter_params=params.get('filter_params'),
            output_dtype=np.uint8 if blur_as_int else np.float64,
        )
        _ = runner_call(
            blur_q,
            view_blurred,
            blur_params
        )
    return timer


def block_heterogeneity(params: dict, entropy_q: Queue, blur_q: Queue) -> TimedTask:
    """Per block (i.e. view) heterogeneity measure based on entropy

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for a single worker
      The data must include:

      source: str
        Path to the tif file to use
      view: tuple
        (x, y, width, height) defining the outer border of the view or block
        to process inner_view: tuple
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
            source=params.get('source'),
            view=view,
            inner_view=params.get('inner_view'),
            categories=params.get('categories'),
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
def get_XT_X(response: str | Band,
             *predictors: Band | str,
             selector: NDArray,
             include_intercept: bool = True,
             verbose: bool = False,
             **mpc_params
             ) -> np.ndarray:
    """Calculate X.T @ X in parallel directly from view of the predictor data

    ..Note::
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
    start_method = mpc_params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    view_size = mpc_params.get('view_size')
    nbr_cpus = mpc_params.get('nbrcpu', cpu_count() - 1)
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
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the aggregation step
        matrix_aggregator = pool.apply_async(
            combine_matrices,
            (output_q,)
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


def get_optimal_betas(*predictors: Band | str,
                      Y: np.ndarray,
                      response: str | Band,
                      selector,
                      include_intercept=True,
                      verbose: bool = False,
                      as_dtype=np.float64,
                      **mpc_params
                      ):
    """
    """
    print(f'get_optimal_betas - {response=}, {predictors=}')
    if not isinstance(response, Band):
        response = Band(source=Source(path=response),
                        bidx=1)
    start_method = mpc_params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS
    view_size = mpc_params.get('view_size')
    nbr_cpus = mpc_params.get('nbrcpu', cpu_count() - 1)
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
        print(f"using {nbr_cpus=}")

    with get_context(start_method).Pool(nbr_cpus) as pool:
        # start the aggregation step
        matrix_aggregator = pool.apply_async(
            combine_matrices,
            (output_q,)
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
                        limit_contribution:float=0.0,
                        no_data: Union[int,float]=0.0,
                        sanitize_predictors:bool=False,
                        verbose:bool=False,
                        **params
                        ) -> dict[Band, str]:
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
    selector = prepare_selector(response,
                                *predictors,
                                block_size=block_size_params["prepare_selector"],
                                verbose=verbose,
                                **params)


    # If selector is empty (meaning all is FALSE) - no need to proceed
    _vals = np.unique(selector)
    if len(_vals) == 1:
        if _vals[0] == False:  # if not _vals[0] is less explicit
            print(f"WARNING: no pixels to fit, selector masks all pixels")
            return None

    print("Check consistency of remaining predictor data...")
    nbr_predictors = len(predictors)
    predictors = check_predictor_consistency(predictors,
                                             selector=selector,
                                             tolerance=limit_contribution,
                                             sanitize=sanitize_predictors,
                                             no_data=no_data,
                                             verbose=verbose,
                                             **params)

    if len(predictors) != nbr_predictors:
        # for details here: see compute_weights
        selector = prepare_selector(response,
                                    *predictors,
                                    block_size=block_size_params["prepare_selector"],
                                    verbose=verbose,
                                    **params)

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
                    limit_contribution:float=0.0,
                    no_data: Union[int,float]=0.0,
                    sanitize_predictors:bool=False,
                    return_linear_dependent_predictors:bool=False,
                    verbose:bool=False,
                    **params
                    ) -> dict[Band, float] | dict[Band, str] | None:
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
    selector = prepare_selector(response,
                                *predictors,
                                block_size=block_size_params["prepare_selector"],
                                verbose=verbose,
                                **params)

    # If selector is empty (meaning all is FALSE) - no need to proceed
    _vals = np.unique(selector)
    if len(_vals) == 1:
        if _vals[0] == False:  # if not _vals[0] is less explicit
            # TODO: remove None as output and raise error or warning
            print(f"WARNING: no pixels to fit, selector masks all pixels")
            return None

    print("Check consistency of remaining predictor data...")
    nbr_predictors = len(predictors)
    predictors = check_predictor_consistency(predictors,
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
        selector = prepare_selector(response,
                                    *predictors,
                                    block_size=block_size_params["prepare_selector"],
                                    verbose=verbose,
                                    **params)
        # NOTE: We do not need to check_predictor_consistency again sine
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
    rank_def = check_rank_deficiency(tpX)

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
    """Partialy calculate the Sum of Squares for the Residuals (SSR)
    """

    response = params.pop("response")
    model = params.pop("model")
    selector = params.pop("selector")
    view = params.get('view')
    window = view_to_window(view)

    # Get data from window
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
        response = Band(source=Source(path=response),bidx=1)
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
                       model=model,)
        block_params.append(bparams)


    manager = Manager()
    ssr_parts = manager.list()
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS

    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:

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
        response = Band(source=Source(path=response),bidx=1)
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
    start_method = params.get('start_method', MPC_STARTER_METHODS[0])
    assert start_method in MPC_STARTER_METHODS

    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', cpu_count() - 1)
    if verbose:
        print(f"using {nbr_cpus=}")
    with get_context(start_method).Pool(nbr_cpus) as pool:

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