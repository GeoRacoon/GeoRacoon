"""
This module contains various helper functions to parallelize

"""
# is_needed
# needs_work (the module is too big!)
# not_tested (partially)
# usedin_both (should be split up!)
from __future__ import annotations

from typing import Any
from copy import copy
from collections.abc import Callable, Collection
from multiprocessing import (Queue, Manager)

import numpy as np

from numpy.typing import NDArray

from .io_ import Source, Band
from .helper import (view_to_window,
                     reduced_mask,
                     aggregated_selector,
                     get_or_set_context,
                     get_nbr_workers, )
from .prepare import create_views, update_view
from .timing import TimedTask


def combine_views(output_params: dict, job_out_q: Queue):
    # not_needed (could become a general combiner method - remove for now)
    # needs_work (docs)
    # not_tested
    """Listens to a queue and writes provided view into a file
    """
    # usedin_both (potentially

    with TimedTask() as timer:
        output_file = output_params.pop('output_file')
        profile = output_params.pop('profile')
        out_band = output_params.pop('band')
        out_tag = output_params.pop('tags')
        verbose = output_params.get('verbose', False)
        if out_band is None:
            out_band = Band(source=Source(path=output_file),
                            bidx=1,
                            tags=out_tag)  # create the file
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
                        if verbose:
                            print(f"\n\nClosing: {out_band.source.path}\n\n")
                        break
                data = output.pop('data')
                view = copy(output.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                if verbose:
                    print(f"Wrote out block {view=}")
                timer.new_lab()
    return timer


def data_writer(writer: Callable, writer_params: dict, aggr_q: Queue) -> TimedTask:
    # is_needed (internally_only)
    # needs_work (make internal; rename io_ has a data_writer method; docs)
    # not_tested
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


def process_block(task: Callable, source: str | Source, bands: Collection[Band] | None, view: tuple[int, int, int, int],
                  task_params: dict, read_params: dict, open_params: dict, out_q: Queue) -> TimedTask:
    # is_needed (internally only)
    # needs_work (check if this can be used as general purpose in all paralellizations)
    # not_tested (should be)
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


def process_masks(task: Callable, bands: Collection[Band], view: tuple[int, int, int, int],
                  task_params: dict, read_params: dict, open_params: dict, aggr_q: Queue,
                  extra_masking_band: Band | None = None) -> TimedTask:
    # is_needed (internally only)
    # needs_work (make internal)
    # not_tested
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
    extra_masking_band:
        Optional `io_.Band` object that is treated as a rasterio mask, i.e. values equal to 0
        .. warning::
          This Band is treated as a mask itself, its own mask is ignored.

    Returns
    -------
    TimedTask
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
        # add the data from the extra masking band as an additional mask
        if extra_masking_band is not None:
            with extra_masking_band.data_reader() as read:
                extra_mask = read(window=window)
            masks.append(extra_mask)
        _ = runner_call(callback=task,
                        params=dict(masks=masks, **task_params),
                        queue=aggr_q,
                        wrapper=lambda x: dict(data=x, view=view))
        # print(f"{view=}\n{data=}\nmask={_}")
    return timer


def runner_call(queue: Queue[Any], callback: Callable, params: dict, wrapper: Callable | None = None) -> dict :
    # is_needed (internally only)
    # needs_work (better docs; make internal; check if can be used for generalization)
    # not_tested (used in tests)
    """Put the results of callback using parameter into the queue

    If provided `wrapper(callback(**params))` is put into the queue.

    Parameters
    ----------
    queue:
    callback:
    params:
    wrapper:

    Returns
    ---------

    """
    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output


def compute_mask(source: str | Source, block_size: tuple[int, int], nodata=0, logic: str = 'all',
                 bands: list[Band] | None = None, verbose: bool = False, **params) -> None:
    # is_needed
    # needs_work (docs)
    # not_tested (used in various tests)
    """Compute the mask of a dataset in parallel and write it it out to the file

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    nodata:
        Supply nodata value to use for mask computation
    logic:
        Either a string or a callable.
        Allowed strings are:
        - `"any"`: Masekd will be each cell for which any of the bands matches the nodata value
        - `"all"`: Masked will be each cell for which all of the bands match the nodata value

        .. note::

            We might, at some point in the future, allow callables here.

            If a callable is provided it takes the data of a window (3D array if multiple
            bands are present, 2D otherwise) and must return a 2D array of
            np.uint8 with 0 for invalid pixels and any value > 0 for valid ones
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

    Returns
    -------
    None
    """
    print(f'compute_mask - {source=}')
    if isinstance(source, str):
        source = Source(path=source)

    source.import_profile()
    width = source.profile.get('width')
    height = source.profile.get('height')

    if bands is None:
        bands = source.get_bands()
    else:
        for band in bands:
            band.source.import_profile()

    # set the per-block parameter
    _, inner_views = create_views(view_size=block_size,
                                  border=(0, 0),
                                  size=(width, height)
                                  )
    block_params = []
    for view in inner_views:
        task_params = dict(
            nodata=nodata,
            logic=logic,
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
        block_params.append(bparams)

    # prepare multiprocessing
    manager = Manager()
    aggr_q = manager.Queue()
    start_method = params.get('start_method', None)
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")

    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
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
            job_timers.append(job.get().get_duration())

        aggr_q.put(dict(signal='kill'))
        pool.close()
        pool.join()
        total_duration = aggregator_job.get().get_duration()


def fill_matrix(matrix: NDArray, aggr_q: Queue) -> tuple[NDArray | None, tuple]:
    # is_needed (internally only)
    # needs_work (docs; make internal)
    # not_tested
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


def prepare_selector(*bands: Band, block_size: tuple[int, int], extra_masking_band: Band | None = None,
                     verbose=False, **params) -> NDArray:
    # is_needed (internally only - also tests)
    # needs_work (make internal; docs)
    # not_tested (not directly)
    """Compute a boolean selector from masks of the provided `io_.Band` objects

    Parameters
    ----------
    bands:
        A collection of strings or `io_.Band` object the specify which bands to use
    block_size:
        Size (width, height) in #pixel of the block that a single job processes
    extra_masking_band: Optional `io_.Band` object thas is treated as a rasterio mask, i.e. values equal to 0
      will be masked.
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
    NDArray
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
                                  size=(width, height)
                                  )
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
            extra_masking_band=extra_masking_band,
        )
        block_params.append(bparams)

    # prepare multiprocessing
    manager = Manager()
    aggr_q = manager.Queue()
    start_method = params.get('start_method', None)
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")

    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
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
        aggr_q.put(dict(signal='kill'))
        pool.close()
        pool.join()
        selector, (timer,) = aggregator_job.get()
        if selector is None:
            print("WARNING: The selector creation retunred no selector: "
                  "All pixel are used!")
            selector = np.full(shape=(height, width), fill_value=True)
        total_duration = timer.get_duration()
    return selector
