"""
This module contains various helper functions to parallelize
"""

from __future__ import annotations

import warnings
from typing import Any
from copy import copy
from collections.abc import Callable, Collection
from multiprocessing import (Queue, Manager)

import numpy as np

from numpy.typing import NDArray

from .io import Source, Band
from .helper import (view_to_window,
                     reduced_mask,
                     aggregated_selector,
                     get_or_set_context,
                     get_nbr_workers, )
from .prepare import create_views, update_view
from .timing import TimedTask


def combine_views(output_params: dict, job_out_q: Queue):
    """Listens to a queue and writes provided view into a file
    """
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
        out_band.export_tags()
        with out_band.data_writer(mode='r+') as write:  # write until done, i.e. receiving no more jobs - kill
            while True:
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
    """Write out data using the context manager `writer`

    This function can be used with the various context managers defined
    in the :class:`~riogrande.io.models.Source` and :class:`~riogrande.io.models.Band` classes,
    such as :meth:`~riogrande.io.models.Source.mask_writer` or
    :meth:`~riogrande.io.models.Band.data_writer`.

    Parameters
    ----------
    writer : Callable
        A :meth:`~riogrande.io.models.Source.data_writer` or
        :meth:`~riogrande.io.models.Source.mask_writer` context manager
        from a :class:`~riogrande.io.models.Source` or :class:`~riogrande.io.models.Band`.
    writer_params : dict
        Keyword arguments that will be passed to the `writer` method
    aggr_q : Queue
        The queue this job listens to.

    Returns
    -------
    :class:`~riogrande.timing.TimedTask`
        Can report the duration of the task.

    See Also
    --------
    :func:`~riogrande.parallel.combine_views` : Listener that writes views into a file.
    :func:`~riogrande.parallel.process_block` : Worker that processes and enqueues a data block.
    """
    with TimedTask() as timer:
        with writer(**writer_params) as write:
            while True:
                job_out = aggr_q.get()
                signal = job_out.get('signal', None)
                if signal:
                    if signal == "kill":
                        break
                data = job_out.pop('data')
                view = copy(job_out.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                timer.new_lab()
    return timer


def process_block(task: Callable, source: str | Source, bands: Collection[Band] | None, view: tuple[int, int, int, int],
                  task_params: dict, read_params: dict, open_params: dict, out_q: Queue) -> TimedTask:
    """Processes a section of the data in the source file.

    This is a general purpose function that can be used to process a large .tif
    in a parallelized manner.

    The view is converted to a :class:`rasterio.windows.Window` via
    :func:`~riogrande.helper.view_to_window`, and the result is enqueued
    using :func:`~riogrande.parallel.runner_call`.

    Parameters
    ----------
    task : Callable
        Function that will be called on the data from the specified band.
        The first argument of the function must be `data`, a :class:`numpy.ndarray`
        that holds the data from this section.
    source : str or Source
        Either a string or a :class:`~riogrande.io.models.Source` object.
    bands : Collection[Band] or None
        A collection of :class:`~riogrande.io.models.Band` objects specifying
        which bands to use.
    view : tuple[int, int, int, int]
        A tuple (x, y, width, height) defining the view of data to extract and
        process.
    task_params : dict
        Keyword arguments that will be passed to the callable `task`
    read_params : dict
        Keyword arguments that are passed to the open method of the `source` object
    open_params : dict
        Keyword arguments that are passed to the reader method of the `source` object
    out_q : Queue
        The queue this job will put the output of the callable `task` into

    Returns
    -------
    :class:`~riogrande.timing.TimedTask`
        Can report the duration of the task.

    See Also
    --------
    :func:`~riogrande.parallel.process_masks` : Analogous function operating on band masks.
    :func:`~riogrande.parallel.runner_call` : Helper that enqueues the callback result.
    """
    with TimedTask() as timer:
        if not isinstance(source, Source):
            source = Source(path=source)
        if bands is None:
            bands = source.get_bands()
            warnings.warn(
                "No specific bands selected, using all"
            )
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
    """Processes a section of the mask for each band

    This is a general purpose function that can be used to process a large .tif
    in a parallelized manner.

    The view is converted to a :class:`rasterio.windows.Window` via
    :func:`~riogrande.helper.view_to_window`, and the result is enqueued
    using :func:`~riogrande.parallel.runner_call`.

    Parameters
    ----------
    task : Callable
        Function that will be called on the data from the specified band.
        The first argument of the function must be `masks`, a list of
        :class:`numpy.ndarray` holding the masks from this section.
    bands : Collection[Band]
        A collection of :class:`~riogrande.io.models.Band` objects specifying
        which bands to use. Their mask readers are obtained via
        :meth:`~riogrande.io.models.Band.get_mask_reader`.
    view : tuple[int, int, int, int]
        A tuple (x, y, width, height) defining the view of data to extract and
        process.
    task_params : dict
        Keyword arguments that will be passed to the callable `task`
    read_params : dict
        Keyword arguments that are passed to the open method of the `source` object
    open_params : dict
        Keyword arguments that are passed to the reader method of the `source` object
    aggr_q : Queue
        The queue this job will put the output of the callable `task` into
    extra_masking_band : Band or None
        Optional :class:`~riogrande.io.models.Band` object that is treated as
        a rasterio mask, i.e. values equal to 0 are masked.

        .. warning::
          This Band is treated as a mask itself, its own mask is ignored.

    Returns
    -------
    :class:`~riogrande.timing.TimedTask`
        Can report the duration of the task.

    See Also
    --------
    :func:`~riogrande.parallel.process_block` : Analogous function operating on band data.
    :func:`~riogrande.parallel.runner_call` : Helper that enqueues the callback result.
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


def runner_call(queue: Queue[Any], callback: Callable, params: dict, wrapper: Callable | None = None) -> dict:
    """Put the results of callback using parameter into the queue

    The function calls ``callback(**params)``, and optionally passes the result
    through a wrapper function if provided.

    Parameters
    ----------
    queue : Queue[Any]
        A :class:`multiprocessing.Queue` into which the result (wrapped or
        unwrapped) will be placed.
    callback : Callable
        A callable object (function or method) to be executed.
    params : dict
        A dictionary of keyword arguments passed to the callback.
    wrapper : Callable or None
        A function applied to the callback result before it is placed into the
        queue. If ``None``, the raw result is placed into the queue.

    Returns
    -------
    dict
        The unwrapped output.

    See Also
    --------
    :func:`~riogrande.parallel.process_block` : Uses this function to enqueue block results.
    :func:`~riogrande.parallel.process_masks` : Uses this function to enqueue mask results.
    """
    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output


def compute_mask(source: str | Source, block_size: tuple[int, int], nodata=0, logic: str = 'all',
                 bands: list[Band] | None = None, verbose: bool = False, **params) -> None:
    """Compute the mask of a dataset in parallel and write it out to the file

    This function is checking validity against the nodata rule across all bands, provided the
    selected logic (via :func:`~riogrande.helper.reduced_mask`).
    The dataset is split into blocks with :func:`~riogrande.prepare.create_views`
    and each block is processed by :func:`~riogrande.parallel.process_block` in a
    :class:`multiprocessing.pool.Pool` obtained via :func:`~riogrande.helper.get_or_set_context`.

    Parameters
    ----------
    source : str or :class:`~riogrande.io.models.Source`
        Path to the tif file or a Source object.
    block_size : tuple[int, int]
        Size (width, height) in #pixel of the block that a single job processes
    nodata : int or float
        Supply nodata value to use for mask computation
    logic : str
        Either a string or a callable.
        Allowed strings are:

        - ``"any"`` : Mask each cell where *any* of the bands matches the nodata value.
        - ``"all"`` : Mask each cell where *all* of the bands match the nodata value.

    bands : list[Band] or None
        An optional selection of :class:`~riogrande.io.models.Band` objects to use.
        If not provided all bands are used.
    verbose : bool
        Print out processing step infos
    **params : dict
        Optional arguments for the multiprocessing:

        - ``nbrcpu`` : int
          Number of CPUs to use, passed to :func:`~riogrande.helper.get_nbr_workers`.
        - ``start_method`` : str
          Starting method for multiprocessing jobs, passed to
          :func:`~riogrande.helper.get_or_set_context`.

    Returns
    -------
    None

    See Also
    --------
    :func:`~riogrande.parallel.prepare_selector` : Build a boolean selector from band masks.
    :func:`~riogrande.helper.reduced_mask` : Per-block mask computation function used internally.
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
        aggr_params = dict(mode='r+')
        aggregator_job = pool.apply_async(
            func=data_writer,   # TODO: this is the only use of the data_writer function I believ
            kwds=dict(
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
    """Fill a matrix with data received through a queue.

    Each received block is placed into `matrix` via
    :func:`~riogrande.prepare.update_view`.

    Parameters
    ----------
    matrix : NDArray
        The :class:`numpy.ndarray` to fill with data.
    aggr_q : Queue
        The :class:`multiprocessing.Queue` this job listens to. Each element
        in the queue must be a ``dict`` containing either:

        - ``{"view": ..., "data": ...}`` specifying where to write what.
        - ``{"signal": "kill"}`` to terminate the process and return the filled matrix.

    Returns
    -------
    matrix, (timer,):
        The first object is the filled :class:`numpy.ndarray`, the second holds a
        :class:`~riogrande.timing.TimedTask` object with duration information.

    See Also
    --------
    :func:`~riogrande.prepare.update_view` : Write a block into a view of an array.
    :func:`~riogrande.parallel.prepare_selector` : Uses this function as the aggregator job.
    """
    with TimedTask() as timer:
        while True:
            output = aggr_q.get()
            signal = output.get('signal', None)
            if signal:
                if signal == "kill":
                    break
            view = output.pop('view')
            block_data = output.pop('data')
            update_view(data=matrix, view=view, block=block_data)
            timer.new_lab()
    return matrix, (timer,)


def prepare_selector(*bands: Band, block_size: tuple[int, int], extra_masking_band: Band | None = None,
                     verbose=False, **params) -> NDArray:
    """Compute a boolean selector from masks of the provided :class:`~riogrande.io.models.Band` objects

    Band masks are aggregated into a boolean selector (via
    :func:`~riogrande.helper.aggregated_selector`) that can be used to identify valid pixels
    across all provided bands. Optionally, an extra masking band may be applied where values
    equal to 0 are masked out.
    The dataset is split into blocks with :func:`~riogrande.prepare.create_views`
    and each block is processed by :func:`~riogrande.parallel.process_masks` in a
    :class:`multiprocessing.pool.Pool` obtained via :func:`~riogrande.helper.get_or_set_context`.
    Results are assembled by :func:`~riogrande.parallel.fill_matrix`.

    Parameters
    ----------
    bands : Band
        A collection of :class:`~riogrande.io.models.Band` objects specifying
        which bands to use.
    block_size : tuple[int, int]
        Size (width, height) in #pixel of the block that a single job processes
    extra_masking_band : Band or None
        Optional :class:`~riogrande.io.models.Band` object that is treated as a rasterio
        mask, i.e. values equal to 0 will be masked.
    verbose : bool
        Print out processing step infos
    **params : dict
        Optional arguments for the multiprocessing:

        - ``nbrcpu`` : int
          The number of CPUs to use, passed to :func:`~riogrande.helper.get_nbr_workers`.
        - ``start_method`` : str
          Starting method for multiprocessing jobs, passed to
          :func:`~riogrande.helper.get_or_set_context`.

    Returns
    -------
    NDArray
       A boolean :class:`numpy.ndarray` that can be used as selector.

    See Also
    --------
    :func:`~riogrande.parallel.compute_mask` : Analogous function that writes the mask to file.
    :func:`~riogrande.helper.aggregated_selector` : Per-block aggregation function used internally.
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
        aggr_params = dict(
            matrix=np.full((height, width), False),
            aggr_q=aggr_q
        )   # set the aggregator parameter - just an all-False selector
        aggregator_job = pool.apply_async(
            func=fill_matrix,
            kwds=aggr_params,
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
