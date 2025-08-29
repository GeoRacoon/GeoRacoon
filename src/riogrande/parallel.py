"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
# is_needed
# needs_work (the module is too big!)
# not_tested (partially)
# usedin_both (should be split up!)
from __future__ import annotations

import warnings
from typing import Any
from collections.abc import Callable, Collection

from copy import copy

from typing import Union

import numpy as np
import rasterio as rio

from multiprocessing import (Queue, Manager)
from numpy.typing import NDArray

from .io_ import Source, Band
from .helper import (view_to_window,
                      reduced_mask,
                      aggregated_selector,
                      check_compatibility,
                      get_or_set_context,
                      get_nbr_workers, )
from .prepare import create_views, update_view
from .timing import TimedTask


# TODO: very important - now it is dependent on ConvSTER which we really dont want.
# Either move this or change it up
from convster.processing import (view_blurred,
                                 view_filtered,
                                 )
from convster.parallel import combine_blurred_categories

# TODO: this could lead to failure but not sure why it does not import
from convster.filters.gaussian import compatible_border_size


def combine_views(output_params: dict,
                  job_out_q: Queue):
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
                            tags=out_tag) # create the file
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

def process_block(task: Callable,
                  source: str | Source,
                  bands: Collection[Band] | None,
                  view: tuple[int, int, int, int],
                  task_params: dict,
                  read_params: dict,
                  open_params: dict,
                  out_q: Queue) -> TimedTask:
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


def process_masks(task: Callable,
                  bands: Collection[Band],
                  view: tuple[int, int, int, int],
                  task_params: dict,
                  read_params: dict,
                  open_params: dict,
                  aggr_q: Queue,
                  extra_masking_band:Band|None=None,
                  ) -> TimedTask:
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


def runner_call(queue: Queue[Any],
                callback: Callable,
                params: dict,
                wrapper: Callable | None = None):
    # is_needed (internally only)
    # needs_work (better docs; make internal; check if can be used for generalization)
    # not_tested (used in tests)
    """Put the results of callback using parameter into the queue

    If provided `wrapper(callback(**params))` is put into the queue.

    """

    output = callback(**params)
    if wrapper is not None:
        queue.put(wrapper(output))
    else:
        queue.put(output)
    return output


# TODO: check how much extract_categories and apply_filter are redundant
def extract_categories(source: str | Source,
                       categories: list,
                       output_file: str,
                       block_size: tuple[int, int],
                       img_filter: None|Callable=None,
                       filter_params: dict|None=None,
                       output_dtype: type|str|None=None,
                       output_params:None|dict = None,
                       filter_output_range:tuple|None=None,
                       verbose: bool = False,
                       **params):
    # is_needed (internally only and in tests)
    # needs_work (make internal?)
    # is_tested
    # usedin_both (potentially)
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
      An optional filter function that can be applied to the data. See e.g.
      skimage.filter.gaussian

      .. note::
        If `img_filter` is not set `filter_params` is ignored.

    filter_params:
      Parameter to pass to the filter callable.

      .. note::
        This argument is mandatory if `img_filter` is provided and ignored
        otherwise.

    block_size:
        Size (width, height) in #pixel of the block that a single job processes

    output_dtype:
      Set the data type of the blurred categories that are returned.
      
      .. warning::
        
        This parameter is deprecated, please use `output_params['as_dtype']`
        instead.

    output_params:
        Keyword arguments for the output file:
        nodata:
          Value to be used as nodata value (if not provided `None` is used)
        as_dtype:
          Data type into which the output of the filer function will be converted
        bigtiff:
          Boolean whether to create a BIGTIFF file or not, for files larger than 4 GB
          TODO: (see apply_filter - on question to make this standard)

          .. note::
            This overwrites `output_dtype`, which will me deprecated in the future

        output_range: tuple
          Specify the range into which the filter output should be mapped.

     filter_output_range: tuple
       Specify the data range expected as output from the applied filter.

       If you expect floats as output but want to set a different range
       than `[0, 1]`, specify it with this parameter.

       .. note::
         Consider setting this if you encounter warning messages issued
         by the `convert_to_dtype` function.

    verbose:
        Print out processing step infos
    **params:
        Optional arguments

        - Data conversion:
              
          output_range: tuple
            Specify the range into which the filter output should be mapped.

        - For the multiprocessing:
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

    if output_params is None:
        output_params = dict()
    # handle deprecated parameters
    blur_as_int = params.pop('blur_as_int', None)
    if blur_as_int is not None:
        if blur_as_int:
            output_params['as_dtype'] = "uint8"
        else:
            output_params['as_dtype'] = "float64"
        warnings.warn("The parameter `blur_as_int` is deprecated, use "
                      f"`output_params['as_dtype']` instead!\nUsing "
                      f"{blur_as_int=} leads to "
                      f"{output_params['as_dtype']=}",
                      category=DeprecationWarning)

    if output_dtype is not None:
        output_params['as_dtype'] = output_dtype
        warnings.warn("The parameter `output_dtype` is deprecated, use "
                      f"`output_params['as_dtype']` instead!\nUsing "
                      f"{output_dtype=} leads to "
                      f"{output_params['as_dtype']=}",
                      category=DeprecationWarning)
    # ---
    if img_filter is not None and filter_output_range is None:
        warnings.warn(
            "It is strongly encouraged to set `filter_output_range` if you are "
            f"using the filter {img_filter}."
        )
    if verbose:
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
    if img_filter is None:  # no filter means blocks with 0 transition region
        border = (0, 0)
    else:
        border = compatible_border_size(**filter_params)
    if verbose:
        print(f"The resulting border size is {border=} pixels")

    # now let's prepare the output parameters:
    count = len(categories)


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
                       filter_output_range=filter_output_range,
                       as_dtype=output_params['as_dtype'],
                       output_range=output_params.get('output_range', None))
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    blur_q = manager.Queue()
    start_method = params.get('start_method', None)
    # get number of workers
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
        # start the blurred category writer task
        blur_combiner = pool.apply_async(combine_blurred_categories,
                                         (blur_output_params, blur_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(block_category_extraction,
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
                 data_as_dtype:type|None=np.uint8,
                 data_output_range:None|NDArray|Collection=None,
                 replace_nan_with: Union[int, float] | None = None,
                 img_filter=None,
                 filter_params:dict|None=None,
                 filter_output_range:Collection|None=(0.,1.),
                 output_dtype:type|None=np.uint8,
                 output_range:tuple|None=None,
                 selector_band: Band | None = None,
                 verbose: bool = False,
                 output_params:None|dict = None,
                 **params
                 )->str:
    # is_needed (only in tests)
    # needs_work (docs - if not jsut deleted; see TODOs)
    # is_tested
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
    data_as_dtype:
      Set the data type that the input data should be converted to before
      applying the filter

      .. note::
        If provided, the loaded data will be rescaled to the range of
        this data type or `out_range` (if provided).

    data_output_range:
      an array or list from which min and max will be used as limits loaded
      data if its data type is changed
    replace_nan_with:
      Replace nan values in the source with the provided value.
      Avoid having areas 'cropped' due to NaN replacement - yet effects filter output.
    img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
    filter_params:
      Parameter to pass to the filter callable
    filter_output_range:
      The range of values the applied filter function can return
    output_dtype:
        Data type into which the output of the filter function will be converted
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
        bigtiff:
          Boolean whether to create a BIGTIFF file or not, for files larger than 4 GB
          TODO: We can think about either making this standard or allowing for the rasteiro implementation from GDAL:
             https://rasterio.readthedocs.io/en/latest/topics/image_options.html
             YET then we need to capitalize the input and it needs to perfectly match the lettering
          .. note::
            This overwrites `output_dtype`, which will me deprecated in the future
    selector_band:
        A band object with categorical data which will be used as a mask for iterative performance of the blurring.
        If provided, the blurring will be done for each categorical data. Use border preserving gaussian filter to
        avoid removing borders between categorical datas of selector_band

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
    # TODO: we should get rid of either output_params['dtype'] or output_dtype as a user input option
    if output_params is None:
        output_params = dict()
    _out_dtype = output_params.get('as_dtype', None)
    if _out_dtype is not None and _out_dtype != output_dtype:
        raise ValueError(f"Provided two different output dtypes: {output_params['as_dtype']=} and {output_dtype=}")
    output_params['as_dtype'] = output_dtype

    # we pass indexes
    indexes = [band.get_bidx() for band in bands]
    profile = source.import_profile()
    border = compatible_border_size(**filter_params)
    if verbose:
        print(f"The resulting border size is {border=} pixels")

    # we need to set nodata to None if input is different dtype than output and input has Nodata e.g. nan to unit8
    # TODO: we can use the rasterio to numpy mapping to check whether input is same as output dtype - else set to None
    output_params['nodata'] = None

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
                       data_as_dtype=data_as_dtype,
                       data_output_range=data_output_range,
                       replace_nan_with=replace_nan_with,
                       img_filter=img_filter,
                       filter_params=filter_params,
                       filter_output_range=filter_output_range,
                       as_dtype=output_dtype,
                       output_range=output_range,
                       selector_band=selector_band,
                       )
        block_params.append(bparams)
    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    blur_q = manager.Queue()
    # get number of cpu's
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    
    start_method = params.get('start_method', None)

    with get_or_set_context(start_method).Pool(nbr_workers) as pool:

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


def compute_mask(source: str | Source,
                 block_size: tuple[int, int],
                 nodata=0,
                 logic: str = 'all',
                 bands: list[Band] | None = None,
                 verbose: bool = False,
                 **params):
    # is_needed
    # needs_work (docs)
    # not_tested (used in various tests)
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

        .. note::

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
    start_method = params.get('start_method', None)
    # get number of workers
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
                     extra_masking_band: Band|None=None,
                     verbose=False,
                     **params) -> NDArray:
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
            extra_masking_band=extra_masking_band,
        )
        block_params.append(bparams)
    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    aggr_q = manager.Queue()
    start_method = params.get('start_method', None)
    # get number of workers
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

def block_category_extraction(params: dict, blur_q: Queue) -> TimedTask:
    # is_needed (internally only)
    # needs_work (make internal; docs)
    # not_tested
    """Per block (i.e. view) category extraction and filter application

    This is a wrapper function to process a selection of a (pot. large)
    tif file and push the results into a multiprocessing queue for
    aggregation.
    The method creates individual layers for each category in a band of
    categorical data and optionally apply an filter callable.

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
      as_dtype:
        Set the data type of the blurred categories that are returned.
      output_range:
        The data range the output will be mapped to (when converting to `output_dtype`)
      img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
      filter_params:
        Parameter to pass to the filter callable, `img_filter`
      filter_output_range:
        Optionally specify the output range the filter method can produce
      
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    with TimedTask() as timer:
        # this is only needed for the entropy part below
        blur_params = dict(
            source=params.get('source'),
            view=params.get('view'),
            inner_view=params.get('inner_view'),
            categories=params.get('categories'),
            img_filter=params.get('img_filter'),
            filter_params=params.get('filter_params'),
            filter_output_range=params.get('filter_output_range'),
            # TODO: we need to consistently use `as_dtype` for the
            #       data type of  returned data
            output_dtype=params.get('as_dtype'),
            output_range=params.get('output_range'),
        )
        _ = runner_call(
            blur_q,
            view_blurred,
            blur_params
        )
    return timer
