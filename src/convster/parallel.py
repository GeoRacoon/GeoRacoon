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

from copy import copy


import numpy as np
import rasterio as rio

from multiprocessing import (Queue, Manager)

from riogrande.io_ import Source, Band
from riogrande.helper import (
    view_to_window,
    output_filename,
    get_or_set_context,
    get_nbr_workers
)

from riogrande.timing import TimedTask
from riogrande.prepare import create_views
from riogrande.io import write_band
from riogrande.parallel import runner_call

from .plotting import plot_entropy
from .processing import (
    view_blurred,
    view_entropy,
    view_interaction
)


def combine_blurred_categories(output_params: dict, blur_q: Queue) -> TimedTask:
    # is_needed
    # needs_work (docs)
    # is_tested
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
        as_dtype = output_params.pop('as_dtype')
        if isinstance(as_dtype, str):
            as_dtype = np.dtype(as_dtype)
        output_file = output_params.pop('output_file')
        # print(f"{output_file=}")
        # print(f"{dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = as_dtype
        # overwrite the profile if explicitely provided:
        profile['nodata'] = output_params.pop('nodata', profile.get('nodata', None))
        profile['count'] = output_params.get('count', profile['count'])
        # check for Bigtiff
        if output_params.pop('bigtiff', False):
            profile['BIGTIFF'] = 'YES'
        verbose = output_params.get('verbose', False)
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
                    write_band(src=dst, bidx=bidx, data=data.astype(as_dtype), window=w,
                               category=band)
                    # NOTE: we might want keep the description unchanged:
                    dst.set_band_description(bidx, f'LC_{band}')
                if verbose:
                    print(f"Wrote out bands for blurred block {inner_view=}")
                timer.new_lab()
        # print(f"\n\n########\n\nProfile")

    return timer

def combine_entropy_blocks(output_params: dict,
                           entropy_q: Queue):
    # is_needed
    # needs_work
    # is_tested
    """Listen to queue (entropy_q) and write computed blocks to a single file.

    This function continuously listens to a queue for entropy blocks and writes
    them to a specified output file.
    It initializes the output file based on the provided parameters and handles
    the writing of data until a termination signal is received.

    Parameters
    ----------
    output_params : dict
        A dictionary containing parameters for output configuration.
        Expected keys include:
        - 'output_dtype': The data type of the output.
        - 'output_file': The path to the output file where data will be written.
        - 'profile': A dictionary containing additional profile settings for the output.
        - 'out_band': (optional) An instance of a Band object for managing output;
          if not provided, a new Band will be created.

    entropy_q : Queue
        A queue from which entropy blocks are read. Each item in the queue
        is expected to be a dictionary containing:
        - 'signal': A control signal (e.g., "kill" to terminate the process).
        - 'data': The actual data block to be written to the output file.
        - 'view': Metadata related to the data block, used for windowing during writing.

    Returns
    -------
    TimedTask
        An instance of TimedTask that tracks the duration of the operation.
        This can be used for performance monitoring or logging purposes.

    Notes
    -----
    The function will block while waiting for items in the queue and will
    terminate gracefully when a "kill" signal is received.
    It is important to ensure that the queue is properly managed to avoid
    deadlocks or resource leaks.
    """

    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        if isinstance(output_dtype, str):
            output_dtype = np.dtype(output_dtype)
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
                # print(f"Wrote out entropy block {view=}")
                timer.new_lab()
    return timer

def combine_interaction_blocks(output_params: dict,
                               interaction_q: Queue):
    # is_needed (internally_only)
    # needs_work (docs; make internal)
    # not_tested
    """Listen to queue (interaction_q) and write computed block to single file
    """

    with TimedTask() as timer:
        output_dtype = output_params.pop('output_dtype')
        if isinstance(output_dtype, str):
            output_dtype = np.dtype(output_dtype)
        output_file = output_params.pop('output_file')
        # print(f"{output_file=}")
        # print(f"{output_dtype=}")
        profile = output_params.pop('profile')
        profile['dtype'] = output_dtype
        out_band = output_params.pop('out_band', None)
        out_tag = output_params.pop('output_tag', dict(category='interaction'))
        verbose = output_params.get('verbose', False)
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
                        if verbose:
                            print(f"\n\nClosing: {out_band.source.path}\n\n")
                        break
                data = output.pop('data')
                view = copy(output.pop('view'))
                w = view_to_window(view)
                write(data, window=w)
                if verbose:
                    print(f"Wrote out interaction block {view=}")
                timer.new_lab()
    return timer

def compute_entropy(source: str | Source,
                    output_file: str,
                    block_size: tuple[int, int],
                    blur_params: dict,  # TODO: is only used to format output_file
                    categories: list | None = None,
                    output_dtype: type | str | None = None,
                    output_range: tuple | None = None, # TODO: Do we want to infer for np.integer? (see compute entropy)
                    normed: bool = True,
                    max_entropy_categories: int | None = None,
                    plot_pdf_preview: bool = True,
                    verbose: bool = False,
                    **params):
    # is_needed (only in example)
    # needs_work (doc)
    # is_tested
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
    output_dtype:
      Set the data type of the entropy file that is returned.
    output_range:
      an array or list from which min and max will be used as limits for the
      returned output.
      By default if `output_dtype` is provided and of type np.floating it will be set to [0, 1], if of type np.integer
      to the min max of possible dtype e.g. uin8 [0, 255]
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
    # handle deprecated parameters
    entropy_as_ubyte = params.pop('entropy_as_ubyte', None)
    if entropy_as_ubyte is not None:
        if entropy_as_ubyte:
            output_dtype = "uint8"
        else:
            output_dtype = "float64"
        warnings.warn("The parameter `entropy_as_ubyte` is deprecated, use "
                      f"`output_dtype` instead!\nUsing "
                      f"{entropy_as_ubyte=} leads to "
                      f"{output_dtype=}",
                      category=DeprecationWarning)
    if output_dtype is None:
        output_dtype = np.uint8
        warnings.warn("No `output_dtype` provided for result!\nUsing "
                      f"{output_dtype=} default instead")

    entropy_output_params = dict(
        input_bands=input_bands,
        # blur_params=blur_params,
        profile=profile,
        output_dtype=output_dtype,
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
                       output_dtype=output_dtype,
                       output_range=output_range,
                       normed=normed,
                       max_entropy_categories=max_entropy_categories,)
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    entropy_q = manager.Queue()
    start_method = params.get('start_method', None)
    # get number of cpu's
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
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
                        output_dtype: type | str | None = None,
                        output_range: tuple | None = None,
                        standardize: bool = False,
                        normed: bool = True,
                        verbose: bool = False,
                        **params):
    # is_needed (only used in tests)
    # needs_work (doc)
    # is_tested
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
    output_dtype:
      Set the data type of the interaction file that is returned.
    output_range:
      an array or list from which min and max will be used as limits for the
      returned output.
      By default if `output_dtype` is provided and of type np.floating it will be set to [0, 1], if of type np.integer
      to the min max of possible dtype e.g. uin8 [0, 255]
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
    # handle deprecated parameters
    interaction_as_ubyte = params.pop('interaction_as_ubyte', None)
    if interaction_as_ubyte is not None:
        if interaction_as_ubyte:
            output_dtype = "uint8"
        else:
            output_dtype = "float64"
        warnings.warn("The parameter `interaction_as_ubyte` is deprecated, use "
                      f"`output_dtype` instead!\nUsing "
                      f"{interaction_as_ubyte=} leads to "
                      f"{output_dtype=}",
                      category=DeprecationWarning)
    if output_dtype is None:
        output_dtype = np.uint8
        warnings.warn("No `output_dtype` provided for result!\nUsing "
                      f"{output_dtype=} default instead")

    interaction_output_params = dict(
        input_bands=input_bands,
        profile=profile,
        output_dtype=output_dtype,
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
                       output_dtype=output_dtype,
                       output_range=output_range,
                       standardize=standardize,
                       normed=normed,)
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = Manager()
    interaction_q = manager.Queue()
    start_method = params.get('start_method', None)
    # get number of workers
    nbr_workers = get_nbr_workers(number=params.pop('nbrcpu', None))
    if verbose:
        print(f"using {nbr_workers=}")
    with get_or_set_context(start_method).Pool(nbr_workers) as pool:
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

def block_entropy(params: dict, entropy_q: Queue) -> TimedTask:
    # is_needed (internally only)
    # needs_work (make internal; docs)
    # not_tested
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

        output_dtype = params.pop('output_dtype')
        output_range = params.pop('output_range')
        normed = params.pop('normed', True)
        max_entropy_categories = params.pop('max_entropy_categories', None)
        entropy_params = dict(
            category_arrays=blurred_data,
            view=view,
            normed=normed,
            max_entropy_categories=max_entropy_categories,
            output_dtype=output_dtype,
            output_range=output_range,
        )
        # This would return the entropy data
        _ = runner_call(
            entropy_q,
            view_entropy,
            entropy_params
        )
    return timer


def block_interaction(params: dict, interaction_q: Queue) -> TimedTask:
    # is_needed (internally only)
    # needs_work (make internal; docs)
    # not_tested
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
        output_dtype = params.pop('output_dtype')
        output_range = params.pop('output_range')
        standardize = params.pop('standardize', False)
        normed = params.pop('normed', True)
        interaction_as_ubyte = params.pop('interaction_as_ubyte', False)
        interaction_params = dict(
            view=view,
            input_dtype=input_dtype,
            standardize=standardize,
            normed=normed,
            category_arrays=blurred_data,
            output_dtype=output_dtype,
            output_range=output_range,
        )
        # This would return the interaction data
        _ = runner_call(
            interaction_q,
            view_interaction,
            interaction_params
        )
    return timer

def block_heterogeneity(params: dict, entropy_q: Queue, blur_q: Queue) -> TimedTask:
    # is_needed
    # needs_work (docs)
    # not_tested
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
      img_filter: Callable
        A filter function that can be applied to the data. See e.g.
        skimage.filter.gaussian
      filter_params:
        Parameter to pass to the filter callable, `img_filter`
      
      Optionally the following parameters can be set:

      entropy_as_ubyte: bool, Default=False
        Should the entropy be normalized and returned as ubyte?
      blur_output_dtype: type|str|None, Default=None
        Sets the data type to which the blurred data should be
        converte before calculating the entropy
      filter_output_range: tuple|None, Default=None
        Sets the data range the filter function (when blurring)
        maximally has.
    entropy_q: multiprocessing.Queue
      The queue to push the entropy maps through
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    # handle deprecated parameters
    blur_as_int = params.pop('blur_as_int', None)
    if blur_as_int is not None:
        if blur_as_int:
            params['blur_output_dtype'] = "uint8"
        else:
            params['blur_output_dtype'] = "float64"
        warnings.warn("The parameter `blur_as_int` is deprecated, use "
                      f"`output_dtype` instead!\nUsing {blur_as_int=} leads to "
                      f"{params['output_dtype']=}",
                      category=DeprecationWarning)
    # ---
    with TimedTask() as timer:
        # this is only needed for the entropy part below
        view = params.get('view')
        blur_params = dict(
            source=params.get('source'),
            view=view,
            inner_view=params.get('inner_view'),
            categories=params.get('categories'),
            img_filter=params.get('img_filter'),
            filter_params=params.get('filter_params'),
            output_dtype=params.get('blur_output_dtype'),
            filter_output_range=params.get('filter_output_range')
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
            output_dtype="uint8" if entropy_as_ubyte else None,
        )
        # This would return the entropy data
        _ = runner_call(
            entropy_q,
            view_entropy,
            entropy_params
        )
    return timer
