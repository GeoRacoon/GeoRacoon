"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
from __future__ import annotations

import warnings

import numpy as np
import rasterio as rio

from copy import copy
from typing import Union

from collections.abc import Callable, Collection
from numpy.typing import NDArray
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

from .processing import (
    view_blurred,
    view_entropy,
    _view_filtered,
    view_interaction
)
from .filters.gaussian import compatible_border_size


def _combine_blurred_categories(output_params: dict, blur_q: Queue) -> TimedTask:
    # TODO: should we think about combining this a bit with "combine_views" in riogrande/parallel?
    #       Or, should we maybe leave them separate as the one in riogrande is really broad (no nodata etc)
    """
    Listen to a queue for blurred category blocks and write them to an output file.

    This function continuously listens to a queue of blurred category data blocks
    and writes each block to the specified output file. Each item received from
    the queue represents a spatial region containing multiple categorical bands.
    The function initializes the output raster according to the provided
    parameters and writes each band sequentially until a termination signal is
    received.

    Parameters
    ----------
    output_params :
        Configuration parameters for the output file. Expected keys include:

        - **as_dtype** : str or numpy.dtype
          The data type to cast written arrays to before saving.
        - **output_file** : str
          Path to the output file to be written.
        - **profile** : dict
          Raster I/O profile used for file creation (e.g., driver, width, height).
        - **nodata** : int or float, optional
          Value used to represent nodata in the output. If not provided,
          inherits from the profile or defaults to ``None``.
        - **count** : int, optional
          Number of bands in the output raster. If provided, overrides
          ``profile["count"]``.
        - **bigtiff** : bool, optional
          If ``True``, forces creation of a BigTIFF file. Defaults to ``False``.
        - **verbose** : bool, optional
          If ``True``, print progress and debug messages during execution.
          Defaults to ``False``.

    blur_q : Queue
        A multiprocessing or threading queue providing blurred data blocks.
        Each item in the queue is expected to be a dictionary with the following
        keys:

        - **signal** : str, optional
          Control signal (e.g., ``"kill"``) used to terminate processing.
        - **data** : dict[str, numpy.ndarray]
          Mapping of category names (band identifiers) to data arrays.
        - **view** : dict
          Metadata describing the spatial region corresponding to the data block
          (used to compute the write window).

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the duration of the operation.
        This can be used for profiling or performance analysis.

    Notes
    -----
    The function blocks while waiting for items in the queue and terminates
    gracefully when an item with ``signal="kill"`` is received. Proper queue
    management is essential to avoid deadlocks or unfinished writes.

    The `write_band` helper function is called for each band to handle writing
    and metadata tagging. Each output band is named ``LC_<band>`` for clarity.
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


def _combine_entropy_blocks(output_params: dict, entropy_q: Queue):
    # TODO: needs testing
    """
    Listen to a queue for entropy blocks and write them to an output file.

    This function continuously monitors a queue for entropy data blocks,
    writing them sequentially to an output file until a termination signal
    is received. It initializes the output file based on the provided
    configuration and ensures proper cleanup when the process ends.

    Parameters
    ----------
    output_params : dict
        Configuration parameters for the output file. Expected keys include:

        - **output_dtype** : str or numpy.dtype
          The data type of the output array.
        - **output_file** : str
          Path to the file where output data will be written.
        - **profile** : dict
          Dictionary of metadata or configuration options for the output.
        - **out_band** : Band, optional
          An existing `Band` object for writing output. If not provided,
          a new `Band` will be created and initialized.

    entropy_q : Queue
        A multiprocessing or threading queue that provides entropy blocks.
        Each item in the queue is expected to be a dictionary with the
        following keys:

        - **signal** : str, optional
          Control signal (e.g., ``"kill"``) to terminate the process.
        - **data** : numpy.ndarray
          The data block to be written to the output.
        - **view** : dict
          Metadata describing the spatial or logical region corresponding
          to the data block (used to compute the output window).

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the total time taken for the operation.
        This object can be used for profiling or performance analysis.

    Notes
    -----
    The function blocks while waiting for new items in the queue. It terminates
    gracefully when a dictionary with ``signal="kill"`` is received. Proper
    queue management is required to prevent deadlocks or resource leaks.
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


def _combine_interaction_blocks(output_params: dict, interaction_q: Queue):
    # TODO: needs testing
    """
    Listen to a queue for interaction blocks and write them to an output file.

    This function continuously monitors a queue for interaction data blocks,
    writing them sequentially to a specified output file until a termination
    signal is received. It initializes the output file according to the
    provided configuration parameters and handles tagged metadata output.

    Parameters
    ----------
    output_params : dict
        Configuration parameters for the output file. Expected keys include:

        - **output_dtype** : str or numpy.dtype
          The data type of the output array.
        - **output_file** : str
          Path to the file where output data will be written.
        - **profile** : dict
          Dictionary of metadata or configuration options for the output file.
        - **out_band** : Band, optional
          An existing `Band` instance to handle writing. If not provided,
          a new `Band` is created using `output_file`.
        - **output_tag** : dict, optional
          Metadata tags for the output file. Defaults to
          ``dict(category='interaction')``.
        - **verbose** : bool, optional
          If ``True``, print progress and debug messages. Defaults to ``False``.

    interaction_q : Queue
        A multiprocessing or threading queue that provides interaction blocks.
        Each item in the queue is expected to be a dictionary with keys:

        - **signal** : str, optional
          Control signal (e.g., ``"kill"``) to terminate processing.
        - **data** : numpy.ndarray
          The data block to be written to the output file.
        - **view** : dict
          Metadata describing the region of the output file corresponding
          to the data block (used for calculating the write window).

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the duration of the operation.
        This object can be used for profiling or performance analysis.

    Notes
    -----
    The function blocks while waiting for new items in the queue and terminates
    gracefully when a message with ``signal="kill"`` is received.
    Proper queue management is required to prevent deadlocks or resource leaks.
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


def _block_category_extraction(params: dict, blur_q: Queue) -> TimedTask:
    """
    Extract and filter category blocks for a specified raster view and push results to a queue.

    This function processes a single spatial block (or "view") from a categorical
    raster dataset, extracting per-category layers and optionally applying a
    spatial filter (e.g., Gaussian blur). The processed block is then pushed to
    a multiprocessing queue for downstream aggregation or writing.

    Parameters
    ----------
    params : dict
        Dictionary containing configuration and input data for processing a single
        raster block. Required keys include:

        - **source** : str
          Path to the source raster file to read from.
        - **view** : tuple(int, int, int, int)
          Outer window coordinates of the block to process, as
          ``(x, y, width, height)``.
        - **inner_view** : tuple(int, int, int, int)
          Usable region of the block (excluding padding or borders).
        - **as_dtype** : str or numpy.dtype
          Desired data type for the output blurred category arrays.
        - **output_range** : tuple(float, float)
          Target data range to map values into when converting to
          ``as_dtype``.
        - **categories** : list[str]
          List of category names or band identifiers to extract.
        - **img_filter** : Callable, optional
          A filter function to apply to each category layer (e.g.,
          ``skimage.filters.gaussian``).
        - **filter_params** : dict, optional
          Keyword arguments to pass to the filter callable.
        - **filter_output_range** : tuple(float, float), optional
          Expected numeric range of the filter’s output. If provided,
          used to scale or normalize the filtered results.

    blur_q : Queue
        A multiprocessing or threading queue to which the blurred, multi-band
        category maps are pushed. Each queue item will typically be a dictionary
        containing processed data arrays and metadata describing the view.

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the duration of the operation.
        Useful for profiling and performance monitoring.

    Notes
    -----
    This function serves as a wrapper that prepares parameters for
    ``view_blurred`` and dispatches the computation through ``runner_call``.
    It is intended for internal use within a multiprocessing workflow that
    coordinates reading, processing, and writing of raster data blocks.

    The queue consumer (e.g., ``combine_blurred_categories``) is responsible for
    collecting and writing the processed results to disk.
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


def block_heterogeneity(params: dict, entropy_q: Queue, blur_q: Queue) -> TimedTask:
    # TODO: I beleive this is not used anywhere actually - DELETE if Needed
    """
    Compute per-block heterogeneity measures based on entropy and push results to queues.

    This function processes a spatial block (or "view") from a categorical raster dataset,
    applying optional filtering and computing a heterogeneity measure based on Shannon
    entropy. The function produces two outputs:
    1. A blurred, multi-band representation of the categorical data pushed to ``blur_q``.
    2. A single-band entropy map pushed to ``entropy_q``.

    Parameters
    ----------
    params : dict
      Dictionary containing all relevant configuration data for processing a single
      raster block. Required keys include:

      - **source** : str
        Path to the input raster file.
      - **view** : tuple(int, int, int, int)
        The full extent of the block to process, as ``(x, y, width, height)``.
      - **inner_view** : tuple(int, int, int, int)
        The usable area of the block, excluding borders.
      - **img_filter** : Callable
        A filter function to apply to each category layer, such as
        ``skimage.filters.gaussian``.
      - **filter_params** : dict
        Parameters to pass to the filter callable.

      Optional keys include:

      - **entropy_as_ubyte** : bool, default=False
        If ``True``, normalize entropy values to the 0–255 range and return
        as unsigned bytes (``uint8``).
      - **blur_output_dtype** : str or numpy.dtype or None, default=None
        Data type to cast blurred data to before entropy computation.
      - **filter_output_range** : tuple(float, float) or None, default=None
        Expected numeric range of the filter’s output, used for normalization.
      - **categories** : list[str], optional
        List of category identifiers (bands) to include in the computation.

    entropy_q : Queue
      A multiprocessing or threading queue to which computed entropy maps
      (heterogeneity measures) are pushed. Each item typically includes a
      dictionary containing ``'data'`` and ``'view'`` fields.

    blur_q : Queue
      A multiprocessing or threading queue to which blurred category maps
      are pushed. Each item typically includes a dictionary mapping category
      names to filtered numpy arrays.

    Returns
    -------
    TimedTask
      A `TimedTask` instance tracking the duration of the operation.
      This can be used for profiling or performance monitoring.

    Notes
    -----
    This function acts as an internal worker for heterogeneity analysis workflows.
    It first produces blurred category data via the ``view_blurred`` function and
    then computes entropy using ``view_entropy``. Both are dispatched asynchronously
    using ``runner_call`` and communicate their results through queues.
    """
    # handle deprecated parameters
    blur_as_int = params.pop('blur_as_int', None)
    if blur_as_int is not None:
        if blur_as_int:
            params['blur_output_dtype'] = "uint8"
        else:
            params['blur_output_dtype'] = "float64"
        # TODO: fix deprecated use of parameters
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


def _block_entropy(params: dict, entropy_q: Queue) -> TimedTask:
    """
    Compute per-block heterogeneity measures based on entropy and push results to a queue.

    This function computes Shannon entropy for each spatial block (or "view") in a raster
    dataset, using pre-blurred or otherwise prepared category data. The resulting entropy
    map is pushed into a multiprocessing queue for aggregation or writing.

    Parameters
    ----------
    params : dict
        Configuration and data required for processing a single raster block.
        Must include the following keys:

        - **source** : str
          Path to the input raster file to process.
        - **view** : tuple(int, int, int, int)
          Full extent of the block to process, as ``(x, y, width, height)``.
        - **inner_view** : tuple(int, int, int, int)
          Usable portion of the block (excluding padding or borders).
        - **input_bands** : list[Band]
          List of `Band` objects providing access to category data for entropy
          computation. Each must support ``get_bidx()`` and ``get_data()`` methods.
        - **output_dtype** : str or numpy.dtype
          Data type for the entropy output (e.g., ``"float32"`` or ``"uint8"``).
        - **output_range** : tuple(float, float)
          Expected value range for the entropy output, used for normalization.

        Optional keys include:

        - **entropy_as_ubyte** : bool, default=False
          If ``True``, normalize entropy values to the 0–255 range and cast to
          unsigned bytes (``uint8``).
        - **normed** : bool, default=True
          Whether to normalize the category probabilities before computing entropy.
        - **max_entropy_categories** : int or None, default=None
          Maximum number of categories to consider in entropy normalization.

    entropy_q : Queue
        A multiprocessing or threading queue used to transmit entropy results.
        Each item added to the queue typically includes a dictionary with keys
        such as ``'data'`` (entropy array) and ``'view'`` (spatial metadata).

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the duration of the operation.
        Useful for profiling and performance monitoring.

    Notes
    -----
    This function extracts category data from the provided input bands within the
    specified window, computes entropy using ``view_entropy``, and sends the result
    to ``entropy_q`` via ``runner_call``.
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


def _block_interaction(params: dict, interaction_q: Queue) -> TimedTask:
    """
    Compute per-block interaction measures between categories and push results to a queue.

    This function processes a single spatial block (or "view") from a raster dataset,
    computing interaction measures between categories (e.g., co-occurrence or overlap)
    based on input band data. The computed interaction map is pushed to a
    multiprocessing queue for aggregation or writing.

    Parameters
    ----------
    params : dict
        Configuration and data required for processing a single raster block.
        Must include the following keys:

        - **source** : str
          Path to the input raster file to process.
        - **view** : tuple(int, int, int, int)
          Full extent of the block to process, as ``(x, y, width, height)``.
        - **inner_view** : tuple(int, int, int, int)
          Usable region of the block (excluding padding or borders).
        - **input_bands** : list[Band]
          List of `Band` objects providing access to category data.
          Each must implement ``get_bidx()`` and ``get_data()`` methods.
        - **output_dtype** : str or numpy.dtype
          Data type for the interaction output (e.g., ``"float32"`` or ``"uint8"``).
        - **output_range** : tuple(float, float)
          Expected range of output values used for normalization or scaling.

        Optional keys include:

        - **input_dtype** : str or numpy.dtype or None, default=None
          Data type of the input category arrays, if conversion is required.
        - **interaction_as_ubyte** : bool, default=False
          If ``True``, normalize the interaction values to the 0–255 range and
          cast to unsigned bytes (``uint8``).
        - **standardize** : bool, default=False
          Whether to standardize category arrays before computing interaction.
        - **normed** : bool, default=True
          Whether to normalize input data (e.g., probability normalization)
          before computing interaction.

    interaction_q : Queue
        A multiprocessing or threading queue used to transmit interaction results.
        Each item in the queue typically includes a dictionary containing the
        computed interaction data and associated spatial metadata.

    Returns
    -------
    TimedTask
        A `TimedTask` instance tracking the duration of the operation.
        Useful for profiling and performance monitoring.

    Notes
    -----
    This function extracts category data from the provided input bands within
    the specified window, computes an interaction measure via the
    ``view_interaction`` function, and sends the results to ``interaction_q``
    using ``runner_call``.
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


def compute_entropy(source: str | Source,
                    output_file: str,
                    block_size: tuple[int, int],
                    blur_params: dict,  # TODO: is only used to format output_file
                    categories: list | None = None,
                    output_dtype: type | str | None = None,
                    output_range: tuple | None = None,
                    # TODO: Do we want to infer for np.integer? (see compute entropy)
                    normed: bool = True,
                    max_entropy_categories: int | None = None,
                    verbose: bool = False,
                    **params):
    """
    Compute entropy-based heterogeneity from categorical raster bands.

    This function orchestrates a parallel computation of entropy-based
    heterogeneity across multiple raster blocks, combining per-block results
    into a single output GeoTIFF. Each block is processed independently via
    multiprocessing workers that read subsets of the input raster, compute
    entropy over selected categorical bands, and write results to a queue
    for aggregation.

    Parameters
    ----------
    source : str or Source
        Path to the categorical raster file (e.g., land-cover map) or an
        initialized `Source` object providing access to the dataset.
    output_file : str
        Path where the final entropy GeoTIFF will be written.
    block_size : tuple of int
        Block dimensions ``(width, height)`` in pixels. Determines the size
        of the chunks processed by each worker.
    blur_params : dict
        Dictionary of parameters for Gaussian blur or other preprocessing.
        Typically includes at least one of ``'sigma'`` or ``'diameter'`` in
        spatial units (e.g., meters). Used primarily for output file naming.
    categories : list, optional
        List of category names or identifiers to include in entropy
        computation. If not provided, all available categories in the source
        raster are used.
    output_dtype : str, type, or None, optional
        Desired data type for the output entropy raster. If not provided,
        defaults to ``numpy.uint8``.
    output_range : tuple of float, optional
        Minimum and maximum output values. If not set, the range is inferred
        from ``output_dtype`` — [0, 1] for floating-point types, or the
        valid range for integer types (e.g., [0, 255] for uint8).
    normed : bool, default=True
        Whether to normalize the entropy values. When True, entropy is scaled
        relative to the maximum possible entropy determined by the number of
        categories.
    max_entropy_categories : int, optional
        Specifies the number of categories to assume for maximum entropy
        normalization. Ignored if ``normed=False``.
    verbose : bool, default=False
        If True, prints progress updates and diagnostic information.
    **params
        Additional optional parameters controlling multiprocessing and output:

        - **nbrcpu** : int
          Number of CPU cores to use. Defaults to the number of available
          cores minus one.
        - **start_method** : str
          Multiprocessing start method (e.g., ``"spawn"`` or ``"fork"``).
        - **entropy_as_ubyte** : bool, optional
          Deprecated. Use ``output_dtype="uint8"`` instead.
        - **compress** : bool, default=False
          If True, compresses the final GeoTIFF using LZW compression.

    Returns
    -------
    output_file : str
        Path to the resulting entropy raster file.

    Notes
    -----
    - Each processing block computes entropy over the given categories using
      the internal `_block_entropy` function and sends results to a queue.
    - The function `_combine_entropy_blocks` merges these intermediate
      results into a single raster file.
    - The operation can be parallelized across multiple CPUs to handle
      large raster datasets efficiently.
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
                       max_entropy_categories=max_entropy_categories, )
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
        entropy_combiner = pool.apply_async(_combine_entropy_blocks,
                                            (entropy_output_params, entropy_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(_block_entropy,
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
    """
    Compute interaction strength between categorical raster bands.

    This function computes pairwise, three-way, or higher-order interaction
    measures among categorical raster bands (e.g., land-cover classes).
    The computation is performed block-wise using multiprocessing, where
    each worker processes a subset of the raster and pushes intermediate
    results to a queue. The results are then combined into a single output
    raster representing spatial interaction strength.

    Parameters
    ----------
    source : str or Source
        Path to the categorical raster file (e.g., a land-cover map) or an
        initialized `Source` object providing access to the dataset.
    output_file : str
        Path where the final interaction raster will be saved.
    block_size : tuple of int
        Block dimensions ``(width, height)`` in pixels that define the size
        of the chunks processed by each multiprocessing worker.
    blur_params : dict
        Dictionary containing parameters for Gaussian blur or other
        preprocessing. Used primarily for formatting the output filename.
        Should include either ``'sigma'`` or ``'diameter'`` in spatial units.
    categories : list, optional
        List of categories (e.g., land-cover types) to include in the
        interaction computation. If not provided, all categories available
        in the source raster are used.
    output_dtype : str, type, or None, optional
        Desired data type for the output raster. If not provided, defaults
        to ``numpy.uint8``.
    output_range : tuple of float, optional
        Minimum and maximum values for scaling the output. If not specified,
        inferred automatically from ``output_dtype``—[0, 1] for floats, or
        the valid integer range for integer types (e.g., [0, 255] for uint8).
    standardize : bool, default=False
        Whether to standardize the data prior to computing interactions.
        This can improve comparability across categories with different
        magnitudes or distributions.
    normed : bool, default=True
        If True, normalizes the computed interaction strengths relative to
        their theoretical maximum values.
    verbose : bool, default=False
        If True, prints detailed progress and diagnostic messages during
        processing.
    **params
        Additional optional parameters controlling multiprocessing and output:

        - **nbrcpu** : int
          Number of CPU cores to use. Defaults to the number of available
          cores minus one.
        - **start_method** : str
          Multiprocessing start method (e.g., ``"spawn"`` or ``"fork"``).
        - **interaction_as_ubyte** : bool, optional
          Deprecated. Use ``output_dtype="uint8"`` instead.
        - **compress** : bool, default=False
          If True, compresses the final GeoTIFF using LZW compression.

    Returns
    -------
    output_file : str
        Path to the resulting interaction raster file.

    Notes
    -----
    - The computation proceeds in parallel by dividing the raster into
      independent processing blocks.
    - Each worker executes an internal `_block_interaction` task and writes
      its results to a multiprocessing queue.
    - The `_combine_interaction_blocks` function merges all block results
      into a single output file.
    - Designed for efficient large-scale raster analysis on multi-core
      systems.
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
                       normed=normed, )
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
        interaction_combiner = pool.apply_async(_combine_interaction_blocks,
                                                (interaction_output_params, interaction_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(_block_interaction,
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


# TODO: check how much extract_categories and apply_filter are redundant
# TODO: I was thinking to delete this and stay with apply_filter
def extract_categories(source: str | Source,
                       categories: list,
                       output_file: str,
                       block_size: tuple[int, int],
                       img_filter: None | Callable = None,
                       filter_params: dict | None = None,
                       output_dtype: type | str | None = None,
                       output_params: None | dict = None,
                       filter_output_range: tuple | None = None,
                       verbose: bool = False,
                       **params):
    """
    Extract per-category maps from a raster, optionally apply a filter, and export to a file.

    This function processes a categorical raster by separating specified categories
    into individual bands. Optionally, a filter function (e.g., Gaussian smoothing)
    can be applied to each category band. The processing is performed block-wise
    with multiprocessing, and the results are combined into a single output raster.

    Parameters
    ----------
    source : str or Source
        Path to the input `.tif` file or a `Source` object providing access to it.
    categories : list
        List of categorical values to separate into individual bands.
    output_file : str
        Path where the resulting raster will be written.
    block_size : tuple of int
        Size of processing blocks in pixels as ``(width, height)``.
    img_filter : callable, optional
        A function to apply to each category band (e.g., ``skimage.filters.gaussian``).
        If None, no filtering is applied and ``filter_params`` is ignored.
    filter_params : dict, optional
        Parameters to pass to `img_filter`. Required if `img_filter` is provided.
    output_dtype : str or type, optional
        Data type of the output bands. Deprecated; use ``output_params['as_dtype']`` instead.
    output_params : dict, optional
        Dictionary of output settings:
        - **as_dtype** : data type for the filtered output (overrides `output_dtype`)
        - **nodata** : value used for missing data (default: None)
        - **bigtiff** : bool, whether to create a BIGTIFF for >4GB files
        - **output_range** : tuple, output value range for data conversion
    filter_output_range : tuple, optional
        Expected value range of the filtered data. Recommended when `img_filter` is used.
    verbose : bool, default=False
        Print processing information and progress.
    **params
        Additional optional arguments:
        - **nbrcpu** : int, number of CPU cores to use (default: available cores minus one)
        - **start_method** : str, multiprocessing start method (e.g., 'spawn' or 'fork')
        - **compress** : bool, if True, compress the final output with LZW

    Returns
    -------
    output_file : str
        Path to the resulting raster file containing extracted and optionally filtered category bands.

    Notes
    -----
    - Deprecated parameters (`blur_as_int`, `output_dtype`) are internally
      mapped to `output_params['as_dtype']` with warnings.
    - Borders for filtering are automatically computed based on the filter kernel.
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
        blur_combiner = pool.apply_async(_combine_blurred_categories,
                                         (blur_output_params, blur_q))

        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(_block_category_extraction,
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


# TODO: This is actually the better function (allows for float etc) - we should keep this (or both)
def apply_filter(source: str | Source,
                 output_file: str,
                 block_size: tuple[int, int],
                 bands: list[Band] | None = None,
                 data_in_range: None | NDArray | Collection = None,
                 data_as_dtype: type | None = np.uint8,
                 data_output_range: None | NDArray | Collection = None,
                 replace_nan_with: Union[int, float] | None = None,
                 img_filter=None,
                 filter_params: dict | None = None,
                 filter_output_range: Collection | None = (0., 1.),
                 output_dtype: type | None = np.uint8,
                 output_range: tuple | None = None,
                 selector_band: Band | None = None,
                 verbose: bool = False,
                 output_params: None | dict = None,
                 **params
                 ) -> str:
    """
    Apply a filter to one or more bands of a raster and export the result.

    This function processes the raster in blocks to allow for memory-efficient
    and parallelized computation. Each block is filtered and then combined into
    a single output raster. Optionally, categorical masking can be used with
    `selector_band`.

    Parameters
    ----------
    source : str or Source
        Path to the input raster file or a `Source` object.
    output_file : str
        Path where the filtered raster will be written.
    block_size : tuple of int
        Size of the processing blocks in pixels as ``(width, height)``.
    bands : list of Band, optional
        Specific bands to process. If None, all bands in the raster are used.
    data_in_range : array-like, optional
        Input range used for rescaling loaded data before filtering.
    data_as_dtype : type, optional, default=np.uint8
        Data type to which input data is converted before filtering.
    data_output_range : array-like, optional
        Output range for data rescaling after conversion to `data_as_dtype`.
    replace_nan_with : int or float, optional
        Value used to replace NaNs in the input before filtering.
    img_filter : callable, optional
        Filter function to apply to the data (e.g., `skimage.filters.gaussian`).
    filter_params : dict, optional
        Parameters to pass to `img_filter`.
    filter_output_range : collection, default=(0., 1.)
        Expected output range of the applied filter function.
    output_dtype : type, optional, default=np.uint8
        Data type of the filtered output raster.
    output_range : tuple, optional
        Value range to rescale the final output raster.
    selector_band : Band, optional
        Optional categorical band used as a mask to apply the filter
        selectively across categories.
    output_params : dict, optional
        Additional output configuration:
        - **as_dtype** : data type for filtered output (overrides `output_dtype`)
        - **nodata** : value for missing data (default: None)
        - **bigtiff** : bool, whether to create a BIGTIFF for >4GB files
          TODO: We can think about either making this standard or allowing for the rasteiro implementation from GDAL:
             https://rasterio.readthedocs.io/en/latest/topics/image_options.html
    verbose : bool, default=False
        Print progress and debug information.
    **params
        Additional optional arguments:
        - **nbrcpu** : int, number of CPU cores for multiprocessing.
        - **start_method** : str, multiprocessing start method.
        - **compress** : bool, whether to compress the final output with LZW.

    Returns
    -------
    output_file : str
        Path to the resulting filtered raster file.

    Notes
    -----
    - Deprecated parameters (`output_dtype` vs `output_params['as_dtype']`) are internally handled with warnings.
    - Borders for filtering are automatically computed from the filter kernel.
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
        assert len(sources) == 1, "Only bands with the same source are " \
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
            func=_combine_blurred_categories,
            kwds=dict(output_params=blur_output_params, blur_q=blur_q)
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(
                func=runner_call,
                kwds=dict(queue=blur_q,
                          callback=_view_filtered,
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
