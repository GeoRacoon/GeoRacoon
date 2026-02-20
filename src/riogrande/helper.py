"""This file provides helper functions including compatibility checks, dtype conversion, parallelization setup
"""

from __future__ import annotations

import os
import json
import warnings

import numpy as np
from numpy.typing import NDArray

import rasterio as rio
from rasterio.windows import Window

from decimal import Decimal
from typing import Any, Union, Tuple, Optional

from collections.abc import Collection

import multiprocessing as mpc
from multiprocessing import context as _context_module

MPC_STARTER_METHODS = ['spawn', 'fork', 'forkserver']


def get_nbr_workers(number: Optional[int] = None) -> int:
    """Determine the number of worker processes to use in mulitprocessing.

    Parameters
    ----------
    number: int or None, optional
        Desired number of workers. If ``None``, the function will use the
        number of CPUs available via :func:`multiprocessing.cpu_count`,
        but never less than 2.

    Returns
    -------
    int
        Number of workers to use (always `>= 2`).

    Notes
    -----
    A warning is emitted when a requested ``number`` is lower than 2 and the
    request is ignored setting the number of used workers to 2.

    See Also
    --------
    :func:`~riogrande.helper.get_or_set_context` : Return a multiprocessing context.
    """
    _min_count = 2  # Hardcoded: some parallelization routines fail when < 2
    if number is None:
        _use = max(_min_count, mpc.cpu_count())
    elif number <= _min_count:
        warnings.warn(
            message=f"For this routine to work properly at least {_min_count} "
                    f"workers are required - the requested {number} are not "
                    "enough and thus the request will be ignored.",
            category=RuntimeWarning
        )
        _use = _min_count
    else:
        _use = int(number)
    return _use


def get_or_set_context(method: Optional[str] = None) -> _context_module.BaseContext:
    """
    Return a multiprocessing context and set the global start method if unset.

    The function tries to be conservative about changing global interpreter state:
    - If `method` is None, it returns a context for the currently configured
      global start method when one exists; otherwise it warns and returns a
      context for a sensible default ('spawn' is used to establish
      compatibility with windows).
    - If `method` is provided and no global start method is set, it attempts to
      set the global start method to `method`. If that attempt races with
      another thread/process, it falls back to returning a context for `method`
      without changing the global start method.
    - If `method` is provided and a different global start method is already
      set, the global start method is not changed; a warning is emitted and a
      context for the requested `method` is returned so callers can still
      create objects using the requested start semantics.

    Parameters
    ----------
    method : {None, 'fork', 'spawn', 'forkserver'}, optional
        Desired multiprocessing start method to use for the returned context.
        If ``None`` the function will:
        - return a context for the currently configured global start method if
          one exists, or
        - emit a ``RuntimeWarning`` and return a context for the configured
          default method (``spawn``) if no global method is set.
        Valid explicit values are ``'fork'``, ``'spawn'`` and ``'forkserver'``
        (availability depends on the platform and Python build). Passing an
        unsupported value raises ``ValueError``.

    Returns
    -------
    multiprocessing.context.BaseContext
        A multiprocessing context object appropriate for creating
        :class:`multiprocessing.Process`, :class:`multiprocessing.pool.Pool`
        and related objects. The returned context will use the start method
        determined by the logic described above. The function always returns a
        context and never mutates an already-set global start method to a
        different value.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported start methods or ``None``.
    RuntimeError
        If the function attempts to set the global start method and the call to
        ``multiprocessing.set_start_method`` raises ``RuntimeError`` for reasons
        other than a race (this is rare); in normal race cases the function
        catches the ``RuntimeError`` and falls back to returning the requested
        context.

    Notes
    -----
    - Calling ``multiprocessing.set_start_method`` can only be done once per
      interpreter process. Once the global start method is set, it cannot be
      changed without restarting the interpreter. This function therefore
      avoids forcibly overwriting an existing different global start method.
    - The returned context is safe to use even when the global start method
      differs, because context objects encapsulate start semantics for the
      created processes independently of global state.
    - On Windows the only available start method is ``'spawn'``; on Unix-like
      systems ``'fork'`` and ``'spawn'`` are commonly available and
      ``'forkserver'`` may be available depending on the platform.
    - Use this helper in library code when you need a guaranteed context but
      do not want to unconditionally mutate global multiprocessing state.

    See Also
    --------
    :func:`~riogrande.helper.get_nbr_workers` : Determine the number of worker processes.

    Examples
    --------
    >>> ctx = get_or_set_context('spawn')
    >>> with ctx.Process(target=worker) as p:
    >>>     p.start()
    >>>     p.join()
    """
    allowed = MPC_STARTER_METHODS + [None, ]
    default_method = MPC_STARTER_METHODS[0]  # default is 'spawn'
    if method not in allowed:
        raise ValueError(f"Unsupported start method: {method!r}")

    # get the current context
    _context = mpc.get_start_method(allow_none=True)

    if _context is None:
        # if method is not None, set the global method and the current context
        if method is not None:
            try:
                mpc.set_start_method(method)  # set starting method
            except RuntimeError:
                # concurrent set; warn and ignore the global context
                warnings.warn(
                    "Race when setting start method; returning requested context.",
                    RuntimeWarning)
            finally:
                _context = method
        else:  # both the gloabl context and method are None:
            # avoid setting the global context, only set locally
            warnings.warn(
                "No multiprocessing start method set and no global either"
                f"— defaulting to local context only with '{default_method}'.",
                RuntimeWarning
            )
            _context = default_method
    else:  # global context is set already
        if method is not None:
            if method != _context:
                warnings.warn(
                    f"Global multiprocessing start method is '{_context}'"
                    f" but requested context is '{method}'"
                    f"— using local context only with '{method}'"
                    "keeping the global unchanged.",
                    RuntimeWarning
                )
                _context = method
            else:  # simply use _context
                pass
        else:  # global is set local in None > use global (_context)
            pass
    # print(f"{mpc.get_start_method()=}")
    # print(f"{_context=}")
    return mpc.get_context(_context)


def serialize(tags: dict[str, Any]) -> dict[str, str]:
    """Convert the values of a dict into JSON

    Each value is serialized using :func:`json.dumps`.

    Parameters
    ----------
    tags:
        Dictionary of tags with string keywords and any-type values,
        which are serializable.

    Returns
    -------
    dict
        Dictionary with tag as key and serialized value as value.

    See Also
    --------
    :func:`~riogrande.helper.deserialize` : Inverse operation; parse JSON back to Python objects.
    :func:`~riogrande.helper.sanitize` : Serialize then deserialize in one step.
    """
    return {tag: json.dumps(obj=value) for tag, value in tags.items()}


def deserialize(tags: dict[str, str]) -> dict[str, Any]:
    """Reads python objects from JSON-encoded values of a dict

    Each value is parsed using :func:`json.loads`.

    Parameters
    ----------
    tags:
        Dictionary with tag as key and serialized values.

    Returns
    -------
    dict
        Dictionary with tag as key and deserialized value as value.

    Notes
    ------
    Inverse operation of :func:`~riogrande.helper.serialize`.

    See Also
    --------
    :func:`~riogrande.helper.serialize` : Convert dict values to JSON strings.
    :func:`~riogrande.helper.sanitize` : Serialize then deserialize in one step.
    """
    return {tag: json.loads(s=value)
            for tag, value in tags.items()}


def sanitize(tags: dict[str, Any]) -> Any:
    """Serializes then deserializes values of a dict

    Convenience wrapper that calls :func:`~riogrande.helper.serialize`
    followed by :func:`~riogrande.helper.deserialize`, ensuring values are
    in the same form they would be when loaded back from a ``.tif`` tag.

    Parameters
    ----------
    tags:
        Dictionary with tag as key and serializable value as value.

    Returns
    ---------
    dict
        Dictionary with tag as key and deserialized value as value.

    See Also
    --------
    :func:`~riogrande.helper.serialize` : Convert dict values to JSON strings.
    :func:`~riogrande.helper.deserialize` : Parse JSON strings back to Python objects.
    """
    return deserialize(serialize(tags))


def match_all(targets: dict, tags: dict) -> bool:
    """Check if all tags in targets are present in tags

    Parameters
    ----------
    targets:
        Dictionary with tags to match to.
    tags:
        Dictionary with tags to check for matching items.

    Returns
    ---------
    bool
        True if all tags in targets are present in tags, otherwise False.

    See Also
    --------
    :func:`~riogrande.helper.match_any` : Return True if *any* tag matches.
    """
    match = True
    for t, v in targets.items():
        if not match:
            break
        if t in tags:
            if tags[t] == v:
                match = True
            else:
                match = False
        else:
            match = False
    return match


def match_any(targets: dict, tags: dict) -> bool:
    """Check if any tag in targets is present in tags

    Parameters
    ----------
    targets:
        Dictionary with tags to match to.
    tags:
        Dictionary with tags to check for matching items.

    Returns
    ---------
    bool
        True if any tags in targets are present in tags, otherwise False.

    See Also
    --------
    :func:`~riogrande.helper.match_all` : Return True only if *all* tags match.
    """
    match = False
    for t, v in targets.items():
        if match:
            break
        if t in tags:
            if tags[t] == v:
                match = True
            else:
                match = False
        else:
            match = False
    return match


def view_to_window(view: None | tuple[int, int, int, int]) -> Window:
    """Conerts a view into a rasterio Window

    Parameters
    ----------
    view:
      tuple (x, y, width, height) defining the view of the data array to update

    Returns
    ---------
    :class:`rasterio.windows.Window`
        Rasterio window object, or ``None`` if `view` is ``None``.
    """
    if view is not None:
        window = Window(view[0], view[1], view[2], view[3])
    else:
        window = None
    return window


def check_units(*sources: str) -> list:
    """Assert that all sources have the same linear units in the coordinate reference system (crs)

    Parameters
    ----------
    sources:
        List of sources (paths to files) from which units are to be compared to each other.

    Returns
    ---------
    list
        All unique units in a list.

    See Also
    --------
    :func:`~riogrande.helper.check_crs` : Check that sources share the same CRS.
    :func:`~riogrande.helper.check_resolution` : Check that sources share the same resolution.
    :func:`~riogrande.helper.check_compatibility` : Run all three checks at once.
    """
    units = []
    for source in sources:
        with rio.open(source) as src:
            crs = src.profile['crs']
            if crs is not None:
                units.append(src.profile['crs'].linear_units.lower())
            else:
                units.append(None)
            if len(set(units)) != 1:
                raise TypeError(f"{source=} has linear units {units[-1]}, "
                                "which is different from the other(s) "
                                f"({units[0]})")
    return units


def check_crs(*sources: str) -> list:
    """Assert that all the sources have the same coordinate reference system (crs)

    Parameters
    ----------
    sources:
        List of sources (paths to files) from which crs are to be compared to each other.

    Returns
    ---------
    list
        All unique crs from sources in a list.

    See Also
    --------
    :func:`~riogrande.helper.check_units` : Check that sources share the same linear units.
    :func:`~riogrande.helper.check_resolution` : Check that sources share the same resolution.
    :func:`~riogrande.helper.check_compatibility` : Run all three checks at once.
    """
    crss = []
    for source in sources:
        with rio.open(source) as src:
            crss.append(str(src.profile.get('crs', None)))
            if len(set(crss)) != 1:
                raise TypeError(f"{source=} has crs {crss[-1]}, which is "
                                f"different from the other(s) ({crss[0]})")
    return crss


def check_resolution(*sources: str) -> list:
    """Assert that all the sources have the same spatial resolution

    Parameters
    ----------
    sources:
        List of sources (paths to files) from which resolutions are to be compared to each other.

    Returns
    ---------
    list
        All unique resolutions from sources in a list.

    See Also
    --------
    :func:`~riogrande.helper.check_units` : Check that sources share the same linear units.
    :func:`~riogrande.helper.check_crs` : Check that sources share the same CRS.
    :func:`~riogrande.helper.check_compatibility` : Run all three checks at once.
    """
    ress = []
    for source in sources:
        with rio.open(source) as src:
            # NOTE: we round 8th digit after the comma here
            ress.append(tuple(map(lambda x: round(x, 8), src.res)))
            if len(set(ress)) != 1:
                raise TypeError(f"{source=} has resolution {ress[-1]}, which "
                                f"is different from the other(s) ({ress[0]})")
    return ress


def check_compatibility(*sources: str) -> Tuple[list, list, list]:
    """Assert that all the sources are compatible with each other.

    The checks include:
        - crs (via :func:`~riogrande.helper.check_crs`)
        - units (via :func:`~riogrande.helper.check_units`)
        - resolution (via :func:`~riogrande.helper.check_resolution`)

    Parameters
    ----------
    sources:
        List of sources (paths to files) from which are to be compared to each other.

    Returns
    ---------
    crss:
        All unique crs from sources in a list (see :func:`~riogrande.helper.check_crs`).
    units:
        All unique units from sources in a list (see :func:`~riogrande.helper.check_units`).
    ress:
        All unique resolutions from sources in a list (see :func:`~riogrande.helper.check_resolution`).

    See Also
    --------
    :func:`~riogrande.helper.check_crs` : Check that sources share the same CRS.
    :func:`~riogrande.helper.check_units` : Check that sources share the same linear units.
    :func:`~riogrande.helper.check_resolution` : Check that sources share the same resolution.
    """
    units = check_units(*sources)
    crss = check_crs(*sources)
    ress = check_resolution(*sources)
    return crss, units, ress


def output_filename(base_name: str, out_type: str, blur_params: None | dict = None) -> str:
    """Construct the filename for the specific output type.

    Parameters
    ----------
    base_name: str
      The basic output name in the form <name>.tif
    out_type: str
      The type of output that will be saved.
      This should be either 'blur' or 'entropy' but any string is accepted
    blur_params: dict
      Output of `get_blur_params`, so 'sigma', 'truncate' and 'diameter'
      are expected keys.

    Returns
    ------
    str:
      The resulting filename of the form
      '<name>_<out_type>_sig_<{sigma}>_diam_<{diameter}>_trunc_<{truncate}>.tif'
    """
    _base_name, _ext = os.path.splitext(base_name)
    _blur_string = ""
    if blur_params is not None:
        for name, value in blur_params.items():
            _blur_string += f"_{name}_{round(value)}"
    return f"{_base_name}_{out_type}{_blur_string}{_ext}"


def dtype_range(dtype: type | str) -> Tuple[int | float, int | float]:
    """Get the range of the specified dtype

    Uses :func:`numpy.iinfo` for integer types and :func:`numpy.finfo`
    for floating-point types.

    .. warning::
      This functions returns min or max as either `int` or `floats`.

      Be sure to convert them back into `dtype` if needed!

    Parameters
    ----------
    dtype:
        A NumPy dtype (e.g. ``np.uint8``, ``np.float32``) or a string
        representation thereof (e.g. ``'uint8'``).

    Returns
    -------
    tuple
        ``(max, min)`` of the dtype's representable range as Python
        ``int`` or ``float``.

    Raises
    ------
    ValueError
        If `dtype` has no defined min/max values.

    See Also
    --------
    :func:`~riogrande.helper.convert_to_dtype` : Convert and optionally rescale an array.
    """
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    # avoid issues of object not callable from rasterio
    elif hasattr(dtype, 'type'):
        dtype = dtype.type
    try:
        _max = int(np.iinfo(dtype).max)
        _min = int(np.iinfo(dtype).min)
    except ValueError:
        try:
            _max = float(np.finfo(dtype).max)
            _min = float(np.finfo(dtype).min)
        except ValueError:
            raise ValueError(f"{dtype=} has no min-/maximal values.")
    return _max, _min


def convert_to_dtype(data: NDArray, as_dtype: None | type | np._dtype | str = None,
                     in_range: None | NDArray | Collection = None,
                     out_range: None | NDArray | Collection | str | type = None) -> NDArray:
    """Converts data to `as_dtype` and optionally rescales it.

    Rescaling is done only if at least one of the ranges is explicitly set.
    If only `in_range` is set then the input range is scaled to the full range
    of the  output data type, `ad_dtype`.
    This behaviour is typically wanted when converting some floating typed data
    in a limited range, e.g. [0, 1] to unsigned integer, e.g. `uint8`, thus
    mapping the range [0,1] to [0, 255].

    In case only `out_range` is set, the full data type range of the input
    data is mapped to the provided `out_range`.
    This is typically used if converting from a "limited" range, like `uint8`
    to a floating data type.

    .. note::

      The default range for any floating type is `[0,1]`!

      This means:

      - If the output data type, `as_dtype` is any subclass of `np.floating`
        and no `out_range` is defined then the output is scaled to the intervarl
        `[0, 1]`.
      - If data is of any `np.floating` type and the data range lies withing
        `[0, 1]` (and `in_range` is not provided) then `in_range` is set to be
        `[0, 1]`.

    Parameters
    ----------
    data:
        Input numpy NDArray
    as_dtype: desired data type to convert to (e.g. np.float64)
        If not provided then at least the `out_range` needs to be set in
        which case the data type remains unchanges, but the data is
        rescaled.
    in_range:
        an array or list from which min and max will be used as input range.
        Min and max are read with :func:`numpy.nanmin` / :func:`numpy.nanmax`.

        .. note::
          You might simply provide the same value as for `data` in order to
          use its min an max for scaling

    out_range:
      an array or list from which min and max will be used as limits for the
      output.
      Alternatively, a data type can be specified, in which case the data
      will be scaled to the full range of the specified data type
      (see :func:`~riogrande.helper.dtype_range`).

    Returns
    ----------
    NDArray
        Converted numpy NDArray with desired data type.

    See Also
    --------
    :func:`~riogrande.helper.dtype_range` : Get the min/max of a NumPy dtype.

    Examples
    --------
    >>> # simple conversion, no rescalingm
    >>> my_data = np.array([0, 0.5, 1.], dtype=np.float64)
    >>> convert_to_dtype(my_data, as_dtype='uint8')
    array([0, 0, 1], dtype=uint8)
    >>> # conversion with rescaling specifying in_range only
    >>> new_data = convert_to_dtype(my_data, as_dtype='uint8', in_range=(0,1))
    >>> new_data
    array([  0, 127, 255], dtype=uint8)
    >>> # convert with scaling specifying out_range only
    >>> convert_to_dtype(data=new_data, as_dtype='float64', out_range=[-1, 1])
    array([-1.        , -0.00392157,  1.        ])
    >>> # only scaling, keeping data type
    >>> convert_to_dtype(data=my_data, in_range=[0,1], out_range=[-1, 1])
    array([-1.,  0.,  1.])
    >>> # scaling with data type as range
    >>> convert_to_dtype(data=my_data, in_range=[0,1], as_dtype='uint16', out_range='uint8')
    array([  0, 127, 255], dtype=uint16)
    """
    # convert to numpy dtype if string was provided
    if isinstance(as_dtype, str):
        as_dtype = np.dtype(as_dtype)

    in_dtype = data.dtype

    rescale = False
    if in_range is not None:
        rescale = True
        if isinstance(in_range, (str, type)):
            # we have a data type
            _inmax, _inmin = dtype_range(in_range)
        else:
            _inmax = float(np.nanmax(in_range))
            _inmin = float(np.nanmin(in_range))
    else:
        # us the full rang of input data type if scaling should be done
        _inmax, _inmin = dtype_range(in_dtype)

    if out_range is not None:
        if not rescale:
            # use the full range in case
            _inmax, _inmin = dtype_range(in_dtype)
        rescale = True
        if isinstance(out_range, (str, type)):
            # we have a data type
            _outmax, _outmin = dtype_range(out_range)
        else:
            _outmax = float(np.nanmax(out_range))
            _outmin = float(np.nanmin(out_range))
    elif rescale:
        # no output range but rescale due to input range
        # use the full range of output data type if scaling should be done
        _outmax, _outmin = dtype_range(as_dtype)
    if rescale:
        # we rescale
        if out_range is None and np.issubdtype(as_dtype, np.floating):
            # we are about to map something to the full float range, this is rather
            # unlikely done on purpose
            warnings.warn(
                f"You are about to rescale data of type '{in_dtype}' in range "
                f"[{_inmin}, {_inmax}] to the full range of '{as_dtype}'. "
                "Consider specifying `out_range` to avoid this."
            )
        # first get the scaling factor
        scale = (Decimal(_outmax) - Decimal(_outmin)) / \
                (Decimal(_inmax) - Decimal(_inmin))
        # now rescale
        out_data = np.array(_outmin).astype(as_dtype) + ((data - _inmin) * float(scale)).astype(as_dtype)

        outmax = float(np.nanmax(out_data))
        outmin = float(np.nanmin(out_data))
        if outmax > _outmax or outmin < outmin:
            warnings.warn(
                f"The rescaled data (range [{outmin}, {outmax}]), exceeds the "
                f"determined output range [{_outmin}, {_outmax}]. "
                "If this is unwanted make sure that the input data does not "
                "exceed the `in_range`."
            )
    else:
        # we simply change the data type - no rescaling
        out_data = data.astype(as_dtype)
    return out_data


def aggregated_selector(masks: list[NDArray], logic: str = 'all') -> NDArray:
    """Turns several rasterio masks into a boolen selector for a numpy array

    Rasterio masks are uint8 numpy arrays where every value > 0 is considered
    a valid cell

    Parameters
    ----------
    masks:
        Arbitrary number of numpy arrays resulting from
        :meth:`rasterio.io.DatasetReader.dataset_mask` or
        :meth:`rasterio.io.DatasetReader.read_masks`.
    logic:
        Determines how the aggreagation should happen.
        If ``'all'`` (the default) a cell is only selected if **all** masks
        consider it valid data — aggregated via :func:`numpy.logical_and`.
        ``'any'`` selects cells which **at least one** mask considers valid
        — aggregated via :func:`numpy.logical_or`.

    Returns
    ----------
    NDArray
        Boolean numpy array as result of logical mask applied.

    See Also
    --------
    :func:`~riogrande.helper.reduced_mask` : Compute a mask from nodata values across bands.
    """
    selector = masks[0] != 0  # values > 0 are selected (i.e. True)
    if logic == 'any':
        _logic = np.logical_or
    else:
        _logic = np.logical_and
    if len(masks) > 1:
        for mask in masks[1:]:
            _logic(selector, mask != 0, out=selector)
    return selector


def reduced_mask(array: NDArray, nodata: float | int | np.nan = 0, logic: str = 'all') -> NDArray:
    """Computes a mask based on the value of several bands

    Parameters
    ----------
    array:
        3D array holding multiple bands of map data
    nodata:
        Nodata value to use. Defaults to 0.
        Pass :data:`numpy.nan` to mask NaN cells (detected via :func:`numpy.isnan`).
    logic:
        Allowed strings are:

        - ``"all"`` : Masked will be each cell for which **all** bands match the nodata value
          (aggregated via :func:`numpy.logical_or` across bands).
        - ``"any"`` : Masked will be each cell for which **any** band matches the nodata value
          (aggregated via :func:`numpy.logical_and` across bands).

    Returns
    ----------
    NDArray
        Boolean numpy array resulting from applied logic.

    See Also
    --------
    :func:`~riogrande.helper.aggregated_selector` : Aggregate rasterio band masks into a selector.

    Examples
    --------
    >>> mydata = np.array([[[2, 4], [0, 1]], [[5, 5], [1, 0]]])
    >>> # only mask if all are nodata
    >>> reduced_mask(mydata)
    array([[1, 1],
           [1, 1]], dtype=uint8)
    >>> # mask if any are nodata
    >>> reduced_mask(mydata, logic='any')
    array([[1, 1],
           [0, 0]], dtype=uint8)
    """
    if logic == 'any':
        _logic = np.logical_and
    else:
        _logic = np.logical_or
    if np.isnan(nodata):
        return _logic.reduce(array=~np.isnan(array), axis=0).astype(np.uint8)
    else:
        return _logic.reduce(array=array != nodata, axis=0).astype(np.uint8)


def count_contribution(data: NDArray, selector: NDArray[np.bool_], no_data: Union[int, float] = 0) -> int:
    """The remaining number of data cells when applying the selector

    Uses :func:`numpy.unique` with ``return_counts=True`` to count valid cells.

    Parameters
    ----------
    data:
      The data to cont the contribution in
    selector:
      A boolean array in the shape of `data` selecting the single cells that
      should be considered
    no_data:
      The value that should be considered as invalid.

      .. note::
        You might also provide :data:`numpy.nan` as no data value
        (detected via :func:`numpy.isnan`).

    Returns
    ----------
    int
       Count of valid cells (pixels in rasterfile).

    See Also
    --------
    :func:`~riogrande.helper.aggregated_selector` : Build a selector from rasterio band masks.
    :func:`~riogrande.helper.reduced_mask` : Build a mask from nodata values across bands.
    """
    if np.isnan(no_data):
        b_vals, b_counts = np.unique(~np.isnan(data[selector]), return_counts=True)
    else:
        b_vals, b_counts = np.unique(data[selector] != no_data, return_counts=True)
    # b_vals is [True, False] and can be used as selector for b_counts
    # thus returning the count of True
    if True in b_vals:
        return int(b_counts[b_vals][0])
    else:
        return 0


def rasterio_to_numpy_dtype(rasterio_dtype: str) -> np.dtype | None:
    """Map Rasterio actual data types to NumPy data types.

    Rasterio types like ``rasterio.dtypes.int16``, ``rasterio.dtypes.float32``
    are mapped to their NumPy equivalents.

    Parameters
    ----------
    rasterio_dtype:
        Output of ``rasterio.open(source).profile['dtype']``, as returned by
        :func:`rasterio.open`.

    Returns
    ----------
    numpy.dtype or None
        Data type as :class:`numpy.dtype`, or ``None`` if the type is unknown.
    """
    dtype_mapping = {
        rio.dtypes.int16: np.int16,
        rio.dtypes.int32: np.int32,
        rio.dtypes.uint8: np.uint8,
        rio.dtypes.uint16: np.uint16,
        rio.dtypes.uint32: np.uint32,
        rio.dtypes.float32: np.float32,
        rio.dtypes.float64: np.float64,
    }
    return dtype_mapping.get(rasterio_dtype, None)
