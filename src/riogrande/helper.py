





MPC_STARTER_METHODS = ['spawn', 'fork', 'forkserver']

def get_nbr_workers(number :Optional[int ] =None )- >int:
    """Determine the number of worker processes to use in mulitprocessing.

    Parameters
    ----------
    number : int or None, optional
        Desired number of workers. If ``None``, the function will use the
        number of CPUs available, but never less than 2.

    Returns
    -------
    int
        Number of workers to use (always `>= 2`).

    Notes
    -----
    A warning is emitted when a requested ``number`` is lower than 2 and the
    request is ignored setting the number of used workers to 2.
    """
    # is_needed
    # no_work
    # not_tested
    # usedin_both (potentially)
    _min_count = 2  # Hardcoded: some parallelization routines fail when < 2
    if number is None:
        _use = max(_min_count, mpc.cpu_count())  # assert the min. count
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
        ``Process``, ``Pool`` and related objects. The returned context will
        use the start method determined by the logic described above. The
        function always returns a context and never mutates an already-set
        global start method to a different value.

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

    Examples
    --------
    >>> ctx = get_or_set_context('spawn')
    >>> with ctx.Process(target=worker) as p:
    >>>     p.start()
    >>>     p.join()
    """
    # is_needed
    # no_work
    # is_tested
    # usedin_both (potentially any usage of mpc)

    allowed = MPC_STARTER_METHODS + [None ,]
    default_method = MPC_STARTER_METHODS[0]  # default is 'spawn'
    if method not in allowed:
        raise ValueError(f"Unsupported start method: {method!r}")

    # get the current context
    _contex t= mpc.get_start_method(allow_none=True)

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


def serialize(tags:dict[str,Any])->dict[str,str]:
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Convert the values of a dict to into JSON
    """
    # is_needed
    # no_work
    # not_tested (indirect calls in tests)
    # usedin_both (io sub-module)
    return {tag: json.dumps(obj=value)
            for tag, value in tags.items()}


def deserialize(tags:dict[str,str])->dict[str,Any]:
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Reads python objects from JSON-encoded values of a dict
    """
    # is_needed
    # no_work
    # not_tested (indirect calls in tests)
    # usedin_both (io sub-module)
    return {tag: json.loads(s=value)
            for tag, value in tags.items()}


def sanitize(tags:dict[str,Any])->Any:
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Serializes then deserializes values of a dict
    """
    # is_needed
    # no_work
    # not_tested (indirect calls in tests)
    # usedin_both (io sub-module)
    return deserialize(serialize(tags))


def match_all(targets:dict, tags:dict)->bool:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Check if all tags in targets are present in tags
    """
    # is_needed
    # no_work
    # is_tested
    # usedin_both (io sub-module)
    match = True
    for t, v in targets.items():
        if not match:
            break  # stop if the last was no match
        if t in tags:  # if tag is present check for value match
            if tags[t] == v:
                match = True
            else:  # if a value is different it is no match
                match = False
        else:  # if a tag is absent it is no match
            match = False
    return match


def match_any(targets:dict, tags:dict)->bool:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Check if any tag in targets is present in tags
    """
    # not_needed (logic covered by match_all)
    # no_work
    # is_tested
    # usedin_both
    match = False
    for t, v in targets.items():
        if match:
            break  # stop if there was a match
        if t in tags:  # if tag is present check for value match
            if tags[t] == v:
                match = True
            else:  # if a value is different it is no match
                match = False
        else:  # if a tag is absent it is no match
            match = False
    return match


def view_to_window(view: None | tuple[int, int, int, int]):
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Conerts a view into a rasterio Window

    Parameters
    ----------
    view:
      tuple (x, y, width, height) defining the view of the data array to update
    """
    # is_needed
    # needs_work (fix doc, dedicated test)
    # not_tested (used in test)
    # usedin_both
    if view is not None:
        window =  Window(view[0],
                         view[1],
                         view[2],
                         view[3])
    else:
        window = None
    return window




def check_units(*sources):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Assert that all sources have the same units
    """
    # is_needed
    # needs_work (fix doc, make internal _...)
    # not_tested
    # usedin_both (used in io submodule)
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


def check_crs(*sources):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Assert that all the sources have the same projection (i.e. same crs)
    """
    # is_needed
    # needs_work (make internal _...)
    # not_tested
    # usedin_both (used in io submodule)
    crss = []
    for source in sources:
        with rio.open(source) as src:
            crss.append(str(src.profile.get('crs', None)))
            if len(set(crss)) != 1:
                raise TypeError(f"{source=} has crs {crss[-1]}, which is "
                                f"different from the other(s) ({crss[0]})")
    return crss


def check_resolution(*sources):
    # TODO: is_needed - no_work - not_tested - usedin_both
    """Assert that all the sources have the same resolution
    """
    # is_needed
    # needs_work (make internal _...)
    # not_tested
    # usedin_both (used in io submodule)
    ress = []
    for source in sources:
        with rio.open(source) as src:
            # NOTE: we round 8th digit after the comma here
            ress.append(tuple(map(lambda x: round(x, 8), src.res)))
            if len(set(ress)) != 1:
                raise TypeError(f"{source=} has resolution {ress[-1]}, which "
                                f"is different from the other(s) ({ress[0]})")
    return ress


def check_compatibility(*sources):
    # TODO: is_needed - no_work - not_tested - usedin_both
    #  --> not sure if this is really needed in both (but its in CLASS io_
    """Assert that all the sources are compatible with each other.

    The checks include:

        - crs
        - units
        - resolution

    """
    # is_needed
    # needs_work (better doc)
    # not_tested (used in tests)
    # usedin_both (used in io submodule and parallel of linfit)
    units = check_units(*sources)
    crss = check_crs(*sources)
    ress = check_resolution(*sources)
    # print(f"{crss=}, {units=}, {ress=}")
    return crss, units, ress


def check_crs_raster(source, reference, verbose=False):
    # TODO: not_needed
    """Compare coordinate reference systems of two raster datasets"""
    # is_needed
    # needs_work (fix doc, dedicated test)
    # not_tested (used in test)
    # usedin_both (used in io submodule)
    with rio.open(source, mode='r') as src:
        src_crs = str(src.crs)
    with rio.open(reference, mode='r') as ref:
        ref_crs = str(ref.crs)

    if src_crs == ref_crs:
        if verbose:
            print(f"Coordinate systems are the same: {src_crs} --> {ref_crs}")
        return True
    else:
        print(f"CRS CHECK FAILING: {src_crs=} - {ref_crs=}")
        return False

def outfile_suffix(filename, suffix, separator:str='_'):
    # TODO: is_needed (for now) - no_work - not_tested - usedin_both
    """Insert suffix into filename and hand back basename_suffix.extension"""
    # is_needed
    # no_work
    # not_tested (used in tests)
    # usedin_both (used in io submodule)
    base, ext = os.path.splitext(filename)
    return f"{base}{separator}{suffix}{ext}"


def strip_suffix(filename:str, separator:str='_'):
    # TODO: not_needed
    """Removes the last suffix from the name (i.e. the last part separated by '_')
    """
    # not_needed
    # no_work
    # not_tested (used in tests)
    # usedin_both (used in io submodule)
    base, ext = os.path.splitext(filename)
    if separator in filename:
        _base = separator.join(filename.split(separator)[:-1])
    else:
        _base = base
    return f"{_base}{ext}"


def output_filename(base_name: str, out_type: str, blur_params: dict):
    # TODO: is_needed - no_work - is_tested - usedin_both
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
    # is_needed
    # needs_work (minor cleanup)
    # not_tested
    # usedin_both
    _base_name, _ext = os.path.splitext(base_name)
    # sig = blur_params['sigma']
    # diam = blur_params['diameter']
    # trunc = blur_params['truncate']
    _blur_string = ""
    for name, value in blur_params.items():
        _blur_string += f"_{name}_{round(value)}"
    # _blur_string = f"sig_{sig}_diam_{diam}_trunc_{trunc}"
    return f"{_base_name}_{out_type}{_blur_string}{_ext}"


def dtype_range(dtype:type|str)->tuple[int|float, int|float]:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Get the range of the specified dtype

    ..warning::
      This functions returns min or max as either `int` or `floats`.

      Be sure to convert them back into `dtype` if needed!

    """
    # is_needed (mostly internal + processing.py)
    # needs_work (adding tests, fix type-hints)
    # not_tested
    # usedin_both (certainly in processing + internal)
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


def convert_to_dtype(data: NDArray,
                     as_dtype:None|type|np._dtype|str=None,
                     in_range:None|NDArray|Collection=None,
                     out_range:None|NDArray|Collection|str|type=None)->NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Converts data to `as_dtype` and optionally rescales it.

    Rescaling is done only if at least one of of the ranges is explicitly set.
    If only `in_range` is set then the input range is scaled to the full range
    of the output output data type, `ad_dtype`.
    This behaviour is typically wanted when converting some floating typed data
    in a limited range, e.g. [0, 1] to unsigned integer, e.g. `uint8`, thus
    mapping the range [0,1] to [0, 255].

    In case only `out_range` is set, the full data type range of the input
    data is mapped to the provided `out_range`.
    This is typically used if converting from a "limited" range, like `uint8`
    to a floating data type.

    Examples:
    >>> # simple conversion, no rescalingm
    >>> y_data = np.array([0, 0.5, 1.], dtype=np.float64)
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

    ..note::

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
    data: input numpy NDArray
    as_dtype: desired data type to convert to (e.g. np.float64)
        If not provided then at least the `out_range` needs to be set in
        which case the data type remains unchanges, but the data is
        rescaled.
    in_range:
        an array or list from which min and max will be used as input range

        ..note::
          You might simply provide the same value as for `data` in order to
          use its min an max for scaling
    out_range:
      an array or list from which min and max will be used as limits for the
      output.
      Alternatively, a data type can be specified, in which case the data
      will be scaled to the full range of the specified data type
    """
    # is_needed
    # needs_work (formatting, fix type-hinting)
    # is_tested
    # usedin_both (could be io module)
    # convert to numpy dtype is string was provided
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


def aggregated_selector(masks:list[NDArray], logic:str='all')->NDArray:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Turns several rasterio masks into a boolen selector for a numpy array

    Rasterio masks are uint8 numpy arrays where every value > 0 is considered
    a valid cell

    Parameters
    ----------
    masks:
        Arbitrary number of numpy arrays resalting from
        `rasterio.io.DatasetReader.dataset_mask` or
        `rasterio.io.DatasetReader.read_masks`
    logic:
        Determines how the aggreagation should happen.
        If `all` (the default) a cell is only selected if **all** masks
        consider it valid data. `logic="any"` will lead to selecting
        all cells which **at least one** mask considers valid
    """
    # is_needed
    # no_work
    # is_tested
    # usedin_both (used in parallel.prepare_selector)
    selector = masks[0]!=0  # values > 0 are selected (i.e. True)
    if logic == 'any':
        _logic = np.logical_or
    else:
        _logic = np.logical_and
    if len(masks) > 1:
        for mask in masks[1:]:
            _logic(selector, mask!=0, out=selector)
    return selector


def reduced_mask(array:NDArray,
                nodata=0,
                logic:str='all',):
    # TODO: is_needed - no_work - is_tested - usedin_both
    """Computes a mask based on the value of several bands

    Parameters
    ----------
    array:
        3D array holding multiple bands of map data
    logic:
        Allowed strings are:
        - `"any"`: Masked will be each cell for which any of the bands matches the nodata value
        - `"all"`: Masked will be each cell for which all of the bands match the nodata value
    """
    # is_needed
    # no_work (create test)
    # not_tested
    # usedin_both (used in parallel.compute_mask)
    if logic=='any':
        _logic = np.logical_and
    else:
        _logic = np.logical_or
    if np.isnan(nodata):
        return _logic.reduce(array=~np.isnan(array), axis=0).astype(np.uint8)
    else:
        return _logic.reduce(array=array!=nodata, axis=0).astype(np.uint8)


def count_contribution(data:NDArray,
                       selector:NDArray[np.bool_],
                       no_data:Union[int, float]=0)->int:
    # TODO: is_needed - no_work - is_tested - usedin_both
    """The remaining number of data cells when applying the selector

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
        You might also provide `np.nan` as no data value.

    """
    # is_needed
    # no_work
    # is_tested
    # usedin_both (io module)
    if np.isnan(no_data):
        b_vals, b_counts = np.unique(~np.isnan(data[selector]), return_counts=True)
    else:
        b_vals, b_counts = np.unique(data[selector]!=no_data, return_counts=True)
    # b_vals is [True, False] and can be used as selector for b_counts
    # thus returning the count of True
    if True in b_vals:
        return int(b_counts[b_vals][0])
    else:
        return 0


def rasterio_to_numpy_dtype(rasterio_dtype):
    # TODO: is_needed (for testing later) - no_work - is_tested - usedin_both
    """
    Map Rasterio actual data types to NumPy data types.

    Rasterio types like rasterio.dtypes.int16, rasterio.dtypes.float32
    are mapped to their NumPy equivalents.
    """
    # not_needed
    # no_work
    # not_tested
    # usedin_both (io module - but not used)
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
