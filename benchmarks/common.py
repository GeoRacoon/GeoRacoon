"""Shared helpers for the GeoRacoon ASV benchmarks.

All benchmark inputs are deterministic (fixed seed for synthetic data) so that
results are comparable across runs and commits. Machine-specific sizing is
resolved from :mod:`~benchmarks.machine_configs` (see ``machine_configs.json``)
keyed by the ASV machine name.
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import rasterio as rio
from rasterio.windows import Window
from rasterio.transform import from_origin

from riogrande.io import coregister_raster
from riogrande.prepare import create_views

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("machine_configs.json")

# Synthetic benchmark rasters are written here and reused across runs.
_BENCH_TMP = os.path.join(tempfile.gettempdir(), "georacoon_benchmarks")


def pretty_name(name):
    """Set the human-readable ASV display name of a benchmark."""
    def decorate(func):
        func.pretty_name = name
        return func
    return decorate


class Machine:
    """Machine-specific benchmark parameter configuration.

    Resolves the ASV machine name from ``~/.asv-machine.json``, merges the
    matching entry over the ``default`` entry in ``machine_configs.json``, and
    exposes a ``get_<key>()`` accessor for every configuration key.
    """

    def __init__(self):
        self._name = self._get_benchmark_machine()
        self._ncpu = os.cpu_count() or 1
        self._config = self._load_config()
        self._bind_config_getters()

    @staticmethod
    def _get_benchmark_machine():
        """Return the single configured ASV machine name, if available."""
        machine_file = Path.home() / ".asv-machine.json"
        if not machine_file.exists():
            return None

        with machine_file.open() as fh:
            data = json.load(fh)

        machines = [
            key for key, value in data.items()
            if key != "version" and isinstance(value, dict)
        ]
        if len(machines) == 1:
            return machines[0]
        return None

    def _load_config(self):
        """Load and resolve the benchmark sizing config for this machine."""
        with CONFIG_PATH.open() as fh:
            configs = json.load(fh)

        if "default" not in configs:
            raise ValueError("machine_configs.json must define 'default'")

        default_config = configs["default"]
        machine_config = configs.get(self._name, {})
        return {**default_config, **machine_config}

    def _bind_config_getters(self):
        """Create ``get_<key>()`` methods for resolved config keys."""
        for key in self._config:
            method_name = f"get_{key}"

            def getter(key=key):
                return self._config[key]

            setattr(self, method_name, getter)

    @property
    def name(self):
        """Configured ASV machine name, or ``None``."""
        return self._name

    @property
    def ncpu(self):
        """Number of logical CPUs available to this process."""
        return self._ncpu

    @property
    def config(self):
        """Fully resolved benchmark sizing configuration."""
        return self._config

    def get_njobs(self):
        """Return ``[2, 3, ..., ncpu - 1]`` for the ``n_jobs`` sweep."""
        return list(range(2, self._ncpu))

    def get_block_fractions(self):
        """Return ``[1, 1/2, 1/3, ..., 1/d]`` for configured denominators."""
        return [1.0 / d for d in self._config["block_fraction_denominators"]]


def data_path(relative_path):
    """Resolve a config-relative data path against the repository root."""
    return str(REPO_ROOT / relative_path)


def _process_tree_rss(proc):
    """Return the summed RSS of ``proc`` and all its (recursive) children."""
    total = 0
    try:
        total += proc.memory_info().rss
    except psutil.Error:
        pass

    try:
        children = proc.children(recursive=True)
    except psutil.Error:
        children = []

    for child in children:
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass

    return total


def peak_rss_while(work_fn, interval=0.01):
    """Run ``work_fn()`` while sampling the process-tree RSS; return the peak.

    The peak covers the main process and all worker/manager children spawned by
    the parallel routines under test. ASV's built-in ``peakmem_`` only samples
    the main process, which misses the multiprocessing workers.

    Parameters
    ----------
    work_fn : callable
        The benchmarked routine, called with no arguments.
    interval : float
        Sampling interval in seconds.

    Returns
    -------
    int
        Peak summed RSS in bytes across the process tree.
    """
    proc = psutil.Process()
    peak = 0
    done = threading.Event()

    def watch():
        nonlocal peak
        while not done.is_set():
            peak = max(peak, _process_tree_rss(proc))
            time.sleep(interval)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        work_fn()
    finally:
        done.set()
        watcher.join()

    return peak


def ensure_bench_dir():
    """Create (if needed) and return the synthetic-data temp directory."""
    os.makedirs(_BENCH_TMP, exist_ok=True)
    return _BENCH_TMP


def _synthetic_blocks(size, block_size):
    """Yield non-overlapping Rasterio windows covering a square raster."""
    block_width = min(size, block_size[0])
    block_height = min(size, block_size[1])
    _, inner_views = create_views(
        view_size=(block_width, block_height),
        border=(0, 0),
        size=(size, size),
    )
    for x, y, width, height in inner_views:
        yield (x, y, width, height), Window(x, y, width, height)


def _block_rng(seed, x, y):
    """Return a deterministic RNG independent of block iteration order."""
    return np.random.default_rng(np.random.SeedSequence([seed, x, y]))


def _synthetic_directory(directory):
    directory = _BENCH_TMP if directory is None else os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    return directory


def synthetic_tif(size, seed=0, dtype="float32", directory=None,
                  block_size=(1000, 1000)):
    """Return the path to a deterministic ``size x size`` single-band GeoTIFF.

    The raster is generated once per input configuration and reused on later
    calls. Data is generated and written one block at a time, so the complete
    raster is never held in memory. Data is random-normal with ``nodata=NaN``.

    Parameters
    ----------
    size : int
        Side length in pixels (the raster is ``size x size``).
    seed : int
        RNG seed; different seeds produce independent rasters.
    dtype : str
        Raster data type.
    directory : path-like or None
        Directory in which to cache the raster. Defaults to the benchmark
        temporary directory.
    block_size : tuple[int, int]
        Width and height of the generation blocks.

    Returns
    -------
    str
        Path to the generated (or cached) GeoTIFF.
    """
    directory = _synthetic_directory(directory)
    path = os.path.join(
        directory, f"synth_{size}_{seed}_{dtype}_{block_size[0]}x{block_size[1]}.tif"
    )
    if os.path.exists(path):
        return path

    dtype = np.dtype(dtype)
    transform = from_origin(0.0, float(size), 1.0, 1.0)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=dtype,
        transform=transform,
        crs="EPSG:32632",
        nodata=np.nan,
        BIGTIFF="YES",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for (x, y, width, height), window in _synthetic_blocks(size, block_size):
            rng = _block_rng(seed, x, y)
            data = rng.normal(size=(height, width)).astype(dtype)
            dst.write(data, 1, window=window)
    return path


def synthetic_filter_tif(size, seed=0, frame_width=31, directory=None,
                         block_size=(1000, 1000)):
    """Return a binary float32 raster with a NaN frame for filter benchmarks.

    The interior contains deterministic zero/one values. The outer frame is
    filled with NaNs so ``bpgaussian`` exercises its border-preserving behavior.
    Data is generated and written one block at a time, so the complete raster
    is never held in memory.

    Parameters
    ----------
    size : int
        Side length in pixels.
    seed : int
        RNG seed for the binary interior.
    frame_width : int
        Width of the NaN frame in pixels.
    directory : path-like or None
        Directory in which to cache the raster. Defaults to the benchmark
        temporary directory.
    block_size : tuple[int, int]
        Width and height of the generation blocks.

    Returns
    -------
    str
        Path to the generated (or cached) GeoTIFF.
    """
    directory = _synthetic_directory(directory)
    path = os.path.join(
        directory,
        f"filter_synth_{size}_{seed}_{frame_width}_"
        f"{block_size[0]}x{block_size[1]}.tif",
    )
    if os.path.exists(path):
        return path

    transform = from_origin(0.0, float(size), 1.0, 1.0)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:32632",
        nodata=np.nan,
        BIGTIFF="YES",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        for (x, y, width, height), window in _synthetic_blocks(size, block_size):
            rng = _block_rng(seed, x, y)
            data = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
            data = data.astype(np.float32)
            local_x = np.arange(x, x + width)[None, :]
            local_y = np.arange(y, y + height)[:, None]
            frame = (
                (local_x < frame_width)
                | (local_x >= size - frame_width)
                | (local_y < frame_width)
                | (local_y >= size - frame_width)
            )
            data[frame] = np.nan
            dst.write(data, 1, window=window)
    return path


def coregistered_tif(source_path, reference_path):
    """Return ``source_path`` coregistered onto ``reference_path``'s grid.

    The reprojection is performed once via
    :func:`~riogrande.io.coregister_raster` and cached on disk under the
    benchmark temp directory, so later calls (including those from fresh ASV
    benchmark subprocesses) reuse the existing file. This is meant to be called
    from benchmark ``setup`` so the coregistration is excluded from timing and
    peak-memory measurement.

    Parameters
    ----------
    source_path : str
        Path of the raster to reproject.
    reference_path : str
        Path of the raster whose grid/resolution is used as the target.

    Returns
    -------
    str
        Path of the (cached) coregistered raster.
    """
    ensure_bench_dir()
    output = os.path.join(
        _BENCH_TMP,
        f"{Path(source_path).stem}_coreg_to_{Path(reference_path).stem}.tif",
    )
    if os.path.exists(output):
        return output
    return coregister_raster(source_path, reference_path, output=output)


_raster_sizes = {}


def raster_size(path):
    """Return the cached ``(width, height)`` of a raster file."""
    if path not in _raster_sizes:
        with rio.open(path) as src:
            _raster_sizes[path] = (src.width, src.height)
    return _raster_sizes[path]


def block_size_from_fraction(fraction, width, height):
    """Return the ``(width, height)`` block for a fraction of a raster's size."""
    return (
        max(1, round(fraction * width)),
        max(1, round(fraction * height)),
    )


def make_temp_output(prefix="georacoon_bench"):
    """Create and return a unique, closed temp file path for output rasters."""
    ensure_bench_dir()
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".tif", dir=_BENCH_TMP)
    os.close(fd)
    return path
