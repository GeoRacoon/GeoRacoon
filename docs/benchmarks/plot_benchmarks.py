# -*- coding: utf-8 -*-
"""
.. _plot_benchmarks:

Parallelization benchmarks
==========================

This example reads the latest ASV benchmark results (``.asv/results/``) for the
synthetic Gaussian filter and multiple linear regression benchmarks and shows
them as heatmaps:
the x-axis is the number of jobs (``n_jobs``), the y-axis is the block size
expressed as a fraction of the raster's total size.

Two figures are produced, one per routine. Each figure has two panels:

- **Wall time**, and
- **Peak memory** (summed RSS of the whole process tree).

Every cell is normalized by the *native* (single-process, no multiprocessing)
baseline, which is reported at ``n_jobs = 1``. Values below 1 are faster / more
memory efficient than the native approach; values above 1 are slower / less
memory efficient.

The plots use the ``10000 x 10000`` synthetic-raster results from the most
recent benchmark run committed to the repository. Raw ASV results for the
other configured sizes remain available for analysis. If no results are
present, run the benchmarks first (``asv run``) and commit ``.asv/results/``.
"""

import json
import os
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

RESULTS_DIR = Path(os.environ["GEORACOON_ASV_RESULTS_DIR"])
INDEX_PATH = RESULTS_DIR / "benchmarks.json"

DOC_SIZE = 10000

FILTER_TIME = "bench_filter.TimeFilterScaling.time_apply_filter_bpgaussian"
FILTER_MEM = "bench_filter.PeakMemFilterScaling.track_apply_filter_bpgaussian_peakmem"
LINREG_TIME = "bench_linreg.TimeComputeWeightsScaling.time_compute_weights"
LINREG_MEM = "bench_linreg.PeakMemComputeWeightsScaling.track_compute_weights_peakmem"

FILTER_TIME_NATIVE = "bench_filter.TimeFilterNativeScaling.time_apply_filter_bpgaussian"
FILTER_MEM_NATIVE = "bench_filter.PeakMemFilterNativeScaling.track_apply_filter_bpgaussian_peakmem"
LINREG_TIME_NATIVE = "bench_linreg.TimeComputeWeightsNativeScaling.time_compute_weights"
LINREG_MEM_NATIVE = "bench_linreg.PeakMemComputeWeightsNativeScaling.track_compute_weights_peakmem"


def _machine_dirs():
    if not RESULTS_DIR.is_dir():
        return []
    return [p for p in RESULTS_DIR.iterdir() if p.is_dir()]


def _result_json(path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _latest_result_file():
    """Return the result file with the newest ``date`` across all machines."""
    candidates = []
    for machine_dir in _machine_dirs():
        files = [
            p for p in machine_dir.glob("*.json") if p.name != "machine.json"
        ]
        candidates.extend(files)

    if not candidates:
        raise RuntimeError(
            "No ASV benchmark results found under '.asv/results/'. Run "
            "'asv run' and commit the results before building the docs."
        )

    return max(
        candidates,
        key=lambda p: (_result_json(p) or {}).get("date", 0),
    )


def _load_grid(benchmark_name, size=None):
    """Return ``(n_jobs, block_fractions, values)`` for a benchmark.

    ``values`` is a 2D array of shape ``(len(block_fractions), len(n_jobs))``
    with ``values[j, i]`` corresponding to ``n_jobs[i]`` and
    ``block_fractions[j]``.
    """
    index = _result_json(INDEX_PATH)
    if index is None or benchmark_name not in index:
        raise RuntimeError(f"Benchmark {benchmark_name!r} not found in index.")

    data = _result_json(_latest_result_file())
    if data is None or benchmark_name not in data.get("results", {}):
        raise RuntimeError(f"No results for {benchmark_name!r}.")

    meta = index[benchmark_name]
    param_names = meta["param_names"]
    columns = data["result_columns"]
    record = data["results"][benchmark_name]

    values = record[columns.index("result")]
    axes = record[columns.index("params")]

    n_jobs = [int(v) for v in axes[param_names.index("n_jobs")]]
    fractions = [float(v) for v in axes[param_names.index("block_fraction")]]

    shape = tuple(len(axis) for axis in axes)
    result_array = np.asarray(values, dtype=float).reshape(shape)
    if "size" in param_names:
        if size is None:
            raise ValueError("A size is required for a scaling benchmark.")
        size_axis = param_names.index("size")
        sizes = [int(v) for v in axes[size_axis]]
        if size not in sizes:
            raise RuntimeError(f"Size {size} not found for {benchmark_name!r}.")
        result_array = np.take(result_array, sizes.index(size), axis=size_axis)

    # ASV stores the first parameter as the outermost axis. Transpose the
    # remaining n_jobs/block_fraction axes into heatmap row/column order.
    n_axis = param_names.index("n_jobs")
    f_axis = param_names.index("block_fraction")
    if "size" in param_names:
        n_axis -= int(n_axis > param_names.index("size"))
        f_axis -= int(f_axis > param_names.index("size"))
    grid = np.asarray(result_array).transpose(f_axis, n_axis)
    return n_jobs, fractions, grid


def _load_native(benchmark_name, size=None):
    """Return a native result, optionally selecting a synthetic size."""
    index = _result_json(INDEX_PATH)
    if index is None or benchmark_name not in index:
        raise RuntimeError(f"Benchmark {benchmark_name!r} not found in index.")

    data = _result_json(_latest_result_file())
    if data is None or benchmark_name not in data.get("results", {}):
        raise RuntimeError(f"No results for {benchmark_name!r}.")

    columns = data["result_columns"]
    record = data["results"][benchmark_name]
    values = record[columns.index("result")]

    if not values:
        raise RuntimeError(f"No values recorded for {benchmark_name!r}.")
    meta = index[benchmark_name]
    param_names = meta["param_names"]
    axes = record[columns.index("params")]
    shape = tuple(len(axis) for axis in axes)
    result_array = np.asarray(values, dtype=float).reshape(shape)
    if "size" in param_names:
        if size is None:
            raise ValueError("A size is required for a scaling benchmark.")
        size_axis = param_names.index("size")
        sizes = [int(v) for v in axes[size_axis]]
        if size not in sizes:
            raise RuntimeError(f"Size {size} not found for {benchmark_name!r}.")
        result_array = np.take(result_array, sizes.index(size), axis=size_axis)
    value = np.asarray(result_array).reshape(-1)[0]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise RuntimeError(f"Native benchmark {benchmark_name!r} is missing.")
    return value


def _fraction_label(fraction):
    frac = Fraction(fraction).limit_denominator(1000)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def _ratio_norm(values):
    """Diverging color norm centered at 1 (native == neutral)."""
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    vmin = min(vmin, 1.0)
    vmax = max(vmax, 1.0)
    if vmin == 1.0:
        vmin = 0.0
    if vmax == 1.0:
        vmax = 2.0
    return TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)


def _heatmap(ax, n_jobs, fractions, values, title, cbar_label):
    norm = _ratio_norm(values)
    image = ax.imshow(
        values, aspect="auto", origin="upper", cmap="coolwarm", norm=norm
    )
    ax.set_xticks(range(len(n_jobs)))
    ax.set_xticklabels([str(n) for n in n_jobs])
    ax.set_yticks(range(len(fractions)))
    ax.set_yticklabels([_fraction_label(f) for f in fractions])
    ax.set_xlabel("n_jobs")
    ax.set_ylabel("block size (fraction of total tif)")
    ax.set_title(title)
    fig = ax.figure
    fig.colorbar(image, ax=ax, label=cbar_label, shrink=0.8)


def _plot_routine(time_name, mem_name, time_native, mem_native, routine_title):
    n_jobs, fractions, times = _load_grid(time_name, size=DOC_SIZE)
    _, _, mem_bytes = _load_grid(mem_name, size=DOC_SIZE)

    time_native_val = _load_native(time_native, size=DOC_SIZE)
    mem_native_val = _load_native(mem_native, size=DOC_SIZE)

    times_ratio = times / time_native_val
    mem_ratio = mem_bytes / mem_native_val

    x_njobs = [1] + n_jobs
    native_col = np.ones((len(fractions), 1))
    times_full = np.concatenate([native_col, times_ratio], axis=1)
    mem_full = np.concatenate([native_col, mem_ratio], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{routine_title} (synthetic {DOC_SIZE} x {DOC_SIZE} raster)")
    _heatmap(
        axes[0], x_njobs, fractions, times_full, "Wall time",
        "ratio to native (< 1 faster)",
    )
    _heatmap(
        axes[1], x_njobs, fractions, mem_full, "Peak memory",
        "ratio to native (< 1 more efficient)",
    )
    fig.tight_layout()
    return fig


# Gaussian filtering
_plot_routine(
    FILTER_TIME,
    FILTER_MEM,
    FILTER_TIME_NATIVE,
    FILTER_MEM_NATIVE,
    "Gaussian filtering (apply_filter + bpgaussian)",
)

# Multiple linear regression
_plot_routine(
    LINREG_TIME,
    LINREG_MEM,
    LINREG_TIME_NATIVE,
    LINREG_MEM_NATIVE,
    "Multiple linear regression (compute_weights)",
)
