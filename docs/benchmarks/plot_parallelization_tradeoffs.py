# -*- coding: utf-8 -*-
"""
Parallel raster computation: trading memory for runtime
========================================================

This example explains two related ideas behind GeoRacoon's block-wise
parallel computations:

* a raster operation can be decomposed into contributions from independent
  spatial blocks; and
* the block size and the number of worker processes are two separate tuning
  dimensions.

The multiple linear regression example below uses the normal equations.  It
does not construct one design matrix for the complete raster.  Instead, it
accumulates the two sufficient statistics needed by ordinary least squares in
two passes over the blocks.

The figures in this example read ASV results for the synthetic filter and
regression benchmarks.  They show runtime and peak process-tree memory
relative to the native, full-raster implementation.  The native result is the
reference case and is shown as ``n_jobs = 1``.  Consequently, a value of 0.5
means half the native runtime or memory, while 1.5 means 150 percent.

The benchmark results are machine-dependent.  They should therefore be read
as an illustration of the available trade-offs, rather than as universal
performance constants.

"""

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import TwoSlopeNorm
from riogrande.prepare import create_views


# %%
# General approach: trading time for memory
# ------------------------------------------
#
# Resource-heavy raster operations can often be split into a set of smaller,
# independent tasks.  Applying a Gaussian filter to a TIFF and fitting a
# multiple linear regression (MLR) model are two examples.  Instead of handling
# the complete raster in one operation, the work is divided into spatial
# blocks.  The blocks can then be processed sequentially or in parallel,
# depending on the available hardware.
#
# This makes it possible to adapt the computation to hardware restrictions by
# trading time for RAM, and vice versa.  Smaller blocks and fewer workers reduce
# the amount of memory required at any one time, but generally increase the
# number of tasks and their scheduling overhead.  Larger blocks and more
# workers can reduce runtime, at the cost of a larger peak memory footprint.


# %%
# Applying a Gaussian filter block-wise
# -------------------------------------
#
# A Gaussian filter requires neighboring pixels, so processing independent
# blocks requires overlap or halo regions at their boundaries.  ``apply_filter``
# computes a halo from the filter parameters and calls ``create_views`` to
# construct two corresponding view lists.  Each worker reads an expanded view
# including the halo, applies the filter, and returns only its inner view for
# writing.  The expanded views overlap, but the inner views tile the output
# without overlap.  At the outer edge of the raster, the halo is reduced to fit.

filter_border = (2, 2)
views, inner_views = create_views(
    view_size=(6, 6),
    border=filter_border,
    size=(24, 24),
)
inner_xs = sorted({view[0] for view in inner_views})
inner_ys = sorted({view[1] for view in inner_views})
center_xs = set(inner_xs[1:3])
center_ys = set(inner_ys[1:3])
center = [
    (view, inner_view)
    for view, inner_view in zip(views, inner_views)
    if inner_view[0] in center_xs and inner_view[1] in center_ys
    and inner_view[:2] != (12, 12)
]
corner = [
    (view, inner_view)
    for view, inner_view in zip(views, inner_views)
    if inner_view[:2] == (18, 18)
]
selected = [*center, *corner]

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 24)
ax.set_ylim(24, 0)
ax.set_aspect("equal")
ax.tick_params(which="both", bottom=False, left=False,
               labelbottom=False, labelleft=False)

for position in range(1, 24):
    ax.axvline(position, color="0.75", linewidth=0.6, zorder=0)
    ax.axhline(position, color="0.75", linewidth=0.6, zorder=0)

for view, _ in selected:
    x, y, width, height = view
    ax.add_patch(Rectangle(
        (x, y), width, height, facecolor="#f4a261", alpha=0.4,
        edgecolor="#e76f51", linewidth=1.5,
    ))

for _, inner_view in selected:
    x, y, width, height = inner_view
    ax.add_patch(Rectangle(
        (x, y), width, height, facecolor="#8ecae6",
        edgecolor="black", linewidth=1.5,
    ))

ax.text(12, -1.0, f"border: {filter_border[0]} pixel",
        ha="center", va="bottom")
ax.legend(handles=[
    Patch(facecolor="#8ecae6", edgecolor="black", label="output block"),
    Patch(facecolor="#f4a261", alpha=0.4, edgecolor="#e76f51",
          label="worker view with border"),
], loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False)
ax.set_title("Block-wise Gaussian filtering")
fig.tight_layout()

# %%
# The illustration shows exemplary blocks that can be processed independently.
# An orange region represents the complete view sent to a worker, including the
# border pixels needed by the filter.  The blue region is the corresponding
# inner block: it is the part of the worker's result that contributes to the
# final raster.  Because only the blue regions are written, these contributions
# tile the output without overlap even though the orange worker views overlap.


# sphinx_gallery_start_ignore
RESULTS_DIR = Path(os.environ["GEORACOON_ASV_RESULTS_DIR"])
INDEX_PATH = RESULTS_DIR / "benchmarks.json"

PARALLEL_TIME = "bench_linreg.TimeComputeWeightsScaling.time_compute_weights"
PARALLEL_MEMORY = (
    "bench_linreg.PeakMemComputeWeightsScaling.track_compute_weights_peakmem"
)
NATIVE_TIME = "bench_linreg.TimeComputeWeightsNativeScaling.time_compute_weights"
NATIVE_MEMORY = (
    "bench_linreg.PeakMemComputeWeightsNativeScaling.track_compute_weights_peakmem"
)
FILTER_TIME = "bench_filter.TimeFilterScaling.time_apply_filter_bpgaussian"
FILTER_MEMORY = (
    "bench_filter.PeakMemFilterScaling.track_apply_filter_bpgaussian_peakmem"
)
FILTER_NATIVE_TIME = (
    "bench_filter.TimeFilterNativeScaling.time_apply_filter_bpgaussian"
)
FILTER_NATIVE_MEMORY = (
    "bench_filter.PeakMemFilterNativeScaling."
    "track_apply_filter_bpgaussian_peakmem"
)


def _read_json(path):
    """Read JSON, returning ``None`` when it cannot be read."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _git_output(*args):
    """Run Git without requiring repository ownership to match the user."""
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _configured_machine():
    """Return the machine selected by ASV's local machine configuration."""
    path = Path.home() / ".asv-machine.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    machines = [
        name for name, value in data.items()
        if name != "version" and isinstance(value, dict)
    ]
    return machines[0] if len(machines) == 1 else None


def _result_files():
    """Return readable ASV result files, excluding machine metadata."""
    if not RESULTS_DIR.is_dir():
        raise RuntimeError(
            "No ASV results found. Run `asv run` before building this example."
        )

    candidates = {}
    for machine_dir in RESULTS_DIR.iterdir():
        if machine_dir.is_dir():
            for path in machine_dir.glob("*.json"):
                if path.name != "machine.json":
                    data = _read_json(path)
                    if isinstance(data, dict) and data.get("commit_hash"):
                        candidates[path] = data
    if not candidates:
        raise RuntimeError(
            "No ASV results found. Run `asv run` before building this example."
        )

    return candidates


def _select_result():
    """Select one machine result for the current checkout or ``main``."""
    candidates = _result_files()
    commits = []
    current = _git_output("rev-parse", "HEAD")
    if current:
        commits.append(current)
    main_log = _git_output("log", "main", "--format=%H")
    if main_log:
        commits.extend(main_log.splitlines())

    available_commits = {data["commit_hash"] for data in candidates.values()}
    commit = next(
        (candidate for candidate in commits if candidate in available_commits),
        None,
    )
    if commit is None:
        # This also supports documentation builds outside a Git checkout.
        commit = max(
            available_commits,
            key=lambda value: max(
                data.get("date", 0)
                for data in candidates.values()
                if data["commit_hash"] == value
            ),
        )

    matches = [
        (path, data) for path, data in candidates.items()
        if data["commit_hash"] == commit
    ]
    requested_machine = os.environ.get("GEORACOON_ASV_MACHINE")
    machine = requested_machine or _configured_machine()
    if machine:
        matches = [item for item in matches if item[0].parent.name == machine]
    if len(matches) != 1:
        available = ", ".join(sorted(path.parent.name for path, _ in matches))
        if not matches:
            raise RuntimeError(
                f"No ASV result for commit {commit[:12]} and machine "
                f"{machine!r}. Available machines: {available or 'none'}."
            )
        raise RuntimeError(
            f"Multiple ASV machines contain commit {commit[:12]} "
            f"({available}). Set GEORACOON_ASV_MACHINE."
        )
    return matches[0]


RESULT_PATH, RESULT_DATA = _select_result()
RESULT_MACHINE = RESULT_PATH.parent.name
RESULT_COMMIT = RESULT_DATA["commit_hash"]


def _result_array(name, size=None):
    """Return one ASV result as its parameter-shaped array."""
    index = _read_json(INDEX_PATH)
    data = RESULT_DATA
    if index is None or name not in index:
        raise RuntimeError(f"ASV result {name!r} is unavailable.")

    metadata = index[name]
    parameters = metadata["param_names"]
    record = data["results"].get(name)
    if record is None:
        raise RuntimeError(f"ASV result {name!r} is unavailable.")

    columns = data["result_columns"]
    values = np.asarray(record[columns.index("result")], dtype=float)
    axes = record[columns.index("params")]
    result = values.reshape(tuple(len(axis) for axis in axes))
    if size is not None and "size" in parameters:
        size_axis = parameters.index("size")
        sizes = [int(value) for value in axes[size_axis]]
        if size not in sizes:
            raise RuntimeError(f"Synthetic raster size {size} is unavailable.")
        result = np.take(result, sizes.index(size), axis=size_axis)
        parameters = [name for name in parameters if name != "size"]
        axes = [axis for name, axis in zip(metadata["param_names"], axes)
                if name != "size"]
    return parameters, axes, result


def _parallel_grid(name, size):
    """Return jobs, block fractions, and a grid in heatmap orientation."""
    parameters, axes, result = _result_array(name, size=size)
    jobs_axis = parameters.index("n_jobs")
    block_axis = parameters.index("block_fraction")
    jobs = [int(value) for value in axes[jobs_axis]]
    fractions = [float(value) for value in axes[block_axis]]
    grid = np.asarray(result).transpose(block_axis, jobs_axis)
    return jobs, fractions, grid


def _native_value(name, size):
    """Return the scalar native result for one synthetic raster size."""
    _, _, result = _result_array(name, size=size)
    return float(np.asarray(result).reshape(-1)[0])


def _fraction_label(value):
    fraction = Fraction(value).limit_denominator(1000)
    return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)


def _plot_relative_heatmap(ax, jobs, fractions, values, title, colorbar_label):
    """Plot a relative benchmark grid."""
    lower = min(1.0, float(np.nanmin(values)))
    upper = max(1.0, float(np.nanmax(values)))
    if lower == 1.0:
        lower = 0.0
    if upper == 1.0:
        upper = 2.0
    image = ax.imshow(
        values,
        aspect="auto",
        origin="upper",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=lower, vcenter=1.0, vmax=upper),
    )
    ax.set_xticks(range(len(jobs)))
    ax.set_xticklabels([str(job) for job in jobs])
    ax.set_yticks(range(len(fractions)))
    ax.set_yticklabels([_fraction_label(value) for value in fractions])
    ax.set_xlabel("n_jobs (1 = native reference)")
    ax.set_ylabel("block size (fraction of raster side)")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)


def _plot_routine(time_name, memory_name, native_time_name,
                  native_memory_name, size, routine_title):
    """Load and plot one routine's runtime and memory benchmark results."""
    jobs, fractions, times = _parallel_grid(time_name, size=size)
    _, _, memory = _parallel_grid(memory_name, size=size)
    native_time = _native_value(native_time_name, size=size)
    native_memory = _native_value(native_memory_name, size=size)

    times = times / native_time
    memory = memory / native_memory
    jobs = [1, *jobs]
    native_column = np.ones((len(fractions), 1))
    times = np.column_stack((native_column, times))
    memory = np.column_stack((native_column, memory))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"{routine_title} (synthetic {size} x {size} raster)")
    _plot_relative_heatmap(
        axes[0], jobs, fractions, times,
        "Wall time relative to native", "relative runtime (native = 1)",
    )
    _plot_relative_heatmap(
        axes[1], jobs, fractions, memory,
        "Peak memory relative to native", "relative memory (native = 1)",
    )
    return fig
# sphinx_gallery_end_ignore


# %%
# Gaussian-filter benchmark results
# ----------------------------------
#
# These heatmaps show the runtime and peak process-tree memory of the block-wise
# filter relative to the native implementation.  The native value is repeated
# as the ``n_jobs = 1`` reference column.
#
_plot_routine(
    FILTER_TIME,
    FILTER_MEMORY,
    FILTER_NATIVE_TIME,
    FILTER_NATIVE_MEMORY,
    10000,
    "Gaussian filtering (apply_filter + bpgaussian)",
)


# %%
# Multiple linear regression as additive block contributions
# ------------------------------------------------------------
#
# Let ``y`` be the response and let ``X`` be the design matrix.  With an
# intercept, ``X`` has one column for every predictor plus a final column of
# ones.  The ordinary least-squares estimate is obtained from the normal
# equations:
#
# .. math::
#
#    \hat{\beta} = (X^T X)^{-1} X^T y.
#
# For a large raster, each usable pixel is one observation.  Loading all
# observations into ``X`` would make the largest array proportional to the
# number of pixels.  Instead, split the observations into spatial blocks
# ``X_1, X_2, ..., X_B`` and corresponding response vectors ``y_b``.  Matrix
# multiplication is additive over rows, so:
#
# .. math::
#
#    X^T X = \sum_{b=1}^{B} X_b^T X_b,
#    \qquad
#    X^T y = \sum_{b=1}^{B} X_b^T y_b.
#
# Each block can therefore be processed independently.  The order in which
# the partial matrices are added does not change the mathematical result,
# although floating-point addition can introduce tiny rounding differences.
#
# The current implementation makes this decomposition explicit in two passes:
#
# 1. ``get_XT_X`` computes ``X_b.T @ X_b`` for every block and adds the
#    results.  The result is the global ``X.T @ X`` matrix.
# 2. ``compute_weights`` inverts this accumulated matrix, then
#    ``get_optimal_betas`` computes and adds ``X_b.T @ y_b`` for every block.
# 3. The coefficient vector is calculated as ``inv(X.T @ X) @ (X.T @ y)``.
#
# The regression state that must be retained between blocks is thus only the
# ``(p + 1) x (p + 1)`` cross-product matrix, where ``p`` is the number of
# predictors, plus two vectors of length ``p + 1``.  This is the key memory
# benefit of the formulation: the accumulated mathematical state depends on
# the number of predictors, not on the number of raster pixels.
#
# This does *not* mean that every byte of peak memory is independent of raster
# size.  A worker temporarily holds the data for its current block, and the
# corresponding design matrix has approximately ``block_pixels x (p + 1)``
# entries.  Peak memory also includes the selector, multiprocessing workers,
# and queued partial results.  Block size and the number of workers therefore
# remain important memory parameters.


# %%
# Two independent tuning dimensions
# ----------------------------------
#
# ``block_size`` controls how much data one worker handles at a time.  Smaller
# blocks reduce the temporary working set, but increase the number of blocks
# and consequently the scheduling and I/O overhead.  Larger blocks usually
# reduce that overhead, while requiring more memory per worker.
#
# ``n_jobs`` controls how many blocks may be processed concurrently.  More
# workers can reduce elapsed time when CPU and I/O resources are available,
# but each worker can have its own block-sized working set.  Increasing
# ``n_jobs`` can therefore increase peak memory substantially.
#
# These controls allow the computation to be adapted to the machine:
#
# * with limited RAM, use fewer workers and smaller blocks, effectively
#   processing a longer sequence of smaller tasks;
# * with more RAM, use larger blocks to reduce overhead; and
# * with more available CPU resources, increase ``n_jobs`` to process more
#   blocks concurrently.
#
# Thus the implementation is not merely a parallel version of the same
# computation.  It exposes a practical runtime--memory trade-off in two
# dimensions while keeping the regression's accumulated state small.


# %%
# Relative runtime and memory
# ---------------------------
#
_plot_routine(
    PARALLEL_TIME,
    PARALLEL_MEMORY,
    NATIVE_TIME,
    NATIVE_MEMORY,
    10000,
    "Multiple linear regression (compute_weights)",
)


# %%
# Reading the plots
# -----------------
#
# The left heatmap shows whether a block-size/worker combination reduces
# elapsed time compared with the native implementation.  The right heatmap
# shows the memory cost of achieving that runtime.  A cell with value 0.75 on
# the runtime plot completes in 75 percent of the native time; a cell with
# value 1.5 on the memory plot uses 150 percent of native peak memory.
#
# The useful configuration is consequently not necessarily the fastest cell.
# On a memory-constrained machine, a slightly slower cell below the available
# memory limit is preferable to a faster cell that cannot run.  Conversely,
# when sufficient RAM and CPU capacity are available, larger blocks and more
# workers may reduce runtime at the cost of a larger peak working set.
