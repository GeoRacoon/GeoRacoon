"""
Convster: Smoothing a Multi-Band Raster
============================================

:mod:`convster` provides parallelised spatial filtering for raster data.
Its main entry point is :func:`~convster.parallel.apply_filter`, which tiles
the image into overlapping blocks and applies any filter function. It will then process
*all bands in a single call*.

This example applies the border-preserving Gaussian filter
:func:`~convster.filters.gaussian.bpgaussian` to a 10-band land-cover
fraction grid of Switzerland at 1 km resolution (aggregated from the
`Copernicus Global Land Service 2015).
Smoothing fraction maps spatially produces regionalised land-cover signals that can feed
downstream analyses.
"""

# %%
# Setup
# -----
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

from riogrande.io import Source, Band

from convster import parallel as cvpara
from convster.filters import bpgaussian

# Fetches the example rasters from Zenodo on first use, then reuses the cache
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from data.fetch import fetch

# %%
# Load the source raster
# -----------------------
#
# We use :class:`~riogrande.io.models.Source` from :mod:`riogrande` to open
# the 10-band input file and import its profile. Convster needs the profile
# to write the filtered output with identical CRS, transform, and nodata.

base_dir = os.getcwd()
lct_file = fetch("examples/switzerland_lc-area-fraction_2015_CGLS-LC100_sinusoidal.tif")

lct_source  = Source(path=lct_file)
lct_profile = lct_source.import_profile()

print(f"Input: {lct_profile['count']} bands, "
      f"{lct_profile['height']} × {lct_profile['width']} px")

# %%
#
# .. note::
#
#    Unlike the :ref:`RioGrande example <sphx_glr_auto_examples_plot_02_riogrande.py>`,
#    no copy of the source file is needed here.
#    :func:`~convster.parallel.apply_filter` only reads from the input and
#    writes to a separate output file, so the original is never modified.

# %%
# Configure the filter
# ---------------------
#
# The sigma is specified in pixel units; here we use a moderate 5-pixel (5 km given the raster resolution)
# radius to capture neighbourhood structure without over-smoothing. We choose to cut the kernel at
# 3 σ (``truncate=3``), which retains 99.7 % of the Gaussian
# weight while keeping the kernel compact enough to avoid excessive padding at
# raster edges.

filter_params = dict(
    sigma=5,
    truncate=3,
    preserve_range=True,
)

# %%
# Apply the filter to all bands at once
# --------------------------------------
#
# :func:`~convster.parallel.apply_filter` iterates over every band automatically
# and writes results to the output file with the same profile as the input.
# Key parameters:
#
# - **block_size** ``(rows, cols)``: the raster is tiled into overlapping
#   100 × 100 pixel blocks so the kernel has enough context at boundaries and
#   no edge artefacts appear. Each block is dispatched to a separate worker (``n_jobs=6``,
#   following the `scikit-learn convention <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_).
# - **data_as_dtype** / **output_dtype**: pixels are cast to
#   ``float32`` for filtering and written back as ``float32`` — half the memory
#   of ``float64`` with negligible precision loss for [0, 1] fraction data.
# - **img_filter** :func:`~convster.filters.gaussian.bpgaussian`: normalises
#   kernel weights by valid (non-NaN) neighbours, so masked raster edges are not
#   pulled towards zero as they would be with a standard Gaussian.

output_file   = os.path.join(base_dir,
                             "../data/examples/"
                             "_tmp_lct_conv_sigma5.tif")
output_source = Source(path=output_file, profile=lct_profile)
output_source.init_source(overwrite=True)

cvpara.apply_filter(
    source=lct_source,
    output_file=output_file,
    block_size=(100, 100),
    data_in_range=None,
    data_as_dtype=np.float32,
    data_output_range=None,
    img_filter=bpgaussian,
    filter_params=filter_params,
    filter_output_range=None,
    output_dtype=np.float32,
    output_range=None,
    selector_band=None,
    n_jobs=6,
)

print(f"Output written to: {output_file}")

# %%
# Compare original vs. filtered for selected bands
# -------------------------------------------------
#
# We pick three bands to illustrate the smoothing effect across land-cover
# types with different spatial patterns.

selected = [1, 5, 8]   # band indices to visualise
output_src = Source(path=output_file)

fig, axes = plt.subplots(2, len(selected), figsize=(14, 7),
                         constrained_layout=True)

for col, bidx in enumerate(selected):
    orig_data = Band(source=lct_source,  bidx=bidx).get_data()
    smth_data = Band(source=output_src,  bidx=bidx).get_data()

    axes[0, col].imshow(orig_data, cmap="PuRd", vmin=0, vmax=1)
    axes[0, col].set_axis_off()
    axes[0, col].set_title(f"Band {bidx}: original", fontsize=9)

    img = axes[1, col].imshow(smth_data, cmap="PuRd", vmin=0, vmax=1)
    axes[1, col].set_axis_off()
    axes[1, col].set_title(f"Band {bidx}: Gaussian σ = 5 px", fontsize=9)

fig.colorbar(img, ax=axes.ravel().tolist(), label="Fraction (0–1)",
             shrink=0.6, pad=0.02)
fig.suptitle("Land-cover fractions before and after Gaussian smoothing",
             fontweight="bold", fontsize=12)
plt.show()