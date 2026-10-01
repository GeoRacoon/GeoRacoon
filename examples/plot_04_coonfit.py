# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

"""
Coonfit: Pixel-wise Regression of NDVI on Land-Cover Fractions
===============================================================

:mod:`coonfit` implements pixel-wise ordinary least squares regression for
raster data, parallelised over spatial blocks.  The core workflow is:

1. :func:`~coonfit.parallel.compute_weights`: solve the OLS normal equations
   and return per-predictor β coefficients keyed by predictor
   :class:`~riogrande.io.models.Band`.
2. :func:`~coonfit.parallel.compute_model`: reconstruct the spatial prediction
   from the fitted weights.
3. :func:`~coonfit.parallel.calculate_rmse` and
   :func:`~coonfit.parallel.calculate_r2`: quantify model accuracy.

This example uses four land-cover fraction bands from the CGLS dataset
introduced in :ref:`sphx_glr_auto_examples_plot_02_riogrande.py` to predict NDVI over Switzerland.
These four classes (forest, grassland, agriculture, and urban) span the main gradient of vegetation density and together
do not sum to one, so there is no multicollinearity issue.
"""

# %%
# Setup
# -----
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import pyplot as plt

from riogrande.io import Source, Band, coregister_raster
from riogrande import parallel as rgpara

from coonfit import parallel as lfpara

# Fetches the example rasters from Zenodo on first use, then reuses the cache
from riogrande.data import fetch

# %%
# Working directory for intermediate data
# ---------------------------------------
#
# The input rasters are obtained with :func:`~riogrande.data.fetch`, which
# downloads them from Zenodo on first use and returns the path to the cached
# file. Their pixel values are never modified. Every raster this example
# derives from them (coregistered NDVI, model) is written to a temporary
# directory, which is removed again at the end of this example.

work_dir_handle = TemporaryDirectory(prefix="georacoon_plot_04_")
work_dir = Path(work_dir_handle.name)

# %%
# Load the predictor bands
# -------------------------
#
# We open the 10-band CGLS fraction grid and tag only the four bands we need.
# The tags are attached to the :class:`~riogrande.io.models.Band` objects
# only, so the fetched file is not modified. Tagging the bands keeps the
# weight dictionary self-documenting.

lct_file_org = fetch("examples/switzerland_lc-area-fraction_2015_CGLS-LC100_sinusoidal.tif")
lct_source = Source(path=lct_file_org)

forest      = Band(source=lct_source, bidx=1, tags=dict(category="forest"))
grassland   = Band(source=lct_source, bidx=3, tags=dict(category="grassland"))
agriculture = Band(source=lct_source, bidx=4,
                   tags=dict(category="agriculture"))
urban       = Band(source=lct_source, bidx=5, tags=dict(category="urban"))

predictors = [forest, grassland, agriculture, urban]

# %%
# Coregister NDVI to the predictor grid
# ---------------------------------------
#
# The NDVI composite has been pre-reprojected to the CRS of the
# CGLS fraction grid, but retains a finer pixel spacing.
# :func:`~riogrande.io.core.coregister_raster` resamples it to match the
# 1 km pixel grid of the predictor source exactly. It only reads the fetched
# NDVI file and writes the result to a new file in the working directory.

ndvi_file_org = fetch("examples/switzerland_ndvi-binned-mean_2015_LANDSAT-8_sinusoidal.tif")
ndvi_file = coregister_raster(source=ndvi_file_org,
                              reference=lct_file_org,
                              output=str(work_dir / "ndvi_coreg_1km.tif"))

ndvi_source = Source(path=ndvi_file)
ndvi_band   = Band(source=ndvi_source, bidx=1)

# %%
# Compute a valid-pixel mask
# ---------------------------
#
# :func:`~riogrande.parallel.compute_mask` iterates over the source in
# 200 × 200 pixel blocks and marks pixels that are ``np.nan`` in any of the
# predictor bands (``logic="all"``). The mask is written back to the source
# file and attached to each band via
# :meth:`~riogrande.io.models.Band.set_mask_reader`, so the fitting step
# skips nodata pixels (border artefacts) automatically.
#
# .. note::
#
#    :func:`~riogrande.parallel.compute_mask` writes the computed dataset
#    mask into the fetched land-cover file itself (via
#    :meth:`~riogrande.io.models.Source.mask_writer`). Only the mask is
#    added; the pixel values remain unchanged.

block_size = (200, 200)
params     = dict(n_jobs=6)  # follows the scikit-learn n_jobs convention

rgpara.compute_mask(lct_source, bands=predictors, logic="all",
                    nodata=np.nan, block_size=block_size, **params)
for pred in predictors:
    pred.set_mask_reader(use="source")

# %%
# Fit the regression
# -------------------
#
# :func:`~coonfit.parallel.compute_weights` solves the OLS normal equations
# per spatial block and returns a dict mapping each predictor
# :class:`~riogrande.io.models.Band` to its fitted β coefficient.
# We expect forest and grassland to carry a positive β, with forest higher
# (dense canopy having a high NDVI vs open grass cover), while agriculture (cropland,
# seasonally low NDVI) and urban (impervious surface) carry negative or near-zero ones.

band_weight = lfpara.compute_weights(
    response=ndvi_band,
    predictors=predictors,
    block_size=block_size,
    include_intercept=True,
    as_dtype=np.float32,
    limit_contribution=0.0,
    no_data=np.nan,
    sanitize_predictors=True,
    return_linear_dependent_predictors=True,
    verbose=False,
    **params,
)

print("Fitted β coefficients (NDVI per unit fraction):")
for band, beta in band_weight.items():
    label = band if isinstance(band, str) else band.tags.get("category", f"band {band.bidx}")
    print(f"  {label:12s}  β = {beta:+.4f}")

# %%
# Reconstruct the model and assess accuracy
# ------------------------------------------
#
# :func:`~coonfit.parallel.compute_model` applies the fitted weights spatially
# to produce a predicted NDVI map.  RMSE and R² are then computed against the
# observed NDVI using a shared valid-pixel selector.

model_file = work_dir / "coonfit_ndvi_lct_model.tif"
lfpara.compute_model(
    predictors=predictors,
    optimal_weights=band_weight,
    output_file=str(model_file),
    block_size=block_size,
    profile=ndvi_source.import_profile(),
    verbose=False,
    **params,
)

_selector = rgpara.prepare_selector(ndvi_band, *predictors, block_size=block_size)
rmse = lfpara.calculate_rmse(response=ndvi_band, model=model_file,
                             selector=_selector, block_size=block_size, **params)
r2   = lfpara.calculate_r2(response=ndvi_band,   model=model_file,
                           selector=_selector,   block_size=block_size, **params)

print(f"RMSE = {rmse:.4f}  |  R² = {r2:.2f}")

# %%
# Visualise observed vs. predicted NDVI
# --------------------------------------
#
# The residual between observed and modelled NDVI highlights pixels where
# four land-cover fractions alone cannot fully explain the NDVI signal.


def show_map(ax, file, title, cmap, limits):
    src  = Source(path=file)
    data = Band(source=src, bidx=1).get_data()
    img  = ax.imshow(data, cmap=cmap, vmin=limits[0], vmax=limits[1])
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    return img


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
img = show_map(axes[0], ndvi_file,  "Observed NDVI",
               cmap="PRGn", limits=(-1, 1))
show_map(axes[1],       model_file, f"Modelled NDVI  (R² = {r2:.2f})",
               cmap="PRGn", limits=(-1, 1))
fig.colorbar(img, ax=axes.ravel().tolist(), label="NDVI", shrink=0.8, pad=0.02)
fig.suptitle("NDVI predicted from forest, grassland, agriculture and urban fractions"
             " - Switzerland 1 km (CGLS 2015)",
             fontweight="bold", fontsize=12)
plt.show()

# %%
# Clean up
# --------
#
# Remove the temporary working directory together with all intermediate
# rasters created in this example.

work_dir_handle.cleanup()
