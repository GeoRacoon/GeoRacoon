"""
RioGrande: Loading and Exploring a Multi-Band Raster
=========================================================

:mod:`riogrande` is the raster I/O backbone of GeoRacoon.
It provides two core objects:
- :class:`~riogrande.io.models.Source` and
- :class:`~riogrande.io.models.Band`
that wrap ``rasterio`` and keep track of file-level metadata (profile, tags, masks) separately from band-level
pixel data.

This example uses a 10-band land-cover fraction grid of Switzerland at 1 km resolution
(aggregated Copernicus Global Land Service) to show how to open a multi-band file,
tag each band with a meaningful label, retrieve bands by tag, and visualise
the full stack at a glance.
"""

# %%
# Setup
# -----
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt

from riogrande.io import Source, Band

# %%
# Open the file and inspect the profile
# ---------------------------------------
#
# A :class:`~riogrande.io.models.Source` wraps a raster file.
# :meth:`~riogrande.io.models.Source.import_profile` returns the standard
# ``rasterio`` profile dict - CRS, transform, dtype, nodata, and band count.

base_dir     = os.getcwd()
lct_file_org = os.path.join(base_dir,
                            "../data/testing/landcover/"
                            "Switzerland_area_frac_grid_1km_CGLS_2015.tif")

lct_source_org = Source(path=lct_file_org)
lct_profile    = lct_source_org.import_profile()

print(f"Shape : {lct_profile['height']} × {lct_profile['width']}")
print(f"Bands : {lct_profile['count']}")
print(f"Dtype : {lct_profile['dtype']}")
print(f"Nodata: {lct_profile['nodata']}")

# %%
# Tag each band and retrieve bands by tag
# ----------------------------------------
#
# :meth:`~riogrande.io.models.Source.set_tags` writes metadata to the file (inplace),
# so we work on a copy to leave the original untouched.
# :meth:`~riogrande.io.models.Source.get_band` can then look up the band by
# any of the tag keys. This is useful when a source holds many bands with different
# roles, for instance raw multisepcral images or land-cover categories.

# Work on a copy so the original file is never altered
lct_file = os.path.join(base_dir,
                        "../data/example/"
                        "_tmp_lct_frac_tagged.tif")
shutil.copy(src=lct_file_org, dst=lct_file)
lct_source = Source(path=lct_file)

# Band index: land-cover class name
lc_labels = {
    1: "forest",
    2: "shrubland",
    3: "grassland",
    4: "agriculture",
    5: "urban",
    6: "barrenland",
    7: "snow and ice",
    8: "water",
    9: "wetland",
    10: "moss and lichen",
}

for bidx, label in lc_labels.items():
    lct_source.set_tags(bidx=bidx, tags=dict(category=label))

# Retrieve three bands by tag and print their mean +/- std
for key in [1, 5, 8]:
    band = lct_source.get_band(category=lc_labels[key])
    data = band.get_data()
    valid = data[~np.isnan(data)]
    print(f"Band '{lc_labels[key]:12s}':  "
          f"mean = {np.mean(valid):.2f},  "
          f"std = {np.std(valid):.2f}")

# %%
# Visualise the full band stack
# ------------------------------
#
# Iterating over the label dict makes it straightforward to build a compact
# overview of all bands.  :meth:`~riogrande.io.models.Band.get_data` reads
# each band's pixel array on demand.

fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)

for ax, label in zip(axes.flat, lc_labels.values()):
    band = lct_source.get_band(category=label)
    data = band.get_data()
    img  = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_title(label, fontsize=9)

fig.colorbar(img, ax=axes.ravel().tolist(), label="Fraction (0–1)",
             shrink=0.6, pad=0.02)
fig.suptitle("Land-cover fraction per class - Switzerland 1 km (CGLS 2015)",
             fontweight="bold", fontsize=12)
plt.show()