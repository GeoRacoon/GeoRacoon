import os
import numpy as np
import rasterio

from landiv_blur import io as lbio
from landiv_blur.helper import nodata_mask_band

# TODO: this entire script can go - we can delete this file

# We want to clip one map with the bounding box of a reference map.
data_path = '../data'
ndvi_map_to_clip = '32U_E5_NDVI_5bins_2014_2016_sos100_eos298.tif'
reference_map_fname = 'reclass_GLC_FCS30_2015_utm32U.tif'
ecoregion_shapefile = 'GEnS_v3_WGS84_GSL_hemi_singlepart.shp'
ecoregion_num = 5

# Create the filenames
to_clip_file = os.path.join(data_path, ndvi_map_to_clip)
clipping_file = os.path.join(data_path, reference_map_fname)
ecoregion_file = os.path.join(data_path, ecoregion_shapefile)
tmp_file = os.path.join(data_path, f"tmp_{ndvi_map_to_clip}")
final_output = os.path.join(data_path, f"result_{ndvi_map_to_clip}")

# Reproject
img_projected = lbio.project_to(source=to_clip_file,
                                reference=clipping_file,
                                output=tmp_file,
                                nodata=np.nan)
print("Raster reprojected")

# Clip to BBox
img_clipped = lbio.clip_to_bounds(source=img_projected,
                                  reference=clipping_file)
print("Clipped to BBox")

# Coregister
img_coreg = lbio._coregister_raster(source=img_clipped,
                                   reference=clipping_file)
print("Coregistered")

# Clip to ecoregion
img_ecoclip = lbio.clip_to_ecoregion(source=img_coreg,
                                     shapefile=ecoregion_file,
                                     ecoregion_number=ecoregion_num,
                                     buffer_meter=-1000)
print("Clipped to Ecoregion")

# Compress final output
lbio.compress_tif(source=img_ecoclip,
                  output=final_output)
print("Compressed file")

# Make internal no-data mask
nodata_mask_band(source=final_output,
                 nodata=np.nan)
print("Mask created output created")

# Remove temporary processing files
for file_name in os.listdir(data_path):
    if file_name.startswith("tmp_"):
        os.remove(os.path.join(data_path, file_name))
