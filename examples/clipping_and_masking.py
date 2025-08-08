import os

from landiv_blur import select as lbselect
from landiv_blur import prepare as lbprep
from landiv_blur import io as lbio


# TODO: this entire script can go - we can delete this file

# We want to clip one map with the bounding box of a reference map.
data_path = '../data/'
map_to_clip_fname = 'ecoregions/gens_v3_recoded.tif'
reference_map_fname = 'ch.tif'
# Create the filenames
to_clip_file = os.path.join(data_path, map_to_clip_fname)
clipping_file = os.path.join(data_path, reference_map_fname)
# If we try without any further considerations:
try:
    clipped_map = lbselect.read_clip(source=to_clip_file,
                                     clipping=clipping_file)
except TypeError as e:
    print(f"ERROR:\n\t\"{e}\"")
# we first need to project our map to clip:
to_clip_rp_file = lbprep.project_to(source=to_clip_file,
                                    reference=clipping_file)
# NOTE: We have created a new .tif file here
print(f"NOTE:\nWe just created the file:\n\t{to_clip_rp_file}")
# Now we can try again with the clipping:
clipped_map = lbselect.read_clip(source=to_clip_rp_file,
                                 clipping=clipping_file)
# now let's see if there is a difference in scale:
scale_factors = lbprep.get_scale_factor(to_clip_rp_file, clipping_file)
print(f"{scale_factors=}")
# TODO: CHECK IF IT IS NOT THE INVERSE!
