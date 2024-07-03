"""
This script demonstrates how to separate a land-cover type .tif file into
separate layers, apply a gaussian filter on each layer and compute the
Shannon entropy for each pixel afterwards.
"""

import numpy as np
from skimage.filters import gaussian
from scipy.stats import entropy

from landiv_blur import helper as lbhelp
from landiv_blur import processing as lbproc


# We start with creating a random land-cover type "map":
rand_data = np.random.randint(8, size=(100, 200)) + 1

###
# Step-by-step guide
###
# Next we construct a map for a single land-cover type from the map
mono_lctype_map = lbproc.select_layer(rand_data, layer=3)
# Now we can apply a filter on the resulting map:
blurred = lbproc.apply_filter(mono_lctype_map, gaussian, sigma=1)
# convert it back to uint8
n_max, _ = lbhelp.dtype_range(np.uint8)
blurred = blurred * n_max
blurred = blurred.astype(np.uint8, copy=False)

# Let's to the same thing for another land-cover type:
mono_lctype_map_2 = lbproc.select_layer(rand_data, layer=2)
blurred_2 = lbproc.apply_filter(mono_lctype_map_2, gaussian, sigma=1)
blurred_2 = blurred_2 * n_max
blurred_2 = blurred_2.astype(np.uint8, copy=False)

# Now we combine them and compute the per pixel Shannon entropy
combined = np.stack([blurred, blurred_2], axis=2)
entropy_layer = entropy(combined, axis=2)

###
# Faster
###
# We can do the same thing with a single command:
entropy_layer_fast = lbproc.get_entropy(
    rand_data,
    layers=[2, 3],
    normed=False,
    img_filter=gaussian,
    filter_params=dict(
        sigma=1
    )
)

# to make sure hat both approaches are equal:
print('Differences:')
print(np.unique(entropy_layer_fast - entropy_layer))
