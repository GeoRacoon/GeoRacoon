"""
This script demonstrates how to separate a land-cover type .tif file into
separate layers, apply a gaussian filter on each layer and compute the
Shannon entropy for each pixel afterwards.
"""

import numpy as np
from skimage.filters import gaussian
from scipy.stats import entropy

from landiv_blur import processing as lbproc


# We start with creating a random land-cover type "map":
rand_map = np.random.randint(8, size=(100, 200)) + 1

###
# Step-by-step guide
###
# Next we construct a map for a single land-cover type from the map
mono_lctype_map = lbproc.filter_for_layer(rand_map, layer=3)
# Now we can apply a filter on the resulting map:
blurred = lbproc.apply_filter(mono_lctype_map, gaussian, sigma=1)

# Let's to the same thing for another land-cover type:
mono_lctype_map_2 = lbproc.filter_for_layer(rand_map, layer=2)
blurred_2 = lbproc.apply_filter(mono_lctype_map, gaussian, sigma=1)

# Now we combine them and compute the per pixel Shannon entropy
combined = np.stack([blurred, blurred_2], axis=2)
entropy_layer = entropy(combined, axis=2)

###
# Faster
###
# We can do the same thing with a single command:
entropy_layer_fast = lbproc.get_entropy(
    rand_map,
    layers=[2, 3],
    img_filter=gaussian,
    sigma=1
)

# to make sure that both approaches are equal:
print(np.all(entropy_layer_fast == entropy_layer_fast))
