import os

import numpy as np

from landiv_blur import prepare as lbprep
from landiv_blur import io as lbio
from landiv_blur import io_ as lbio_
from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lb_filter

# we use the test data in this example
DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'data'
))
# generated data we write here
INTERIM_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'results'
))
lct_map = os.path.join(DATA_DIR, 'testing', 'landcover',
                 'Switzerland_CLC_2012_reclass8.tif')
_ndvi_map = os.path.join(DATA_DIR, 'testing', 'ndvi',
                 'Switzerland_NDVI_binning_2015.tif')

# ###
# Before we start we need to resample the ndvi map as it has a different resolution
# scale it down to 100x100m (from 30x30)
ndvi_map = os.path.join(INTERIM_DIR, 'resampled_NDVI_CH_map_toDELETE.tif')
lbio.coregister_raster(_ndvi_map, lct_map, output=ndvi_map)
# now we are good to go

# ###
# First we want to compute and save per-land-cover type blurred bands
# ###
# setting some parameter
blurred_tif = os.path.join(INTERIM_DIR, 'blurred_CH_map_toDELETE.tif')
categories = [1,2,3,4,5,6]  # what categories to extract
diameter = 5000  # this is in meter
scale = 100  # meter per pixel
_diameter = diameter / scale
truncate = 3  # property of the gaussian filter
block_size = (500, 400)  # how big of a block schould a single job handle
                         # (in pixels)
blur_output_dtype = np.uint8  # data type to use for computing the entropy array
blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
filter_params = dict(sigma=blur_params['sigma'], truncate=blur_params['truncate'])
# now we can start with generating the blurred layers
source = lbio_.Source(path=lct_map)
blurred_tif = lbpara.extract_categories(
    source=source,
    categories=categories,
    output_file=blurred_tif,
    img_filter=lb_filter.gaussian,  # which filter function to apply
    filter_params=filter_params,
    output_dtype=blur_output_dtype,
    block_size=block_size,
    compress = True
)
# we directly create a Source object for this file
blurred_source = lbio_.Source(path=blurred_tif)

# ###
# Second we prepare the mask for this blurred tif
# ###
# we only want pixels for which at least one category has a non-zero value
lbpara.compute_mask(source=blurred_source, block_size=block_size)

# ###
# Next we compute the per cell entropy of all these categories
# ###
# setting some parameter
entropy_tif = os.path.join(INTERIM_DIR, 'entropy_CH_map_toDELETE.tif')
normed = True  # express entropy relative to maximal possible value (fraction)
entropy_as_ubyte = True  # map the resuliting values [0, 1] to [0, 255]
entropy_tif = lbpara.compute_entropy(
    source=blurred_source,
    output_file=entropy_tif,
    block_size=block_size,  # we could choose bigger blocks here!
    blur_params=blur_params.copy(),
    categories=categories,  # we could use a different selection
    entropy_as_ubyte=entropy_as_ubyte,
    normed=normed,
)
# also create a Source object for this file
entropy_source = lbio_.Source(path=entropy_tif)

# ###
# Finally we perform a multiple linear regression to predict the ndiv values
# ###
# here also we need to set some parameter
include_intercept = True  # we want to fit the intercept as well
as_dtype = np.float64  # data type to use for the weights
block_size = (500, 500)
# define the response
response_band = lbio_.Band(source=lbio_.Source(path=ndvi_map), bidx=1)
# and define what to use as predictors
predictors = blurred_source.get_bands()  # we use all bands with the blurred
                                         # categories
# define an entropy band
entropy_band = entropy_source.get_band(category='entropy')  # use tags to det-
                                                            # ermine the index
predictors.append(entropy_band)  # add this to the predictors as well
# now we need to specfiy what mask each band should use
for pred_band in predictors:
    pred_band.set_mask_reader(use='source')
# and finally we compute the optimal weights
optimal_weights = lbpara.compute_weights(response=response_band,
                                         predictors=predictors,
                                         include_intercept=include_intercept,
                                         block_size=block_size,
                                         as_dtype=as_dtype,
                                         limit_contribution=0.5,
                                         sanitize_predictors=True,
                                         verbose=True,
                                         )
print(f"{optimal_weights=}")
