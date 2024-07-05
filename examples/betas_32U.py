import os

import numpy as np
import rasterio as rio

from landiv_blur import io as lbio
from landiv_blur import parallel as lbpara
from landiv_blur import inference as lbinf

data_path = os.path.join('..', 'data', 'ldiv', 'project01_ndiv_scale')
procdata_path = os.path.join(data_path, 'process_datasets')
lctdata_path = os.path.join(procdata_path,
                            'landcover_smoothed_global',
                            'europe')
ndvidata_path = os.path.join(procdata_path, 'ndvi_clipped_global', 'europe')


def get_optimal_weights():
    """Calculate the transposed product of a predictor matrix
    """
    # data format of output and interim matrices
    as_dtype = np.float64
    # size (in pixels) of single slices to process in parallel
    view_size = (4000, 4000)
    # if the intercept should be fitted separately
    include_intercept = True
    # do some print statements
    verbose = True
    
    # prepare the input
    response = os.path.join(ndvidata_path,
            '32U_E05_buffer_0_ndvi_2014_2016.tif')
    lctblurred_file =  os.path.join(
        lctdata_path,
        'lc_heterogeneity_32U_blurred_diameter_5000_sigma_833_truncate_3_compress.tif'
    )
    predictors = (
        # TODO: there are landcover bands that mask 100%: check
        #(lctblurred_file, 1),
        (lctblurred_file, 2),
        (lctblurred_file, 3),
        #(lctblurred_file, 4),
        (lctblurred_file, 5),
        #(lctblurred_file, 6),
        (
            os.path.join(lctdata_path,
                        'lc_heterogeneity_32U_entropy_diameter_5000_sigma_833_truncate_3_compress.tif'),
            1
        )
    )
    # firs step: use the response mask and enrich it with the predictor masks
    #            to create a data selector
    print("Creating selector...")
    # TODO: selector is calculated within get_XT_X again: remove from fct!
    selector = lbinf.prepare_selector(response,
                                      *predictors)
    print("\tdone!")
    # compute (X^T X) in parallel 
    print("Calculate X.T @ X...")
    tpX = lbpara.get_XT_X(response,
                          *predictors,
                          selector=selector,
                          include_intercept=include_intercept,
                          verbose=verbose,
                          view_size=view_size,
                          )
    print("\tdone!")
    # invert the transposed product
    print("Inverting X.T @ X...")
    Y = np.linalg.inv(tpX)
    print("\tdone!")
    # compute Y @ X.T @ y in parallel (Y see above, and y is the response data)
    print("Calculate Y @ X.T @ y (optimal weights)...")
    betas = lbpara.get_optimal_betas(*predictors,
                                     Y=Y,
                                     response=response,
                                     selector=selector,
                                     include_intercept=include_intercept,
                                     verbose=verbose,
                                     as_dtype=as_dtype,
                                     view_size=view_size,
                                     )
    print("\tdone!")
    return betas


if __name__ == '__main__':
    betas = get_optimal_weights()
    print(f"{betas=}")
