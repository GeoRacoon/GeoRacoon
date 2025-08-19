# - - - - - - - - - - - - - - - - - - - - - - - -
# Example use case
# - - - - - - - - - - - - - - - - - - - - - - - -
#
# Descritption scenario:
# We have a dataset on land surface temperature (LST from MODIS) for the mean summer LST for the Alps.
# We want to know what the adiabatic gradient ist in that region, meaning the change of elevation
from defusedxml.ElementTree import parse
# Therefore we need to fit a model where we have LST as the response and elevation as the predictor.
# Yet the gradient might be slightly different in different regions and also climate zones. To make regions comparable,
# and fit one model - we need to remove the region climate. We will do this using a convolution to estimate mean regional climate.

# Steps:
#   1) ...
# - - - - - - - - - - - - - - - - - - - - - - - -

import os
import shutil
from landiv_blur.io_ import Source, Band
from landiv_blur import parallel as ldpara
from landiv_blur import filters as ldfilter

# Parameters
base_dir = os.path.dirname(__file__)
lst_file_org = os.path.join(base_dir, "../data/example/lst_day_mean_summer_2015_MODISLST8D_alps.tif")
topo_file = os.path.join(base_dir, "../data/example/topo_mean_COP90_alps.tif")

params = dict(nbrcpu=4)


def main():

    # 1. Setup sources
    lst_file = os.path.join(base_dir, "../data/example/_tmp_lst_day_mean_summer_2015_MODISLST8D_alps.tif")
    shutil.copy(lst_file_org, lst_file)

    lst_source = Source(path=lst_file)
    lst_profile = lst_source.import_profile()
    lst_band = Band(lst_source, bidx=1)

    data_type = lst_profile.get("dtype")
    print(data_type)


    # 1. Convolution

    # parameters model
    kernel_truncate = 3 # in sigma
    kernel_m_sigma = 20000 # in meters
    resolution = 1000 # 1,000 m ~Modis LST
    kernel_pixel_sigma = kernel_m_sigma / resolution


    params_filter = dict(
        sigma=kernel_pixel_sigma,
        truncate=kernel_truncate,
        preserve_range=True,
    )

    # TODO: We need to implement that files will be initated if they dont exist
    lst_conv_file = os.path.join(base_dir, f"../data/example/lst_conv_{kernel_m_sigma}m_day_mean_summer_2015_MODISLST8D_alps.tif")
    lst_conv_source = Source(path=lst_conv_file,
                                   profile=lst_profile)
    lst_conv_source.init_source(overwrite=True)
    lst_conv_band = Band(lst_conv_source, bidx=1)

    ldpara.apply_filter(
        source=lst_source,
        output_file=lst_conv_file,
        block_size=(200, 200), # check again
        data_in_range=None,
        data_as_dtype=data_type,
        data_output_range=None,
        img_filter=ldfilter.bpgaussian, # border preserving gaussian
        filter_params=params_filter,
        filter_output_range=None,
        output_dtype=data_type,
        output_range=None,
        selector_band=None,
        **params
    )

    # Remove convoluted from
    lst_band.subtract(band=lst_conv_band)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    main()