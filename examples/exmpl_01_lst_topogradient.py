# - - - - - - - - - - - - - - - - - - - - - - - -
# Example use case
# - - - - - - - - - - - - - - - - - - - - - - - -
#
# Descritption scenario:
# We have a dataset on land surface temperature (LST from MODIS) for the mean summer LST for the Alps.
# We want to know what the lapse rate in that region, meaning the change of elevation
# Therefore we need to fit a model where we have LST as the response and elevation as the predictor.
# Yet the gradient might be slightly different in different regions and also climate zones. To make regions comparable,
# and fit one model - we need to remove the region climate. We will do this using a convolution to estimate mean regional climate.

# Steps:
#   1) ...
# - - - - - - - - - - - - - - - - - - - - - - - -

import os
import shutil
import numpy as np
from landiv_blur.io_ import Source, Band
from landiv_blur import parallel as ldpara
from landiv_blur import filters as ldfilter

# Parameters
base_dir = os.path.dirname(__file__)
lst_file_org = os.path.join(base_dir, "../data/example/lst_day_mean_summer_2015_MODISLST8D_alps.tif")
lct_file = os.path.join(base_dir, "../data/example/lc_frac_plots_cgls2015_alps.tif")
topo_file_org = os.path.join(base_dir, "../data/example/topo_mean_COP90_alps.tif")
count_file = os.path.join(base_dir, "../data/example/countries_alps.tif")

params = dict(nbrcpu=6)
block_size = (200, 200)


def main():

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1. Data preparation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    lst_file = os.path.join(base_dir, "../data/example/_tmp_lst_diff_alps.tif")
    shutil.copy(lst_file_org, lst_file)

    lst_source = Source(path=lst_file)
    lst_profile = lst_source.import_profile()
    lst_band = Band(lst_source, bidx=1)

    data_type = lst_profile.get("dtype")


    # Elevation preparation
    topo_file = os.path.join(base_dir, "../data/example/_tmp_elevation_diff_alps.tif")
    shutil.copy(topo_file_org, topo_file)
    elev_cat = "elevation_mean"
    topo_source = Source(path=topo_file)
    topo_profile = topo_source.import_profile()
    topo_source.set_tags(bidx=1,
                         tags=dict(
                             category=elev_cat
                         ))
    elev_band = topo_source.get_band(category=elev_cat)


    # - - - - - - - - - - - - - - - - - - - - - - - -
    # 2. Convolution
    # - - - - - - - - - - - - - - - - - - - - - - - -
    # parameters model
    kernel_truncate = 3 # in sigma
    kernel_m_sigma = 30000 # in meters
    resolution = 1000 # 1,000 m ~Modis LST
    kernel_pixel_sigma = kernel_m_sigma / resolution


    params_filter = dict(
        sigma=kernel_pixel_sigma,
        truncate=kernel_truncate,
        preserve_range=True,
    )

    # 2.1 LST
    # TODO: We need to implement that files will be initated if they dont exist
    lst_conv_file = os.path.join(base_dir, f"../data/example/_tmp_lst_conv_{kernel_m_sigma}m_alps.tif")
    lst_conv_source = Source(path=lst_conv_file,
                                   profile=lst_profile)
    lst_conv_source.init_source(overwrite=True)
    lst_conv_band = Band(lst_conv_source, bidx=1)

    ldpara.apply_filter(
        source=lst_source,
        output_file=lst_conv_file,
        block_size=block_size, # check again
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


    # 2.2 Topography
    elev_conv_file = os.path.join(base_dir, f"../data/example/elev_conv_{kernel_m_sigma}m_alps.tif")
    elev_conv_source = Source(path=elev_conv_file,
                                   profile=topo_profile)
    elev_conv_source.init_source(overwrite=True)
    elev_conv_band = Band(elev_conv_source, bidx=1)

    ldpara.apply_filter(
        source=topo_source,
        output_file=elev_conv_file,
        bands=[elev_band],
        block_size=block_size,
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
    elev_band.subtract(band=elev_conv_band)

    # - - - - - - - - - - - - - - - - - - - - - - - -
    # 3. Fit model
    # - - - - - - - - - - - - - - - - - - - - - - - -
    predictors = []

    # TODO: for now leave the lc computation out
    # lct_source = Source(path=lct_file)
    # lct_profile = lct_source.import_profile()
    # lc_categories = list(range(1, lct_profile['count'] + 1))
    # lct_band_list = [lct_source.get_band(bidx=band_id) for band_id in lc_categories]
    # # Mask and Predictors
    # ldpara.compute_mask(lct_source,
    #                     bands=lct_band_list,
    #                     logic='all',
    #                     nodata=0.0,
    #                     block_size=block_size,
    #                     **params)
    # for band in lct_band_list:
    #     band.set_mask_reader(use='source')
    # predictors.extend(lct_band_list)


    # Masking and predictors
    ldpara.compute_mask(topo_source,
                        bands=[elev_band],
                        logic='all',
                        nodata=np.nan,
                        block_size=block_size,
                        **params)
    elev_band.set_mask_reader(use='source')
    predictors.append(elev_band)


    band_weight = ldpara.compute_weights(
        response=lst_band,
        predictors=predictors,
        block_size=block_size,
        include_intercept=False,
        as_dtype=data_type,
        limit_contribution=0.0,
        no_data=np.nan,
        sanitize_predictors=True,
        return_linear_dependent_predictors=True,
        verbose=False,
        # extra_masking_band=ecoreg_masking_band, ( maybe countries pixels)
        **params
    )

    print(band_weight)

    beta_elev = band_weight[elev_band]
    lapse_rate = beta_elev * 100 # tranform from /meter to /100 km
    print(f"\nLapse rate is:{lapse_rate}")


    # - - - - - - - - - - - - - - - - - - - - - - - -
    # 4. (Reverse) Compute Model
    # - - - - - - - - - - - - - - - - - - - - - - - -
    model_file = os.path.join(base_dir, f"../data/example/_tmp_model_conv_{kernel_m_sigma}_m.tif")

    model_data_tif = ldpara.compute_model(
        predictors=predictors,
        optimal_weights=band_weight,
        output_file=model_file,
        block_size=block_size,
        profile=lst_profile,
        # selector_band=ecoreg_band,
        verbose=False,
        **params)
    model_source = Source(path=model_data_tif)
    model_band = model_source.get_band(bidx=1)
    model_band.add(band=lst_conv_band)

    #
    lst_org_source = Source(path=lst_file_org)
    lst_org_band = Band(lst_org_source, bidx=1)
    ldpara.compute_mask(lst_org_source,
                        bands=[lst_org_band],
                        logic='all',
                        nodata=np.nan,
                        block_size=block_size,
                        **params)
    lst_org_band.set_mask_reader(use='source')

    _selector_all = ldpara.prepare_selector(lst_org_band, *predictors,
                                            block_size=block_size, )
    rmse = ldpara.calculate_rmse(response=lst_org_band,
                                 model=model_data_tif,
                                 selector=_selector_all,
                                 block_size=block_size,
                                 **params)
    print(f"{rmse=}")
    r2 = ldpara.calculate_r2(response=lst_org_band,
                             model=model_data_tif,
                             selector=_selector_all,
                             block_size=block_size,
                             **params)
    print(f"{r2=}")


    # for f in [model_file, lst_file, topo_file]:
    #     if os.path.exists(f):
    #         os.remove(f)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    main()