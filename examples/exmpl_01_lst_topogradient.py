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
#   1) Set up data and Get data
#   2) Convolution
#   3) Fit model
#   4) (Reverse) Compute Model
#   5) Results
#
# - - - - - - - - - - - - - - - - - - - - - - - -

import os
import shutil
from unicodedata import category

import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt

# Our package(s)
from riogrande.io import Source, Band
from riogrande import parallel as rgpara
from convster import parallel as cvpara
from convster.filters import bpgaussian
from linfit import parallel as lfpara

# Parameters
base_dir = os.path.dirname(__file__)
lst_file_org = os.path.join(base_dir, "../data/example/lst_day_mean_summer_2015_MODISLST8D_alps.tif")
topo_file_org = os.path.join(base_dir, "../data/example/elevation_mean_COP90_alps.tif")
# lct_file = os.path.join(base_dir, "../data/example/lc_frac_plots_cgls2015_alps.tif")
# count_file = os.path.join(base_dir, "../data/example/countries_alps.tif")

params = dict(nbrcpu=6)
block_size = (200, 200)
data_type = np.float32

def main():


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1. Data preparation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + " | " * 10 + "Data preparation" + " | " * 10, end="\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.1) Make copies of data (to no alter original datasets)

    # Land Surface Temperature data
    lst_file = os.path.join(base_dir, "../data/example/_tmp_lst_diff_alps.tif")
    shutil.copy(src=lst_file_org, dst=lst_file)

    # Elevation data
    topo_file = os.path.join(base_dir, "../data/example/_tmp_elevation_diff_alps.tif")
    shutil.copy(src=topo_file_org, dst=topo_file)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.2) Set up objects (class Source and Band

    # 1.2.1) LST get Source and use Band (idx=1), as there is only one band present.
    lst_source = Source(path=lst_file)
    lst_profile = lst_source.import_profile()
    lst_band = Band(source=lst_source, bidx=1)

    # 1.2.1) Similar with elevantion, but we need to set the tag
    # (so later for the computation of weights/betas we have a name for the predictor)
    topo_source = Source(path=topo_file)
    topo_profile = topo_source.import_profile()

    # Set tags and get Band (not by idx but by tag)
    elev_cat = "elevation_mean"
    topo_source.set_tags(bidx=1,
                         tags=dict(
                             category=elev_cat
                         ))
    elev_band = topo_source.get_band(category=elev_cat)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 2. Convolution
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + " | " * 10 + "Convolution" + " | " * 10, end="\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 2.1) Parameters for convolution model
    kernel_truncate = 3 # in sigma
    kernel_m_sigma = 30000 # in meters
    resolution = 1000 # 1,000 m ~Modis LST
    kernel_pixel_sigma = kernel_m_sigma / resolution

    params_filter = dict(
        sigma=kernel_pixel_sigma,
        truncate=kernel_truncate,
        preserve_range=True,
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 2.2 LST convolution

    # This is done to model regional cliamte provided the filter parameters from above.
    # (Such provides an example with arbitrary sigma in meters, as a user you may want to optimize this programmatically)

    # Initate empty file for later
    lst_conv_file = os.path.join(base_dir, f"../data/example/_tmp_lst_conv_{kernel_m_sigma}m_alps.tif")
    lst_conv_source = Source(path=lst_conv_file, profile=lst_profile)
    lst_conv_source.init_source(overwrite=True)
    lst_conv_band = Band(lst_conv_source, bidx=1)

    # 2.2.1 Actual convolution
    # TODO: We need to implement that files will be initated if they dont exist
    cvpara.apply_filter(
        source=lst_source,
        output_file=lst_conv_file,
        block_size=block_size,
        data_in_range=None,
        data_as_dtype=data_type,
        data_output_range=None,
        img_filter=bpgaussian, # border preserving gaussian
        filter_params=params_filter,
        filter_output_range=None,
        output_dtype=data_type,
        output_range=None,
        selector_band=None,
        **params
    )

    # 2.2.1 Remove convoluted from original
    # As the convolution simulates the regional climate, the difference will show the deviation from this
    lst_band.subtract(band=lst_conv_band)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 2.3 Topography convolution

    # It is key to also convolute the elevation for our purpose, given that otherwise we look at the local deviation,
    # from the regional cliamte, whereas here we would have the absolute values.
    # Again we are interested in the deviation of elevation from the regional determining elevation

    # Initate empty file again
    elev_conv_file = os.path.join(base_dir, f"../data/example/_tmp_elev_conv_{kernel_m_sigma}m_alps.tif")
    elev_conv_source = Source(path=elev_conv_file, profile=topo_profile)
    elev_conv_source.init_source(overwrite=True)
    elev_conv_band = Band(elev_conv_source, bidx=1)

    # 2.3.1 Actual convolution (again)
    cvpara.apply_filter(
        source=topo_source,
        output_file=elev_conv_file,
        bands=[elev_band],
        block_size=block_size,
        data_in_range=None,
        data_as_dtype=data_type,
        data_output_range=None,
        img_filter=bpgaussian, # border preserving gaussian
        filter_params=params_filter,
        filter_output_range=None,
        output_dtype=data_type,
        output_range=None,
        selector_band=None,
        **params
    )

    # 2.3.2 ... and calculate difference again
    elev_band.subtract(band=elev_conv_band)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 3. Fit model
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + " | " * 10 + "Model Fitting" + " | " * 10, end="\n")

    # At this point we want to actually fit the model so we get the lapse rate we are so interested in.

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 3.1 Predictor setup
    predictors = []

    # We want to compute a mask which speeds up things later and helps us identify which values we are not interested in
    rgpara.compute_mask(topo_source,
                        bands=[elev_band],
                        logic='all',
                        nodata=np.nan,
                        block_size=block_size,
                        **params)
    # set the mask to source if you want it to be
    elev_band.set_mask_reader(use='source')

    # 3.1.1 Add predictor to predictors to fit later
    predictors.append(elev_band)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 3.2 Full model fit (very simple form)
    band_weight = lfpara.compute_weights(
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
    print(" - "*20, end="\n")
    print(f"Model results: {band_weight=}", end="\n")


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 4. (Reverse) Compute Model
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + " | " * 10 + "Model Computing & Assessment" + " | " * 10, end="\n")

    # We want to check how good our overall model explains our data
    # We will therefore compute the model and then add the previously removed convolution to compare with our original data

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 4.1 Compute the model
    model_file = os.path.join(base_dir, f"../data/example/_tmp_model_conv_{kernel_m_sigma}_m.tif")
    model_data_tif = lfpara.compute_model(
        predictors=predictors,
        optimal_weights=band_weight,
        output_file=model_file,
        block_size=block_size,
        profile=lst_profile,
        # selector_band=ecoreg_band,
        verbose=False,
        **params)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 4.2 Get the accuracy assessment for the FULL model

    # Set up the filter for the data
    lst_org_source = Source(path=lst_file_org)
    lst_org_band = Band(lst_org_source, bidx=1)
    rgpara.compute_mask(lst_org_source,
                        bands=[lst_org_band],
                        logic='all',
                        nodata=np.nan,
                        block_size=block_size,
                        **params)
    lst_org_band.set_mask_reader(use='source')

    _selector_all = rgpara.prepare_selector(lst_org_band, *predictors,
                                            block_size=block_size, )

    # ATTENTION: the R2 might already get high, as the convouted image explains quite a lot, so we want to calculate both R2
    # 1) for the full model, 2) for the residual model we actually fit above

    # 4.2.1
    rmse = lfpara.calculate_rmse(response=lst_band, # Here we need the diff band
                                 model=model_data_tif,
                                 selector=_selector_all,
                                 block_size=block_size,
                                 **params)

    r2 = lfpara.calculate_r2(response=lst_band,
                             model=model_data_tif,
                             selector=_selector_all,
                             block_size=block_size,
                             **params)

    print(" - "*20, end="\n")
    print(f"Residual {rmse=:.2f} | {r2=:.2f}")

    # 4.2.2 Accuracy for residuals actually fit
    model_source = Source(path=model_data_tif)
    model_band = model_source.get_band(bidx=1)
    model_band.add(band=lst_conv_band)


    r2 = lfpara.calculate_r2(response=lst_org_band,
                             model=model_data_tif,
                             selector=_selector_all,
                             block_size=block_size,
                             **params)

    print(" - "*20, end="\n")
    print(f"Overall {rmse=:.2f} | {r2=:.2f}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 4.3 Residuals
    # TODO: it would be nice to add the residuals as an extra band directly to the model tiff.
    # This is not implementable yet, as there will be now second band created when out_band is used,
    # We can think about doing this --> for now I just create a new file
    resid_file = os.path.join(base_dir, f"../data/example/_tmp_resid_model_conv_{kernel_m_sigma}_m.tif")
    resid_source = Source(path=resid_file, profile=lst_profile)
    resid_source.init_source(overwrite=True)
    resid_band = Band(source=resid_source, bidx=1)

    # Residual calculation
    lst_org_band.subtract(band=model_band,
                          out_band=resid_band)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 5. Lapse Rate results (model weights)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print(" | " * 10 + "Results" + " | " * 10, end="\n")

    beta_elev = band_weight[elev_band]
    lapse_rate = beta_elev * 1000 # tranform from /meter to /km
    print(" - "*20, end="\n")
    print(f"Our LAPSE RATE is:  {lapse_rate:.2f}/km", end="\n\n")
    print("\t NOTE: The actual lapse rate is descriped in literature being between -5 to -6°C/kilometer (or -0.5 to -0.6/100 meters).\n"
          "\t In the European Alps (a study in northern Italy), found the annual rate to range between -5.4 to -5.8°C per year.\n"
          "\t (Source: https://doi.org/10.1175/1520-0442(2003)016%3C1032:SASVOA%3E2.0.CO;2")


    # for f in [model_file, lst_file, topo_file]:
    #     if os.path.exists(f):
    #         os.remove(f)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 6. Plotting results
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Set up plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    axes = axes.flatten()

    # function making later plotting more neat and easy
    def plot_map(file: str, ax_n: int, title: str, limits: tuple, bidx=1) -> None:
        src = rio.open(file)
        data = src.read(bidx)
        ax = axes[ax_n]
        ax.set_axis_off()
        img = ax.imshow(data, cmap="RdBu_r", vmin=limits[0], vmax=limits[1])
        ax.set_title(title)
        fig.colorbar(img, ax=ax, label="°C", shrink=0.4)

    # LST original
    plot_map(file=lst_file_org, ax_n=0, title="Land Surface Temperature (LST)", limits=(0, 50))

    # LST Convolution
    plot_map(file=lst_conv_file, ax_n=1, title="LST Convolution", limits=(0, 50))

    # Model Full
    plot_map(file=model_file, ax_n=2, title="Complete Model (Conv + Lapse Rate)", limits=(0, 50))

    # Residuals
    plot_map(file=resid_file, ax_n=3, title="Residuals", limits=(-10, 10))

    fig.savefig(os.path.join(base_dir, "exmpl_01_results_plot.pdf"), format="pdf",
                bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
