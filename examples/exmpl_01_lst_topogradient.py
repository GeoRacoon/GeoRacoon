# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

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
# Data handling:
#   The input rasters are obtained with `riogrande.data.fetch`, which downloads
#   them from Zenodo on first use and returns the path to the cached file.
#   Their pixel values are never modified. All intermediate rasters are
#   written to a temporary directory that is removed when the script ends.
#
# - - - - - - - - - - - - - - - - - - - - - - - -

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt

# Our package(s)
from riogrande.io import Source, Band
from riogrande import parallel as rgpara
from convster import parallel as cvpara
from convster.filters import bpgaussian
from coonfit import parallel as lfpara

# Fetches the example rasters from Zenodo on first use, then reuses the cache
from riogrande.data import fetch

# Parameters
params = dict(n_jobs=6)
block_size = (200, 200)
data_type = np.float32


def main():

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1. Data preparation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    print("\n" + " | " * 10 + "Data preparation" + " | " * 10, end="\n")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # 1.1) Get the data (paths to the cached, fetched files)
    lst_file_org = fetch(
        "examples/alps_lst-day-mean_summer_2015_MOD11A2_sinusoidal.tif")
    topo_file_org = fetch(
        "examples/alps_elevation-mean_GLO90DEM_sinusoidal.tif")

    # All intermediate rasters are written to this temporary directory,
    # which is removed automatically once the block is left.
    with TemporaryDirectory(prefix="georacoon_exmpl_01_") as tmp_dir:
        work_dir = Path(tmp_dir)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 1.2) Set up objects (class Source and Band)

        # 1.2.1) LST get Source and use Band (idx=1), as there is only one band present.
        lst_source = Source(path=lst_file_org)
        lst_profile = lst_source.import_profile()
        lst_band = Band(source=lst_source, bidx=1)

        # 1.2.2) Similar with elevation, but we need to set the tag
        # (so later for the computation of weights/betas we have a name for the predictor)
        topo_source = Source(path=topo_file_org)
        topo_profile = topo_source.import_profile()

        # The tag is set on the Band object only, so the fetched file is not modified
        elev_cat = "elevation_mean"
        elev_band = Band(source=topo_source, bidx=1,
                         tags=dict(category=elev_cat))

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 2. Convolution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print("\n" + " | " * 10 + "Convolution" + " | " * 10, end="\n")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 2.1) Parameters for convolution model
        kernel_truncate = 3  # in sigma
        kernel_m_sigma = 30000  # in meters
        resolution = 1000  # 1,000 m ~Modis LST
        kernel_pixel_sigma = kernel_m_sigma / resolution

        params_filter = dict(
            sigma=kernel_pixel_sigma,
            truncate=kernel_truncate,
            preserve_range=True,
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 2.2 LST convolution

        # This is done to model regional cliamte provided the filter parameters from above.
        # (Such provides an example with arbitrary sigma in meters, as a user you may want to optimize this programmatically)

        # Initate empty file for later
        lst_conv_file = work_dir / f"lst_conv_{kernel_m_sigma}m_alps.tif"
        lst_conv_source = Source(path=lst_conv_file, profile=lst_profile)
        lst_conv_source.init_source(overwrite=True)
        lst_conv_band = Band(lst_conv_source, bidx=1)

        # 2.2.1 Actual convolution
        # TODO: We need to implement that files will be initated if they dont exist
        cvpara.apply_filter(
            source=lst_source,
            output_file=str(lst_conv_file),
            block_size=block_size,
            data_in_range=None,
            data_as_dtype=data_type,
            data_output_range=None,
            img_filter=bpgaussian,  # border preserving gaussian
            filter_params=params_filter,
            filter_output_range=None,
            output_dtype=data_type,
            output_range=None,
            selector_band=None,
            **params
        )

        # 2.2.2 Remove convoluted from original
        # As the convolution simulates the regional climate, the difference will show the deviation from this.
        # The result is written to a new band (out_band), the fetched file stays unchanged.
        lst_diff_file = work_dir / "lst_diff_alps.tif"
        lst_diff_source = Source(path=lst_diff_file, profile=lst_profile)
        lst_diff_source.init_source(overwrite=True)
        lst_diff_band = Band(lst_diff_source, bidx=1)
        lst_band.subtract(band=lst_conv_band, out_band=lst_diff_band)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 2.3 Topography convolution

        # It is key to also convolute the elevation for our purpose, given that otherwise we look at the local deviation,
        # from the regional cliamte, whereas here we would have the absolute values.
        # Again we are interested in the deviation of elevation from the regional determining elevation

        # Initate empty file again
        elev_conv_file = work_dir / f"elev_conv_{kernel_m_sigma}m_alps.tif"
        elev_conv_source = Source(path=elev_conv_file, profile=topo_profile)
        elev_conv_source.init_source(overwrite=True)
        elev_conv_band = Band(elev_conv_source, bidx=1)

        # 2.3.1 Actual convolution (again)
        cvpara.apply_filter(
            source=topo_source,
            output_file=str(elev_conv_file),
            bands=[elev_band],
            block_size=block_size,
            data_in_range=None,
            data_as_dtype=data_type,
            data_output_range=None,
            img_filter=bpgaussian,  # border preserving gaussian
            filter_params=params_filter,
            filter_output_range=None,
            output_dtype=data_type,
            output_range=None,
            selector_band=None,
            **params
        )

        # 2.3.2 ... and calculate difference again (into a new, tagged band)
        elev_diff_file = work_dir / "elevation_diff_alps.tif"
        elev_diff_source = Source(path=elev_diff_file, profile=topo_profile)
        elev_diff_source.init_source(overwrite=True)
        elev_diff_band = Band(elev_diff_source, bidx=1,
                              tags=dict(category=elev_cat))
        elev_band.subtract(band=elev_conv_band, out_band=elev_diff_band)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 3. Fit model
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print("\n" + " | " * 10 + "Model Fitting" + " | " * 10, end="\n")

        # At this point we want to actually fit the model so we get the lapse rate we are so interested in.

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 3.1 Predictor setup
        predictors = []

        # We want to compute a mask which speeds up things later and helps us identify which values we are not interested in
        rgpara.compute_mask(elev_diff_source,
                            bands=[elev_diff_band],
                            logic='all',
                            nodata=np.nan,
                            block_size=block_size,
                            **params)
        # set the mask to source if you want it to be
        elev_diff_band.set_mask_reader(use='source')

        # 3.1.1 Add predictor to predictors to fit later
        predictors.append(elev_diff_band)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 3.2 Full model fit (very simple form)
        band_weight = lfpara.compute_weights(
            response=lst_diff_band,
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

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 4. (Reverse) Compute Model
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print("\n" + " | " * 10 + "Model Computing & Assessment" + " | " * 10, end="\n")

        # We want to check how good our overall model explains our data
        # We will therefore compute the model and then add the previously removed convolution to compare with our original data

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 4.1 Compute the model
        model_file = work_dir / f"model_conv_{kernel_m_sigma}m.tif"
        model_data_tif = lfpara.compute_model(
            predictors=predictors,
            optimal_weights=band_weight,
            output_file=str(model_file),
            block_size=block_size,
            profile=lst_profile,
            # selector_band=ecoreg_band,
            verbose=False,
            **params)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 4.2 Get the accuracy assessment for the FULL model

        # Set up the filter for the data
        # NOTE: compute_mask writes the computed dataset mask into the fetched
        # LST file itself (via Source.mask_writer). Only the mask is added,
        # the pixel values remain unchanged.
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
        rmse = lfpara.calculate_rmse(response=lst_diff_band,  # Here we need the diff band
                                     model=model_data_tif,
                                     selector=_selector_all,
                                     block_size=block_size,
                                     **params)

        r2 = lfpara.calculate_r2(response=lst_diff_band,
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

        rmse = lfpara.calculate_rmse(response=lst_org_band,
                                     model=model_data_tif,
                                     selector=_selector_all,
                                     block_size=block_size,
                                     **params)

        r2 = lfpara.calculate_r2(response=lst_org_band,
                                 model=model_data_tif,
                                 selector=_selector_all,
                                 block_size=block_size,
                                 **params)

        print(" - "*20, end="\n")
        print(f"Overall {rmse=:.2f} | {r2=:.2f}")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 4.3 Residuals
        # TODO: it would be nice to add the residuals as an extra band directly to the model tiff.
        # This is not implementable yet, as there will be now second band created when out_band is used,
        # We can think about doing this --> for now I just create a new file
        resid_file = work_dir / f"resid_model_conv_{kernel_m_sigma}m.tif"
        resid_source = Source(path=resid_file, profile=lst_profile)
        resid_source.init_source(overwrite=True)
        resid_band = Band(source=resid_source, bidx=1)

        # Residual calculation
        lst_org_band.subtract(band=model_band,
                              out_band=resid_band)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 5. Lapse Rate results (model weights)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        print(" | " * 10 + "Results" + " | " * 10, end="\n")

        beta_elev = band_weight[elev_diff_band]
        lapse_rate = beta_elev * 1000  # tranform from /meter to /km
        print(" - "*20, end="\n")
        print(f"Our LAPSE RATE is:  {lapse_rate:.2f}/km", end="\n\n")
        print("\t NOTE: The actual lapse rate is descriped in literature being between -5 to -6°C/kilometer (or -0.5 to -0.6/100 meters).\n"
              "\t In the European Alps (a study in northern Italy), found the annual rate to range between -5.4 to -5.8°C per year.\n"
              "\t (Source: https://doi.org/10.1175/1520-0442(2003)016%3C1032:SASVOA%3E2.0.CO;2")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 6. Plotting results
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Set up plot
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
        axes = axes.flatten()

        # function making later plotting more neat and easy
        def plot_map(file: str | Path, ax_n: int, title: str, limits: tuple, bidx=1) -> None:
            with rio.open(file) as src:
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

        # Uncomment to save the figure to the current working directory
        # fig.savefig(Path.cwd() / "exmpl_01_results_plot.pdf", format="pdf",
        #             bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    main()
