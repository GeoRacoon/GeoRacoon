# Data

This folder holds raster fixtures (GeoTIFF) used by this repository's `examples/` scripts and
`tests/` suite. 

> [!NOTE]
> None of this data is shipped with the `georacoon` package.
> It exists only for contributors working from a git clone.
> 
> The files are hosted on Zenodo at [doi:10.5281/zenodo.22307203](https://doi.org/10.5281/zenodo.22307203) (a concept DOI that
> always resolves to the latest version) as two zip archives, `examples.zip` and `test.zip`.
> `data/fetch.py` downloads and hash-checks the relevant archive on first use, extracting only
> the requested file into the matching `data/examples/` or `data/test/` path. Both the
> `examples/` scripts and the `tests/` suite call this automatically, so no manual download step
> is needed.

All datasets below are third-party data, redistributed here under their own upstream
licenses (**not** GeoRacoon's MIT license — see [`../LICENSE`](../LICENSE) for the code).

Naming pattern: `area-of-interest_data-description_(year)_dataset[_projection].tif`

## `examples/`

| File | Dataset / product                                               | Provider | License | Alterations applied here                                                                                                                                                                               |
|---|-----------------------------------------------------------------|---|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `alps_elevation-mean_GLO90DEM_sinusoidal.tif` | Copernicus DEM GLO-90 (90 m global DEM)                         | Copernicus / ESA| [Copernicus DEM License](https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf) (free, attribution required) | Cropped to Alps extent; spatially resampled to ~1km pixel size in a sinusoidal projection; meaned elevation data for resampling|
| `alps_lst-day-mean_summer_2015_MOD11A2_sinusoidal.tif` | MODIS Terra Land Surface Temperature MOD11A2 V6.1 8-Day average | NASA LP DAAC / USGS | Public domain (no restrictions on reuse/redistribution; attribution requested) | Cropped to Alps extent; averaged to a summer daytime mean, 2015. |
| `switzerland_lc-area-fraction_2015_CGLS-LC100_sinusoidal.tif` | Copernicus Global Land Cover 100 m (CGLS-LC100), 2015 epoch     | Copernicus Land Monitoring Service / VITO | [Copernicus free & open data policy](https://zenodo.org/records/3938963) (reproduction, distribution, adaptation permitted; source must be acknowledged) | Cropped to Switzerland extent; converted to a 10-band per-class area-fraction grid at 1 km resolution, Sinusoidal CRS.                       |
| `switzerland_ndvi-binned-mean_2015_LANDSAT-8_sinusoidal.tif` | Landsat-8 NDVI composite 2015                                   | NASA LP DAAC / USGS | [Public domain](https://www.usgs.gov/centers/eros/data-citation) (federally created data; no copyright restriction; USGS credit requested) | Reprojected from `switzerland_ndvi-binned-mean_2015_LANDSAT-8_epsg3035.tif`  to Sinusoidal CRS. |

## `test/`

| File | Dataset / product                                     | Provider | License | Alterations applied here                                                                                                                                                                           |
|---|-------------------------------------------------------|---|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `switzerland_lc-8-reclass_2012_CLC_epsg3035.tif` | CORINE Land Cover (CLC) 2012                          | European Environment Agency (EEA) / Copernicus Land Monitoring Service | Copernicus / EEA free re-use (attribution required) | Cropped to Switzerland extent; reclassified from the full CLC nomenclature down to 8 aggregate classes. |
| `switzerland_lc-area-fraction_2015_CGLS-LC100_epsg2056.tif` | Copernicus Global Land Cover 100 m (CGLS-LC100), 2015 | Copernicus Land Monitoring Service / VITO | [Copernicus free & open data policy](https://zenodo.org/records/3938963) (reproduction, distribution, adaptation permitted; source must be acknowledged) | Reprojected from `examples/switzerland_lc-area-fraction_2015_CGLS-LC100_sinusoidal.tif` to EPSG:2056 (Swiss LV95)                                                                                  |
| `switzerland_ndvi-binned-mean_2015_LANDSAT-8_epsg3035.tif` | Landsat-8 composite 2015                              | NASA LP DAAC / USGS | [Public domain](https://www.usgs.gov/centers/eros/data-citation) (federally created data; no copyright restriction; USGS credit requested) | Cloud mask; calculated NDVI and DOY-binned, median-per-bin, mean-across-bins; reprojected to EPSG:3035.                                                                        |
