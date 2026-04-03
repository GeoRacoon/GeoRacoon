---
title: 'GeoRacoon: A Python package for Geographic Raster operations'
tags:
  - Python
  - GIS
  - parallelization
  - MLR
  - remote sensing
authors:
  - name:
      given-names: Simon
      surname: Landauer
    orcid: 0009-0002-5031-8378
    affiliation: 1
  - name:
      given-names: Pascal A.
      surname: Niklaus
    orcid: 0000-0002-2360-1357
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name:
      given-names: Jonas I. 
      surname: Liechti
    orcid: 0000-0003-3447-3060
    affiliation: 2 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
affiliations:
 - name: Department of Evolutionary Biology and Environmental Studies, University of Zurich, 8057 Zurich, Switzerland
   index: 1
   ror: 02crff812
 - name: T4D GmbH, 8045 Zurich, Switzerland
   index: 2
   ror: 055j0y167
date: 06 March 2026
bibliography: paper.bib

---

# Summary
The Python package `GeoRacoon` is aimed at supporting analyses and work with
large spatial raster data.
It consists of 3 sub-packages: `riogrande`, a class based extension of rasterio
[@Gillies_2019] with extended functionalities for tag based raster object
management; `convster`, a fully parallelized module for convolution of spatial
raster data (e.g. satellite imagery); `coonfit`, a multiple linear regression
module using analytical solutions, yet fully parallelized to allow for large
data analysis.


# Statement of need

Analyses of global-scale raster datasets — such as satellite-derived land-cover maps, vegetation indices, and climate
variables — frequently involve spatial filtering (convolution) and pixel-wise statistical modelling. When applied to
rasters comprising billions of pixels, these operations become computationally prohibitive without parallelization and
out-of-core processing strategies.

At the time of development, to the authors' knowledge, no Python package provided a turnkey, parallelized workflow for
Gaussian convolution of large categorical rasters with correct handling of nodata boundaries. Existing tools either
required users to implement parallelization themselves or did not support border-preserving convolution for categorical
data. Similarly, fitting pixel-wise multiple linear regression (MLR) across full rasters — where spatial bands serve as
predictors and response — lacked an accessible, parallelized Python solution using the analytical normal-equations
approach. Finally, working with multi-band rasters carrying rich metadata (e.g., land-cover category labels, temporal
identifiers) required manual bookkeeping of band indices; no library offered tag-based band selection out of the box.

`GeoRacoon` addresses these gaps. Its three sub-packages evolved from the practical needs of a research project on
landscape diversity effects on vegetation productivity [@Landauer_2025_preprint], where large-scale spatial convolution,
derived heterogeneity metrics, and raster-based regression were essential. During development, `riogrande` emerged as a
foundational extension of `rasterio` [@Gillies_2019] to manage large volumes of imagery in an object-oriented, tag-based
manner — improving usability for broader applications beyond the original research context.


# State of the field

Several established tools address parts of the geospatial raster processing pipeline. `rasterio` [@Gillies_2019]
provides the standard Python interface for geospatial raster I/O, built on top of GDAL [@GDAL_2025], but offers no
high-level analytical operations such as convolution or regression. `xarray` [@Hoyer_2017] and its geospatial extension
`rioxarray` enable lazy, chunked computation via Dask for large arrays, yet do not provide turnkey pipelines for
categorical convolution, entropy-based heterogeneity metrics, or pixel-wise MLR. Cloud-based platforms such as Google
Earth Engine [@Gorelick_2017] excel at large-scale analysis but are tied to proprietary infrastructure and do not
support custom filter implementations or analytical regression decomposition. `scipy.ndimage` [@Virtanen_2020] and
`scikit-image` [@van_der_Walt_2014] provide spatial filters — including Gaussian — but operate entirely in-memory,
do not handle geospatial metadata or nodata boundaries, and lack block-parallel decomposition.

`GeoRacoon` differentiates itself by combining block-parallel processing with correct nodata handling, tag-based
metadata management, and analytical MLR — all within a single, pip-installable Python package that requires no external
computing infrastructure.


# Software design

`GeoRacoon` is organized into three sub-packages with clear separation of concerns:
`riogrande` provides the foundation and parallelization infrastructure,
`convster` implements spatial convolution and derived metrics,
and `coonfit` provides parallelized multiple linear regression.
The packages build on `rasterio` [@Gillies_2019], NumPy [@Harris_2020], `scikit-image` [@van_der_Walt_2014], and SciPy [@Virtanen_2020].

**Parallelization model.**  
All computationally intensive operations follow a common block-parallel architecture.
Rasters are decomposed into spatial blocks via a view system, with configurable overlap (borders) to prevent edge
artifacts in filter operations.
Workers are dispatched via Python's `multiprocessing.Pool`, each processing a block
independently, while a dedicated aggregator process collects results from a shared queue and writes them to disk.
This producer-consumer pattern enables processing of rasters that exceed available memory without requiring external
distributed computing frameworks.

**`riogrande` — tag-based raster management.**  
The `Source` and `Band` classes extend `rasterio` with a structured metadata layer.
Band-level metadata is stored as JSON-serialized key-value pairs in a dedicated GeoTIFF tag namespace,
enabling tag-based band selection (e.g., querying by land-cover category or date) instead of manual index tracking.
Additional functionality includes configurable mask strategies per band, compatibility checks across rasters
(CRS, resolution, units), data type conversion with range rescaling, and file compression utilities.

**`convster` — parallelized spatial convolution.**  
The convolution module follows a filter-agnostic design: any callable matching the expected signature can be used as the spatial filter.
The default Gaussian filter wraps `skimage.filters.gaussian`, and a custom border-preserving variant (`bpgaussian`)
correctly handles nodata boundaries through a normalization-correction approach.
Kernel dimensions are estimated adaptively from the filter's impulse response,
and border sizes are computed automatically to ensure artifact-free block-parallel processing.
Beyond raw convolution, `convster` provides derived spatial metrics including Shannon entropy and an interaction index inspired
by the Simpson index, computed from filtered categorical layers.

**`coonfit` — parallelized multiple linear regression.**  
The regression module exploits the decomposability of the normal equations:
since $X^TX = \sum_b X_b^TX_b$ and $X^Ty = \sum_b X_b^Ty_b$ over spatial blocks $b$, both terms can be computed in parallel.
The workflow proceeds in three stages:
(1) parallel accumulation of the block-wise $X^TX$ matrices,
(2) serial inversion of the resulting (small) matrix, and
(3) parallel computation of the regression coefficients.
Rank-deficiency detection, predictor consistency validation, and model evaluation metrics (RMSE, $R^2$)
are provided as part of the pipeline.


# Research impact statement

The package supported carrying out a research project on landscape diversity effects on vegetation productivity
at a global scale [@Landauer_2025_preprint].

# AI usage disclosure

No generative AI tools were used in the development of this software.
AI-assisted tools were used to format some of the source code docstrings
and to support the drafting of portions of this manuscript, which were
subsequently reviewed and edited by the authors.


# Acknowledgements

[//]: # (# TODO: Jonas - Not sure if we need or should put this here)
The initial stages of development for this package were supported by the European Union’s Horizon 2020
research and innovation programme under the Marie Sklodowska-Curie (MSC) grant agreement No 847585 
as well as a grant of the University of Zurich Research Priority Program Global Change and Biodiversity (URPP GCB).


# References
