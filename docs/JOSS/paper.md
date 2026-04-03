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


# Research impact statement

The package supported carrying out a research project on landscape diversity using effects on vegetation productivity
on a global scale leading [@Landauer_2025_preprint].

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing
of this manuscript, or the preparation of supporting materials.


# Acknowledgements

[//]: # (# TODO: Jonas - Not sure if we need or should put this here)
The initial stages of development for this package was supported by the European Union’s Horizon 2020
research and innovation programme under the Marie Sklodowska-Curie (MSC) grant agreement No 847585 
as well as a grant of the University of Zurich Research Priority Program Global Change and Biodiversity (URPP GCB).


# References
