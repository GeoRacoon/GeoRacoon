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
The Python package `GeoRacoon` is aimed at supporting analyses and work with large spatial raster data. It consists of 
3 sub-packages: `riogrande`, a class based extension of rasterio [@gillies_2019] with extended functionalities
for tag based raster object management; `convster`, a fully parallelized module for convolution of spatial raster data
(e.g. satellite imagery); `coonfit`, a multiple linear regression module using analytical solutions, yet fully
parallelized to allow for large data analysis.


# Statement of need



At the time of development, to the authors knowledge, there existed no comprehensive software which would allow for 
seamless gaussian convolution of large raster datasets. In addition, as convolution is a relatively frequently used 
process in image analysis - scalability of such a process is of great need. 

... 
Within the process of developing `GeoRacoon` the 3 sub-packages evolved as a necessity to tackle a research project on ...
Yet, due to the lack of existing software - the development will fill a gap by ...
During the development process `riogrande` evolved as a necessary extension to `GeoRaccon` to handle vast amounts of 
imagery data in easier, object based way - allowing for even better usability for broader use in other applications after
the depolyment of 


# State of the field                                                                                                                  


# Software design


# Research impact statement


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


# References
