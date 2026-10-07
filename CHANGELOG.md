<!--
SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>

SPDX-License-Identifier: MIT
-->

# Changelog

All notable changes to GeoRacoon follow [Conventional Commits](https://www.conventionalcommits.org/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### ⚠ BREAKING CHANGES

* None — this release is fully backward compatible. The removal of the
  deprecated `nbrcpu` argument will constitute the next breaking release
  (2.0.0).

### Features

* **riogrande,convster,coonfit**: add `n_jobs` keyword argument to all parallel
  entry points, replacing `nbrcpu` (#166)
* **riogrande**: support scikit-learn-style negative values in
  `get_nbr_workers` (#166)
* **data**: Zenodo-backed data fetching with local caching via `pooch`
  for test and example data (`data/fetch.py`) (#172, #177)
* **data**: rename and restructure datasets under `data/example/` and
  `data/testing/` with systematic
  `<domain>_<metric>_<period>_<product>_<crs>.tif` naming (#168)
* **examples**: dedicated examples per subpackage: `plot_02_riogrande`,
  `plot_03_convster`, `plot_04_coonfit` (#151)
* **examples**: full RMSE accuracy assessment in the MODIS LST
  topographic-gradient example (#156)
* **convster**: precise lossy-cast detection (`_is_lossy_cast`) replaces
  unconditional filter-output warnings

### Bug Fixes

* **docs**: absolute path for logo location (25ff819)

### Performance

* **ci**: caching of Zenodo data fetches speeds up test runs (#177)

### Documentation

* **docs**: `CONTRIBUTING.md` and contributing page (#154)
* **docs**: homepage/introduction rework with PyPI installation path
  promoted (#150)

### Other

* **licensing**: REUSE/SPDX compliance: `LICENSES/`, file headers, `reuse`
  CI workflow (#173)
* **ci**: cross-PR coverage and status-check workflows

### Deprecated

* **riogrande,convster,coonfit**: `nbrcpu` keyword argument in favor of
  `n_jobs`; emits a warning and will be removed in 2.0.0 (#166)

## [1.0.0] - 2026-06-26

First stable release of the GeoRacoon umbrella packages `riogrande`
(raster I/O and parallel processing), `convster` (filtering and entropy
computations), and `coonfit` (zonal statistics and fitting).

[Unreleased]: https://github.com/GeoRacoon/landiv/compare/1.0.0...HEAD
[1.0.0]: https://github.com/GeoRacoon/landiv/releases/tag/1.0.0
