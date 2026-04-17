<div align="center">

<img src="./docs/_static/georacoonPin.svg" alt="GeoRacoon Logo" width="400">

<p>
  <a href="https://github.com/GeoRacoon/georacoon/tree/python-coverage-comment-action-data">
    <img src="https://github.com/GeoRacoon/georacoon/raw/python-coverage-comment-action-data/badge.svg" alt="Coverage">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/releases/latest">
    <img src="https://img.shields.io/github/v/release/GeoRacoon/GeoRacoon?label=release" alt="Latest Release">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>
<h1>GeoRacoon</h1>
<p>Out and about<br><><br>ready to tackle Geographic Raster</p>

  <br>
  <br>
<pre><small>Born from a collaboration between:</small>
<br>

<a href="https://www.ieu.uzh.ch/en/research/ecology/soil.html"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.cd.uzh.ch/dam/jcr:9528e314-fbb5-4ede-b7a5-3446bf8d9337/UZH_Logo_white.svg"><source media="(prefers-color-scheme: light)" srcset="https://www.cd.uzh.ch/dam/jcr:e2f01a3c-e263-427a-91d7-723fc337af4b/uzh-logo.svg"><img alt="UZH logo" src="https://www.cd.uzh.ch/dam/jcr:e2f01a3c-e263-427a-91d7-723fc337af4b/uzh-logo.svg" width="290" style="vertical-align: middle;"></picture></a>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://github.com/t4d-gmbh"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/logo_with_Ds_wb.svg"><source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/logo_with_Ds.svg"><img alt="T4D Logo" src="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/logo_color.svg" width="220" style="vertical-align: middle;"></picture></a>
</pre>
</div>

## Quickstart

This project is an extension of [RasterIO](https://rasterio.readthedocs.io/en/stable/) (rio) allowing to work with Sources (GeoTIFFS) and Bands as objects, which easily incorporate the use of tags.

_GeoRacoon_ provides 3 packages, `riogrande`, `convster` and `coonfit`, which facilitate (in our opinion) working with TIFF files.

- **RioGrande** provides the great heart of the GeoRacoon and adds functionality for parallel processing using Windows, dataset compatibility checks, data type 
conversion, mask and selector creation as well as simple file compression.  
- **Convster** allows for (gaussian) convolution of raster files using parallelized processing, _coon-style_. 
While gaussian and border-preserving gaussian filters are default parameters, other filters can be used.
- **CoonFit** allows to fit linear moodls the _coon-way_, meaning parallelized and fast, while understanable due to the
reliance on matrix operations

<!-- quickstart -->


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Features](#features)
- [Authors](#authors)
- [Release History](#release-history)
- [License](#license)

## Installation

<!-- installation-start -->
GeoRacoon can be installed directly from GitHub into your virtual environment.
Simply run:

```
pip install git+https://github.com/ReoRacoon/georacoon.git
```

<details>
<summary><b>Local install</b></summary>

Alternatively, you can clone the repository and install the package from your
local copy.
This might be the desired strategy if you intend to work on the source code, in
which case you can replace `install` by `install -e` and modify the files
in-place.


To install:
1. Clone this repository
1. `cd` into the repository

On macOS and Linux:

```sh
$ python -m pip install .
```

On Windows:

```sh
PS> python -m pip install .
```
</details>

_Note:_
_This package relies on [rasterio](https://rasterio.readthedocs.io/en/latest/index.html)_
_which partially depends on [libgdal](https://gdal.org/)._
_If you follow the installation instructions below you will attempt to install_
_rasterio from the Python Package Index in which case the libgdal library_
_will be shipped along._
_However, if you encounter any issues with the installation of rasterio, head_
_over to the [rasterio installation instructions](https://rasterio.readthedocs.io/en/stable/installation.html) for more details._

_We also use the python package `gdal` which depends on `libgdal` that has been installed (see comment above)._
_It is important to install matching version, so first check with `gdalinfo --version` what version of `libgdal`_
_you have installed and then install the corresponding python package with `pip install gdal==x.x.x`._

<!-- installation-end -->

## Usage

<!-- usage-start -->

<details>
<summary><b>RioGrande — open a GeoTIFF and work with Sources and Bands</b></summary>

```python
from riogrande.io import Source, Band

# Open a GeoTIFF as a Source
source = Source("elevation.tif")

# Import the file's profile (size, CRS, dtype, …)
profile = source.import_profile()

# Tag a band and retrieve it by tag later
source.set_tags(bidx=1, tags={"category": "elevation_mean"})
band = source.get_band(category="elevation_mean")

# Or just grab a band by index
band = source.get_band(bidx=1)
print(band.tags)
```

</details>

<details>
<summary><b>Convster — apply a spatial filter to a raster</b></summary>

```python
from riogrande.io import Source
from convster import parallel as cvpara
from convster.filters import bpgaussian  # border-preserving Gaussian

source = Source("landcover.tif")

# Kernel: 30 km sigma, 1 km resolution → 30 pixels
params_filter = dict(sigma=30, truncate=3, preserve_range=True)

cvpara.apply_filter(
    source=source,
    output_file="landcover_blurred.tif",
    block_size=(200, 200),
    img_filter=bpgaussian,
    filter_params=params_filter,
    data_as_dtype="float32",
    nbrcpu=4,
)
```

</details>

<details>
<summary><b>CoonFit — fit a linear model and generate a prediction raster</b></summary>

```python
import numpy as np
from riogrande.io import Source, Band
from coonfit import parallel as lfpara

# Set up response and predictor bands
response_band = Band(Source("lst.tif"), bidx=1)

pred_source = Source("elevation.tif")
pred_source.set_tags(bidx=1, tags={"category": "elevation"})
predictor_band = pred_source.get_band(category="elevation")

# Fit the model — returns a dict of {band: weight}
weights = lfpara.compute_weights(
    response=response_band,
    predictors=[predictor_band],
    block_size=(200, 200),
    include_intercept=True,
    no_data=np.nan,
    nbrcpu=4,
)
print(weights)

# Apply the fitted weights to produce a prediction raster
lfpara.compute_model(
    predictors=[predictor_band],
    optimal_weights=weights,
    output_file="lst_predicted.tif",
    block_size=(200, 200),
    nbrcpu=4,
)
```

</details>

Head over to the [examples/](examples/) folder for a full end-to-end walk-through.

For more examples, please refer to the project's [documentation page](docs).

<!-- usage-end -->

## Technologies

RioGrande uses the following technologies and tools:

- [Python](https://www.python.org/): ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


## Features

RioGrande currently has the following set of features:

- Support for...
- ...

## Contributing

To contribute to the development of RioGrande, follow the steps below:

1. Fork RioGrande from <https://github.com/yourusername/yourproject/fork>
2. Create your feature branch (`git checkout -b feature-new`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some new feature'`)
5. Push to the branch (`git push origin feature-new`)
6. Create a new pull request


## Authors
[<img src="https://github.com/simonsaysenjoy.png" width="60" height="60" style="border-radius:50%">](https://github.com/simonsaysenjoy)
[<img src="https://github.com/j-i-l.png" width="60" height="60" style="border-radius:50%">](https://github.com/j-i-l)
## Release History

- 1.0.0
    - First working version

## License

RioGrande is distributed under the < license > license.

## Acknowledgements

Mention
