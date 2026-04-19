<div align="center">

<img src="./docs/_static/georacoonPin.svg" alt="GeoRacoon Logo" width="400">

<p>
  <a href="https://github.com/GeoRacoon/GeoRacoon/releases/latest">
    <img src="https://img.shields.io/github/v/release/GeoRacoon/GeoRacoon?label=Release" alt="Release">
  </a>
  <a href="https://georacoon.github.io/GeoRacoon" target="_blank">
    <img src="https://img.shields.io/badge/Docs-online-blue.svg" alt="Docs">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>
<p>
  <a href="https://github.com/GeoRacoon/GeoRacoon/actions/workflows/deploy.yml">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/deploy-ubuntu.json" alt="Ubuntu">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/actions/workflows/deploy.yml">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/deploy-fedora.json" alt="Fedora">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/actions/workflows/deploy.yml">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/deploy-macos.json" alt="macOS">
  </a>
  <a href="https://github.com/GeoRacoon/GeoRacoon/actions/workflows/deploy.yml">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/deploy-windows.json" alt="Windows">
  </a>
</p>
<p>
  <a href="https://github.com/GeoRacoon/georacoon/tree/python-coverage-comment-action-data">
    <img src="https://github.com/GeoRacoon/georacoon/raw/python-coverage-comment-action-data/badge.svg" alt="Coverage">
  </a>
</p>
<h1>GeoRacoon</h1>
<p>Out and about<br><><br>ready to tackle Geographic Raster</p>

  <br>
  <br>
<pre><small>Born from a collaboration between:</small>
<br>

<a href="https://www.ieu.uzh.ch/en/research/ecology/soil.html"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.cd.uzh.ch/dam/jcr:9528e314-fbb5-4ede-b7a5-3446bf8d9337/UZH_Logo_white.svg"><source media="(prefers-color-scheme: light)" srcset="https://www.cd.uzh.ch/dam/jcr:e2f01a3c-e263-427a-91d7-723fc337af4b/uzh-logo.svg"><img alt="UZH logo" src="https://www.cd.uzh.ch/dam/jcr:e2f01a3c-e263-427a-91d7-723fc337af4b/uzh-logo.svg" width="290" style="vertical-align: middle;"></picture></a>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<a href="https://github.com/t4d-gmbh"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/wb/T4D_design_develop.svg"><source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/bw/T4D_design_develop.svg"><img alt="T4D Logo" src="https://raw.githubusercontent.com/t4d-gmbh/.github/main/static/logo/bw/T4D_design_develop.svg" width="220" style="vertical-align: middle;"></picture></a>
</pre>
</div>

# Quickstart

This project is an extension of [RasterIO](https://rasterio.readthedocs.io/en/stable/) (rio) allowing to work with Sources (GeoTIFFS) and Bands as objects, which easily incorporate the use of tags.

_GeoRacoon_ provides 3 packages, `riogrande`, `convster` and `coonfit`, which facilitate (in our opinion) working with TIFF files.

- **RioGrande** provides the great heart of the GeoRacoon and adds functionality for parallel processing using Windows, dataset compatibility checks, data type 
conversion, mask and selector creation as well as simple file compression.  
- **Convster** allows for (Gaussian) convolution of raster files using parallelized processing, _coon-style_. 
While Gaussian and border-preserving Gaussian filters are default parameters, other filters can be used.
- **CoonFit** allows to fit linear models the _coon-way_, meaning parallelized and fast, while understandable due to the
reliance on matrix operations

<!-- quickstart -->

## Installation

**Supported Python versions:**

[![Python 3.14](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/python-3.14.json)](https://github.com/GeoRacoon/GeoRacoon/actions/workflows/status.yml)
[![Python 3.13](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/python-3.13.json)](https://github.com/GeoRacoon/GeoRacoon/actions/workflows/status.yml)
[![Python 3.12](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/python-3.12.json)](https://github.com/GeoRacoon/GeoRacoon/actions/workflows/status.yml)
[![Python 3.11](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/python-3.11.json)](https://github.com/GeoRacoon/GeoRacoon/actions/workflows/status.yml)
[![Python 3.10](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/j-i-l/b692f90cf16ebc9a3646ce218df856e8/raw/python-3.10.json)](https://github.com/GeoRacoon/GeoRacoon/actions/workflows/status.yml)

### Setting up a virtual environment

We recommend installing GeoRacoon in a virtual environment to avoid dependency conflicts.

On macOS and Linux:

```sh
$ python -m venv .venv
$ source .venv/bin/activate
```

On Windows:

```sh
PS> python -m venv .venv
PS> .venv\Scripts\activate
```

### Installing GeoRacoon

GeoRacoon can be installed directly from GitHub into your virtual environment.
Simply run:

```
pip install git+https://github.com/GeoRacoon/georacoon.git
```

<details>
<summary><b>Development install</b></summary>

Alternatively, you can clone the repository and install the package from your
local copy.
This is the recommended strategy if you intend to work on the source code,
allowing you to modify the files in-place.

To install:
1. Clone this repository
1. `cd` into the repository

On macOS and Linux:

```sh
$ python -m pip install -e .
```

On Windows:

```sh
PS> python -m pip install -e .
```

Alternatively, if you're using `uv`:

```sh
$ git clone https://github.com/GeoRacoon/GeoRacoon.git
$ cd GeoRacoon
$ uv sync
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

### Alternative installation using uv

If you prefer using [uv](https://docs.astral.sh/uv/) for faster package management, you can install GeoRacoon as follows:

```sh
$ uv pip install git+https://github.com/GeoRacoon/georacoon.git
```

For development installation with uv:

```sh
$ git clone https://github.com/GeoRacoon/GeoRacoon.git
$ cd GeoRacoon
$ uv pip install -e .
```

## Usage

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

For more examples, please refer to the project's [documentation page](https://georacoon.github.io/GeoRacoon).


## Contributing

We welcome contributions from the community!

Here are some guidelines to help you get started:

1. **Seeking Support:** 
   If you need help with one of the GeoRacoon packages, you can seek support at [the issue page](https://github.com/GeoRacoon/GeoRacoon/issues) on this GitHub repository. 
   Before you open a new issue, please first have a look at the existing - also the closed - ones, maybe you are not to first to run into it!
   
   If you decide to open a new issue, please describe your problem in detail and include a minimal reproducible example if possible.
   
2. **Reporting Issues or Problems:** 
   If you encounter any issues, problems or otherwise unexpected behaviour with GeoRacoon, please report them on [the issue page](https://github.com/GeoRacoon/GeoRacoon/issues).
   Before you open a new issue, please first have a look at the existing - also the closed - ones, maybe you are not to first to run into it!
   When reporting an issue, include as much detail as possible, including steps to reproduce the issue, your operating system and R version, and any error messages you received.

3. **Software Contributions:**
   We encourage contributions directly via pull requests on the GeoRacoon repository.
   Before starting your work, please first create an issue describing the contribution you wish to make. 
   This allows us to discuss and agree on the best way to integrate your contribution into the package.

   In case you are unsure about how to proceed  with a contribution, you can follow these steps:

   1. Fork GeoRacoon from <https://github.com/GeoRacoon/GeoRacoon/fork>
   2. Create your feature branch (`git checkout -b feature-new`)
   3. Make your changes
   4. Commit your changes (`git commit -am 'Add some new feature'`)
   5. Push to the branch (`git push origin feature-new`)
   6. Create a new pull request


## Authors
[<img src="https://github.com/simonsaysenjoy.png" width="60" height="60" style="border-radius:50%">](https://github.com/simonsaysenjoy)
[<img src="https://github.com/j-i-l.png" width="60" height="60" style="border-radius:50%">](https://github.com/j-i-l)

<!-- ## Citation

Citation information will be added here.

-->

## License

GeoRacoon is distributed under a MIT license: [LICENSE](./LICENSE).
