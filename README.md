# RioGrande

![coverage](https://img.shields.io/badge/coverage-75%25-yellowgreen) 
![version](https://img.shields.io/badge/version-1.0.0-blue)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) (all of these not linked)

This package is an extinsion of [RasterIO](https://rasterio.readthedocs.io/en/stable/) (rio) allowing to work with Sources (GeoTIFFS) and Bands as objects, which easily incorporate the use of tags.
RioGrande adds functionality for parallel processing using Windows, dataset compatibility checks, data type conversion, mask and selector creation as well as simple file compression.


<img alt="raster image" height="200" src="../../results/test_france.png" width="200"/>

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Features](#features)
- [Authors](#authors)
- [Release History](#release-history)
- [License](#license)

## Installation

On macOS and Linux:

```sh
$ python -m pip install riogrande
```

On Windows:

```sh
PS> python -m pip install riogrande
```

## Usage

To run RioGrande, fire up a terminal window and run the following command:

```sh
$ <project>
```

Here are a few examples of using the riogrande library in your code:

```python
from riogrande.io_ import Source, Band

s = Source("example.tif")
s

b1 = s.get_band(bidx=1)
b1

b1.tags

...
```

For more examples, please refer to the project's [documentation page](docs).

## Technologies

RioGrande uses the following technologies and tools:

- [Python](https://www.python.org/): ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


## Features

RioGrande currently has the following set of features:

- Support for...
- ...

## Contributing

To contribute to the development of RioGrande, follow the steps below:

1. Fork RioGrandefrom <https://github.com/yourusername/yourproject/fork>
2. Create your feature branch (`git checkout -b feature-new`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some new feature'`)
5. Push to the branch (`git push origin feature-new`)
6. Create a new pull request


## Authors
<a href="https://github.com/GeoRacoon">
<img src="../../images/georacoon_v02_202509.svg" alt="GeoRacoon Logo" width="50">
</a>

## Release History

- 1.0.0
    - First working version

## License

RioGrande is distributed under the < license > license.

## Acknowledgements

Mention