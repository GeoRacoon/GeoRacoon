# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/GeoRacoon/GeoRacoon/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/convster/filters/gaussian.py |       46 |        0 |    100% |           |
| src/convster/processing.py       |      195 |       65 |     67% |96, 254, 567, 569, 659-678, 882-933, 1002-1023, 1066-1073, 1121-1129, 1192-1223 |
| src/coonfit/exceptions.py        |        4 |        0 |    100% |           |
| src/coonfit/inference.py         |      142 |        8 |     94% |196-198, 261, 266-267, 321, 764 |
| src/coonfit/parallel\_helpers.py |      142 |       58 |     59% |90-109, 133-136, 345, 406-458 |
| src/riogrande/helper.py          |      191 |       15 |     92% |181-183, 353, 407, 437, 579-580, 682, 766, 770-771, 815, 819, 882-891 |
| src/riogrande/io/core.py         |      138 |       12 |     91% |365-367, 382, 421-422, 453-463, 532 |
| src/riogrande/io/exceptions.py   |       10 |        0 |    100% |           |
| src/riogrande/io/models.py       |      414 |       79 |     81% |113, 131, 238-249, 347-350, 368-369, 432, 440, 449, 515, 627-631, 655-656, 697-704, 738-747, 750, 754, 949, 964, 981-1042, 1189, 1239, 1244, 1423, 1546, 1644 |
| src/riogrande/parallel.py        |      177 |       65 |     63% |46, 48, 51, 54, 62-75, 107-120, 167-168, 175-176, 238-239, 241-242, 246-247, 249-255, 287-292, 345, 386, 445-456, 569 |
| src/riogrande/prepare.py         |       92 |        2 |     98% |  118, 125 |
| src/riogrande/timing.py          |       22 |        0 |    100% |           |
| **TOTAL**                        | **1573** |  **304** | **81%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/GeoRacoon/GeoRacoon/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/GeoRacoon/GeoRacoon/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/GeoRacoon/GeoRacoon/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/GeoRacoon/GeoRacoon/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FGeoRacoon%2FGeoRacoon%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/GeoRacoon/GeoRacoon/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.