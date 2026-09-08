# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/GeoRacoon/GeoRacoon/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/convster/filters/gaussian.py |       46 |        0 |    100% |           |
| src/convster/processing.py       |      195 |       65 |     67% |101, 259, 572, 574, 664-683, 887-938, 1007-1028, 1071-1078, 1126-1134, 1197-1228 |
| src/coonfit/exceptions.py        |        4 |        0 |    100% |           |
| src/coonfit/inference.py         |      142 |        8 |     94% |201-203, 266, 271-272, 326, 769 |
| src/coonfit/parallel\_helpers.py |      147 |       60 |     59% |96-115, 139-142, 326-330, 362, 423-475 |
| src/riogrande/helper.py          |      194 |       15 |     92% |191-193, 363, 417, 447, 589-590, 692, 776, 780-781, 825, 829, 892-901 |
| src/riogrande/io/core.py         |      138 |       12 |     91% |370-372, 387, 426-427, 458-468, 537 |
| src/riogrande/io/exceptions.py   |       10 |        0 |    100% |           |
| src/riogrande/io/models.py       |      414 |       79 |     81% |118, 136, 243-254, 352-355, 373-374, 437, 445, 454, 520, 632-636, 660-661, 702-709, 743-752, 755, 759, 954, 969, 986-1047, 1194, 1244, 1249, 1428, 1551, 1649 |
| src/riogrande/parallel.py        |      185 |       64 |     65% |51, 53, 56, 59, 67-80, 112-125, 173-174, 180-182, 242-243, 246-249, 292-297, 350, 360, 396, 461-472, 596 |
| src/riogrande/prepare.py         |       92 |        2 |     98% |  123, 130 |
| src/riogrande/timing.py          |       22 |        0 |    100% |           |
| **TOTAL**                        | **1589** |  **305** | **81%** |           |


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