import os
import pytest

FIXTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'data'
))

ALL_MAPS = pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'testing', 'landcover',
                 'Switzerland_CLC_2012_reclass8.tif'),
    os.path.join(FIXTURE_DIR, 'testing', 'ndvi',
                 'Switzerland_NDVI_binning_2015.tif'),
)
