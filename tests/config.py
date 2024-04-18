import os
import pytest

FIXTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../',
    'data'
))

ALL_MAPS = pytest.mark.datafiles(
    os.path.join(FIXTURE_DIR, 'ch.tif'),
)
