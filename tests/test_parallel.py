import builtins
from functools import partial

from time import sleep

import numpy as np
import multiprocessing as mproc
import itertools
import random
import rasterio as rio

from landiv_blur import helper as lbhelp
from landiv_blur import io as lbio
from landiv_blur import io_ as lbio_
from landiv_blur import processing as lbproc
from landiv_blur import prepare as lbprep
from landiv_blur import inference as lbinf
from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lbf_gauss
from landiv_blur.helper import rasterio_to_numpy_dtype

from .conftest import ALL_MAPS, get_file, set_mpc_strategy

from matplotlib import pyplot as plt

