# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

import numpy as np
import rasterio as rio

from matplotlib import pyplot as plt

from riogrande import helper as rghelp
from riogrande import io as rgio
from riogrande import prepare as rgprep

from .conftest import ALL_MAPS, get_file, set_mpc_strategy

