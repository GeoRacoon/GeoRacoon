"""This module defines functions that might be helpful when working
with rasterio
"""
from __future__ import annotations

import os
import json
import warnings

import numpy as np
import rasterio as rio

from rasterio.windows import Window

from typing import Any, Union, Dict, List

from collections.abc import Collection

from numpy.typing import NDArray

from decimal import Decimal

import multiprocessing as mpc
from multiprocessing import context as _context_module
from typing import Optional










