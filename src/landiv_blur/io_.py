from __future__ import annotations
import os

import numpy as np
import rasterio as rio

from rasterio.windows import Window
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import partial
from numpy.typing import NDArray
from collections.abc import Callable

from typing import Union

from ._exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
    InvalidMaskSelectorError,
)

from .io import (
    NS,
    get_tags,
    set_tags,
    get_bidx,
    match_all,
    find_bidxs,
    compress_tif,
    load_block
)
from ._helper import (
    check_compatibility as _check_compatibility,
    count_contribution,
)


# TODO: I feel: all is_needed - but needs a bit of work

