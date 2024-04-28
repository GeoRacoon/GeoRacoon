"""
This package contains functions for specific filters that can be applied to
blur land-cover type maps.

Whenever possible the filters are not implemented directly but imported from
packages with efficient implementations.
In particular implementations from https://scikit-image.org are used.

Usually, within a specific sub-module a filter method of the same name exists.
So, for example, within `filters.gaussian`, `skimage.filters.gaussian` is
available as `gaussian`.
"""

from .gaussian import gaussian as _gauss_filter
from .gaussian import get_kernel_diameter as _gauss_get_kd
from .gaussian import get_kernel_size as _gauss_get_ks


_filters = [_gauss_filter, ]
_get_kernel_diam = [_gauss_get_kd, ]
_get_kernel_size = [_gauss_get_ks, ]
