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
