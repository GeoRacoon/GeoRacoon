"""
Functions used to create windows/ views for parallelization processes in 2D raster maps.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray


def update_view(data: NDArray, view: tuple[int, int, int, int], block: ArrayLike) -> None:
    """Update a view from the data array with a block

    The block must exactly match the shape of the view:
    `block.shape == (view[3], view[2])`, where
    `view = (x, y, width, height)`.

    Parameters
    ----------
    data:
      The array that we want to update
    view:
      tuple (x, y, width, height) defining the view of the data array to update
    block:
      np.array with the updated values.

    Returns
    --------
    None

    See Also
    --------
    :func:`~riogrande.prepare.get_view` : Read a rectangular view from an array.
    :func:`~riogrande.prepare.create_views` : Generate a set of views covering an array.

    Examples
    --------
    >>> data = np.zeros((5, 5))
    >>> block = np.ones((2, 3))
    >>> update_view(data, (1, 2, 3, 2), block)
    >>> data
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    """
    data[slice(view[1], view[1] + view[3]), slice(view[0], view[0] + view[2])] = block


def create_views(view_size: tuple[int, int], border: tuple[int, int], size: tuple[int, int]) -> tuple[list, ...]:
    """Returns a set of views on which an operation can be applied independently

    Parameters
    ----------
    view_size:
      The size (width, height) in pixels of a single view (excluding borders)
    border:
      The border size (width, height) in number of pixels along each axis
    size:
      The total size of the map in number of pixels (width, height)

    Return
    ------
    tuple
      The first element is a list of tuples (x, y, width, height) defining each
      view on which to apply an operation.
      The second element is a list of tuples (x, y, width, height) defining for
      each view the usable region.
      A region is usable if it does not contain any artificial border effects
      that were introduced from splitting up a bigger view into smaller chunks

    Notes
    -----
    - Handles cases where the region cannot be divided evenly by `view_size`.
      The last row/column of views may be smaller (`leftovers`) and are still included.
      Uses :func:`math.floor` to compute the number of full blocks.
    - Borders on the outer edges are reduced to fit within the total size.

    See Also
    --------
    :func:`~riogrande.prepare.update_view` : Write a block into a view of an array.
    :func:`~riogrande.prepare.get_view` : Read a rectangular view from an array.
    :func:`~riogrande.prepare.relative_view` : Express an inner view relative to an outer view.

    Examples
    --------
    >>> views, usable = create_views((5, 5), (1, 1), (9, 9))
    >>> len(views), len(usable)
    (4, 4)
    >>> views
    [(0, 0, 6, 6), (3, 0, 6, 5), (0, 3, 6, 6), (3, 3, 6, 6)]
    >>> usable
    [(0, 0, 4, 4), (4, 0, 5, 4), (0, 4, 4, 5), (4, 4, 5, 5)]
    """
    assert all(len(x) == 2 for x in (view_size, border, size)), \
        f"{len(view_size)=},{len(border)=},{len(size)=} all need to be of " \
        "length 2 (width, height)"
    # calculate the leftovers along both axes
    leftovers = list(map(lambda x: x[0] % x[1], zip(size, view_size)))
    # number of full block along each axis that do not cover the leftovers
    nbr_views = list(map(lambda x: math.floor((x[1] - x[2]) / x[0]),
                         zip(view_size, size, leftovers)))

    xstarts = []
    ystarts = []
    heights = []
    widths = []
    inner_xs = []
    inner_ys = []
    inner_h = []
    inner_w = []
    for i in range(nbr_views[0]):  # horizontally
        if 0 < i < nbr_views[0] - 1:
            hpadding = 2 * border[0]
        elif nbr_views[0] == 1:
            hpadding = 0
        else:
            hpadding = border[0]
        for j in range(nbr_views[1]):  # vertically
            if 0 < j < nbr_views[1] - 1:
                vpadding = 2 * border[1]
            elif nbr_views[1] == 1:
                vpadding = 0
            else:
                vpadding = border[1]
            xstarts.append(max(0, i * view_size[0] - border[0]))
            widths.append(view_size[0] + hpadding)
            ystarts.append(max(0, j * view_size[1] - border[1]))
            heights.append(view_size[1] + vpadding)
            # the useable inner view
            # starting points form a regular grid
            inner_xs.append(i * view_size[0])
            inner_ys.append(j * view_size[1])
            # used is always the view_size
            inner_w.append(view_size[0])
            inner_h.append(view_size[1])
            # handle the leftover
            # prepare for horiz. leftover pixels
            if i == nbr_views[0] - 1 and leftovers[0]:
                inner_w[-1] = leftovers[0]
                widths[-1] = leftovers[0] + 2 * border[0]
            # prepare for vertical. leftover pixels
            if j == nbr_views[1] - 1 and leftovers[1]:
                inner_h[-1] = leftovers[1]
                heights[-1] = leftovers[1] + 2 * border[1]
    if leftovers[0]:
        # add the last column of leftovers
        for j in range(nbr_views[1]):  # vertically since column
            _width = border[0]
            if 0 < j < nbr_views[1] - 1:
                vpadding = 2 * border[1]
            else:
                vpadding = border[1]
            xstarts.append(size[0] - (view_size[0] + _width))
            ystarts.append(max(0, j * view_size[1] - border[1]))
            widths.append(view_size[0] + _width)
            heights.append(view_size[1] + vpadding)
            # the useable inner view
            inner_xs.append(size[0] - view_size[0])
            inner_ys.append(max(0, j * view_size[1]))
            inner_w.append(view_size[0])
            inner_h.append(view_size[1])
            # NOTE: this is not the corner, but the last of the regular blocks
            if j == nbr_views[1] - 1 and leftovers[1]:
                inner_h[-1] = leftovers[1]
                heights[-1] = leftovers[1] + border[1]
    if leftovers[1]:
        # add the last row of leftovers
        for i in range(nbr_views[0]):  # horizontally since row
            _height = border[1]
            if 0 < i < nbr_views[0] - 1:
                hpadding = 2 * border[0]
            else:
                hpadding = border[0]
            xstarts.append(max(0, i * view_size[0] - border[0]))
            ystarts.append(size[1] - (view_size[1] + _height))
            widths.append(view_size[0] + hpadding)
            heights.append(view_size[1] + _height)
            # the useable inner view
            inner_xs.append(i * view_size[0])
            inner_ys.append(size[1] - view_size[1])
            inner_w.append(view_size[0])
            inner_h.append(view_size[1])
            if i == nbr_views[0] - 1 and leftovers[0]:
                inner_w[-1] = leftovers[0]
                widths[-1] = leftovers[0] + 2 * border[0]
    if all(leftovers):
        # add the outer corner block
        xstarts.append(size[0] - (view_size[0] + border[0]))
        ystarts.append(size[1] - (view_size[1] + border[1]))
        widths.append(view_size[0] + border[0])
        heights.append(view_size[1] + border[1])
        inner_xs.append(size[0] - view_size[0])
        inner_ys.append(size[1] - view_size[1])
        inner_w.append(view_size[0])
        inner_h.append(view_size[1])

    return (list(zip(xstarts, ystarts, widths, heights)),
            list(zip(inner_xs, inner_ys, inner_w, inner_h)))


def get_view(data: NDArray, view: tuple[int, int, int, int]) -> NDArray:
    """Return a recatangular view of the data array

    ..Note::
      data.shape == height, width!

    Parameters
    ----------
    data:
      np.array to return the view from
    view:
      tuple (x, y, width, height) defining the view

    Returns
    -------
    NDArray
        A view (slice) of `data` with shape `(height, width)` as specified
        by the `view` tuple.

    See Also
    --------
    :func:`~riogrande.prepare.update_view` : Write a block into a view of an array.
    :func:`~riogrande.prepare.create_views` : Generate a set of views covering an array.

    Examples
    --------
    >>> arr = np.arange(16).reshape(4, 4)
    >>> arr
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> get_view(arr, (1, 1, 2, 2))
    array([[ 5,  6],
           [ 9, 10]])
    """
    return data[slice(view[1], view[1] + view[3]),
                slice(view[0], view[0] + view[2])]


def relative_view(view: tuple[int, int, int, int],
                  inner_view: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return the `inner_view` relative to `view`

    Given two rectangular regions defined as `(x, y, width, height)`,
    this function returns the coordinates of `inner_view` relative
    to the origin of `view`.

    Parameters
    ----------
    view :
        A 4-tuple `(x, y, width, height)` defining the outer view.
    inner_view :
        A 4-tuple `(x, y, width, height)` defining a subregion inside `view`.

    Returns
    --------
    tuple
        A 4-tuple `(x, y, width, height)` giving the position and size of
        `inner_view` relative to `view`. The width and height are unchanged.

    See Also
    --------
    :func:`~riogrande.prepare.create_views` : Generate outer and inner view pairs.

    Examples
    --------
    >>> outer = (10, 20, 100, 50)
    >>> inner = (15, 30, 20, 10)
    >>> outer
    (10, 20, 100, 50)
    >>> inner
    (15, 30, 20, 10)
    >>> relative_view(outer, inner)
    (5, 10, 20, 10)
    """
    return (inner_view[0] - view[0], inner_view[1] - view[1],
            inner_view[2], inner_view[3])
