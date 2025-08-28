"""
Add sth here
"""
from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

import math


def update_view(data:NDArray,
                view:tuple[int,int,int,int],
                block:ArrayLike)->None:
    # TODO: is_needed - needs_work - is_tested - usedin_both
    """Update a view from the data array with a block

    ..Note::
       `block.shape must be equal to height and width of `view`, so
       `block.shape == (view[3], view[2])`

    Parameters
    ----------
    data:
      The array that we want to update
    view:
      tuple (x, y, width, height) defining the view of the data array to update
    block:
      np.array with the updated values.

    """
    # is_needed
    # no_work
    # not_tested (used in test)
    # usedin_both

    # data[slice(view[0], view[0] + view[2]),
    #      slice(view[1], view[1] + view[3])] = block
    data[slice(view[1], view[1] + view[3]),
         slice(view[0], view[0] + view[2])] = block


def create_views(view_size:tuple[int, int],
                 border:tuple[int, int],
                 size:tuple[int, int])->tuple[list, ...]:
    # TODO: is_needed - needs_work - is_tested - usedin_both
    """Returns a set of views on which the filter can be applied independently

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

    tuple:
      The first element is a list of tuples (x, y, width, height) defining each
      view on which to apply a filter.
      The second element is a list of tuples (x, y, width, height) defining for
      each view the usable region.
      A region is usable if it does not contain any artificial border effects
      that were introduced from splitting up a bigger view into smaller chunks
    """
    # is_needed
    # no_work
    # is_tested
    # usedin_both
    assert all(len(x) == 2 for x in (view_size, border, size)), \
           f"{len(view_size)=},{len(border)=},{len(size)=} all need to be of "\
           "length 2 (width, height)"
    # calculate the leftovers along both axes
    leftovers = list(map(lambda x: x[0] % x[1], zip(size, view_size)))
    # number of full block along each axis that do not cover the leftovers
    nbr_views = list(map(lambda x: math.floor((x[1] - x[2]) / x[0]),
                         zip(view_size, size, leftovers)))
    # print("\n")
    # print(f"\t{size=}")
    # print(f"\t{view_size=}")
    # print(f"\t{leftovers=}")
    # print(f"\t{nbr_views=}")
    # print(f"\t{border=}")
    # print("\n")

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
            # TODO: could this not be handled in next if?
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
    # print(f"{xstarts=}\n{ystarts=}")
    # print(f"{widths=}\n{heights=}")
    # print(f"{inner_xs=}\n{inner_ys=}")
    # print(f"{inner_w=}\n{inner_h=}")
    return (
        list(zip(xstarts, ystarts, widths, heights)),
        list(zip(inner_xs, inner_ys, inner_w, inner_h))
    )


def get_view(data:NDArray, view:tuple[int,int,int,int])->NDArray:
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    # TODO: This was moved here to riogrand as it makes more sense, despite it may only be used in processing
    """Return a view of the data array

    ..Note::
      data.shape == height, width!

    Parameters
    ----------
    data:
      np.array to return the view from
    view:
      tuple (x, y, width, height) defining the view

    """
    # is_needed
    # not_tested (used in test)
    # no_work
    # usedin_processing (maybe both)

    # return data[slice(view[0], view[0] + view[2]),
    #             slice(view[1], view[1] + view[3])]
    return data[slice(view[1], view[1] + view[3]),
                slice(view[0], view[0] + view[2])]


def relative_view(view:tuple[int,int,int,int],
                  inner_view:tuple[int,int,int,int])->tuple[int,int,int,int]:
    # TODO: is_needed - needs_work - is_tested - usedin_processing
    # TODO: This was moved here to riogrand as it makes more sense, despite it may only be used in processing
    """Return the `inner_view` relative to `view`

    Parameters
    ----------
    view:
      (x, y, width, height) defining a view
    inner_view:
      (x, y, width, height) defining a view
    """
    # is_needed
    # needs_work (better docs)
    # not_tested (used in tests)
    # usedin_processing (maybe both)
    return (inner_view[0] - view[0],
            inner_view[1] - view[1],
            inner_view[2],
            inner_view[3])


