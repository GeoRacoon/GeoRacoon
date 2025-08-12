"""
Submodule providing the necessary functions to allow an efficient processing
of a categorical maps, like land-cover types or similar.
"""
from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

import math

def overhead_fraction(nbr_blocks:int,
                      border:int,
                      size:tuple[int,int])->float:
    """Compute the fraction of area that is processed multiple times.

    This function reports the relation between pixels that are processed
    multiple times and the total amount of pixels.
    A pixel that is processed n times will count (n-1) times to the amount of
    pixels that are processed multiple times. Thus an `overhead_fraction` of
    `1.0` means that on average each pixel is processed twice.

    Clearly one should aim for a configuration with an `overhead_fraction` as
    low as possible.

    Parameters
    ----------
    nbr_blocks: scalar
      The number of block along one axis
    border: int
      The border size in number of pixels
    size: tuple
      The total size of the map in number of pixels (width, height)

    """
    # not_needed (thought that might be an interesting measure to have)
    # no_work
    # not_tested
    # usedin_processing
    total_area = size[0] * size[1]

    h_overhead = (nbr_blocks - 1) * (2 * border) * size[0]
    v_overhead = (nbr_blocks - 1) * (2 * border) * size[1]
    corners = 4 * border**2
    overhead_area = h_overhead + v_overhead + corners
    return overhead_area / total_area


def update_views(data:NDArray,
                 views:list[tuple[int,int,int,int]],
                 blocks:list[ArrayLike])->None:
    """Update the data array with a sequence of views.

    Note that the updates are applied in order of the provided list from first
    to last.

    Parameters
    ----------
    data:
      The map from which we want to get views from
    views:
      a list of tuples (x, y, width, height) defining each view for where the
      data array should be updated.

    """
    for view, block in zip(views, blocks):
        update_view(data, view, block)

    return None


def update_view(data:NDArray,
                view:tuple[int,int,int,int],
                block:ArrayLike)->None:
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


def recombine_blocks(blocks:list[tuple[ArrayLike, tuple[int,int,int,int]]],
                     output:NDArray)->NDArray:
    """Write a sequence of blocks onto an output array

    Parameters
    ----------
    blocks:
      iterable of blocks each being a tuple `(data, view)`
      where data is an np.array with the data to use in the update
      and view a tuple with
      (vertical start, horiz start, height, width) of the view
      to update
    output:
      numpy array in which the block will be updated

    Returns
    -------
    ArrayLike

      The array provided in `output` with the updated blocks

    """
    # not_needed
    # no_work
    # not_tested
    # usedin_both (potentially)
    for data, view in blocks:
        update_view(output, view, block=data)
    return output


def get_blur_params(diameter=None, sigma=None, truncate=None):
    """
    .. note::
        The default of truncate is 3

    """
    # is_needed
    # needs_work (docs)
    # not_tested (used in tests)
    # usedin_processing

    # use default value of 3 for truncate
    truncate = truncate or 3
    if diameter:
        if sigma:
            truncate = 0.5 * diameter / sigma
        else:
            if truncate:
                sigma = 0.5 * diameter / truncate
    else:
        if sigma:
            diameter = 2 * sigma * truncate
        else:
            # TODO: this test should be done when parsing the input arguments
            raise TypeError("Either the `diameter` or the `sigma` parameter "
                            " need to be provided. We got:\n"
                            f"- {diameter=}\n- {sigma=}")
    return dict(diameter=diameter, sigma=sigma, truncate=truncate)
