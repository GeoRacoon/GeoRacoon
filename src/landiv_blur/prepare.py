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
    # TODO: not_needed - no_work - not_tested
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
    # TODO: not_needed
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








def get_view(data:NDArray, view:tuple[int,int,int,int])->NDArray:
    # TODO: is_needed - needs_work - is_tested - usedin_processing
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
    # TODO: not_needed
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



