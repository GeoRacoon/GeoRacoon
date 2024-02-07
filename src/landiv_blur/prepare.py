"""
Submodule providing the necessary functions to setup an efficient processing
of a land-cover type map.
"""
import numpy as np


def overhead_fraction(nbr_blocks, border, size):
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
    total_area = size[0] * size[1]

    h_overhead = (nbr_blocks - 1) * (2 * border) * size[0]
    v_overhead = (nbr_blocks - 1) * (2 * border) * size[1]
    corners = 4 * border**2
    overhead_area = h_overhead + v_overhead + corners
    return overhead_area / total_area


def get_blocks(nbr_blocks, border, size):
    """Returns a set blocks on which the filter can be applied independently.

    Parameters
    ----------
    nbr_blocks: scalar
      The number of block along one axis
    border: int
      The border size in number of pixels
    size: tuple
      The total size of the map in number of pixels (width, height)

    Return
    ------

    list:
      A list of tuples (x, y, width, height) defining each block on which to
      apply a filter
    """
    for i, s in enumerate(size):
        assert int(s / nbr_blocks) == s / nbr_blocks, \
               f"{size[i]=} needs to be a multiple of {nbr_blocks=}"
    block_sizes = list(map(lambda x: int(x / nbr_blocks), size))
    vstarts = []
    hstarts = []
    heights = []
    widths = []
    for i in range(nbr_blocks):
        if 0 < i < nbr_blocks - 1:
            vpadding = 2 * border
        else:
            vpadding = border
        for j in range(nbr_blocks):
            if 0 < j < nbr_blocks - 1:
                hpadding = 2 * border
            else:
                hpadding = border
            vstarts.append(max(0, i * block_sizes[0] - border))
            hstarts.append(max(0, j * block_sizes[1] - border))
            heights.append(block_sizes[0] + vpadding)
            widths.append(block_sizes[1] + hpadding)
    return list(zip(vstarts, hstarts, heights, widths))
