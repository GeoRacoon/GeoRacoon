"""
Submodule providing the necessary functions to setup an efficient processing
of a land-cover type map.
"""


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

def update_views(data, views, blocks):
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

def update_view(data, view, block):
    """Update a view from the data array with a block

    ..Note::
       `block.shape must be equal to height and width of `view`, so
       `block.shape == (view[2], view[3])`

    Parameters
    ----------
    data:
      The map from which we want to get views from
    view:
      tuple (x, y, width, height) defining the view of the data array to update
    block:
      np.array with the updated values.

    """
    data[slice(view[0], view[0] + view[2]),
         slice(view[1], view[1] + view[3])] = block


def create_views(nbr_views, border, size):
    """Returns a set of views on which the filter can be applied independently

    Parameters
    ----------
    nbr_views: tuple of scalars
      The number of view along each axis
    border: tuple of int
      The border size in number of pixels along each axis
    size: tuple
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
    assert all(len(x) == 2 for x in (nbr_views, border, size)), \
           f"{len(nbr_views)=},{len(border)=},{len(size)=} all need to be of "\
           "length 2 (vertical, horizontal)"
    for i, s in enumerate(size):
        view_size = int(s / nbr_views[i])
        assert view_size == s / nbr_views[i], \
               f"{size[i]=} needs to be a multiple of {nbr_views[i]=}"
        assert border[i] <= view_size, \
               f"{border[i]=} cannot be bigger the {view_size=}"
    view_sizes = list(map(lambda i: int(size[i] / nbr_views[i]),
                          range(len(size))))
    vstarts = []
    hstarts = []
    heights = []
    widths = []
    inner_vs = []
    inner_hs = []
    inner_h = []
    inner_w = []
    for i in range(nbr_views[0]):  # vertically
        if 0 < i < nbr_views[0] - 1:
            vpadding = 2 * border[0]
        else:
            vpadding = border[0]
        for j in range(nbr_views[1]):  # horizontally
            if 0 < j < nbr_views[1] - 1:
                hpadding = 2 * border[1]
            else:
                hpadding = border[1]
            vstarts.append(max(0, i * view_sizes[0] - border[0]))
            hstarts.append(max(0, j * view_sizes[1] - border[1]))
            heights.append(view_sizes[0] + vpadding)
            widths.append(view_sizes[1] + hpadding)
            # the useable inner view
            inner_vs.append(max(0, i * view_sizes[0]))
            inner_hs.append(max(0, j * view_sizes[1]))
            inner_h.append(view_sizes[0])
            inner_w.append(view_sizes[1])
    return (
        list(zip(vstarts, hstarts, heights, widths)),
        list(zip(inner_vs, inner_hs, inner_h, inner_w))
    )


def get_view(data, view):
    """Return a view of the data array

    Parameters
    ----------
    data:
      np.array to return the view from
    view:
      tuple (x, y, width, height) defining the view

    """
    return data[slice(view[0], view[0] + view[2]),
                slice(view[1], view[1] + view[3])]

def relative_view(view, inner_view):
    return (inner_view[0] - view[0],
            inner_view[1] - view[1], 
            inner_view[2],
            inner_view[3])


def recombine_blocks(blocks, output):
    """Write a sequence of blocks onto an output array

    Parameters
    ----------
    blocks:
      iterable of blocks each being a tuple `(data, view)`
      where data is an np.array with the data to use in the update
      and view a tuple with
      (vertical start, horiz start, height, widht) of the view
      to update
    """
    for data, view in blocks:
        update_view(output, view, block=data)
    return output
