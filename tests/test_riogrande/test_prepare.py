from riogrande import prepare as rgprep


def test_view_definition():
    """Make sure that the views returned are correct
    """
    size = (9, 12)
    border = (1, 2)
    view_size = (3, 4)
    # in a matrix of 9x9 with 3 views and a border size of 1 we expect a.o.:
    expected_views = [
        # in the reference corner:
        (0, 0, view_size[0] + border[0], view_size[1] + border[1]),
        (2, 2, 5, 8),  # view towards the center
        (2, 0, 5, 6),  # view vertically centered on horizontal left border
        (5, 6, 4, 6),  # view opposite to the reference corner
    ]
    views, inner_views = rgprep.create_views(view_size=view_size,
                                             border=border,
                                             size=size)
    for eb in expected_views:
        assert eb in views, f"View {eb} not in {views=}"
    inner_expected_views = [
        (0, 0, 3, 4),  # view in the reference corner
        (3, 4, 3, 4),  # view towards center
        (3, 0, 3, 4),  # view vertically centered on horizontal left border
        (6, 8, 3, 4),  # view opposite to the reference corner
    ]
    for ieb in inner_expected_views:
        assert ieb in inner_views, f"View {ieb} not in inner {inner_views=}"
