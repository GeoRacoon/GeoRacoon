from landiv_blur import prepare as lbprep


def test_block_definition():
    """Make sure that the blocks returned are correct
    """
    msize = 9
    size = (msize, msize)
    border = 1
    nbr_blocks = 3
    # in a matrix of 9x9 with 3 blocks and a border size of 1 we expect a.o.:
    expected_blocks = [
        (0, 0, 4, 4),  # block in the reference corner
        (2, 2, 5, 5),  # block in the center computing the center 3x3 square
        (2, 0, 5, 4),  # block vertically centered on horizontal border
        (5, 5, 4, 4),  # block opposite to the reference corner
    ]
    blocks = lbprep.get_blocks(nbr_blocks, border, size)
    for eb in expected_blocks:
        assert eb in blocks, f"Block {eb} not in {blocks=}"
