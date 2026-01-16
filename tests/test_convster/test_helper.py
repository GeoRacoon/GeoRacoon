import numpy as np
from convster.helper import (first_nonzero,
                             last_nonzero)


def test_first_nonzero():
    a = np.array([
        [0, 0, 3, 0],
        [0, 0, 4, 0],
        [0, 2, 0, 0],
        [0, 3, 5, 4]
    ])
    np.testing.assert_array_equal(first_nonzero(a, axis=1), np.array([2, 2, 1, 1]))
    np.testing.assert_array_equal(first_nonzero(a, axis=0), np.array([-1, 2, 0, 3]))


def test_last_nonzero():
    a = np.array([
        [0, 0, 3, 0],
        [0, 0, 4, 0],
        [0, 2, 0, 0],
        [0, 3, 5, 4]
    ])
    np.testing.assert_array_equal(last_nonzero(a, axis=1), np.array([2, 2, 1, 3]))
    np.testing.assert_array_equal(last_nonzero(a, axis=0), np.array([-1, 3, 3, 3]))