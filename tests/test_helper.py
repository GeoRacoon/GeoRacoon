import pytest

import numpy as np
from landiv_blur.helper import (
    match_all,
    match_any,
    count_contribution
)

def test_matching():
    """Make sure our matching functions work as expected
    """
    d1 = dict(
        a = 1,
        b = 2,
        c = 3,
    )
    d2 = dict(
        a = 1,
        b = 2,
    )
    d3 = dict(
        c = 3,
        d = 4
    )
    # not all of d1 is in d2
    assert not match_all(d1, d2)
    # all of d2 is in d1
    assert match_all(d2, d1)
    # some of d1 is in d3
    assert match_any(d1, d3)
    # some of d3 is in d1
    assert match_any(d3, d1)
    # none of d2 is in d3
    assert not match_any(d2, d3)

def test_count_contrib():
    """
    """
    a = np.array([[1., 2., 2.],
                  [3., 0., 2.],
                  [0., 1., 2.]])
    b_array = np.array([[True, False, False],
                        [False, False, False],
                        [True, True, False]])
    no_data = 0.0
    counts = count_contribution(data=a,
                                selector=b_array,
                                no_data=no_data)
    # we have 3 unmasked whereof 2 are not no_data
    assert counts == 2
    # set no_data to np.nan
    no_data = np.nan
    # now all unmasked should count
    counts = count_contribution(data=a,
                                selector=b_array,
                                no_data=no_data)
    assert counts == 3
    # set a np.nan
    a[0][0] = np.nan
    counts = count_contribution(data=a,
                                selector=b_array,
                                no_data=no_data)
    # now only 2 out of the unmased should be there
    assert counts == 2
    # now with no valid data
    a[b_array] = 0.0
    counts = count_contribution(data=a,
                                selector=b_array,
                                no_data=0.0)
    # now only 2 out of the unmased should be there
    assert counts == 0
