import pytest

import numpy as np
import random

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur.helper import (
    match_all,
    match_any,
    count_contribution,
    convert_to_dtype,
    check_rank_deficiency
)

from .conftest import ALL_MAPS, get_file

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

def test_convert_to_dtype_basics():
    """
    """
    # array containing min and max of uint16
    a = np.array([[2,4,4],[3,0.,2], [20,128.,255.]], dtype=np.uint8)
    # no rescaling since no range is given
    b = convert_to_dtype(data=a, as_dtype=np.uint16)
    assert np.max(b) == np.iinfo(np.uint8).max
    assert np.min(b) == np.iinfo(np.uint8).min
    # now using uint16 as output range
    b_scaled = convert_to_dtype(data=a, as_dtype=np.uint16, out_range='uint16')
    assert np.max(b_scaled) == np.iinfo(np.uint16).max
    assert np.min(b_scaled) == np.iinfo(np.uint16).min
    # convert and scale to custom range
    c = convert_to_dtype(data=a, as_dtype=np.float32, out_range=[0,1])
    assert np.max(c) == 1
    assert np.min(c) == 0
    # rescale a float in [0,1] to a float in [0,1] > do nothing
    d = convert_to_dtype(data=c, as_dtype='float32')
    np.testing.assert_equal(c, d)
    # rescale float to float with identical range > do noting
    d_scaled = convert_to_dtype(data=c, as_dtype=np.float32, in_range=[0,1],
                         out_range=[0,1])
    np.testing.assert_equal(c, d_scaled)
    # convert and scale a float back to an uint8 full range
    print(f"{c=}")

    # d_scaled should be all 0 only where c==1, we get 1
    d_expect = np.zeros_like(d_scaled)
    d_expect[c==1] = 1
    e_scaled = convert_to_dtype(data=d, as_dtype=np.uint8, in_range=[0,1])
    np.testing.assert_equal(e_scaled, a)
    # convert without scaling
    e = convert_to_dtype(data=d, as_dtype=np.uint8)
    # converting [0,1] to uint8 gives 0 unless the value is 1
    e_expect = np.zeros_like(d)
    e_expect[d==1] = 1
    np.testing.assert_equal(e, e_expect)

    # simply rescale [0,1] to [0,0.5] - doubling it should bring us back
    c_scaled = convert_to_dtype(c, in_range=[0, 1], out_range=[0, 0.5])
    np.testing.assert_equal(c, 2*c_scaled)

    # using strings to set the dtype
    dtype_str = 'float64'
    d = convert_to_dtype(data=a, as_dtype=dtype_str, out_range=(0,1))
    # convert back to uint8
    a_reconv = convert_to_dtype(data=d, in_range=(0,1), as_dtype='uint8')
    np.testing.assert_equal(a, a_reconv)
    # expecting a TypeError error with badly formatted string
    dtype_str_broken = 'foat64'
    with pytest.raises(TypeError, match=dtype_str_broken):
        d = convert_to_dtype(data=a, as_dtype=dtype_str_broken, out_range=(0,1))


def test_convert_to_dtype_range_handling():
    """Make sure input data is handled properly
    """
    # floats scaled form [0,1] to  [0, 1] range should not change anything
    a = np.array([[0,0,0],[0.5, 0.5, 0.5], [1,1,1]], dtype=np.float64)
    a_converted = convert_to_dtype(data=a, as_dtype=np.float64, in_range=[0,1],
                                   out_range=[0.0, 1.0])
    np.testing.assert_equal(a, a_converted)
    # we want to emit a warning is something is converted to the full float range
    with pytest.warns(match='full range'):  # make sure we get a warning
        a_converted = convert_to_dtype(data=a, as_dtype=np.float64,
                                in_range=[0.0, 1.0])
    a_wrong = a.copy()
    # this should be scaled properly
    a_wrong[2][2] += 0.01
    with pytest.warns(match='exceeds'):  # make sure we get a warning
        a_converted = convert_to_dtype(data=a_wrong, as_dtype=np.float64,
                                in_range=[0,1],
                                out_range=[0.0, 1.0])



@ALL_MAPS
def test_convert_to_dtype_real_range_handling(datafiles):
    """Make sure datatypes are properly converted
    """
    ch_tif = get_file(pattern="Switzerland_CLC_*.tif", datafiles=datafiles)
    ch_data = lbio.load_map(ch_tif)['data']
    ch_range = np.nanmin(ch_data), np.nanmax(ch_data)
    print(ch_range)

    lctypes = lbproc.get_categories(ch_data)
    sigma = 10
    truncate = 3
    params = dict(
        sigma=sigma,
        truncate=truncate
    )
    # TODO

def test_rank_deficiency():
    """Test some examples to identify rank deficiency
    """
    N = 20

    # Test: zero column issue
    X = np.random.rand(50, N)
    col_zero = random.randint(0, N-1)
    X[:, col_zero] = 0

    b_symm = X.T @ X
    res = check_rank_deficiency(b_symm)
    print(f"{col_zero=} {res=}")
    assert sorted([col_zero]) == sorted([k for k in res.keys()])

    # Test: dependency issue
    X = np.random.rand(50, N)
    col_ldep = random.sample(range(0, N-1), 2)
    X[:, col_ldep[1]] = X[:, col_ldep[0]] * 2

    b_symm = X.T @ X
    res = check_rank_deficiency(b_symm)
    print(f"{col_ldep=} {res=}")
    assert sorted(col_ldep) == sorted([k for k in res.keys()])

    # Test: zero and dependency issue
    X = np.random.rand(50, N)
    col_issue = random.sample(range(0, N - 1), 3)
    X[:, col_issue[0]] = 0
    X[:, col_issue[1]] = X[:, col_issue[2]] * 2

    b_symm = X.T @ X
    res = check_rank_deficiency(b_symm, return_by_issue_type=True)
    print(f"{col_issue=} {res=}")
    assert sorted([col_issue[0]]) == sorted(res["all_zero"])
    assert sorted(col_issue[1:3]) == sorted(res["linear_dependent"])
