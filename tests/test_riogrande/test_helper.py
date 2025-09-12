import pytest

import sys
import subprocess
import json
import os
import platform

import numpy as np
import multiprocessing as mpc

from riogrande.helper import (
    match_all,
    match_any,
    count_contribution,
    convert_to_dtype,
    get_nbr_workers
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


def test_get_nbr_workers_sequence(monkeypatch):
    """Test sequence for get_nbr_workers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch multiprocessing.cpu_count for deterministic
        behaviour.

    Notes
    -----
    This single test runs the following checks in order:
    - number is None: uses patched cpu_count and respects min_count.
    - number <= min_count: emits RuntimeWarning and returns min_count.
    - number > min_count: returns the provided number as int.
    - negative min_count: cpu_count still respected when number is None.
    """
    # 1) number is None -> uses cpu_count
    monkeypatch.setattr(mpc, "cpu_count", lambda: 4)
    assert get_nbr_workers(None) == 4

    # cpu_count less than min_count -> returns min_count
    monkeypatch.setattr(mpc, "cpu_count", lambda: 1)
    assert get_nbr_workers(None) == 2

    # 2) number <= min_count -> warns and returns min_count
    monkeypatch.setattr(mpc, "cpu_count", lambda: 8)

    with pytest.warns(RuntimeWarning,
                      match='will be ignored'):
        res = get_nbr_workers(1)
        assert res == 2

    # 3) number > min_count -> returns provided number (as int)
    assert get_nbr_workers(5) == 5
    assert get_nbr_workers(int(3.0)) == 3


def run_in_subprocess(pycode):
    cmd = [sys.executable, "-c", pycode]
    env = os.environ.copy()
    p = subprocess.run(cmd,
                       env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def make_worker_script(test_body):
    script = f'''
import multiprocessing
import warnings
import json
warnings.simplefilter("always", RuntimeWarning)

from riogrande.helper import get_or_set_context

result = {{}}
try:
    {test_body}
except Exception as e:
    result["error"] = repr(e)

print(json.dumps(result))
    '''
    return script


@pytest.mark.usefixtures()
class TestEnsureOrContext:
    @pytest.mark.skipif(platform.system() == "Windows", reason="Requires 'fork' support on non-Windows")
    def test_method_none_when_global_set(self):
        test_body = '''
    multiprocessing.set_start_method("fork")
    ctx = get_or_set_context(None)
    result["ctx_method"] = ctx.get_start_method()
    result["global"] = multiprocessing.get_start_method()
    '''
        code = make_worker_script(test_body)
        rc, out, err = run_in_subprocess(code)
        assert rc == 0, f"subprocess failed: {err}"
        j = json.loads(out)
        assert j.get("error") is None
        assert j["global"] == "fork"
        assert j["ctx_method"] == "fork"

    def test_method_none_when_global_unset_defaults_spawn(self):
        test_body = '''
    cur = multiprocessing.get_start_method(allow_none=True)
    result["initial"] = cur
    ctx = get_or_set_context(None)
    result["ctx_method"] = ctx.get_start_method()
    result["global_after"] = multiprocessing.get_start_method(allow_none=True)
    '''
        code = make_worker_script(test_body)
        rc, out, err = run_in_subprocess(code)
        assert rc == 0, err
        j = json.loads(out)
        assert j["initial"] is None
        assert j["ctx_method"] == "spawn"
        assert j["global_after"] is None

    def test_method_set_and_global_unset_sets_global(self):
        test_body = '''
    cur = multiprocessing.get_start_method(allow_none=True)
    result["initial"] = cur
    ctx = get_or_set_context("spawn")
    result["ctx_method"] = ctx.get_start_method()
    result["global_after"] = multiprocessing.get_start_method(allow_none=True)
    '''
        code = make_worker_script(test_body)
        rc, out, err = run_in_subprocess(code)
        assert rc == 0, err
        j = json.loads(out)
        assert j["initial"] is None
        assert j["ctx_method"] == "spawn"
        assert j["global_after"] == "spawn"

    @pytest.mark.skipif(platform.system() == "Windows", reason="Requires 'fork' support on non-Windows")
    def test_method_set_and_global_set_different_warns_but_returns_requested(self):
        test_body = '''
    multiprocessing.set_start_method("fork")
    cur = multiprocessing.get_start_method()
    result["global_initial"] = cur
    ctx = get_or_set_context("spawn")
    result["ctx_method"] = ctx.get_start_method()
    result["global_after"] = multiprocessing.get_start_method()
    '''
        code = make_worker_script(test_body)
        rc, out, err = run_in_subprocess(code)
        assert rc == 0, err
        j = json.loads(out)
        assert j["global_initial"] == "fork"
        assert j["global_after"] == "fork"
        assert j["ctx_method"] == "spawn"

    def test_invalid_method_raises(self):
        test_body = '''
    try:
        get_or_set_context("invalid_method")
        result["raised"] = False
    except Exception as e:
        result["raised"] = True
        result["err_type"] = type(e).__name__
        result["err_repr"] = repr(e)
    '''
        code = make_worker_script(test_body)
        rc, out, err = run_in_subprocess(code)
        assert rc == 0, err
        j = json.loads(out)
        assert j["raised"]
        assert j["err_type"] == "ValueError"
