import sys
import subprocess
import json
import os
import pytest
import platform
import multiprocessing as mpc

from landiv_blur.helper import get_nbr_workers


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
    assert get_nbr_workers(None, min_count=2) == 2

    # 2) number <= min_count -> warns and returns min_count
    monkeypatch.setattr(mpc, "cpu_count", lambda: 8)

    with pytest.warns(RuntimeWarning,
                      match='will be ignored'):
        res = get_nbr_workers(1, min_count=2)
        assert res == 2

    # 3) number > min_count -> returns provided number (as int)
    assert get_nbr_workers(5, min_count=2) == 5
    assert get_nbr_workers(int(3.0), min_count=2) == 3

    # 4) negative min_count handled (cpu_count used when number is None)
    monkeypatch.setattr(mpc, "cpu_count", lambda: 2)
    assert get_nbr_workers(None, min_count=-1) == 2


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

from landiv_blur.helper import get_or_set_context

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
