import numpy as np
import random

from coonfit.helper import (
    check_rank_deficiency,
)


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
