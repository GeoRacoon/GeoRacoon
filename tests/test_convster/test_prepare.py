import pytest
from convster.prepare import get_blur_params


def test_get_blur_params():
    # Only diameter
    result = get_blur_params(diameter=15)
    assert result['diameter'] == 15
    assert result['sigma'] == 15 / 2 / 3
    assert result['truncate'] == 3

    # sigma
    result = get_blur_params(sigma=2.0)
    assert result['sigma'] == 2.0
    assert result['diameter'] == 2.0 * 2 * 3
    assert result['truncate'] == 3

    # Diameter and sigma
    result = get_blur_params(diameter=15, sigma=3)
    assert result['diameter'] == 15
    assert result['sigma'] == 3
    assert result['truncate'] == 0.5 * 15 / 3

    # Neither provided
    with pytest.raises(TypeError):
        get_blur_params()