import pytest
import convster as cvr


def test_package_import():
    """
    Simple test to verify package import
    """
    assert cvr._answer_to_everything == 42


