import pytest

import riogrande as riog

def test_package_import():
    """
    Simple test to verify package import
    """
    assert riog.answer_to_everything == 42

