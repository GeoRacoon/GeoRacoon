# SPDX-FileCopyrightText: 2026 Jonas I. Liechti <j-i-l@t4d.ch>
# SPDX-FileCopyrightText: 2026 Simon Landauer <georacccoon@proton.me>
#
# SPDX-License-Identifier: MIT

import pytest
import riogrande as riog


def test_package_import():
    """
    Simple test to verify package import
    """
    assert riog._answer_to_everything == 42

