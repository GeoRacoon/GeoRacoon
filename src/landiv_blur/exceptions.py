"""This module defines custom exceptions
"""


class InferenceError(Exception):
    """Some steps of a multiple linear regression have invalid parameters
    """
    pass


class BandSelectionAmbiguousError(Exception):
    """If multiple bands match the provided tags
    """
    pass

class BandSelectionNoMatchError(Exception):
    """No band found with matching tags
    """
    pass
