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

class SourceNotSavedError(Exception):
    """This source cannot be found at the specified location.

    Either the path is wrong or the related data is not (yet) saved to file.
    """
    pass

class UnknownExtensionError(Exception):
    """Unclear how to handle a file with the given extension.
    """
    pass
