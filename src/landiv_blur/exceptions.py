"""This module defines custom exceptions
"""
# TODO: all of these get use at a certain point


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

class InvalidMaskSelectorError(Exception):
    """An selector string was used that did not match to any mask reader.
    """
    pass

class InvalidPredictorError(Exception):
    """Not all of the provided predictors are valid.
    """
    pass
