"""This module defines custom exceptions
"""
# TODO: all of these get use at a certain point


class InferenceError(Exception):
    """Some steps of a multiple linear regression have invalid parameters
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    pass

class BandSelectionAmbiguousError(Exception):
    """If multiple bands match the provided tags
    """
    # is_needed
    # no_work
    # not_tested (no need - used in tests)
    # usedin_both (in io mostly - relevant for both)
    pass

class BandSelectionNoMatchError(Exception):
    """No band found with matching tags
    """
    # is_needed
    # no_work
    # not_tested (no need - used in tests)
    # usedin_both (in io mostly - relevant for both)
    pass

class SourceNotSavedError(Exception):
    """This source cannot be found at the specified location.

    Either the path is wrong or the related data is not (yet) saved to file.
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_both (in io mostly - relevant for both)
    pass

class UnknownExtensionError(Exception):
    """Unclear how to handle a file with the given extension.
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_both (in io mostly - relevant for both)
    pass

class InvalidMaskSelectorError(Exception):
    """An selector string was used that did not match to any mask reader.
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_both (in io mostly - relevant for both)
    pass

class InvalidPredictorError(Exception):
    """Not all of the provided predictors are valid.
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    pass
