"""This module defines custom exceptions
"""


class InvalidPredictorError(Exception):
    """Not all the provided predictors are valid.
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    pass


class InferenceError(Exception):
    """Some steps of a multiple linear regression have invalid parameters
    """
    # is_needed
    # no_work
    # not_tested (no need)
    # usedin_linfit
    pass
