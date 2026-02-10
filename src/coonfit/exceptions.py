"""This module defines custom exceptions
"""


class InvalidPredictorError(Exception):
    """Not all the provided predictors are valid.
    """
    pass


class InferenceError(Exception):
    """Some steps of a multiple linear regression have invalid parameters
    """
    pass
