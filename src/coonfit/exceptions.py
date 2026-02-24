"""
Custom exceptions for the coonfit inference workflow.

Exceptions defined here are raised when the multiple linear regression
pipeline encounters invalid predictor data or configuration errors, such as
predictors with insufficient valid pixels or ill-conditioned regression setups.
"""


class InvalidPredictorError(Exception):
    """Not all the provided predictors are valid.
    """
    pass


class InferenceError(Exception):
    """Some steps of a multiple linear regression have invalid parameters
    """
    pass
