"""
Custom exceptions for the riogrande I/O layer.

Exceptions defined here are raised when operations on :class:`~riogrande.io.models.Source`
and :class:`~riogrande.io.models.Band` objects encounter invalid or ambiguous
states, such as band selection failures, missing source files, or unrecognised
file extensions.
"""


class BandSelectionAmbiguousError(Exception):
    """Multiple bands match the provided tags
    """
    pass


class BandSelectionNoMatchError(Exception):
    """No band found with matching tags
    """
    pass


class SourceNotSavedError(Exception):
    """Source cannot be found at the specified location.
    Either the path is wrong or the related data is not (yet) saved to file.
    """
    pass


class UnknownExtensionError(Exception):
    """Handeling of file with the given extension unclear.
    """
    pass


class InvalidMaskSelectorError(Exception):
    """A selector string was used that did not match to any mask reader.
    """
    pass
