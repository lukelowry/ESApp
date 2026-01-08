"""Custom exception classes for the SAW wrapper."""


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class PowerWorldError(Error):
    """Raised when PowerWorld reports an error following a SimAuto call."""

    pass


class COMError(Error):
    """Raised when attempting to call a SimAuto function results in an
    error.
    """

    pass


class CommandNotRespectedError(Error):
    """Raised if a command sent into PowerWorld is not respected, but
    PowerWorld itself does not raise an error. This exception should
    be used with helpers that double-check commands.
    """

    pass