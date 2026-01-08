"""Custom exception classes for the SAW wrapper."""


class Error(Exception):
    """
    Base class for exceptions in this module.
    """

    pass


class PowerWorldError(Error):
    """
    Raised when PowerWorld reports an error following a SimAuto call.
    This class can parse the error message to provide more context.
    """

    def __init__(self, message: str):
        self.raw_message = message
        self.source = None
        self.message = message

        parts = message.split(":", 1)
        if len(parts) == 2:
            self.source = parts[0].strip()
            self.message = parts[1].strip()

        super().__init__(message)

    @staticmethod
    def from_message(message: str):
        """Factory method to create a specific PowerWorldError subclass."""
        lower_msg = message.lower()
        if "cannot be retrieved through simauto" in lower_msg:
            return SimAutoFeatureError(message)
        
        # Common prerequisite errors (missing data, setup, or invalid state)
        if (
            "no active" in lower_msg 
            or "not found" in lower_msg 
            or "could not be found" in lower_msg
            or "requires setup" in lower_msg
            or "is not online" in lower_msg
            or "at least one" in lower_msg
            or "no directions set" in lower_msg
            or "out-of-range" in lower_msg
        ):
            return PowerWorldPrerequisiteError(message)
            
        if "not registered" in lower_msg:
            return PowerWorldAddonError(message)
        # Add more specific error checks here as they are identified.
        return PowerWorldError(message)


class SimAutoFeatureError(PowerWorldError):
    """
    Raised when a specific SimAuto feature is not supported for the given
    object or in the current context (e.g., trying to read an object type
    that SimAuto doesn't allow reading).
    """
    pass


class PowerWorldPrerequisiteError(PowerWorldError):
    """
    Raised when a command fails because some prerequisite condition or
    data is not met in the case (e.g., no active contingencies for a
    contingency-related command).
    """
    pass


class PowerWorldAddonError(PowerWorldError):
    """
    Raised when a command fails because a required PowerWorld add-on
    (like TransLineCalc) is not registered or licensed.
    """
    pass


class COMError(Error):
    """
    Raised when attempting to call a SimAuto function results in an
    error.
    """

    pass


class CommandNotRespectedError(Error):
    """
    Raised if a command sent into PowerWorld is not respected, but
    PowerWorld itself does not raise an error. This exception should
    be used with helpers that double-check commands.
    """

    pass