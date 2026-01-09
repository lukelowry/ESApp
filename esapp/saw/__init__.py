"""
SimAuto Wrapper (:mod:`esapp.saw`)
==================================
This package provides a low-level, object-oriented interface for communicating
with the PowerWorld Simulator Automation Server (SimAuto).

The primary entry point is the :class:`~.SAW` class, which is a composite
class built from numerous mixins. Each mixin corresponds to a specific
functional area of the PowerWorld API, such as power flow, contingency
analysis, or transient stability. This modular design keeps the codebase
organized and makes the API easier to navigate.

The package also defines custom exception classes for handling COM and
PowerWorld-specific errors, along with helper functions for data conversion.
"""
from .saw import SAW
from ._exceptions import (
    PowerWorldError,
    COMError,
    CommandNotRespectedError,
    Error,
    SimAutoFeatureError,
    PowerWorldPrerequisiteError,
    PowerWorldAddonError,
)
from ._helpers import (
    df_to_aux,
    convert_to_windows_path,
    convert_list_to_variant,
    convert_df_to_variant,
    convert_nested_list_to_variant,
    create_object_string,
)


# To make them available from the saw module directly
__all__ = [
    "SAW",
    "PowerWorldError",
    "COMError",
    "CommandNotRespectedError",
    "Error",
    "SimAutoFeatureError",
    "PowerWorldPrerequisiteError",
    "PowerWorldAddonError",
    "df_to_aux",
    "convert_to_windows_path",
    "convert_list_to_variant",
    "convert_df_to_variant",
    "convert_nested_list_to_variant",
    "create_object_string",
]
