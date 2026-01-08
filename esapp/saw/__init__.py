"""
SimAuto Wrapper (:mod:`esapp.saw`)
==================================

This module provides the low-level interface for communicating with the
PowerWorld Simulator Automation Server (SimAuto). The primary entry point
is the :class:`~.SAW` class. It also defines custom exception classes
for handling COM and PowerWorld-specific errors.
"""
from .saw import SAW
from ._exceptions import PowerWorldError, COMError, CommandNotRespectedError, Error
from ._helpers import (
    df_to_aux,
    convert_to_windows_path,
    convert_list_to_variant,
    convert_df_to_variant,
    convert_nested_list_to_variant,
)


# To make them available from the saw module directly
__all__ = [
    "SAW",
    "PowerWorldError",
    "COMError",
    "CommandNotRespectedError",
    "Error",
    "df_to_aux",
    "convert_to_windows_path",
    "convert_list_to_variant",
    "convert_df_to_variant",
    "convert_nested_list_to_variant",
]
