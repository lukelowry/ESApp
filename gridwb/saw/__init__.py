"""
saw is short for SimAuto Wrapper. This package provides a class, SAW, for
interfacing with PowerWorld's Simulator Automation Server (SimAuto).
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
