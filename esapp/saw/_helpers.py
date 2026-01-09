"""Helper functions for data conversion for SimAuto COM interface."""

import json
from pathlib import PureWindowsPath
from typing import List

import pythoncom
import win32com
from win32com.client import VARIANT


def df_to_aux(fp, df, object_name: str):
    """Convert a dataframe to PW aux/axd data section.
    
    Parameters
    ----------
    fp : file
        File handler.
    df : pandas.DataFrame
        DataFrame to convert.
    object_name : str
        PowerWorld object type.
    """
    # write the header
    fields = ",".join(df.columns.tolist())
    header = f"DATA ({object_name}, [{fields}])"
    header_chunks = header.split(",")
    i = 0
    line_width = 0
    max_width = 86
    working_line = []
    container = []
    while True:
        if line_width + len(header_chunks[i]) <= max_width:
            working_line.append(header_chunks[i])
            line_width += len(header_chunks[i])
            i += 1
        else:
            container.append(",".join(working_line))
            working_line = []
            line_width = 0
        if i == len(header_chunks):
            if len(working_line):
                container.append(",".join(working_line))
            break
    container = [ls + "," for ls in container[:-1]] + [container[-1]]
    container = [container[0]] + ["    " + ls for ls in container[1:]]  # add tab to each line

    # write the remaining part
    container.append("{")
    container.extend(json.dumps(row, separators=(" ", ": "))[1:-1] for row in df.values.tolist())
    container.append("}\r\n")
    fp.write("\n".join(container))


def convert_to_windows_path(p):
    """Given a path, p, convert it to a Windows path."""
    return str(PureWindowsPath(p))


def convert_list_to_variant(list_in: list) -> VARIANT:
    """Given a list, convert to a variant array."""
    # noinspection PyUnresolvedReferences
    return VARIANT(pythoncom.VT_VARIANT | pythoncom.VT_ARRAY, list_in)


def convert_df_to_variant(df):
    """Given a DataFrame, convert to a variant array for Rect functions."""
    data_as_list = df.values.tolist()
    return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_VARIANT, data_as_list)


def convert_nested_list_to_variant(list_in: list) -> List[VARIANT]:
    """Given a list of lists, convert to a variant array."""
    return [convert_list_to_variant(sub_array) for sub_array in list_in]


def create_object_string(object_type: str, *keys) -> str:
    """
    Helper to format a PowerWorld object string identifier.

    This function creates strings formatted like '[BUS 1]' or '[BRANCH 1 2 "1"]'
    which are used to identify objects in SimAuto script commands.

    Parameters
    ----------
    object_type : str
        The type of object (e.g. "Bus", "Gen", "Branch").
    *keys : Any
        The key values identifying the object. Strings will be automatically
        enclosed in double quotes if they are not already quoted.

    Returns
    -------
    str
        Formatted string like '[ObjectType key1 key2 ...]'.
    """
    parts = [object_type.upper()]
    for key in keys:
        if isinstance(key, str):
            # Check if already quoted with " or '
            if (len(key) >= 2) and ((key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'"))):
                parts.append(key)
            else:
                parts.append(f'"{key}"')
        else:
            parts.append(str(key))
    
    return f"[{' '.join(parts)}]"