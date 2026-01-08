"""
Parses the PowerWorld 'Case Objects Fields' Text File and generates a Python
module (components.py) containing the structured data.

This script is designed to replace the old method of using an Excel file,
providing a more robust and automated way to process the raw text data.
"""

import os
from collections import OrderedDict

# Constants and helpers adapted from the legacy components_dev.py script
# to ensure the generated output format is consistent.

# Problematic Objects to exclude from generation
excludeObjects = [
    'AlarmOptions', 'GenMWMaxMin_GenMWMaxMinXYCurve',
    'GenMWMax_SolarPVBasic1',
    'GenMWMax_SolarPVBasic2', 'GenMWMax_TemperatureBasic1', 'GenMWMax_WindBasic',
    'GenMWMax_WindClass1', 'GenMWMax_WindClass2', 'GenMWMax_WindClass3',
    'GenMWMax_WindClass4', 'GICGeographicRegionSet', 'GIC_Options',
    'LPOPFMarginalControls', 'MvarMarginalCostValues', 'MWMarginalCostValues',
    'NEMGroupBranch', 'NEMGroupGroup', 'NEMGroupNode', 'PieSizeColorOptions',
    'PWBranchDataObject', 'RT_Study_Options', 'SchedSubscription',
    'TSFreqSummaryObject', 'TSModalAnalysisObject', 'TSSchedule',
    'Exciter_Generic', 'Governor_Generic',
    'InjectionGroupModel_GenericInjectionGroup', 'LoadCharacteristic_Generic',
    'WeatherPathPoint', 'TSTimePointSolutionDetails'
]

# Problematic Fields to exclude from generation
excludeFields = [
    'BusMarginalControl', 'BusMCMVARValue', 'BusMCMWValue', 'LoadGrounded',
    'GEDateIn', 'GEDateOut'
]

# Data Type Mapping from PowerWorld types to Python types
dtypemap = {"String": "str", "Real": "float", "Integer": "int"}


def fix_pw_string(name: str) -> str:
    """
    Converts a Python-safe attribute name back to the PowerWorld string format.
    Example: 'ThreeWindingTransformer' -> '3WindingTransformer'
             'Bus__Num' -> 'Bus:Num'

    Args:
        name (str): The Python-safe attribute name.

    Returns:
        str: The PowerWorld-compatible field name string.
    """
    new_name = "3" + name[5:] if name.startswith("Three") else name # Handle 'Three' prefix
    new_name = new_name.replace('__', ':') # Convert double underscore back to colon
    new_name = new_name.replace('___', ' ')
    return new_name


def sanitize_for_python(name: str) -> str:
    """
    Converts a PowerWorld field name to a Python-safe attribute name.
    Example: '3WindingTransformer' -> 'ThreeWindingTransformer',
             'Bus:Num' -> 'Bus__Num',
             'Use Pattern' -> 'Use_Pattern'

    Args:
        name (str): The PowerWorld field name.

    Returns:
        str: A Python-safe attribute name.
    """
    new_name = name.replace(":", "__") # Convert colons to double underscores
    new_name = new_name.replace(" ", "___")
    if new_name and new_name[0] == '3': # Handle '3' prefix
        new_name = 'Three' + new_name[1:] 
    return new_name


def pw_to_dict(filepath: str) -> OrderedDict:
    """
    Parses the 'Case Objects Fields'tab-delimited text file into a structured dictionary.

    The file format is semi-structured:
    - Object types are on non-indented lines.
    - Fields for an object type are on subsequent indented lines.
    - Tabs are used for alignment, so splitting by tab is the primary method.

    Args:
        filepath (str): The path to the 'Case Objects Fields' text file.

    Returns:
        OrderedDict: A dictionary where keys are object types and values
                     contain properties and a list of fields for that object.
    """
    # These headers correspond to the columns in the PW Raw file.
    HEADERS = [
        "Object Type", "SUBDATA Allowed", "Key/Required Fields", "Variable Name",
        "Concise Variable Name", "Type of Variable", "Description",
        "Available Field List", "Enterable", "Data Maintainer Support",
        "Data Maintainer Inheritance"
    ]

    data = OrderedDict()
    current_object_type_name = None

    with open(filepath, 'r', encoding='utf-8') as f:
        # The first line of the file is the header, which we can skip.
        next(f, None)

        for line_num, line in enumerate(f, 2):
            line = line.rstrip('\n')
            if not line.strip():
                continue

            # A line not starting with a tab is a new Object Type.
            if not line.startswith('\t'):
                parts = line.split('\t')

                current_object_type_name = parts[0].strip()

                # Check if the object type name is invalid (empty, single character, or in excludeObjects).
                # If invalid, reset current_object_type_name to None and skip processing this object.
                # This prevents subsequent indented lines from being associated with an invalid or excluded object.
                if not current_object_type_name or \
                   len(current_object_type_name) <= 1 or \
                   current_object_type_name in excludeObjects:
                    current_object_type_name = None # Clear the current object type
                    continue
                # If the object type is valid, proceed to process its properties and fields.

                properties = OrderedDict()
                for i, part in enumerate(parts):
                    if i < len(HEADERS) and part.strip():
                        properties[HEADERS[i]] = part.strip()

                data[current_object_type_name] = {
                    'properties': properties,
                    'fields': []
                }

            # A line starting with a tab is a field of the current object.
            else:
                if not current_object_type_name:
                    continue

                parts = line.split('\t')
                field_data = OrderedDict()

                # Map parts to headers based on their position (index).
                # The initial empty parts from tabs correctly offset the data.
                for i, part in enumerate(parts):
                    if i < len(HEADERS):
                        field_data[HEADERS[i]] = part.strip()

                # Determine the raw variable name to use.
                vname_raw = field_data.get("Variable Name")
                # Legacy parser drops rows without a Variable Name.
                if not vname_raw:
                    continue

                # Apply field-level exclusions
                if not vname_raw or vname_raw in excludeFields or '/' in vname_raw:
                    continue

                # Sanitize for Python variable name
                vname_py = sanitize_for_python(vname_raw)

                # Store the determined Python-safe name for later use in generate_components
                field_data["_PythonVariableName"] = vname_py
                data[current_object_type_name]['fields'].append(field_data)

    return data


def generate_components(data: OrderedDict, output_path: str) -> None:
    """
    Generates a Python module with classes for each PowerWorld object type.

    This function replicates the output format of the legacy `components_dev.py`
    script, creating a structured, class-based representation of PowerWorld
    components and their fields.

    Args:
        data (OrderedDict): The structured data from the PW Raw file.
        output_path (str): The path for the output Python script.
    """
    # Preamble containing the base classes for the component definitions.
    preamble = """#
# -*- coding: utf-8 -*-
# This file is auto-generated by generate_components.py.
# Do not edit this file manually, as your changes will be overwritten.

from .gobject import *
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(preamble)

        for obj_name, obj_data in data.items():
            if obj_name in excludeObjects:
                continue

            cls_name = sanitize_for_python(obj_name.split(" ")[0])
            f.write(f'\n\nclass {cls_name}(GObject):')

            # Pre-calculate priorities and sort fields to match legacy behavior
            for field in obj_data['fields']:
                p_str = field.get("Key/Required Fields", "")
                p = 10  # Default to Optional
                if '1' in p_str: p = 1
                elif '2' in p_str: p = 2
                elif '3' in p_str: p = 3
                elif 'A' in p_str: p = 4
                elif 'B' in p_str: p = 5
                elif 'C' in p_str: p = 6
                elif '**' in p_str: p = 7
                field['_PriorityValue'] = p

            obj_data['fields'].sort(key=lambda x: x['_PriorityValue'])

            for field in obj_data['fields']:
                vname_py = field.get("_PythonVariableName")
                p = field['_PriorityValue']
                dtype_str = field.get("Type of Variable", "")
                desc = field.get("Description", "")

                dtype = dtypemap.get(dtype_str, "str")

                f.write(f'\n\t{vname_py} = ("{fix_pw_string(vname_py)}", {dtype}, FieldPriority.')

                if p <= 3:
                    f.write('PRIMARY')
                elif 3 < p <= 7:
                    f.write('SECONDARY')
                else:
                    f.write('OPTIONAL')
                
                if p == 7:
                    f.write(' | FieldPriority.REQUIRED')
                
                f.write(")\n\t")

                f.write(r'"""')
                f.write(desc.replace("\\", "/"))
                f.write(r'"""')

            # This special member is defined last and is used by GObject.__new__
            # to store the PowerWorld object type string.
            f.write(f"\n\n\tObjectString = '{obj_name}'\n")


if __name__ == "__main__":

    RAW_IN = 'PWRaw'
    OUT_PY = 'components.py'

    # Get the directory where the script is located to build robust relative paths.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_FILE_PATH = os.path.join(script_dir, RAW_IN)
    OUTPUT_PY_PATH = os.path.join(script_dir, OUT_PY)

    parsed_data = pw_to_dict(RAW_FILE_PATH)
    print(f"\nParsing complete.\n")

    generate_components(parsed_data, OUTPUT_PY_PATH)
    print(f"Successfully Produced  -> components.py!\n")