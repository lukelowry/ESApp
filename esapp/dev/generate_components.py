"""
Parses the PowerWorld 'Case Objects Fields' Text File and generates a Python
module (components.py) containing the structured data.
"""
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Optional


class FieldRole(Flag):
    """Maps to PWRaw Key/Required column symbols."""
    STANDARD = 0
    PRIMARY_KEY = auto()        # *
    ALTERNATE_KEY = auto()      # *A*
    COMPOSITE_KEY_1 = auto()    # *1*
    COMPOSITE_KEY_2 = auto()    # *2*
    COMPOSITE_KEY_3 = auto()    # *3*
    SECONDARY_ID = auto()       # *2B*
    CIRCUIT_ID = auto()         # *4B*
    BASE_VALUE = auto()         # **
    STANDARD_FIELD = auto()     # <


@dataclass
class FieldDefinition:
    """Represents a single field/variable within a PowerWorld object type."""
    variable_name: str
    python_name: str
    concise_name: str
    data_type: str
    description: str
    role: FieldRole
    enterable: bool
    available_list: str = ""

    @property
    def is_primary(self) -> bool:
        return bool(self.role & (
            FieldRole.PRIMARY_KEY | FieldRole.COMPOSITE_KEY_1 |
            FieldRole.COMPOSITE_KEY_2 | FieldRole.COMPOSITE_KEY_3
        ))

    @property
    def is_secondary(self) -> bool:
        return bool(self.role & (
            FieldRole.ALTERNATE_KEY | FieldRole.SECONDARY_ID |
            FieldRole.CIRCUIT_ID | FieldRole.BASE_VALUE
        ))

    @property
    def is_base_value(self) -> bool:
        return bool(self.role & FieldRole.BASE_VALUE)


@dataclass
class ObjectTypeDefinition:
    """Represents a PowerWorld object type (e.g., Gen, Bus, Load)."""
    name: str
    subdata_allowed: bool
    fields: list = field(default_factory=list)


excludeObjects = [
    'AlarmOptions', 'GenMWMaxMin_GenMWMaxMinXYCurve',
    'GenMWMax_SolarPVBasic1', 'GenMWMax_SolarPVBasic2',
    'GenMWMax_TemperatureBasic1', 'GenMWMax_WindBasic',
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

excludeFields = [
    'BusMarginalControl', 'BusMCMVARValue', 'BusMCMWValue', 'LoadGrounded',
    'GEDateIn', 'GEDateOut'
]

dtypemap = {"String": "str", "Real": "float", "Integer": "int"}


def fix_pw_string(name: str) -> str:
    """Converts a Python-safe attribute name back to the PowerWorld string format."""
    new_name = "3" + name[5:] if name.startswith("Three") else name
    new_name = new_name.replace('__', ':')
    new_name = new_name.replace('___', ' ')
    return new_name


def sanitize_for_python(name: str) -> str:
    """Converts a PowerWorld field name to a Python-safe attribute name."""
    new_name = name.replace(":", "__")
    new_name = new_name.replace(" ", "___")
    if new_name and new_name[0] == '3':
        new_name = 'Three' + new_name[1:]
    return new_name


def strip_quotes(value: str) -> str:
    """Strips surrounding single quotes from a value."""
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def sanitize_description(desc: str) -> str:
    """
    Sanitizes a description string for use in a Python triple-quoted docstring.
    
    Handles:
    - Backslashes (replaced with forward slashes)
    - Embedded double quotes (escaped to prevent docstring termination)
    - Triple quotes (escaped)
    """
    desc = desc.replace("\\", "/")
    desc = desc.replace('"""', r'\"\"\"')
    desc = desc.replace('"', r'\"')
    return desc


def parse_key_symbol(symbol: str) -> FieldRole:
    """
    Parses Key/Required column symbols into FieldRole.
    
    Symbols can be combined (e.g., '*1*<' means COMPOSITE_KEY_1 + STANDARD_FIELD).
    Order matters: check specific patterns before generic ones.
    """
    symbol = symbol.strip()
    role = FieldRole.STANDARD
    
    if '*1*' in symbol:
        role |= FieldRole.COMPOSITE_KEY_1
    elif '*2B*' in symbol:
        role |= FieldRole.SECONDARY_ID
    elif '*4B*' in symbol:
        role |= FieldRole.CIRCUIT_ID
    elif '*2*' in symbol:
        role |= FieldRole.COMPOSITE_KEY_2
    elif '*3*' in symbol:
        role |= FieldRole.COMPOSITE_KEY_3
    elif '*A*' in symbol:
        role |= FieldRole.ALTERNATE_KEY
    elif '**' in symbol:
        role |= FieldRole.BASE_VALUE
    elif '*' in symbol and not any(x in symbol for x in ['*1*', '*2*', '*3*', '*A*', '**', '*2B*', '*4B*']):
        role |= FieldRole.PRIMARY_KEY
    
    if '<' in symbol:
        role |= FieldRole.STANDARD_FIELD
    
    return role


def parse_enterable(value: str) -> bool:
    """
    Parses the Enterable column value.
    Returns True if the field is user-editable.
    """
    value = strip_quotes(value.strip().lower())
    return value in ('yes', 'edit mode only')


def get_sort_key(field_def: FieldDefinition) -> int:
    """Returns sort priority based on FieldRole."""
    role = field_def.role
    if role & FieldRole.COMPOSITE_KEY_1 or role & FieldRole.PRIMARY_KEY:
        return 0
    elif role & FieldRole.COMPOSITE_KEY_2:
        return 1
    elif role & FieldRole.COMPOSITE_KEY_3:
        return 2
    elif role & FieldRole.ALTERNATE_KEY:
        return 3
    elif role & FieldRole.SECONDARY_ID or role & FieldRole.CIRCUIT_ID:
        return 4
    elif role & FieldRole.BASE_VALUE:
        return 5
    return 10


def get_column(parts: list, index: int, strip_q: bool = False) -> str:
    """Safely extracts a column value from parts list."""
    value = parts[index].strip() if index < len(parts) else ""
    return strip_quotes(value) if strip_q else value


def pw_to_dict(filepath: str) -> OrderedDict:
    """Parses the PWRaw TSV file into structured ObjectTypeDefinition instances."""
    data = OrderedDict()
    current_obj: Optional[ObjectTypeDefinition] = None

    with open(filepath, 'r', encoding='utf-8') as f:
        next(f, None)

        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            parts = line.split('\t')

            if not line.startswith('\t'):
                obj_name = parts[0].strip()

                if not obj_name or len(obj_name) <= 1 or obj_name in excludeObjects:
                    current_obj = None
                    continue

                subdata = get_column(parts, 1).lower() == 'yes'
                current_obj = ObjectTypeDefinition(name=obj_name, subdata_allowed=subdata)
                data[obj_name] = current_obj

            elif current_obj is not None:
                var_name = get_column(parts, 3)
                if not var_name or var_name in excludeFields or '/' in var_name:
                    continue

                key_str = get_column(parts, 2)
                enterable = parse_enterable(get_column(parts, 8))

                field_def = FieldDefinition(
                    variable_name=var_name,
                    python_name=sanitize_for_python(var_name),
                    concise_name=get_column(parts, 4),
                    data_type=get_column(parts, 5),
                    description=get_column(parts, 6, strip_q=True),
                    role=parse_key_symbol(key_str),
                    enterable=enterable,
                    available_list=get_column(parts, 7, strip_q=True)
                )
                current_obj.fields.append(field_def)

    return data


def _build_field_priority_flags(field_def: FieldDefinition) -> str:
    """Builds the FieldPriority flag string for a field definition."""
    flags = []

    if field_def.is_primary:
        flags.append('FieldPriority.PRIMARY')
    elif field_def.is_secondary:
        flags.append('FieldPriority.SECONDARY')
    else:
        flags.append('FieldPriority.OPTIONAL')

    if field_def.is_base_value:
        flags.append('FieldPriority.REQUIRED')

    if field_def.enterable:
        flags.append('FieldPriority.EDITABLE')

    return ' | '.join(flags)


def generate_components(data: OrderedDict, output_path: str) -> None:
    """Generates components.py with classes for each PowerWorld object type."""
    preamble = """#
# -*- coding: utf-8 -*-
# This file is auto-generated by generate_components.py.
# Do not edit this file manually, as your changes will be overwritten.

from .gobject import *
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(preamble)

        for obj_name, obj_def in data.items():
            cls_name = sanitize_for_python(obj_name.split(" ")[0])
            f.write(f'\n\nclass {cls_name}(GObject):')

            obj_def.fields.sort(key=get_sort_key)

            for field_def in obj_def.fields:
                dtype = dtypemap.get(field_def.data_type, "str")
                pw_name = fix_pw_string(field_def.python_name)
                flags = _build_field_priority_flags(field_def)
                safe_desc = sanitize_description(field_def.description)

                f.write(f'\n\t{field_def.python_name} = ("{pw_name}", {dtype}, {flags})')
                f.write(f'\n\t"""{safe_desc}"""')

            f.write(f"\n\n\tObjectString = '{obj_name}'\n")


# =============================================================================
# TS Field Generation
# =============================================================================

# Maps PowerWorld object type prefixes to the GObject class name
TS_OBJECT_MAPPING = {
    'TSBus': 'Bus',
    'TSGen': 'Gen',
    'TSACLine': 'Branch',
    'TSLoad': 'Load',
    'TSShunt': 'Shunt',
    'TSArea': 'Area',
    'TSSub': 'Substation',
    'TSSystem': 'System',
    'TSInjectionGroup': 'InjectionGroup',
}


@dataclass
class TSFieldDefinition:
    """Represents a Transient Stability result field."""
    pw_field_name: str      # Full PowerWorld field name (e.g., "TSBusVPU")
    concise_name: str       # Short display name (e.g., "TSVpu")
    description: str        # Human-readable description
    python_attr: str        # Python-safe attribute name (e.g., "VPU")
    object_type: str        # Which GObject it belongs to (e.g., "Bus")


def extract_ts_fields(filepath: str) -> dict[str, list[TSFieldDefinition]]:
    """
    Parses PWRaw and extracts all TS result fields grouped by object type.

    Returns:
        dict mapping object type -> list of TSFieldDefinition
    """
    ts_fields: dict[str, list[TSFieldDefinition]] = {}

    # Track which fields we've seen to avoid duplicates
    seen_fields: set[str] = set()

    with open(filepath, 'r', encoding='utf-8') as f:
        next(f, None)  # Skip header

        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue

            parts = line.split('\t')

            # Only process field lines (start with tab)
            if not line.startswith('\t'):
                continue

            var_name = get_column(parts, 3)
            if not var_name:
                continue

            # Check if this is a TS result field (starts with TS prefix)
            matched_type = None
            for prefix, obj_type in TS_OBJECT_MAPPING.items():
                if var_name.startswith(prefix):
                    matched_type = obj_type
                    break

            if not matched_type:
                continue

            # Skip save/configuration fields, only want result fields
            if 'TSSave' in var_name or 'TSResult' in var_name:
                continue

            # Skip duplicate fields
            if var_name in seen_fields:
                continue
            seen_fields.add(var_name)

            concise_name = get_column(parts, 4)
            description = get_column(parts, 6, strip_q=True)

            # Extract the suffix after the prefix to create Python attribute name
            # e.g., TSBusVPU -> VPU, TSGenDelta -> Delta
            prefix_len = len([p for p in TS_OBJECT_MAPPING.keys() if var_name.startswith(p)][0])
            python_attr = var_name[prefix_len:]

            # Sanitize for Python (handle colons, etc.)
            python_attr = sanitize_for_python(python_attr)

            # Skip empty attribute names
            if not python_attr:
                continue

            field_def = TSFieldDefinition(
                pw_field_name=var_name,
                concise_name=concise_name,
                description=description,
                python_attr=python_attr,
                object_type=matched_type
            )

            if matched_type not in ts_fields:
                ts_fields[matched_type] = []
            ts_fields[matched_type].append(field_def)

    return ts_fields


def generate_ts_fields(ts_fields: dict[str, list[TSFieldDefinition]], output_path: str) -> None:
    """
    Generates ts_fields.py with comprehensive TS field classes for intellisense.
    """
    preamble = '''#
# -*- coding: utf-8 -*-
# This file is auto-generated by generate_components.py.
# Do not edit this file manually, as your changes will be overwritten.
#
# Transient Stability Field Constants for IDE Intellisense
#
# Usage:
#   from esapp.ts_fields import TS
#   wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.Speed])
#

from dataclasses import dataclass


@dataclass(frozen=True)
class TSField:
    """
    Represents a Transient Stability result field.

    Attributes:
        name: The PowerWorld field name string
        description: Human-readable description of the field
    """
    name: str
    description: str = ""

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"TSField({self.name!r})"


class TS:
    """
    Transient Stability Field Constants for Intellisense.

    Provides IDE autocomplete for all available TS result fields organized
    by object type (Bus, Gen, Branch, Load, Shunt, Area, etc.).

    Example:
        >>> from esapp.ts_fields import TS
        >>> from esapp.grid import Gen
        >>> wb.dyn.watch(Gen, [TS.Gen.P, TS.Gen.Speed, TS.Gen.Delta])
    """
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(preamble)

        # Sort object types for consistent output
        for obj_type in sorted(ts_fields.keys()):
            fields = ts_fields[obj_type]

            f.write(f'\n    class {obj_type}:\n')
            f.write(f'        """TS result fields for {obj_type} objects."""\n')

            # Sort fields by Python attribute name
            for field_def in sorted(fields, key=lambda x: x.python_attr):
                safe_desc = sanitize_description(field_def.description)
                f.write(f'        {field_def.python_attr} = TSField("{field_def.pw_field_name}", "{safe_desc}")\n')

        f.write('\n')

    print(f"Generated {output_path} with {sum(len(v) for v in ts_fields.values())} TS fields")


if __name__ == "__main__":
    RAW_IN = 'PWRaw'
    OUT_PY = 'components.py'
    TS_OUT_PY = 'ts_fields.py'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_FILE_PATH = os.path.join(script_dir, RAW_IN)
    OUTPUT_PY_PATH = os.path.join(script_dir, OUT_PY)
    TS_OUTPUT_PATH = os.path.join(script_dir, TS_OUT_PY)

    parsed_data = pw_to_dict(RAW_FILE_PATH)
    print(f"\nParsing complete.\n")

    generate_components(parsed_data, OUTPUT_PY_PATH)
    print(f"Successfully Produced  -> components.py!\n")

    # Generate TS fields
    ts_fields = extract_ts_fields(RAW_FILE_PATH)
    generate_ts_fields(ts_fields, TS_OUTPUT_PATH)
    print(f"Successfully Produced  -> ts_fields.py!\n")