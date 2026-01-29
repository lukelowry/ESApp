"""Enum types and constants for SimAuto wrapper.

This module defines standardized types for string literals used throughout
the SAW module, replacing hardcoded strings with type-safe enumerations.
"""

from enum import Enum
from typing import Union


class YesNo(str, Enum):
    """Boolean flag values for PowerWorld commands.

    PowerWorld uses "YES" and "NO" strings for boolean parameters in
    script commands rather than true/false.
    """
    YES = "YES"
    NO = "NO"

    @classmethod
    def from_bool(cls, value: bool) -> "YesNo":
        """Convert a Python boolean to YesNo enum.

        Parameters
        ----------
        value : bool
            The boolean value to convert.

        Returns
        -------
        YesNo
            YesNo.YES if value is True, YesNo.NO otherwise.
        """
        return cls.YES if value else cls.NO

    def __str__(self):
        return self.value


class FilterKeyword(str, Enum):
    """Special filter keywords for PowerWorld commands.

    These keywords are passed unquoted to PowerWorld, unlike custom
    filter names which must be quoted.
    """
    SELECTED = "SELECTED"
    AREAZONE = "AREAZONE"
    ALL = "ALL"


class SolverMethod(str, Enum):
    """Power flow solution methods.

    These are the available solver algorithms for the SolvePowerFlow command.
    """
    RECTNEWT = "RECTNEWT"        # Rectangular Newton-Raphson (default)
    POLARNEWT = "POLARNEWT"      # Polar Newton-Raphson
    GAUSSSEIDEL = "GAUSSSEIDEL"  # Gauss-Seidel
    FASTDEC = "FASTDEC"          # Fast Decoupled
    ROBUST = "ROBUST"            # Robust solver
    DC = "DC"                    # DC power flow


class LinearMethod(str, Enum):
    """Linear calculation methods for sensitivity analysis.

    Used in PTDF, LODF, shift factor, and related calculations.
    """
    DC = "DC"      # DC linear method (most common default)
    AC = "AC"      # AC linear method
    DCPS = "DCPS"  # DC linear with post-solution adjustment


class FileFormat(str, Enum):
    """File format types for import/export operations."""
    CSV = "CSV"                  # Comma-separated values
    CSVCOLHEADER = "CSVCOLHEADER"  # CSV with column headers
    CSVNOHEADER = "CSVNOHEADER"  # CSV without headers
    AUX = "AUX"                  # PowerWorld auxiliary format
    AUXCSV = "AUXCSV"            # Hybrid auxiliary/CSV format
    TAB = "TAB"                  # Tab-separated format
    PTI = "PTI"                  # PTI/PSS-E format
    TXT = "TXT"                  # Text format
    PWB = "PWB"                  # PowerWorld case format
    AXD = "AXD"                  # Oneline diagram format
    GE = "GE"                    # GE EPC format
    CF = "CF"                    # Custom format
    UCTE = "UCTE"                # UCTE format
    AREVAHDB = "AREVAHDB"        # AREVA HDB format
    OPENNETEMS = "OPENNETEMS"    # OPENNET EMS format


class InterfaceLimitSetting(str, Enum):
    """Interface limit configuration values."""
    AUTO = "AUTO"    # Automatic limit calculation
    NONE = "NONE"    # No limit applied


class ObjectType(str, Enum):
    """PowerWorld object type identifiers.

    These are used for filtering and operations on specific element types.
    """
    BUS = "BUS"
    BRANCH = "BRANCH"
    GEN = "GEN"
    LOAD = "LOAD"
    SHUNT = "SHUNT"
    AREA = "AREA"
    ZONE = "ZONE"
    OWNER = "OWNER"
    INTERFACE = "INTERFACE"
    INJECTIONGROUP = "INJECTIONGROUP"
    BUSSHUNT = "BUSSHUNT"
    SUPERBUS = "SUPERBUS"
    TRANSFORMER = "TRANSFORMER"
    LINE = "LINE"
    SUPERAREA = "SUPERAREA"


class KeyFieldType(str, Enum):
    """Key field types for result output."""
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    LABEL = "LABEL"


class StarBusHandling(str, Enum):
    """Star bus handling options for case append operations."""
    NEAR = "NEAR"  # Map to nearest bus (default)
    MAX = "MAX"    # Map to maximum impedance bus


class MultiSectionLineHandling(str, Enum):
    """Multi-section line handling options for case append operations."""
    MAINTAIN = "MAINTAIN"        # Maintain multisection line structure (default)
    EQUIVALENCE = "EQUIVALENCE"  # Convert to equivalent circuits


class IslandReference(str, Enum):
    """Island reference options for sensitivity analysis."""
    EXISTING = "EXISTING"  # Use existing island configuration
    NO = "NO"              # No area reference


class OnelineLinkMode(str, Enum):
    """Oneline diagram linking modes."""
    LABELS = "LABELS"    # Link objects by labels (default)
    NUMBERS = "NUMBERS"  # Link objects by numbers


class ShuntModel(str, Enum):
    """Shunt model types for line tapping operations."""
    CAPACITANCE = "CAPACITANCE"
    INDUCTANCE = "INDUCTANCE"


class BranchDeviceType(str, Enum):
    """Branch device types for bus splitting operations."""
    LINE = "Line"
    BREAKER = "Breaker"


class TSGetResultsMode(str, Enum):
    """Mode for saving transient stability results."""
    SINGLE = "SINGLE"
    SEPARATE = "SEPARATE"
    JSIS = "JSIS"


class JacobianForm(str, Enum):
    """Jacobian matrix coordinate forms."""
    RECTANGULAR = "R"  # AC Jacobian in Rectangular coordinates
    POLAR = "P"        # AC Jacobian in Polar coordinates
    DC = "DC"          # B' matrix (DC approximation)


class BranchDistanceMeasure(str, Enum):
    """Branch distance measurement types for topology analysis."""
    REACTANCE = "X"   # Use reactance as distance measure
    IMPEDANCE = "Z"   # Use impedance magnitude as distance measure


class BranchFilterMode(str, Enum):
    """Branch filter modes for topology traversal."""
    ALL = "ALL"         # All branches
    SELECTED = "SELECTED"  # Only selected branches
    CLOSED = "CLOSED"   # Only closed branches


class ScalingBasis(str, Enum):
    """Scaling basis for load/generation scaling operations."""
    MW = "MW"       # Absolute MW/MVAR values
    FACTOR = "FACTOR"  # Multiplier factor


class ObjectIDHandling(str, Enum):
    """Object ID handling modes for contingency export."""
    NO = "NO"           # Standard object references
    YES_MS_3W = "YES_MS_3W"  # Include multi-section and 3-winding IDs


class RatingSetPrecedence(str, Enum):
    """Rating set precedence for weather-based ratings."""
    NORMAL = "NORMAL"  # Use normal rating set
    CTG = "CTG"        # Use contingency rating set


class RatingSet(str, Enum):
    """Rating set identifiers for branch limits."""
    DEFAULT = "DEFAULT"  # Use default rating
    NO = "NO"           # Don't update rating
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"
    I = "I"
    J = "J"
    K = "K"
    L = "L"
    M = "M"
    N = "N"
    O = "O"


class FieldListColumn(str, Enum):
    """Column names for GetFieldList results.

    PowerWorld returns field metadata with these column headers. Different
    Simulator versions may return different subsets of these columns.
    """
    KEY_FIELD = "key_field"
    INTERNAL_FIELD_NAME = "internal_field_name"
    FIELD_DATA_TYPE = "field_data_type"
    DESCRIPTION = "description"
    DISPLAY_NAME = "display_name"
    ENTERABLE = "enterable"

    @classmethod
    def base_columns(cls) -> list:
        """Returns the standard 5-column format (most common)."""
        return [
            cls.KEY_FIELD.value,
            cls.INTERNAL_FIELD_NAME.value,
            cls.FIELD_DATA_TYPE.value,
            cls.DESCRIPTION.value,
            cls.DISPLAY_NAME.value,
        ]

    @classmethod
    def old_columns(cls) -> list:
        """Returns the legacy 4-column format (older Simulator versions)."""
        return [
            cls.KEY_FIELD.value,
            cls.INTERNAL_FIELD_NAME.value,
            cls.FIELD_DATA_TYPE.value,
            cls.DESCRIPTION.value,
        ]

    @classmethod
    def new_columns(cls) -> list:
        """Returns the extended 6-column format (newer Simulator versions)."""
        return [
            cls.KEY_FIELD.value,
            cls.INTERNAL_FIELD_NAME.value,
            cls.FIELD_DATA_TYPE.value,
            cls.DESCRIPTION.value,
            cls.DISPLAY_NAME.value,
            cls.ENTERABLE.value,
        ]


class SpecificFieldListColumn(str, Enum):
    """Column names for GetSpecificFieldList results.

    PowerWorld returns specific field metadata with these column headers.
    """
    VARIABLENAME_LOCATION = "variablename:location"
    FIELD = "field"
    COLUMN_HEADER = "column header"
    FIELD_DESCRIPTION = "field description"
    ENTERABLE = "enterable"

    @classmethod
    def base_columns(cls) -> list:
        """Returns the standard 4-column format."""
        return [
            cls.VARIABLENAME_LOCATION.value,
            cls.FIELD.value,
            cls.COLUMN_HEADER.value,
            cls.FIELD_DESCRIPTION.value,
        ]

    @classmethod
    def new_columns(cls) -> list:
        """Returns the extended 5-column format (newer Simulator versions)."""
        return [
            cls.VARIABLENAME_LOCATION.value,
            cls.FIELD.value,
            cls.COLUMN_HEADER.value,
            cls.FIELD_DESCRIPTION.value,
            cls.ENTERABLE.value,
        ]


# Type aliases for flexibility - allows either enum or raw string
FilterType = Union[FilterKeyword, str]


def format_filter(filter_name: FilterType) -> str:
    """Format a filter name for use in PowerWorld commands.

    Special filter keywords (SELECTED, AREAZONE, ALL) are passed unquoted,
    while custom filter names are quoted.

    Parameters
    ----------
    filter_name : FilterType
        The filter name to format. Can be a FilterKeyword enum or a string.

    Returns
    -------
    str
        The formatted filter string for use in script commands.
    """
    if not filter_name:
        return ""

    # Handle enum values
    if isinstance(filter_name, FilterKeyword):
        return filter_name.value

    # Handle string values - check if it's a special keyword
    if filter_name in (FilterKeyword.SELECTED.value, FilterKeyword.AREAZONE.value, FilterKeyword.ALL.value):
        return filter_name

    # Custom filter name - needs quotes
    return f'"{filter_name}"'


def format_filter_selected_only(filter_name: FilterType) -> str:
    """Format a filter name, treating only SELECTED as special.

    Only SELECTED is passed unquoted; other values including AREAZONE and ALL
    are quoted like custom filter names.

    Parameters
    ----------
    filter_name : FilterType
        The filter name to format.

    Returns
    -------
    str
        The formatted filter string.
    """
    if not filter_name:
        return ""

    if isinstance(filter_name, FilterKeyword) and filter_name == FilterKeyword.SELECTED:
        return filter_name.value

    if filter_name == FilterKeyword.SELECTED.value:
        return filter_name

    return f'"{filter_name}"'


def format_filter_areazone(filter_name: FilterType) -> str:
    """Format a filter name, treating SELECTED and AREAZONE as special.

    SELECTED and AREAZONE are passed unquoted; ALL and custom names are quoted.

    Parameters
    ----------
    filter_name : FilterType
        The filter name to format.

    Returns
    -------
    str
        The formatted filter string.
    """
    if not filter_name:
        return ""

    if isinstance(filter_name, FilterKeyword):
        if filter_name in (FilterKeyword.SELECTED, FilterKeyword.AREAZONE):
            return filter_name.value
        return f'"{filter_name.value}"'

    if filter_name in (FilterKeyword.SELECTED.value, FilterKeyword.AREAZONE.value):
        return filter_name

    return f'"{filter_name}"'
