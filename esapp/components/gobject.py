"""
Defines the base components for creating structured grid object schemas.

This module provides the `GObject` enum, a specialized base class used to define
the data model for various power system components like buses, generators, and
lines. It uses a unique pattern within the `Enum`'s `__new__` method to
dynamically construct a schema, including field names, data types, and keys,
from the class definition itself.

The `FieldPriority` flag is used to categorize these fields, for example, to
distinguish primary keys from other data attributes.
"""

from enum import Enum, Flag, auto

class FieldPriority(Flag):
    """
    A Flag enumeration to define the characteristics of a GObject field.

    These flags can be combined (e.g., `REQUIRED | EDITABLE`) to specify
    multiple attributes for a single field.
    """
    PRIMARY   = auto()  #: Field is part of the primary key for the object.
    SECONDARY = auto()  #: Field is part of a secondary key.
    REQUIRED  = auto()  #: Field is required for data retrieval or updates.
    OPTIONAL  = auto()  #: Field is optional.
    EDITABLE  = auto()  #: Field is user-modifiable.


class GObject(Enum):
    """
    A base class for defining the schema of a power system grid object.

    This class uses a custom `Enum` implementation to parse its own members
    at definition time, creating a structured schema for a grid component.
    Subclasses should define their members to build the schema.

    The class automatically populates `_FIELDS`, `_KEYS`, and `_TYPE` attributes
    based on its member definitions. These are exposed through the `fields()`,
    `keys()`, and `TYPE()` classmethods.

    Example:
    --------
    .. code-block:: python

        class Bus(GObject):
            # The first member defines the PowerWorld object type string.
            _ = 'Bus'

            # Subsequent members define the object's fields.
            # (FieldName, DataType, Priority)
            Number = 'BusNum', int, FieldPriority.PRIMARY
            Name = 'BusName', str, FieldPriority.REQUIRED | FieldPriority.EDITABLE
            PUVolt = 'BusPUVolt', float, FieldPriority.OPTIONAL

    """

    # Called when each field of a subclass is parsed by python
    def __new__(cls, *args):
        """Dynamically construct Enum members to build a class-level schema."""
        # Initialize _FIELDS, _KEYS, _SECONDARY, and _EDITABLE lists if they don't exist on the class itself
        if '_FIELDS' not in cls.__dict__:
            cls._FIELDS = []
        if '_KEYS' not in cls.__dict__:
            cls._KEYS = []
        if '_SECONDARY' not in cls.__dict__:
            cls._SECONDARY = []
        if '_EDITABLE' not in cls.__dict__:
            cls._EDITABLE = []

        # The object type string name is the only argument for this member
        if len(args) == 1:
            cls._TYPE = args[0]

            # Set integer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = value

            return obj

        # Everything else is a field with (name, dtype, priority)
        else:
            field_name_str, field_dtype, field_priority = args

            # Set integer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = (value,  field_name_str, field_dtype, field_priority)

            # Add to appropriate Lists
            cls._FIELDS.append(field_name_str)

            # A field is a key if it's PRIMARY.
            if field_priority & FieldPriority.PRIMARY == FieldPriority.PRIMARY:
                cls._KEYS.append(field_name_str)

            # A field is a secondary identifier if it's SECONDARY.
            if field_priority & FieldPriority.SECONDARY == FieldPriority.SECONDARY:
                cls._SECONDARY.append(field_name_str)

            # A field is editable if it has the EDITABLE flag.
            if field_priority & FieldPriority.EDITABLE == FieldPriority.EDITABLE:
                cls._EDITABLE.append(field_name_str)

            return obj

    def __repr__(self) -> str:
        # For the type-defining member, show the type.
        if isinstance(self._value_, int):
            return f'<{self.__class__.__name__}.{self.name}: TYPE={self.__class__.TYPE()}>'
        # For field members, show the field info.
        return f'<{self.__class__.__name__}.{self.name}: Field={self._value_[1]}>'

    def __str__(self) -> str:
        # For the type-defining member, it has no string field name.
        if isinstance(self._value_, int):
            return self.name
        # For field members, return the PowerWorld field name string.
        return str(self._value_[1])

    @classmethod
    def keys(cls):
        return getattr(cls, '_KEYS', [])

    @classmethod
    def fields(cls):
        return getattr(cls, '_FIELDS', [])

    @classmethod
    def secondary(cls):
        """Secondary identifier fields (used with primary keys to identify records)."""
        return getattr(cls, '_SECONDARY', [])

    @classmethod
    def editable(cls):
        return getattr(cls, '_EDITABLE', [])

    @classmethod
    def identifiers(cls):
        """All identifier fields: primary keys + secondary keys."""
        return set(getattr(cls, '_KEYS', [])) | set(getattr(cls, '_SECONDARY', []))

    @classmethod
    def settable(cls):
        """Fields that can be set: identifiers (primary + secondary keys) + editable fields."""
        return cls.identifiers() | set(getattr(cls, '_EDITABLE', []))

    @classmethod
    def TYPE(cls):
        return getattr(cls, '_TYPE', 'NO_OBJECT_NAME')

    @classmethod
    def is_editable(cls, field_name: str) -> bool:
        """Check if a field is user-modifiable."""
        return field_name in getattr(cls, '_EDITABLE', [])

    @classmethod
    def is_settable(cls, field_name: str) -> bool:
        """Check if a field can be set (either a key or editable)."""
        return field_name in cls.settable()
