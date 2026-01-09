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
    based on its member definitions. These are exposed through the `fields`,
    `keys`, and `TYPE` class properties.

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
        # Initialize _FIELDS and _KEYS lists if they don't exist on the class itself
        if '_FIELDS' not in cls.__dict__:
            cls._FIELDS = []
        if '_KEYS' not in cls.__dict__:
            cls._KEYS = []
        
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

            return obj
    
    def __repr__(self) -> str:
        # For the type-defining member, show the type.
        if isinstance(self._value_, int):
            return f'<{self.__class__.__name__}.{self.name}: TYPE={self.__class__.TYPE}>'
        # For field members, show the field info.
        return f'<{self.__class__.__name__}.{self.name}: Field={self._value_[1]}>'
    
    def __str__(self) -> str:
        # For the type-defining member, it has no string field name.
        if isinstance(self._value_, int):
            return self.name
        # For field members, return the PowerWorld field name string.
        return str(self._value_[1])
    
    @classmethod
    @property
    def keys(cls):
        return getattr(cls, '_KEYS', [])
    
    @classmethod
    @property
    def fields(cls):
        return getattr(cls, '_FIELDS', [])
    
    @classmethod
    @property
    def TYPE(cls):
        return getattr(cls, '_TYPE', 'NO_OBJECT_NAME')