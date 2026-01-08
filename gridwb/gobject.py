
from enum import Enum, Flag, auto

class FieldPriority(Flag):
    PRIMARY   = auto()
    SECONDARY = auto()
    REQUIRED  = auto()
    OPTIONAL  = auto()
    EDITABLE  = auto()


class GObject(Enum):

    # Called when each field of a subclass is parsed by python
    def __new__(cls, *args):
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
        return str(self._value_)
    
    def __str__(self) -> str:
        return f'Field String: {self._value_[1]}'
    
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