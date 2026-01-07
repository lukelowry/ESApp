from enum import Enum, Flag, auto

class FieldPriority(Flag):
    PRIMARY   = auto()
    SECONDARY = auto()
    REQUIRED  = auto()
    OPTIONAL  = auto()
    EDITABLE  = auto()


class GObject(Enum):

    # Called when each field of subclass is parsed by python
    def __new__(cls, *args):
        
        # The object type string name 
        if len(args) == 1:

            cls._TYPE = args[0]
            
             # Set intgeer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = value

            return obj
        
        # Everything else is field
        else:
            # Look at raw field name and priority
            field_name_str, field_dtype, field_priority = args 

            # Check if fields class function has been initialized
            if not hasattr(cls, '_FIELDS'):
                cls._FIELDS = []
            if not hasattr(cls, '_KEYS'):
                cls._KEYS = []

            # Add to appropriate Lists
            cls._FIELDS.append(field_name_str)
            if field_priority & FieldPriority.PRIMARY == FieldPriority.PRIMARY:
                cls._KEYS.append(field_name_str)


            # Set intgeer and name as member value
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = (value,  field_name_str, field_dtype, field_priority)

            return obj
    
    def __repr__(self) -> str:
        return str(self._value_)
    
    def __str__(self) -> str:
        return f'Field String: {self._value_[1]}'
    
    @classmethod
    @property
    def keys(cls):
        '''
        Get the properly formatted string names of all fields
        '''
        if not hasattr(cls, '_KEYS'):
            return []
        return cls._KEYS
    
    @classmethod
    @property
    def fields(cls):
        '''
        Get the properly formatted string names of all fields
        '''
        if not hasattr(cls, '_FIELDS'):
            return []
        return cls._FIELDS 
    
    @classmethod
    @property
    def TYPE(cls):
        if not hasattr(cls, '_TYPE'):
            return 'NO_OBJECT_NAME'
        return cls._TYPE