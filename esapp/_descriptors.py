"""
Descriptors for PowerWorld settings.

Provides lightweight, Pythonic attribute access to PowerWorld option flags
without repetitive boilerplate setter/getter methods.
"""

from .components import Sim_Solution_Options, GIC_Options_Value
from .saw._enums import YesNo


class SolverOption:
    """Descriptor mapping a Python attribute to a Sim_Solution_Options field."""

    def __init__(self, key: str, is_bool: bool = True):
        self.key = key
        self.is_bool = is_bool

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj[Sim_Solution_Options, self.key][self.key].iloc[0]
        if self.is_bool:
            return val == YesNo.YES
        return val

    def __set__(self, obj, value):
        if self.is_bool:
            obj[Sim_Solution_Options, self.key] = YesNo.from_bool(value)
        else:
            obj[Sim_Solution_Options, self.key] = value


class GICOption:
    """Descriptor mapping a Python attribute to a GIC_Options_Value entry."""

    def __init__(self, key: str, is_bool: bool = True):
        self.key = key
        self.is_bool = is_bool

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        df = obj._pw[GIC_Options_Value, "ValueField"]
        row = df[df['VariableName'] == self.key]
        if row.empty:
            return None
        val = row['ValueField'].iloc[0]
        if self.is_bool:
            return val == YesNo.YES
        return val

    def __set__(self, obj, value):
        if self.is_bool:
            value = YesNo.from_bool(value)
        obj._pw.esa.EnterMode("EDIT")
        obj._pw.esa.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            [self.key, value]
        )
        obj._pw.esa.EnterMode("RUN")
