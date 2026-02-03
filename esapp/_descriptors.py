"""
Descriptors for PowerWorld settings.

Provides lightweight, Pythonic attribute access to PowerWorld option flags
without repetitive boilerplate setter/getter methods.
"""

from .components import Sim_Solution_Options
from .saw._enums import YesNo


class SolverOption:
    """Descriptor for Sim_Solution_Options fields.

    Supports bool (YES/NO), int, and float option types via the
    ``is_bool`` parameter (default True).

    Usage as a class-level attribute on PowerWorld::

        class PowerWorld(Indexable):
            do_one_iteration = SolverOption('DoOneIteration')
            max_iterations   = SolverOption('MaxItr', is_bool=False)

        pw.do_one_iteration = True   # sets YES/NO
        pw.max_iterations = 100      # sets raw value
        pw.do_one_iteration          # reads back as bool
        pw.max_iterations            # reads back as-is
    """

    def __init__(self, key: str, is_bool: bool = True):
        self.key = key
        self.is_bool = is_bool

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj[Sim_Solution_Options, self.key]
        if self.is_bool:
            return val == YesNo.YES
        return val

    def __set__(self, obj, value):
        if self.is_bool:
            obj[Sim_Solution_Options, self.key] = YesNo.from_bool(value)
        else:
            obj[Sim_Solution_Options, self.key] = value


class GICOption:
    """Descriptor for GIC_Options_Value settings.

    Usage as a class-level attribute on GIC::

        class GIC:
            pf_include = GICOption('IncludeInPowerFlow')
            calc_mode  = GICOption('CalcMode', is_bool=False)

        gic.pf_include = True        # sets via _set_gic_option
        gic.calc_mode = 'SnapShot'   # non-bool option
    """

    def __init__(self, key: str, is_bool: bool = True):
        self.key = key
        self.is_bool = is_bool

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj.get_gic_option(self.key)
        if self.is_bool and val is not None:
            return val == YesNo.YES
        return val

    def __set__(self, obj, value):
        obj._set_gic_option(self.key, value)
