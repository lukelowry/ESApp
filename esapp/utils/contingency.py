"""
Contingency Builder Utilities
=============================

Provides fluent builder tools for constructing transient stability
contingency event sequences.

Classes
-------
ContingencyBuilder
    Fluent builder for TS contingencies with method chaining.
SimAction
    Enumeration of standard simulation action strings.
"""

from enum import Enum
from typing import List, Tuple, Union, Any

from pandas import DataFrame

__all__ = ['ContingencyBuilder', 'SimAction']


class SimAction(str, Enum):
    """Enumeration of standard simulation actions to prevent magic string errors."""
    FAULT_3PB = "FAULT 3PB SOLID"
    CLEAR_FAULT = "CLEARFAULT"
    OPEN = "OPEN"
    CLOSE = "CLOSE"

class ContingencyBuilder:
    """
    Fluent builder for Transient Stability (TS) contingencies.

    Constructs a timeline of events to be simulated using method chaining.

    Parameters
    ----------
    name : str
        Unique name for the contingency.
    runtime : float, optional
        Simulation duration in seconds (default: 10.0).

    Attributes
    ----------
    name : str
        The contingency name.
    runtime : float
        Simulation end time in seconds.

    Example
    -------
    >>> builder = ContingencyBuilder("GenTrip", runtime=5.0)
    >>> builder.at(1.0).fault_bus("101").at(1.1).clear_fault("101")
    """

    def __init__(self, name: str, runtime: float = 10.0):
        self.name = name
        self.runtime = runtime
        self._current_time: float = 0.0
        self._events: List[Tuple[float, str, str, str]] = []

    def at(self, t: float) -> 'ContingencyBuilder':
        """
        Set the current time cursor for subsequent events.

        Parameters
        ----------
        t : float
            Time in seconds (must be non-negative).

        Returns
        -------
        ContingencyBuilder
            Self for method chaining.

        Raises
        ------
        ValueError
            If time is negative.
        """
        if t < 0:
            raise ValueError(f"Time cannot be negative: {t}")
        self._current_time = t
        return self

    def add_event(self, obj_type: str, who: str, action: Union[str, SimAction]) -> 'ContingencyBuilder':
        """
        Add a generic event at the current time cursor.

        Parameters
        ----------
        obj_type : str
            PowerWorld object type (e.g., "Bus", "Gen", "Branch").
        who : str
            Object identifier string.
        action : Union[str, SimAction]
            Action to perform (e.g., SimAction.OPEN or "OPEN").

        Returns
        -------
        ContingencyBuilder
            Self for method chaining.
        """
        act_str = action.value if isinstance(action, SimAction) else str(action)
        self._events.append((self._current_time, obj_type, who, act_str))
        return self

    def fault_bus(self, bus: Any) -> 'ContingencyBuilder':
        """Apply a 3-phase solid fault to a bus at the current time."""
        return self.add_event("Bus", str(bus), SimAction.FAULT_3PB)

    def clear_fault(self, bus: Any) -> 'ContingencyBuilder':
        """Clear the fault at a bus at the current time."""
        return self.add_event("Bus", str(bus), SimAction.CLEAR_FAULT)

    def trip_gen(self, bus: Any, gid: str = "1") -> 'ContingencyBuilder':
        """Trip (open) a generator at the current time."""
        return self.add_event("Gen", f"{bus} '{gid}'", SimAction.OPEN)

    def trip_branch(self, f_bus: Any, t_bus: Any, ckt: str = "1") -> 'ContingencyBuilder':
        """Trip (open) a branch at the current time."""
        return self.add_event("Branch", f"{f_bus} {t_bus} '{ckt}'", SimAction.OPEN)

    def to_dataframes(self) -> Tuple[DataFrame, DataFrame]:
        """
        Generates DataFrames matching the ESA GObject schemas.

        Returns:
            Tuple[DataFrame, DataFrame]: (Contingency Definition, Element Definitions)
        """
        # 1. Contingency Header
        ctg_df = DataFrame({
            'TSCTGName': [self.name],
            'StartTime': [0.0],
            'EndTime': [self.runtime],
            'CTGSkip': ['NO']
        })

        # 2. Elements
        if not self._events:
            return ctg_df, DataFrame()

        # Vectorized list creation is generally fast enough here
        ele_rows = [
            {
                'TSCTGName': self.name,
                'TSEventString': f"{action} {obj_type} {who}",
                'TSTimeInSeconds': t,
                'WhoAmI': f"{obj_type} {who}",
                'TSTimeInCycles': t * 60.0,
            }
            for t, obj_type, who, action in self._events
        ]

        return ctg_df, DataFrame(ele_rows)
