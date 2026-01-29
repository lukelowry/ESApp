"""
Power system utilities and general-purpose helpers.

This module provides:
- Injection vectors for sensitivity analysis and load modeling
- Function decorators for debugging and profiling
"""

from __future__ import annotations

from functools import wraps
from time import time
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

__all__ = [
    'InjectionVector',
    'timing',
]

# =============================================================================
# Decorators
# =============================================================================

F = TypeVar('F', bound=Callable)


def timing(func: F) -> F:
    """
    Decorator that prints the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function that prints timing information.

    Examples
    --------
    >>> @timing
    ... def slow_function():
    ...     time.sleep(1)
    ...
    >>> slow_function()
    'slow_function' took: 1.0012 sec
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        elapsed = time() - start
        print(f'{func.__name__!r} took: {elapsed:.4f} sec')
        return result
    return wrapper


class InjectionVector:
    """
    Normalized injection vector for power system sensitivity studies.

    Represents a pattern of power injections across system buses,
    normalized so that total supply equals total demand plus losses.
    Useful for computing power transfer distribution factors (PTDFs)
    and line outage distribution factors (LODFs).

    Parameters
    ----------
    loaddf : pandas.DataFrame
        DataFrame containing at least a 'BusNum' column for all buses.
    losscomp : float, default 0.05
        Loss compensation factor. Supply is scaled up by (1 + losscomp)
        to account for system losses.

    Attributes
    ----------
    loaddf : pandas.DataFrame
        Internal DataFrame with 'Alpha' column for injection values,
        indexed by BusNum.
    losscomp : float
        Loss compensation factor.

    Examples
    --------
    >>> inj = InjectionVector(bus_df, losscomp=0.05)
    >>> inj.supply(101, 102)  # Set buses 101, 102 as supply
    >>> inj.demand(201)       # Set bus 201 as demand
    >>> alpha = inj.vec       # Get normalized injection vector
    """

    def __init__(self, loaddf: DataFrame, losscomp: float = 0.05) -> None:
        self.loaddf = loaddf.copy()
        self.loaddf['Alpha'] = 0.0
        self.loaddf = self.loaddf.set_index('BusNum')
        self.losscomp = losscomp

    @property
    def vec(self) -> NDArray[np.float64]:
        """
        Get the current injection vector as a numpy array.

        Returns
        -------
        np.ndarray
            Injection values for all buses in bus number order.
        """
        return self.loaddf['Alpha'].to_numpy()

    def supply(self, *busids: int) -> None:
        """
        Set specified buses as supply points (positive injection).

        The injection vector is automatically normalized after this call.

        Parameters
        ----------
        *busids : int
            Bus numbers to set as supply points.
        """
        self.loaddf.loc[list(busids), 'Alpha'] = 1.0
        self.norm()

    def demand(self, *busids: int) -> None:
        """
        Set specified buses as demand points (negative injection).

        The injection vector is automatically normalized after this call.

        Parameters
        ----------
        *busids : int
            Bus numbers to set as demand points.
        """
        self.loaddf.loc[list(busids), 'Alpha'] = -1.0
        self.norm()

    def norm(self) -> None:
        """
        Normalize the injection vector.

        Scales supply and demand so that:
        - Total supply = (1 + losscomp) * total demand
        - Supply buses sum to 1.0
        - Demand buses sum to -1.0

        This ensures power balance accounting for system losses.
        """
        alpha = self.vec
        is_supply = alpha > 0
        is_demand = alpha < 0

        supply_sum = np.sum(alpha[is_supply])
        demand_sum = -np.sum(alpha[is_demand])

        if supply_sum > 0:
            self.loaddf.loc[is_supply, 'Alpha'] /= supply_sum / (1 + self.losscomp)
        if demand_sum > 0:
            self.loaddf.loc[is_demand, 'Alpha'] /= demand_sum
