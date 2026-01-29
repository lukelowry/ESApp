"""
Power system utilities for sensitivity analysis and load modeling.

This module provides tools for constructing injection vectors and
modifying Y-bus matrices with load/generation models.
"""

from typing import Optional, Sequence

import numpy as np
from numpy import sum as npsum
from numpy.typing import NDArray
from pandas import DataFrame
import scipy.sparse as sp

__all__ = [
    'InjectionVector',
    'ybus_with_loads',
]


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
        is_demand = ~is_supply

        supply_sum = npsum(alpha[is_supply])
        demand_sum = -npsum(alpha[is_demand])

        if supply_sum > 0:
            self.loaddf.loc[is_supply, 'Alpha'] /= supply_sum / (1 + self.losscomp)
        if demand_sum > 0:
            self.loaddf.loc[is_demand, 'Alpha'] /= demand_sum


def ybus_with_loads(
    Y: sp.spmatrix,
    buses: Sequence,
    loads: Sequence,
    gens: Optional[Sequence] = None
) -> sp.spmatrix:
    """
    Modify Y-bus matrix to include constant impedance load/generation models.

    Converts P/Q injections at each bus into equivalent shunt admittances
    based on bus voltages and adds them to the Y-bus diagonal. This creates
    a linearized load model suitable for small-signal analysis.

    Parameters
    ----------
    Y : scipy.sparse matrix
        Original Y-bus admittance matrix.
    buses : sequence
        Bus component objects with attributes:
        - BusNum: Bus number
        - BusLoadMW: Active load (MW)
        - BusLoadMVR: Reactive load (MVAr)
        - BusPUVolt: Per-unit voltage magnitude
    loads : sequence
        Load component objects (currently unused, load data comes from buses).
    gens : sequence, optional
        Generator component objects. Generators without dynamic models
        (not GENROU) are treated as negative constant impedance loads.
        Each must have: BusNum, GenMW, GenMVR, BusPUVolt, TSGenMachineName,
        GenStatus.

    Returns
    -------
    scipy.sparse matrix
        Modified Y-bus matrix with load/generation admittances added.

    Notes
    -----
    - Uses 100 MVA base for per-unit conversion.
    - Constant impedance model: Y_load = S* / |V|^2
    - Generators with GENROU models and 'Closed' status are skipped
      (assumed handled by dynamic simulation).

    Examples
    --------
    >>> Y_modified = ybus_with_loads(Ybus, buses, loads, gens=generators)
    """
    Y = Y.copy()
    basemva = 100.0

    # Map bus number to Y-bus index
    bus_to_idx = {b.BusNum: i for i, b in enumerate(buses)}

    for bus in buses:
        idx = bus_to_idx[bus.BusNum]

        # Net load at bus (per-unit)
        p_pu = bus.BusLoadMW / basemva if bus.BusLoadMW > 0 else 0.0
        q_pu = bus.BusLoadMVR / basemva
        s_pu = p_pu + 1j * q_pu

        # Voltage magnitude
        vmag = bus.BusPUVolt

        # Constant impedance admittance
        y_load = s_pu.conjugate() / vmag**2

        Y[idx, idx] += y_load

    # Add generators without dynamic models as negative load
    if gens is not None:
        for gen in gens:
            # Skip generators with dynamic models
            if gen.TSGenMachineName == 'GENROU' and gen.GenStatus == 'Closed':
                continue

            idx = bus_to_idx[gen.BusNum]

            p_pu = gen.GenMW / basemva
            q_pu = gen.GenMVR / basemva
            s_pu = p_pu + 1j * q_pu

            vmag = gen.BusPUVolt
            y_gen = s_pu.conjugate() / vmag**2

            # Negative admittance (generation = negative load)
            Y[idx, idx] -= y_gen

    return Y
