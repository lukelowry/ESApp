from numpy import sum
from pandas import DataFrame


class InjectionVector:
    """Represents a normalized injection vector for power system sensitivity studies."""

    def __init__(self, loaddf: DataFrame, losscomp=0.05) -> None:
        """Initializes the InjectionVector.

        Parameters
        ----------
        loaddf : pandas.DataFrame
            A DataFrame containing at least a 'BusNum' column for all buses in the system.
        losscomp : float, optional
            Loss compensation factor. For an increased injection, generation will be
            increased to compensate for losses. Defaults to 0.05.
        """
        self.loaddf = loaddf.copy()

        self.loaddf['Alpha'] = 0
        self.loaddf = self.loaddf.set_index('BusNum')

        self.losscomp = losscomp
    
    @property
    def vec(self):
        """Returns the current injection vector as a NumPy array.

        Returns
        -------
        numpy.ndarray
            The injection vector.
        """
        return self.loaddf['Alpha'].to_numpy()
    
    def supply(self, *busids):
        """Sets the specified buses as supply points (positive injection).

        The 'Alpha' value for these buses will
        """
        self.loaddf.loc[busids, 'Alpha'] = 1
        self.norm()

    def demand(self, *busids):
        """Sets the specified buses as demand points (negative injection).

        :param busids: Variable number of bus IDs.
        """
        self.loaddf.loc[busids, 'Alpha'] = -1
        self.norm()
    
    def norm(self):
        """Normalizes the vector so that total supply equals total demand plus losses."""
        # Normalize Positive
        isPos = self.vec>0
        posSum = sum(self.vec[isPos])
        negSum = -sum(self.vec[~isPos])

        self.loaddf.loc[isPos,'Alpha'] /= posSum/(1+self.losscomp) if posSum>0 else 1
        self.loaddf.loc[~isPos,'Alpha'] /= negSum if negSum>0 else 1


def ybus_with_loads(Y, buses, loads, gens=None):
    """
    Modifies a Y-Bus matrix to include constant impedance load and generation models.

    This function converts P/Q injections into equivalent shunt admittances based on 
    the current bus voltages and adds them to the diagonal of the Y-Bus matrix.

    :param Y: The original sparse Y-Bus matrix (scipy.sparse).
    :param buses: List of Bus component objects.
    :param loads: List of Load component objects.
    :param gens: Optional list of Gen component objects. Generators without dynamic 
        models (e.g., GENROU) are treated as negative constant impedance loads.
    :return: The modified sparse Y-Bus matrix.
    :rtype: scipy.sparse.base.spmatrix
    """

    # Copy so don't modify
    Y = Y.copy()

    # Map the bus number to its Y-Bus Index
    # TODO Do a sort by Bus Num to gaurentee order
    busPosY = {b.BusNum: i for i, b in enumerate(buses)}

    # For Per-Unit Conversion
    basemva = 100

    for bus in buses:

        # Location in YBus
        busidx = busPosY[bus.BusNum]

        # Net Load at Bus
        pumw = bus.BusLoadMW/basemva if bus.BusLoadMW > 0 else 0
        pumvar = bus.BusLoadMVR/basemva if bus.BusLoadMVR > 0 or bus.BusLoadMVR < 0 else 0
        puS = pumw + 1j*pumvar

        # V at Bus
        vmag = bus.BusPUVolt

        # Const Impedenace Load/Gen
        constAdmit = puS.conjugate()/vmag**2

        # Add to Ybus
        Y[busidx][busidx] += constAdmit # TODO determine if to use + or -!


    # Add Generators without models as negative load (if closed)
    if gens is not None:
        for gen in gens:

            if gen.TSGenMachineName == 'GENROU' and gen.GenStatus=='Closed':
                continue
            else:
                basemva = 100
                # Net Load at Bus
                pumw = gen.GenMW/basemva
                pumvar = gen.GenMVR/basemva
                puS = pumw + 1j*pumvar

                # V at Bus
                vmag =gen.BusPUVolt

                # Const Impedenace Load/Gen
                constAdmit = puS.conjugate()/vmag**2

                # Location in YBus
                busidx = busPosY[gen.BusNum]

                # Negative Admittance
                Y[busidx][busidx] -= constAdmit

    return Y
