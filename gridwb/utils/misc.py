from numpy import sum
from pandas import DataFrame
from ..grid.components import Gen, Load, Bus

class InjectionVector:

    def __init__(self, loaddf: DataFrame, losscomp=0.05) -> None:
        '''An Instance Representing an Injection Vector
        params:
        - loaddf: Poor Naming but should just be df with 'BusNum' Column for all buses
        - losscomp: For an increased injection, generation will be increased to compensate losses
        '''
        self.loaddf = loaddf.copy()

        self.loaddf['Alpha'] = 0
        self.loaddf = self.loaddf.set_index('BusNum')

        self.losscomp = losscomp
    
    @property
    def vec(self):
        return self.loaddf['Alpha'].to_numpy()
    
    def supply(self, *busids):
        self.loaddf.loc[busids, 'Alpha'] = 1
        self.norm()

    def demand(self, *busids):
        self.loaddf.loc[busids, 'Alpha'] = -1
        self.norm()
    
    def norm(self):

        # Normalize Positive
        isPos = self.vec>0
        posSum = sum(self.vec[isPos])
        negSum = -sum(self.vec[~isPos])

        self.loaddf.loc[isPos,'Alpha'] /= posSum/(1+self.losscomp) if posSum>0 else 1
        self.loaddf.loc[~isPos,'Alpha'] /= negSum if negSum>0 else 1



def ybus_with_loads(Y, buses: list[Bus], loads: list[Load], gens=None):
    '''
    If a list of Generators are passed it will add
    the generation as negative impedence for gens without GENROU models
    '''

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

            gen: Gen
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



