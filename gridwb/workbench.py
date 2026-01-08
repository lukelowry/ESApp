from .grid import *
from .apps import GIC, Network, ForcedOscillation
from .indextool import IndexTool
from .adapter import Adapter

import numpy as np

class GridWorkBench(Adapter, IndexTool):
    """
    Main entry point for interacting with the PowerWorld grid model.
    """
    def __init__(self, fname=None):
        """
        Initialize the GridWorkBench.

        Parameters
        ----------
        fname : str, optional
            Path to the PowerWorld case file (.pwb).
        """
        if fname is None:
            return
        self.fname = fname 

        self.open()

        # Applications
        #self.dyn = Dynamics(self)
        #self.statics = Statics(self)
        self.gic = GIC(self)
        self.network = Network(self)
        self.modes = ForcedOscillation(self)

    def save(self):
        """
        Save the open PowerWorld file.
        """
        self.esa.SaveCase()

    def write_voltage(self,V):
        """
        Given Complex 1-D vector write to PowerWorld.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector.
        """
        V_df =  np.vstack([np.abs(V), np.angle(V,deg=True)]).T

        self[Bus,['BusPUVolt', 'BusAngle']] = V_df

