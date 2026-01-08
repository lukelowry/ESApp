from .grid import Bus
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
        self.network = Network(self)
        self.gic = GIC(self)
        self.modes = ForcedOscillation(self)
        #self.dyn = Dynamics(self)
        #self.statics = Statics(self)


