
import numpy as np
from pandas import DataFrame

from .grid.components import *
from .apps import GIC, Network, ForcedOscillation

from .indextool import IndexTool
from .adapter import Adapter

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

    def __getitem__(self, arg):
        """
        Local Indexing of retrieval.

        Parameters
        ----------
        arg : tuple
            Indexing arguments (ObjectType, Fields).

        Returns
        -------
        pd.DataFrame or pd.Series
            The retrieved data.
        """
        return self[arg]
    
    def __setitem__(self, args, value) -> None:
        """
        Write Power World Objects and Fields.

        Parameters
        ----------
        args : tuple
            Indexing arguments (ObjectType, Fields).
        value : any
            The value(s) to write.
        """
        self[args] = value

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


        
    ''' LOCATION FUNCTIONS '''

    def busmap(self):
        """
        Returns a Pandas Series indexed by BusNum to the positional value of each bus
        in matricies like the Y-Bus, Incidence Matrix, Etc.

        Returns
        -------
        pd.Series
            Series mapping BusNum to index.
        """
        return self.network.busmap()
    
    
    def buscoords(self, astuple=True):
        """
        Retrive dataframe of bus latitude and longitude coordinates based on substation data.

        Parameters
        ----------
        astuple : bool, optional
            Whether to return as a tuple of (Lon, Lat). Defaults to True.

        Returns
        -------
        pd.DataFrame or tuple
            Coordinates data.
        """
        A, S = self[Bus, 'SubNum'],  self[Substation, ['Longitude', 'Latitude']]
        LL = A.merge(S, on='SubNum') 
        if astuple:
            return LL['Longitude'], LL['Latitude']
        return LL
    
    
    ''' Syntax Sugar '''



    
