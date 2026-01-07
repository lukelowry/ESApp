
import numpy as np
from pandas import DataFrame

from .grid.components import *
from .apps import GIC, Network, ForcedOscillation

from .indextool import IndexTool
from .adapter import Adapter

class GridWorkBench:
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

        self.io = IndexTool(fname)
        self.io.open()

        # Applications
        #self.dyn = Dynamics(self.io)
        #self.statics = Statics(self.io)
        self.gic = GIC(self.io)
        self.network = Network(self.io)
        self.modes = ForcedOscillation(self.io)
        self.func = Adapter(self.io)

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
        return self.io[arg]
    
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
        self.io[args] = value

    def save(self):
        """
        Save the open PowerWorld file.
        """
        self.io.save()

    def voltage(self, asComplex=True):
        """
        The vector of voltages in PowerWorld.

        Parameters
        ----------
        asComplex : bool, optional
            Whether to return complex values. Defaults to True.

        Returns
        -------
        pd.Series or tuple
            Series of complex values if asComplex=True, 
            else tuple of (Vmag, Angle in Radians).
        """
        v_df = self.io[Bus, ['BusPUVolt','BusAngle']] 

        vmag = v_df['BusPUVolt']
        rad = v_df['BusAngle']*np.pi/180

        if asComplex:
            return vmag * np.exp(1j * rad)
        
        return vmag, rad
    
    def write_voltage(self,V):
        """
        Given Complex 1-D vector write to PowerWorld.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector.
        """
        V_df =  np.vstack([np.abs(V), np.angle(V,deg=True)]).T

        self.io[Bus,['BusPUVolt', 'BusAngle']] = V_df

    def pflow(self, getvolts=True) -> DataFrame:
        """
        Solve Power Flow in external system.
        By default bus voltages will be returned.

        Parameters
        ----------
        getvolts : bool, optional
            Flag to indicate the voltages should be returned after power flow, 
            defaults to True.

        Returns
        -------
        pd.DataFrame or None
            Dataframe of bus number and voltage if requested.
        """
        # Solve Power Flow through External Tool
        self.io.pflow()

        # Request Voltages if needed
        if getvolts:
            return self.voltage()
        
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
        A, S = self.io[Bus, 'SubNum'],  self.io[Substation, ['Longitude', 'Latitude']]
        LL = A.merge(S, on='SubNum') 
        if astuple:
            return LL['Longitude'], LL['Latitude']
        return LL
    
    
    ''' Syntax Sugar '''

    def lines(self):
        """
        Retrieves and returns all transmission line data. Convenience function.

        Returns
        -------
        pd.DataFrame
            Transmission line data.
        """
        # Get Data
        branches = self.io[Branch, :]

        # Return requested Records
        return branches.loc[branches['BranchDeviceType']=='Line']
    
    def xfmrs(self):
        """
        Retrieves and returns all transformer data. Convenience function.

        Returns
        -------
        pd.DataFrame
            Transformer data.
        """
        # Get Data
        branches = self.io[Branch, :]

        # Return requested Records
        return branches.loc[ branches['BranchDeviceType']=='Transformer']
         
    def ybus(self, dense=False):
        """
        Returns the Y-Bus Matrix.

        Parameters
        ----------
        dense : bool, optional
            Whether to return a dense array. Defaults to False (sparse).

        Returns
        -------
        Union[np.ndarray, csr_matrix]
            The Y-Bus matrix.
        """
        return self.io.esa.get_ybus(dense)
    
 

    
