
import numpy as np
from pandas import DataFrame

from .grid.components import *
from .apps import GIC, Network, ForcedOscillation

from .indextool import IndexTool
from .adapter import Adapter

class GridWorkBench:
    def __init__(self, fname=None):

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
        '''Local Indexing of retrieval'''
        return self.io[arg]
    
    def __setitem__(self, args, value) -> None:
        '''Write Power World Objects and Fields'''
        self.io[args] = value

    def save(self):
        '''Save Open the Power World File'''
        self.io.save()

    def voltage(self, asComplex=True):
        '''
        Description
            The vector of voltages in power world
        Parameters
            dtype: only returns compelx vector at this moment.
        Returns
            complex=True -> Series of complex values if complex=True
            complex=False -> Tuple of Vmag and Angle (In Radians)
        '''
        v_df = self.io[Bus, ['BusPUVolt','BusAngle']] 

        vmag = v_df['BusPUVolt']
        rad = v_df['BusAngle']*np.pi/180

        if asComplex:
            return vmag * np.exp(1j * rad)
        
        return vmag, rad
    
    def write_voltage(self,V):
        '''
        Given Complex 1-D vector write to power world
        '''

        V_df =  np.vstack([np.abs(V), np.angle(V,deg=True)]).T

        self.io[Bus,['BusPUVolt', 'BusAngle']] = V_df

    def pflow(self, getvolts=True) -> DataFrame | None:
        """
        Solve Power Flow in external system.
        By default bus voltages will be returned.

        :param getvolts: flag to indicate the voltages should be returned after power flow, 
            defaults to True
        :type getvolts: bool, optional
        :return: Dataframe of bus number and voltage if requested
        :rtype: DataFrame or None
        """

        # Solve Power Flow through External Tool
        self.io.pflow()

        # Request Voltages if needed
        if getvolts:
            return self.voltage()
        
    ''' LOCATION FUNCTIONS '''

    def busmap(self):
        '''
        Returns a Pandas Series indexed by BusNum to the positional value of each bus
        in matricies like the Y-Bus, Incidence Matrix, Etc.

        Example usage:
        branches['BusNum'].map(busmap)
        '''
        return self.network.busmap()
    
    
    def buscoords(self, astuple=True):
        '''Retrive dataframe of bus latitude and longitude coordinates based on substation data'''
        A, S = self.io[Bus, 'SubNum'],  self.io[Substation, ['Longitude', 'Latitude']]
        LL = A.merge(S, on='SubNum') 
        if astuple:
            return LL['Longitude'], LL['Latitude']
        return LL
    
    
    ''' Syntax Sugar '''

    def lines(self):
        '''
        Retrieves and returns all transmission line data. Convenience function.
        '''

        # Get Data
        branches = self.io[Branch, :]

        # Return requested Records
        return branches.loc[branches['BranchDeviceType']=='Line']
    
    def xfmrs(self):
        '''
        Retrieves and returns all transformer data. Convenience function.
        '''

        # Get Data
        branches = self.io[Branch, :]

        # Return requested Records
        return branches.loc[ branches['BranchDeviceType']=='Transformer']
         
    def ybus(self, dense=False):
        '''Returns the sparse Y-Bus Matrix'''
        return self.io.esa.get_ybus(dense)
    
 

    
