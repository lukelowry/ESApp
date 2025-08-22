

# WorkBench Imports
from gridwb.workbench.core.powerworld import PowerWorldIO
from gridwb.workbench.grid.components import Branch, Bus
from .app import PWApp
from ..core import Context

from scipy.sparse import diags, lil_matrix
import numpy as np
from pandas import Series

from enum import Enum

# Types of support branch weights
class BranchType(Enum):
    LENGTH = 1
    RES_DIST = 2 # Resistance Distance
    PROPAGATION = 3
    
    

# Constructing Network Matricies and other metrics
class Network(PWApp):

    io: PowerWorldIO

    def __init__(self, context: Context) -> None:
        super().__init__(context)

        # Incidence
        self.A = None

    def busmap(self):
        '''
        Returns a Pandas Series indexed by BusNum to the positional value of each bus
        in matricies like the Y-Bus, Incidence Matrix, Etc.

        Example usage:
        branches['BusNum'].map(busmap)
        '''
        busNums = self.io[Bus]
        return Series(busNums.index, busNums['BusNum'])

    def incidence(self, remake=True):
        '''
        Details
            Returns incidence matrix. If alreayd made, retrieves from cache
            instead of making it again.

        Returns:
            Sparse Incidence Matrix of the branch network of the grid.

        Dimensions: (Number of Branches)x(Number of Buses)
        '''

        # If already made, don't remake
        if self.A is not None and not remake:
            return self.A
        


        # Retrieve
        branches = self.io[Branch,['BusNum', 'BusNum:1']]

        # Column Positions 
        bmap    = self.busmap()
        fromBus = branches['BusNum'].map(bmap).to_numpy()
        toBus   = branches['BusNum:1'].map(bmap).to_numpy()

        # Lengths and indexers
        nbranches = len(branches)
        branchIDs = np.arange(nbranches)

        # Sparse Arc-Incidence Matrix
        A = lil_matrix((nbranches,len(bmap)))
        A[branchIDs, fromBus] = -1
        A[branchIDs, toBus]   = 1

        self.A = A

        # Don't use until a sparse operation
        #A = self.io.esa.get_incidence_matrix()

        return A

    def laplacian(self, weights: BranchType | np.ndarray):
        '''
        Description:
            Uses the systems incident matrix and creates
            a laplacian with branch weights as W
        Parameters:
            W: 1-D array of weights
        Returns:
            Sparse Laplacian
        '''

        match weights:
            case BranchType.LENGTH:      #  m^-2
                W = 1/self.lengths()**2
            case BranchType.RES_DIST:       #  ohms^-2
                # NOTE squaring makes this very poor. May need Gramian
                W = 1/self.zmag()#**2
            case BranchType.PROPAGATION: #  m^-2
                ell = self.lengths()
                GAM = self.gamma()
                W = (1/ell**2 - GAM**2)/1e6  
            case _:
                W = weights

        A = self.incidence() 

        return A.T@diags(W)@A
    
    ''' Branch Weights '''

  
    def lengths(self, longer_xfmr_lens=False):
        '''
        Returns lengths of each branch in kilometers.

        Parameters
            longer_xfmr_lens: Use a ficticious length that is approximately
            the same length as it would be if it was a branch
            If False, lengths are assumed to be 1 meter.
        '''

        # This is distance in kilometers
        field = 'LineLengthByParameters:2'
        ell = self.io[Branch,field][field]

        # Assume XFMR 1 meter long
        if longer_xfmr_lens:

            branches = self.io[Branch, ['LineX', 'LineR', 'BranchDeviceType', 'LineLengthByParameters:2']]

            #lines = branches.loc[branches['BranchDeviceType']=='Line']
            #xfmrs = branches.loc[branches['BranchDeviceType']!='Line']
            lines = branches.loc[branches['LineLengthByParameters:2'] > 0]
            xfmrs = branches.loc[branches['LineLengthByParameters:2'] <= 0]

            lineZ = np.abs(lines['LineR'] + 1j*lines['LineX'])
            xfmrZ = np.abs(xfmrs['LineR'] + 1j*xfmrs['LineX'])

            # Average Ohms per km for lines
            ZperKM = (lineZ/lines['LineLengthByParameters:2']).mean()

            # Impedence Magnitude of Transformers
            psuedoLength = (xfmrZ/ZperKM).to_numpy()


            ell.loc[ell==0] = psuedoLength
        else:
            ell.loc[ell==0] = 0.001

        return ell
    
    def zmag(self):
        '''
        Steady-state phase delays of the branches, approximated
        as the angle of the complex value.
        Units
            Radians
        Min/Max
            -pi/2, 0
        '''
        Y = self.ybranch() 

        return 1/np.abs(Y)
      
    def ybranch(self, asZ=False):
        '''
        Return Admittance of Lines in Complex Form
        '''

        branches = self.io[Branch, ['LineR:2', 'LineX:2']]
        R = branches['LineR:2']
        X = branches['LineX:2']
        Z = R + 1j*X 

        if asZ:
            return Z
        return 1/Z
    
    def yshunt(self):
        '''
        Return Admittance of Lines in Complex Form
        '''

        branches = self.io[Branch, ['LineG', 'LineC']]
        G = branches['LineG']
        B = branches['LineC']
 
        return G + 1j*B

    def gamma(self):
        '''Returns approximation of propagation constants for each branch'''

        # Length (Set Xfmr to 1 meter)
        ell = self.lengths()

        # Series Parameters
        Z = self.ybranch(asZ=True)
        Y = self.yshunt()


        # Correct Zero-Values
        Z[Z==0] = 0.000446+ 0.002878j
        Y[Y==0] = 0.000463j

        # By Length TODO check the mult/division order here.
        Z /= ell # Series Value
        Y /= ell # Shunt Value
        

        # Propagation Parameter
        return  np.sqrt(Y*Z)