

# WorkBench Imports
from ..io import Context, PowerWorldIO
from ..grid.components import Branch, Bus, DCTransmissionLine, Gen
from .app import PWApp

from scipy.sparse import diags, lil_matrix
import numpy as np
from pandas import Series, concat
from enum import Enum


# Types of support branch weights
class BranchType(Enum):
    LENGTH = 1
    RES_DIST = 2 # Resistance Distance
    DELAY = 3
    


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

    def incidence(self, remake=True, hvdc=False):
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
        fields = ['BusNum', 'BusNum:1']
        branches = self.io[Branch,fields][fields]

        if hvdc:
            hvdc_branches = self.io[DCTransmissionLine,fields][fields]
            branches = concat([branches,hvdc_branches], ignore_index=True)

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

    def laplacian(self, weights: BranchType | np.ndarray, longer_xfmr_lens=True, len_thresh=0.01, hvdc=False):
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
            case BranchType.LENGTH:    #  m^-2
                W = 1/self.lengths(longer_xfmr_lens, len_thresh, hvdc)**2
            case BranchType.RES_DIST:  #  ohms^-2
                W = 1/self.zmag(hvdc) 
            case BranchType.DELAY:
                W = 1/self.delay()**2  # 1/s^2
            case _:
                W = weights

        A = self.incidence(hvdc=hvdc) 

        LAP =  A.T@diags(W)@A

        return LAP.tocsc()
    
    ''' Branch Weights '''

  
    def lengths(self, longer_xfmr_lens=False, length_thresh_km = 0.01,hvdc=False):
        '''
        Returns lengths of each branch in kilometers.

        Parameters
            longer_xfmr_lens: Use a ficticious length that is approximately
            the same length as it would be if it was a branch
            If False, lengths are assumed to be 1 meter.

            length_thresh_km: Branches less than 0.1km will use the 'equivilent elc. dist'.
            This is an option because various devices may actually have zero distance

            hvdc: if True, this will also include hvdc lines
        '''

        # This is distance in kilometers
        # Just found out that this can be EITHER?? so have to figure 
        # out which to use. Porbably prefer first field
        field = ['LineLengthByParameters', 'LineLengthByParameters:2']
        ell = self.io[Branch,field][field]

        ell_user = ell['LineLengthByParameters']
        ell.loc[ell_user>0,'LineLengthByParameters:2'] = ell.loc[ell_user>0,'LineLengthByParameters']
        ell = ell['LineLengthByParameters:2']

        if hvdc:
            field = 'LineLengthByParameters'
            hvdc_ell = self.io[DCTransmissionLine,field][field]
            ell = concat([ell, hvdc_ell], ignore_index=True)

        # Calculate the equivilent distance if same admittance of a line
        if longer_xfmr_lens:

            fields = ['LineX:2', 'LineR:2']
            branches = self.io[Branch, fields][fields]

            isLongLine = ell > length_thresh_km
            lines = branches.loc[isLongLine]
            xfmrs = branches.loc[~isLongLine]

            lineZ = np.abs(lines['LineR:2'] + 1j*lines['LineX:2'])
            xfmrZ = np.abs(xfmrs['LineR:2'] + 1j*xfmrs['LineX:2'])

            # Average Ohms per km for lines
            ZperKM = (lineZ/ell).mean()

            # BUG Mean is probably a bad way, since the line lengths are very diverse.

            # Impedence Magnitude of Transformers
            psuedoLength = (xfmrZ/ZperKM).to_numpy()


            ell.loc[~isLongLine] = psuedoLength

        # Assume XFMR 10 meter long
        else:
            ell.loc[ell==0] = 0.1

        return ell
    
    def zmag(self, hvdc=False):
        '''
        Steady-state phase delays of the branches, approximated
        as the angle of the complex value.
        Units
            Radians
        Min/Max
            -pi/2, 0
        '''
        Y = self.ybranch(hvdc=hvdc) 

        return 1/np.abs(Y)
      
    def ybranch(self, asZ=False, hvdc=False):
        '''
        Return Admittance of Lines in Complex Form
        '''

        branches = self.io[Branch, ['LineR:2', 'LineX:2']]



        R = branches['LineR:2']
        X = branches['LineX:2']
        Z = R + 1j*X 

        if hvdc: # Just add small impedence for HVDC
            cnt = len(self.io[DCTransmissionLine])
            Zdc = Z[:cnt].copy()
            Zdc[:] = 0.001
            Z = concat([Z, Zdc], ignore_index=True)

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
    
    
    def delay(self, min_delay=10e-6):
        '''
        Return Effective delay of branches
        The minimum delay permitted is 10 us
        '''

        w = 2*np.pi*60

        # EDGE SERIES RESISTANCE & INDUCTANCE
        Z = self.ybranch(asZ=True)

        # EFFECTIVE EDGE SHUNT ADMITTANCE
        Ybus = self.io.esa.get_ybus()
        SUM = np.ones(Ybus.shape[0])
        AVG = np.abs(self.incidence())/2 
        Y = AVG@Ybus@SUM 

        # NOTE Do I need to make G =0?

        # Propagation Constant
        gam = np.sqrt(Z*Y)
        beta = np.imag(gam)

        # EFFECTIVE DELAY (SQUARED)
        tau = beta/w

        # Enforce lower bound
        tau[tau<min_delay] = min_delay

        # Propagation Parameter
        return tau
    

