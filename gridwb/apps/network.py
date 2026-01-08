from ..indextool import IndexTool
from ..grid.components import Branch, Bus, DCTransmissionLine
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

    io: IndexTool

    def __init__(self, io: IndexTool) -> None:
        super().__init__(io)

        # Incidence
        self.A = None


    def busmap(self):
        '''
        Returns a Pandas Series indexed by BusNum to the positional value of each bus.

        Useful for mapping bus numbers to matrix indices.

        Returns
        -------
        pd.Series
            Mapping from BusNum to matrix index.
        '''
        busNums = self.io[Bus]
        return Series(busNums.index, busNums['BusNum'])

    def incidence(self, remake=True, hvdc=False):
        '''
        Returns the sparse incidence matrix of the branch network.

        Parameters
        ----------
        remake : bool, optional
            If True, recalculates the matrix even if cached. Defaults to True.
        hvdc : bool, optional
            If True, includes HVDC lines. Defaults to False.

        Returns
        -------
        scipy.sparse.lil_matrix
            Sparse Incidence Matrix of the branch network (Branches x Buses).
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

    def laplacian(self, weights: BranchType, longer_xfmr_lens=True, len_thresh=0.01, hvdc=False):
        '''
        Uses the systems incident matrix and creates a laplacian with branch weights.

        Parameters
        ----------
        weights : BranchType
            Type of weights to use (LENGTH, RES_DIST, DELAY).
        longer_xfmr_lens : bool, optional
            If True, uses fictitious lengths for transformers. Defaults to True.
        len_thresh : float, optional
            Threshold for short lines in km. Defaults to 0.01.
        hvdc : bool, optional
            If True, includes HVDC lines. Defaults to False.

        Returns
        -------
        scipy.sparse.csc_matrix
            Sparse Laplacian matrix.
        '''

        if weights == BranchType.LENGTH:    #  m^-2
            W = 1/self.lengths(longer_xfmr_lens, len_thresh, hvdc)**2
        elif weights == BranchType.RES_DIST:  #  ohms^-2
            W = 1/self.zmag(hvdc) 
        elif weights == BranchType.DELAY:
            W = 1/self.delay()**2  # 1/s^2
        else:
            W = weights

        A = self.incidence(hvdc=hvdc)

        LAP =  A.T@diags(W)@A

        return LAP.tocsc()
    
    ''' Branch Weights '''

  
    def lengths(self, longer_xfmr_lens=False, length_thresh_km = 0.01,hvdc=False):
        '''
        Returns lengths of each branch in kilometers.

        Parameters
        ----------
        longer_xfmr_lens : bool, optional
            Use a ficticious length for transformers. Defaults to False.
        length_thresh_km : float, optional
            Minimum length threshold in km. Defaults to 0.01.
        hvdc : bool, optional
            If True, includes HVDC lines. Defaults to False.

        Returns
        -------
        pd.Series
            Lengths of branches.
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
        Steady-state phase delays of the branches, approximated as the angle of the complex value.

        Parameters
        ----------
        hvdc : bool, optional
            If True, includes HVDC lines. Defaults to False.

        Returns
        -------
        pd.Series
            Phase delays (radians).
        '''
        Y = self.ybranch(hvdc=hvdc) 

        return 1/np.abs(Y)
      
    def ybranch(self, asZ=False, hvdc=False):
        '''
        Return Admittance (or Impedance) of Lines in Complex Form.

        Parameters
        ----------
        asZ : bool, optional
            If True, returns Impedance (Z). If False, returns Admittance (Y). Defaults to False.
        hvdc : bool, optional
            If True, includes HVDC lines. Defaults to False.

        Returns
        -------
        pd.Series
            Complex admittance or impedance.
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
        Return Shunt Admittance of Lines in Complex Form.

        Returns
        -------
        pd.Series
            Complex shunt admittance.
        '''

        branches = self.io[Branch, ['LineG', 'LineC']]
        G = branches['LineG']
        B = branches['LineC']
 
        return G + 1j*B

    def gamma(self):
        '''
        Returns approximation of propagation constants for each branch.

        Returns
        -------
        pd.Series
            Propagation constants.
        '''

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
    
    
    def delay(self, min_delay=10e-4):
        '''
        Return Effective delay of branches.

        Parameters
        ----------
        min_delay : float, optional
            Minimum delay permitted. Defaults to 10e-4.

        Returns
        -------
        pd.Series
            Effective delay.
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



        # NOTE The issue I am seeing is that this value tau
        # is very very small in most cases. Dividing it by w
        # makes it even smaller.
        # So when it 1/t^2 is calculated, there is an overflow of precision.
        # Therefore here (for now) we will actually just use beta
        # for stability purposes

        # EFFECTIVE DELAY
        tau = beta#/w

        # Enforce lower bound
        tau[tau<min_delay] = min_delay

        # Propagation Parameter
        return tau
    
