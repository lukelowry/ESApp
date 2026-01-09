from ..indexable import Indexable

import numpy as np

# Constructing Network Matricies and other metrics
class ForcedOscillation(Indexable):

    # Need to make a DEF folder

    def sgwt_def(self, WS):
        '''
        Description
            Performs DEF integration on SGWT coefficient signal
        Parameters
            WS: Complex Wavelet Coefficients (Buses x Time x Scale)
        Returns
            Ed: Real-Valued Integrated DEF (Buses x Time x Scale)
        '''

        # Time Derivative
        dS = np.diff(WS,axis=1,n=1)

        # Squared Magnitude of coefficients
        dS = np.abs(dS)**2

        # Integrate over time
        Ediss = np.cumsum(dS, axis=1)
        
        return Ediss
    
    def orthodox_def(self):
        
        pass

    
