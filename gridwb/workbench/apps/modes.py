

# WorkBench Imports
from ..io.powerworld import PowerWorldIO
from .app import PWApp
from ..io import Context

import numpy as np

# Constructing Network Matricies and other metrics
class ForcedOscillation(PWApp):

    io: PowerWorldIO

    def __init__(self, context: Context) -> None:
        super().__init__(context)

           # TEMPORARY

    # Need to make a DEF folder

    def sgwt_def(self, WS):
        
        # OMEGA
        omega = (WS[:,2:] - WS[:,:-2])/WS[:,1:-1]

        # VELOCITY dw/dk
        Vel = omega[:,:,2:] - omega[:,:,:-2]

        # Acceleration
        Acc = Vel * omega[:,:,1:-1]

        # Differential Power
        dS = WS[:,1:-1,1:-1]*Acc

        # Integrate
        S = np.cumsum(dS, axis=1)
        P = S.real
        Q = S.imag
        S = np.abs(S)

        return P, Q, S
    
    def orthodox_def(self):
        
        pass

    
