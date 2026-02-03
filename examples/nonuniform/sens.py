"""
GIC sensitivity analysis for non-uniform electric fields.

Provides standalone functions for computing:
- Interface flow sensitivity to transformer GIC currents (dBound/dI)
- E-field to GIC Jacobian (dI/dE)

These functions operate on matrices produced by ``PowerWorld.gic.model()``
and require a live ``PowerWorld`` instance for bus category data.

Example
-------
>>> from esapp import PowerWorld
>>> from esapp.utils import GIC, jac_decomp
>>> from examples.nonuniform.sens import dBounddI, dIdE
>>>
>>> pw = PowerWorld("case.pwb")
>>> pw.gic.model()
>>> H = pw.gic.H
>>> J = pw.jacobian(dense=True)
>>> V = pw.voltage(complex=False)[0].to_numpy()
>>> eta = ...  # injection vector
>>> PX = pw.gic.Px
>>> sens = dBounddI(pw, eta, PX, J, V)
"""

import numpy as np
from scipy.sparse import hstack
from scipy.sparse.linalg import inv as sinv

from esapp.components import Bus
from esapp.utils.gic import jac_decomp

__all__ = ['dBounddI', 'dIdE', 'signdiag']


def signdiag(x):
    """
    Create diagonal matrix of signs.

    Parameters
    ----------
    x : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Diagonal matrix with sign(x) on diagonal.
    """
    return np.diagflat(np.sign(x))


def dBounddI(pw, eta, PX, J, V):
    """
    Compute interface sensitivity with respect to transformer GIC currents.

    Parameters
    ----------
    pw : PowerWorld
        Live PowerWorld instance (used to retrieve bus categories).
    eta : np.ndarray
        Injection vector (n x 1).
    PX : np.ndarray or sparse matrix
        Transformer to loaded-bus mapping (n x m).
    J : np.ndarray
        Full AC power flow Jacobian at boundary.
    V : np.ndarray
        Bus voltage magnitudes (n x 1).

    Returns
    -------
    np.ndarray
        Sensitivity vector (1 x n).
    """
    buscat = pw[Bus, ['BusCat']]['BusCat']
    slk = buscat == 'Slack'
    pv = buscat == 'PV'
    pq = ~(slk | pv)

    dPdT, dPdV, dQdT, dQdV = jac_decomp(J)

    A = hstack([dPdT[:, ~slk], dPdV[:, pq]])
    B = hstack([dQdT[pq][:, ~slk], dQdV[pq][:, pq]])

    Vdiag = np.diagflat(V[pq])

    return (1 / (eta.T @ eta)) @ eta.T @ A @ B.T @ sinv((B @ B.T).tocsc()) @ Vdiag @ PX[pq]


def dIdE(H, E=None, i=None):
    """
    Compute Jacobian between mesh E-field and absolute transformer GICs.

    Parameters
    ----------
    H : np.ndarray or sparse matrix
        H-matrix (e.g., from ``wb.gic.H`` after calling ``model()``).
    E : np.ndarray, optional
        Electric field vector. If provided and i is None, computes i = H @ E.
    i : np.ndarray, optional
        Signed neutral transformer currents. Required if E is not provided.

    Returns
    -------
    np.ndarray
        Jacobian matrix (rows: transformers, cols: E-field components).

    Raises
    ------
    ValueError
        If neither E nor i is provided.
    """
    if E is not None:
        if i is None:
            i = H @ E
    elif i is None:
        raise ValueError("Either E or i must be provided")

    F = signdiag(i)
    return F @ H
