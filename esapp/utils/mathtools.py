"""
Mathematical utilities for linear algebra and spectral analysis.

This module provides functions for matrix decomposition, eigenvalue
analysis, graph Laplacian construction, and matrix transformations
commonly used in power systems and signal processing applications.
"""

import numpy as np
from numpy import block, diag, real, imag
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import schur
from typing import Union

__all__ = [
    'MU0',
    'takagi',
    'eigmax',
    'sorteig',
    'periodiclap',
    'pathlap',
    'periodicincidence',
    'pathincidence',
    'normlap',
    'hermitify',
]

# =============================================================================
# Physical Constants
# =============================================================================

MU0: float = 1.256637e-6
"""Permeability of free space (H/m)."""


# =============================================================================
# Matrix Decomposition
# =============================================================================

def takagi(M: NDArray[np.complexfloating]) -> tuple[NDArray, NDArray]:
    """
    Perform Takagi factorization of a complex symmetric matrix.

    For a complex symmetric matrix M (where M = M^T, not M = M^H),
    the Takagi factorization finds a unitary matrix U and non-negative
    real diagonal values such that M = U @ diag(sigma) @ U^T.

    Parameters
    ----------
    M : np.ndarray
        Complex symmetric matrix of shape (n, n).

    Returns
    -------
    U : np.ndarray
        Unitary matrix of shape (n, n).
    sigma : np.ndarray
        Non-negative singular values of shape (n,).

    Notes
    -----
    This implementation uses the real Schur decomposition of an
    augmented real matrix to compute the factorization.

    References
    ----------
    .. [1] Takagi, T. (1925). "On an algebraic problem related to an
           analytic theorem of Carathéodory and Fejér".
    """
    n = M.shape[0]
    augmented = block([
        [-real(M), imag(M)],
        [imag(M), real(M)]
    ])
    D, P = schur(augmented)
    pos = diag(D) > 0
    sigma = diag(D[pos, pos])
    U = P[n:, pos] + 1j * P[:n, pos]
    return U, sigma.diagonal()


# =============================================================================
# Eigenvalue Analysis
# =============================================================================

def eigmax(L: Union[NDArray, sp.spmatrix]) -> float:
    """
    Find the largest eigenvalue of a matrix.

    Optimized for sparse symmetric matrices using ARPACK.

    Parameters
    ----------
    L : np.ndarray or scipy.sparse matrix
        Input matrix (should be symmetric for meaningful results).

    Returns
    -------
    float
        The largest eigenvalue.

    Notes
    -----
    Uses scipy.sparse.linalg.eigsh with 'LA' (largest algebraic)
    selection, which is efficient for sparse matrices.
    """
    return eigsh(L, k=1, which='LA', return_eigenvectors=False)[0]


def sorteig(
    eigenvalues: NDArray,
    eigenvectors: NDArray
) -> tuple[NDArray, NDArray]:
    """
    Sort eigenvalue decomposition by magnitude.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues.
    eigenvectors : np.ndarray
        Matrix of eigenvectors (columns).

    Returns
    -------
    sorted_eigenvalues : np.ndarray
        Eigenvalues sorted by absolute value (ascending).
    sorted_eigenvectors : np.ndarray
        Corresponding eigenvectors.
    """
    idx = np.argsort(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]


# =============================================================================
# Graph Laplacian Construction
# =============================================================================

def pathlap(N: int, periodic: bool = False) -> NDArray:
    """
    Create the graph Laplacian for a path or cycle graph.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, default False
        If True, creates a cycle graph (first and last nodes connected).
        If False, creates a path graph.

    Returns
    -------
    np.ndarray
        The Laplacian matrix of shape (N, N).

    Notes
    -----
    - For a path graph: L[i,i] = 2 for interior nodes, 1 for endpoints.
    - For a cycle graph: L[i,i] = 2 for all nodes.
    - Off-diagonal entries are -1 for adjacent nodes.

    See Also
    --------
    periodiclap : Alias with periodic=True default.
    """
    O = np.ones(N)
    L = sp.diags(
        [2 * O, -O[:1], -O[:1]],
        offsets=[0, 1, -1],
        shape=(N, N)
    ).toarray()

    if periodic:
        L[0, -1] = -1
        L[-1, 0] = -1
    else:
        L[0, 0] = 1
        L[-1, -1] = 1

    return L


def periodiclap(N: int, periodic: bool = True) -> NDArray:
    """
    Create a periodic discrete graph Laplacian.

    Alias for pathlap with periodic=True as default.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, default True
        Whether the graph is periodic (cycle) or not (path).

    Returns
    -------
    np.ndarray
        The Laplacian matrix.

    See Also
    --------
    pathlap : Primary implementation.
    """
    return pathlap(N, periodic=periodic)


def pathincidence(N: int, periodic: bool = False) -> NDArray:
    """
    Create the incidence matrix for a path or cycle graph.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, default False
        If True, creates a cycle graph incidence matrix.
        If False, creates a path graph incidence matrix.

    Returns
    -------
    np.ndarray
        The incidence matrix.

    Notes
    -----
    For a path graph: shape is (N, N-1) with N-1 edges.
    For a cycle graph: shape is (N, N) with N edges.
    Each column has +1 at source node and -1 at target node.

    See Also
    --------
    periodicincidence : Alias with periodic=True default.
    """
    O = np.ones(N)
    B = sp.diags(
        [O, -O[:1]],
        offsets=[0, 1],
        shape=(N, N)
    ).toarray()

    if periodic:
        B[-1, 0] = -1

    return B


def periodicincidence(N: int, periodic: bool = True) -> NDArray:
    """
    Create a periodic discrete graph incidence matrix.

    Alias for pathincidence with periodic=True as default.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, default True
        Whether the graph is periodic.

    Returns
    -------
    np.ndarray
        The incidence matrix.

    See Also
    --------
    pathincidence : Primary implementation.
    """
    return pathincidence(N, periodic=periodic)


# =============================================================================
# Matrix Transformations
# =============================================================================

def normlap(
    L: Union[NDArray, sp.spmatrix],
    return_scaling: bool = False
) -> Union[NDArray, tuple[NDArray, sp.dia_matrix, sp.dia_matrix]]:
    """
    Compute the normalized Laplacian of a matrix.

    The normalized Laplacian is defined as:
        L_norm = D^{-1/2} @ L @ D^{-1/2}

    where D is the diagonal matrix of L's diagonal entries.

    Parameters
    ----------
    L : np.ndarray or scipy.sparse matrix
        Input Laplacian matrix.
    return_scaling : bool, default False
        If True, also return the scaling matrices.

    Returns
    -------
    L_norm : np.ndarray
        The normalized Laplacian.
    D : scipy.sparse.dia_matrix, optional
        Diagonal scaling matrix (sqrt of original diagonal).
        Only returned if return_scaling=True.
    D_inv : scipy.sparse.dia_matrix, optional
        Inverse diagonal scaling matrix.
        Only returned if return_scaling=True.

    Notes
    -----
    The normalized Laplacian has eigenvalues in [0, 2] for
    undirected graphs and is useful for spectral clustering.
    """
    Yd = np.sqrt(L.diagonal())
    Di = sp.diags(1 / Yd)

    if return_scaling:
        D = sp.diags(Yd)
        return Di @ L @ Di, D, Di
    else:
        return Di @ L @ Di


def hermitify(A: Union[NDArray, sp.spmatrix]) -> NDArray:
    """
    Convert a complex symmetric matrix to Hermitian form.

    For a complex symmetric matrix (A = A^T), this function produces
    a Hermitian matrix (A_H = A_H^H) by taking the average of
    conjugate transposes.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix
        Input complex symmetric matrix.

    Returns
    -------
    np.ndarray
        The Hermitian form of the matrix.

    Notes
    -----
    Useful for converting admittance matrices to a form suitable
    for eigenvalue algorithms that require Hermitian input.
    """
    if isinstance(A, np.ndarray):
        return (np.triu(A).conjugate() + np.tril(A)) / 2
    else:
        return (np.triu(A.A).conjugate() + np.tril(A.A)) / 2
