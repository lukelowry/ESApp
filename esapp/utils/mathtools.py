from abc import ABC

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import schur

import numpy as np
from numpy import block, diag, real, imag

# Constants
MU0 = 1.256637e-6


def takagi(M):
   """
   Performs the Takagi factorization of a complex symmetric matrix.

   Parameters
   ----------
   M : np.ndarray
       Complex symmetric matrix.

   Returns
   -------
   tuple
       (U, Sigma) where M = U * diag(Sigma) * U^T.
   """
   n = M.shape[0]
   D, P = schur(block([[-real(M),imag(M)],[imag(M),real(M)]]))
   pos = diag(D) > 0
   Sigma = diag(D[pos,pos])
   # Note: The arithmetic below is technically not necessary
   U = P[n:,pos] + 1j*P[:n,pos]
   return U, Sigma.diagonal()



def eigmax(L):
    """
    Finds the largest eigenvalue of a matrix (intended for sparse Laplacians).

    Parameters
    ----------
    L : Union[np.ndarray, sp.spmatrix]
        The input matrix.

    Returns
    -------
    float
        The largest eigenvalue.
    """
    return eigsh(L, k=1, which='LA', return_eigenvectors=False)[0]


def sorteig(Lam, U):
    """
    Sorts eigenvalue decomposition by eigenvalue magnitude (least to greatest).

    Parameters
    ----------
    Lam : np.ndarray
        Eigenvalues.
    U : np.ndarray
        Eigenvectors.

    Returns
    -------
    tuple
        (Sorted Lam, Sorted U).
    """
    idx = np.argsort(np.abs(Lam))
    return Lam[idx], U[:,idx]

# TODO rename to 'pathlap' so periodicity is an option
def periodiclap(N, periodic=True):
    """
    Creates a branchless periodic discrete graph Laplacian.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, optional
        Whether the graph is periodic. Defaults to True.

    Returns
    -------
    np.ndarray
        The Laplacian matrix.
    """

    O = np.ones(N)

    L = sp.diags(
        [2*O, -O[:1], -O[:1]],
        offsets=[0, 1, -1], 
        shape=(N,N)
    ).toarray()

    if periodic:
        L[0, -1] = -1
        L[-1, 0] = -1
    else:
        L[0, 0] = 1
        L[-1, -1] = 1

    return L

def periodicincidence(N, periodic=True):
    """
    Creates a branchless periodic discrete graph incidence matrix.

    Parameters
    ----------
    N : int
        Number of nodes.
    periodic : bool, optional
        Whether the graph is periodic. Defaults to True.

    Returns
    -------
    np.ndarray
        The incidence matrix.
    """

    O = np.ones(N)

    L = sp.diags(
        [O, -O[:1]],
        offsets=[0, 1], 
        shape=(N,N)
    ).toarray()

    if periodic:
        L[-1, 0] = -1

    return L

# Matrix Helper Functions
def normlap(L, retD=False):
    """
    Returns the normalized Laplacian of a square matrix.

    Parameters
    ----------
    L : Union[np.ndarray, sp.spmatrix]
        Input square Laplacian matrix.
    retD : bool, optional
        Whether to return the diagonal scaling matrices. Defaults to False.

    Returns
    -------
    Union[np.ndarray, tuple]
        Normalized Laplacian, or (NormL, D, Di) if retD is True.
    """

    # Get Diagonal and Invert for convenience
    Yd = np.sqrt(L.diagonal())
    Di = sp.diags(1/Yd)

    # Return Normalized Laplacian with or without scaled diag
    if retD:
        D = sp.diags(Yd)
        return Di@L@Di, D, Di
    else:
        return Di@L@Di


def hermitify(A):
    """
    Converts a complex symmetric matrix to a Hermitian matrix.

    Parameters
    ----------
    A : Union[np.ndarray, sp.spmatrix]
        Input complex symmetric matrix.

    Returns
    -------
    np.ndarray
        The Hermitian version of the matrix.
    """

    if isinstance(A, np.ndarray):
        return (np.triu(A).conjugate() + np.tril(A))/2
    else:
        return (np.triu(A.A).conjugate() + np.tril(A.A))/2


class Operator(ABC):
    """Abstract Mathematical Operator Object."""

    def __init__(self) -> None:
        pass
    

class DifferentialOperator(Operator):
    """
    Finite difference operator generator for 2D grids.
    Only Supports 2D Fortran Style Ordering.
    """

    def __init__(self, shape, order='F') -> None:
        """
        Initialize the DifferentialOperator.

        Parameters
        ----------
        shape : tuple
            (nx, ny) dimensions of the grid.
        order : str, optional
            Memory ordering. Defaults to 'F'.
        """
        self.shape = shape
        self.nx, self.ny = shape
        self.nElement = self.nx*self.ny

        self.D = [-1, 1] 

    def newop(self):
        """Create empty operator matrices."""
        return np.zeros((self.nElement, self.nElement)), np.zeros((self.nElement, self.nElement))
    
    def aslil(self, Dx, Dy):
        """Convert to LIL sparse format."""
        return sp.lil_matrix(Dx), sp.lil_matrix(Dy)

    def flatidx(self, x, y):
        """
        Convert 2D coordinates to flat index.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.
        """
        return y*self.nx + x
    
    def flattoloc(self, idx):
        return idx%self.nx, idx//self.nx
    
    def up(self, idx):
        return idx + self.nx
        
    def down(self, idx):
        return idx - self.nx
    
    def right(self, idx):
        return idx + 1
    
    def left(self, idx):
        return idx - 1
    
    def elementiter(self):
        """Iterate through each tensor element to get index and position."""

        for yi in np.arange(self.ny):
            for xi in np.arange(self.nx):
                yield xi, yi, self.flatidx(xi, yi)
        
    def central_diffs(self) -> None:
        """
        Produces central difference gradient operators for a vector field.

        Returns
        -------
        tuple
            (Dx, Dy) sparse matrices.
        """

        Dx, Dy = self.newop()

        for xi, yi, idx in self.elementiter():

            if xi==0 or xi==self.nx-1: continue
            if yi==0 or yi==self.ny-1: continue

            # Selectors
            dx = [ self.left(idx) , self.right(idx) ]
            dy = [ self.down(idx) , self.up(idx)    ]

            Dx[idx , dx] += self.D
            Dy[idx , dy] += self.D

        return self.aslil(Dx/2, Dy/2)


    def forward_diffs(self) -> None:
        """
        Produces forward difference gradient operators for a vector field.

        Returns
        -------
        tuple
            (Dx, Dy) sparse matrices.
        """

        Dx, Dy = self.newop()

        for xi, yi, idx in self.elementiter():

            # Selectors
            dx = [idx , self.right(idx)]
            dy = [idx , self.up(idx)]

            # Add Y Differential to Tile
            if xi < self.nx-1:
                Dx[idx , dx] += self.D

            # Add to Adjacent Tiles
            if yi < self.ny-1:
                Dy[idx , dy] += self.D

        return self.aslil(Dx, Dy)
    
    def backward_diffs(self) -> None:
        """
        Produces backward difference gradient operators for a vector field.

        Returns
        -------
        tuple
            (Dx, Dy) sparse matrices.
        """

        Dx, Dy = self.newop()

        for xi, yi, idx in self.elementiter():

            # Selectors
            dx = [idx, self.left(idx)]
            dy = [idx, self.down(idx)]

            if xi != 0:
                Dx[idx , dx] += self.D

            if yi != 0:
                Dy[idx , dy] += self.D

        return self.aslil(Dx, Dy)
    
    def partial(self):
        """
        Return centered partial operators for a 2D vector field tensor.

        Returns
        -------
        tuple
            (Dx, Dy) sparse matrices.
        """

        Dxf, Dyf = self.forward_diffs()
        Dxb, Dyb = self.backward_diffs()

        return Dxb - Dxf, Dyb - Dyf
    
    def divergence(self):
        """
        Central Difference Based Finite Divergence.

        Returns
        -------
        sp.spmatrix
            Divergence operator.
        """

        Dx, Dy = self.partial()
        return sp.hstack([Dx, Dy])
    
    def curl(self):
        """
        Central Difference Based Finite Curl.

        Returns
        -------
        sp.spmatrix
            Curl operator.
        """
    
        Dx, Dy = self.partial()
        return sp.hstack([Dy, -Dx])
    
    def laplacian(self):
        """
        Central Difference Based Discrete Laplacian.

        Returns
        -------
        sp.spmatrix
            Laplacian operator.
        """

        Dxf, Dyf = self.forward_diffs()
        return Dxf.T@Dxf + Dyf.T@Dyf
    
    def J(self):
        """Complex Unit Equivilent and/or hodge star."""

        n = self.nElement
        I = sp.eye(n)
        return sp.bmat([
            [None, -I  ],
            [I   , None]
        ])
    
    def ext_der(self):
        """Calculate exterior derivative of linear function/operator."""
        # Used outside class up top
        pass

    


class MeshSelector:
    """Helper for selecting regions of a 2D mesh."""

    def __init__(self, dop: DifferentialOperator) -> None:
        """
        Initialize the MeshSelector.

        Parameters
        ----------
        dop : DifferentialOperator
            The operator defining the mesh dimensions.
        """

        self.SELECTOR = np.full(dop.nElement, False)

        nsel = lambda n: (self.SELECTOR.copy() for i in range(n))

        # Sides Including Corners
        self.LEFT, self.RIGHT, self.UP, self.DOWN = nsel(4)

        # Primary Indexing
        for xi, yi, idx in dop.elementiter(): 
            self.LEFT[idx] =  (xi==0)
            self.RIGHT[idx] =  (xi==dop.nx-1)
            self.UP[idx] =  (yi==dop.ny-1)
            self.DOWN[idx] =  (yi==0)

            # CENTRAL2[idx] = ~((xi==1) or (yi==1) or (xi==dop.nx-2) or (yi==dop.ny-2))

        # Secondary Indexing
        self.ALLCRNR = (self.LEFT|self.RIGHT)&(self.UP|self.DOWN)
        self.BOUND = self.LEFT|self.RIGHT|self.UP|self.DOWN
        self.CENTRAL = ~self.BOUND 


        # TODO Generic versions of above
