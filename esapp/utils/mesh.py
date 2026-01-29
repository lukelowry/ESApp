"""
Discrete geometry utilities for meshes and structured grids.

This module provides tools for working with:
- Unstructured meshes (PLY file I/O, graph operations)
- Structured 2D grids (finite difference operators)

Both mesh types support computing incidence matrices, Laplacians, and
other discrete differential operators.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

__all__ = [
    'Mesh',
    'extract_unique_edges',
    'Grid2D',
    'GridSelector',
]


# =============================================================================
# Unstructured Mesh Utilities
# =============================================================================

def extract_unique_edges(faces: list[list[int]]) -> NDArray[np.int_]:
    """
    Extract unique edges from a list of mesh faces.

    Each face is a list of vertex indices forming a polygon. Edges are
    extracted by connecting consecutive vertices (including last to first).
    Duplicate edges are removed, and each edge is stored with the smaller
    vertex index first.

    Parameters
    ----------
    faces : list of list of int
        Mesh faces, where each face is a list of vertex indices.

    Returns
    -------
    np.ndarray
        An (M, 2) array of unique edges, sorted lexicographically.
        Column 0 contains the smaller vertex index for each edge.

    Examples
    --------
    >>> faces = [[0, 1, 2], [1, 2, 3]]
    >>> extract_unique_edges(faces)
    array([[0, 1],
           [0, 2],
           [1, 2],
           [1, 3],
           [2, 3]])
    """
    unique_edges = set()

    for face in faces:
        n = len(face)
        for i in range(n):
            u = face[i]
            v = face[(i + 1) % n]
            edge = (u, v) if u < v else (v, u)
            unique_edges.add(edge)

    return np.array(sorted(unique_edges), dtype=np.int_)


@dataclass
class Mesh:
    """
    A 3D mesh consisting of vertices and polygonal faces.

    This class represents an unstructured mesh and provides methods for
    loading from PLY files and computing graph-theoretic properties like
    incidence matrices and Laplacians.

    Attributes
    ----------
    vertices : list of tuple
        List of (x, y, z) vertex coordinates.
    faces : list of list of int
        List of faces, where each face is a list of vertex indices.

    Examples
    --------
    >>> mesh = Mesh.from_ply("model.ply")
    >>> L = mesh.to_laplacian()
    >>> xyz = mesh.get_xyz()
    """
    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]

    @classmethod
    def from_ply(cls, filepath: str) -> "Mesh":
        """
        Load a mesh from a PLY file.

        Supports both ASCII and binary_little_endian PLY formats.
        Extracts vertex positions (x, y, z) and face connectivity.

        Parameters
        ----------
        filepath : str
            Path to the PLY file.

        Returns
        -------
        Mesh
            The loaded mesh.

        Raises
        ------
        ValueError
            If the PLY format is not supported.

        Notes
        -----
        Supported vertex property types: char, uchar, short, ushort,
        int, uint, float, double.
        """
        import struct

        with open(filepath, 'rb') as f:
            # Parse header
            header_ended = False
            fmt = "ascii"
            vertex_count = 0
            face_count = 0
            vertex_props = []
            current_element = None

            while not header_ended:
                line = f.readline().strip()
                if not line:
                    break
                line_str = line.decode('ascii', errors='ignore')

                if line_str == "end_header":
                    header_ended = True
                    break

                parts = line_str.split()
                if not parts:
                    continue

                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    current_element = parts[1]
                    if current_element == "vertex":
                        vertex_count = int(parts[2])
                    elif current_element == "face":
                        face_count = int(parts[2])
                elif parts[0] == "property":
                    if current_element == "vertex":
                        vertex_props.append((parts[2], parts[1]))

            # Parse body
            vertices = []
            faces = []

            if fmt == "ascii":
                lines = f.readlines()
                for i in range(vertex_count):
                    parts = lines[i].strip().split()
                    v = (float(parts[0]), float(parts[1]), float(parts[2]))
                    vertices.append(v)

                for i in range(face_count):
                    parts = lines[vertex_count + i].strip().split()
                    vertex_indices = [int(x) for x in parts[1:]]
                    faces.append(vertex_indices)

            elif fmt == "binary_little_endian":
                np_type_map = {
                    'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
                    'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8'
                }
                dtype_fields = [
                    (name, np_type_map.get(type_str, 'f4'))
                    for name, type_str in vertex_props
                ]
                vertex_dtype = np.dtype(dtype_fields)

                vertex_data = f.read(vertex_count * vertex_dtype.itemsize)
                v_arr = np.frombuffer(vertex_data, dtype=vertex_dtype)

                if all(n in v_arr.dtype.names for n in ('x', 'y', 'z')):
                    vertices = list(zip(v_arr['x'], v_arr['y'], v_arr['z']))
                else:
                    names = v_arr.dtype.names
                    vertices = list(zip(
                        v_arr[names[0]], v_arr[names[1]], v_arr[names[2]]
                    ))

                for _ in range(face_count):
                    n = struct.unpack('<B', f.read(1))[0]
                    indices = list(struct.unpack(f'<{n}i', f.read(n * 4)))
                    faces.append(indices)
            else:
                raise ValueError(f"Unsupported PLY format: {fmt}")

        return cls(vertices=vertices, faces=faces)

    def get_incidence_matrix(self) -> csc_matrix:
        """
        Construct the oriented incidence matrix for the mesh graph.

        The incidence matrix B has shape (|V|, |E|) where each column
        represents an edge with +1 at the source vertex and -1 at the
        target vertex.

        Returns
        -------
        scipy.sparse.csc_matrix
            The incidence matrix B.

        Notes
        -----
        The edge orientation is determined by vertex index ordering
        (smaller index is source, larger is target).
        """
        edges = extract_unique_edges(self.faces)
        num_verts = len(self.vertices)
        num_edges = len(edges)

        row = edges.ravel()
        col = np.repeat(np.arange(num_edges), 2)
        data = np.tile([1.0, -1.0], num_edges)

        return csc_matrix((data, (row, col)), shape=(num_verts, num_edges))

    def get_xyz(self) -> NDArray[np.float64]:
        """
        Get vertex coordinates as a numpy array.

        Returns
        -------
        np.ndarray
            An (N, 3) array of vertex coordinates.
        """
        return np.array(self.vertices, dtype=np.float64)

    def to_laplacian(self) -> csc_matrix:
        """
        Compute the graph Laplacian matrix.

        The Laplacian is computed as L = B @ B.T where B is the
        incidence matrix. This produces the combinatorial Laplacian
        with diagonal entries equal to vertex degrees.

        Returns
        -------
        scipy.sparse.csc_matrix
            The graph Laplacian matrix L.
        """
        B = self.get_incidence_matrix()
        return B @ B.T


# =============================================================================
# Structured Grid Utilities
# =============================================================================

class Grid2D:
    """
    Finite difference operator generator for regular 2D grids.

    Generates sparse matrix operators for gradient, divergence, curl,
    and Laplacian on a rectangular grid with Fortran-style (column-major)
    indexing.

    Parameters
    ----------
    shape : tuple of int
        Grid dimensions (nx, ny).

    Attributes
    ----------
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.
    size : int
        Total number of grid points (nx * ny).

    Examples
    --------
    >>> grid = Grid2D((10, 10))
    >>> Dx, Dy = grid.gradient()
    >>> L = grid.laplacian()
    >>> div = grid.divergence()

    Notes
    -----
    Grid points are indexed in Fortran order (column-major), so point
    (x, y) maps to flat index y * nx + x. This matches numpy's 'F' order.
    """

    def __init__(self, shape: tuple[int, int]) -> None:
        self.nx, self.ny = shape
        self.size = self.nx * self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions (nx, ny)."""
        return (self.nx, self.ny)

    def flat_index(self, x: int, y: int) -> int:
        """
        Convert 2D coordinates to flat index.

        Parameters
        ----------
        x : int
            X coordinate (0 to nx-1).
        y : int
            Y coordinate (0 to ny-1).

        Returns
        -------
        int
            Flat index in column-major order.
        """
        return y * self.nx + x

    def grid_coords(self, idx: int) -> tuple[int, int]:
        """
        Convert flat index to 2D coordinates.

        Parameters
        ----------
        idx : int
            Flat index.

        Returns
        -------
        tuple of int
            (x, y) coordinates.
        """
        return idx % self.nx, idx // self.nx

    def iter_points(self) -> Iterator[tuple[int, int, int]]:
        """
        Iterate over all grid points.

        Yields
        ------
        tuple of int
            (x, y, flat_index) for each grid point in column-major order.
        """
        for y in range(self.ny):
            for x in range(self.nx):
                yield x, y, self.flat_index(x, y)

    def _build_gradient_x(self, scheme: str = 'forward') -> csr_matrix:
        """Build x-direction gradient operator."""
        n = self.size
        rows, cols, data = [], [], []

        for x, y, idx in self.iter_points():
            if scheme == 'forward':
                if x < self.nx - 1:
                    rows.extend([idx, idx])
                    cols.extend([idx, idx + 1])
                    data.extend([-1.0, 1.0])
            elif scheme == 'backward':
                if x > 0:
                    rows.extend([idx, idx])
                    cols.extend([idx - 1, idx])
                    data.extend([-1.0, 1.0])
            elif scheme == 'central':
                if 0 < x < self.nx - 1:
                    rows.extend([idx, idx])
                    cols.extend([idx - 1, idx + 1])
                    data.extend([-0.5, 0.5])

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def _build_gradient_y(self, scheme: str = 'forward') -> csr_matrix:
        """Build y-direction gradient operator."""
        n = self.size
        rows, cols, data = [], [], []
        step = self.nx  # y-direction step in flat indexing

        for x, y, idx in self.iter_points():
            if scheme == 'forward':
                if y < self.ny - 1:
                    rows.extend([idx, idx])
                    cols.extend([idx, idx + step])
                    data.extend([-1.0, 1.0])
            elif scheme == 'backward':
                if y > 0:
                    rows.extend([idx, idx])
                    cols.extend([idx - step, idx])
                    data.extend([-1.0, 1.0])
            elif scheme == 'central':
                if 0 < y < self.ny - 1:
                    rows.extend([idx, idx])
                    cols.extend([idx - step, idx + step])
                    data.extend([-0.5, 0.5])

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def gradient(self, scheme: str = 'forward') -> tuple[csr_matrix, csr_matrix]:
        """
        Build gradient operators for a scalar field.

        Parameters
        ----------
        scheme : {'forward', 'backward', 'central'}
            Finite difference scheme to use.

        Returns
        -------
        Dx : scipy.sparse.csr_matrix
            X-direction gradient operator.
        Dy : scipy.sparse.csr_matrix
            Y-direction gradient operator.

        Notes
        -----
        For a scalar field u, the gradient is (Dx @ u, Dy @ u).

        Forward differences use: du/dx ≈ u[i+1] - u[i]
        Backward differences use: du/dx ≈ u[i] - u[i-1]
        Central differences use: du/dx ≈ (u[i+1] - u[i-1]) / 2
        """
        return self._build_gradient_x(scheme), self._build_gradient_y(scheme)

    def divergence(self) -> csr_matrix:
        """
        Build divergence operator for a vector field.

        Returns
        -------
        scipy.sparse.csr_matrix
            Divergence operator of shape (n, 2n).

        Notes
        -----
        For a vector field (u, v) stored as a stacked vector [u; v],
        the divergence is: div(u, v) = du/dx + dv/dy

        Uses centered differences (backward - forward).
        """
        Dxf, Dyf = self.gradient('forward')
        Dxb, Dyb = self.gradient('backward')
        Dx = Dxb - Dxf
        Dy = Dyb - Dyf
        return sp.hstack([Dx, Dy], format='csr')

    def curl(self) -> csr_matrix:
        """
        Build 2D curl operator for a vector field.

        Returns
        -------
        scipy.sparse.csr_matrix
            Curl operator of shape (n, 2n).

        Notes
        -----
        For a 2D vector field (u, v), the curl gives the scalar:
        curl(u, v) = dv/dx - du/dy

        Uses centered differences (backward - forward).
        """
        Dxf, Dyf = self.gradient('forward')
        Dxb, Dyb = self.gradient('backward')
        Dx = Dxb - Dxf
        Dy = Dyb - Dyf
        return sp.hstack([Dy, -Dx], format='csr')

    def laplacian(self) -> csr_matrix:
        """
        Build the discrete Laplacian operator.

        Returns
        -------
        scipy.sparse.csr_matrix
            Laplacian operator of shape (n, n).

        Notes
        -----
        Computed as L = Dx.T @ Dx + Dy.T @ Dy using forward differences,
        which gives the standard 5-point stencil for interior points.
        """
        Dx, Dy = self.gradient('forward')
        return Dx.T @ Dx + Dy.T @ Dy

    def hodge_star(self) -> csr_matrix:
        """
        Build the 2D Hodge star operator.

        The Hodge star rotates vectors by 90 degrees, equivalent to
        multiplication by the imaginary unit in the complex plane.

        Returns
        -------
        scipy.sparse.csr_matrix
            Hodge star operator of shape (2n, 2n).

        Notes
        -----
        For a vector field [u; v], returns [-v; u].
        """
        n = self.size
        I = sp.eye(n, format='csr')
        Z = csr_matrix((n, n))
        return sp.bmat([[Z, -I], [I, Z]], format='csr')


class GridSelector:
    """
    Boolean masks for selecting regions of a structured grid.

    Provides masks for boundary and interior point selection,
    useful for applying boundary conditions in finite difference schemes.

    Parameters
    ----------
    grid : Grid2D
        The grid to create selectors for.

    Attributes
    ----------
    left : np.ndarray
        Boolean mask for left boundary (x = 0).
    right : np.ndarray
        Boolean mask for right boundary (x = nx-1).
    bottom : np.ndarray
        Boolean mask for bottom boundary (y = 0).
    top : np.ndarray
        Boolean mask for top boundary (y = ny-1).
    corners : np.ndarray
        Boolean mask for corner points.
    boundary : np.ndarray
        Boolean mask for all boundary points.
    interior : np.ndarray
        Boolean mask for interior (non-boundary) points.

    Examples
    --------
    >>> grid = Grid2D((10, 10))
    >>> sel = GridSelector(grid)
    >>> u[sel.boundary] = 0  # Apply Dirichlet BC
    >>> interior_values = u[sel.interior]
    """

    def __init__(self, grid: Grid2D) -> None:
        n = grid.size
        self.left = np.zeros(n, dtype=bool)
        self.right = np.zeros(n, dtype=bool)
        self.bottom = np.zeros(n, dtype=bool)
        self.top = np.zeros(n, dtype=bool)

        for x, y, idx in grid.iter_points():
            self.left[idx] = (x == 0)
            self.right[idx] = (x == grid.nx - 1)
            self.bottom[idx] = (y == 0)
            self.top[idx] = (y == grid.ny - 1)

        self.corners = (self.left | self.right) & (self.bottom | self.top)
        self.boundary = self.left | self.right | self.bottom | self.top
        self.interior = ~self.boundary


# Backwards compatibility aliases
DifferentialOperator = Grid2D
MeshSelector = GridSelector
