"""
Unit tests for the Grid2D class in esapp.utils.mesh.

Tests verify the incidence-matrix-based construction of discrete
operators (gradient, divergence, curl, Laplacian) and boundary
region masks on structured 2D grids.

USAGE:
    pytest tests/test_grid2d.py -v
"""
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from esapp.utils.mesh import Grid2D


# =============================================================================
# Grid construction and basic properties
# =============================================================================

class TestGrid2DConstruction:
    """Tests for Grid2D initialization and basic properties."""

    def test_shape_and_size(self):
        """Grid2D stores correct dimensions."""
        grid = Grid2D((5, 4))
        assert grid.nx == 5
        assert grid.ny == 4
        assert grid.size == 20
        assert grid.shape == (5, 4)

    def test_edge_counts(self):
        """Grid2D computes correct edge counts."""
        grid = Grid2D((5, 4))
        # Horizontal edges: (nx-1) * ny = 4 * 4 = 16
        assert grid.n_edges_x == 16
        # Vertical edges: nx * (ny-1) = 5 * 3 = 15
        assert grid.n_edges_y == 15
        assert grid.n_edges == 31

    def test_flat_index_roundtrip(self):
        """flat_index and grid_coords are inverses."""
        grid = Grid2D((7, 5))
        for x in range(grid.nx):
            for y in range(grid.ny):
                idx = grid.flat_index(x, y)
                assert grid.grid_coords(idx) == (x, y)

    def test_iter_points_covers_all(self):
        """iter_points visits every grid point exactly once."""
        grid = Grid2D((4, 3))
        points = list(grid.iter_points())
        assert len(points) == 12
        indices = [p[2] for p in points]
        assert sorted(indices) == list(range(12))

    def test_minimal_grid(self):
        """A 2x2 grid has correct structure."""
        grid = Grid2D((2, 2))
        assert grid.size == 4
        assert grid.n_edges_x == 2   # 1 * 2
        assert grid.n_edges_y == 2   # 2 * 1
        assert grid.n_edges == 4

    def test_single_row(self):
        """A 1D horizontal grid (nx > 1, ny = 1)."""
        grid = Grid2D((5, 1))
        assert grid.n_edges_x == 4
        assert grid.n_edges_y == 0
        assert grid.n_edges == 4

    def test_single_column(self):
        """A 1D vertical grid (nx = 1, ny > 1)."""
        grid = Grid2D((1, 5))
        assert grid.n_edges_x == 0
        assert grid.n_edges_y == 4
        assert grid.n_edges == 4


# =============================================================================
# Incidence matrix
# =============================================================================

class TestIncidenceMatrix:
    """Tests for the incidence matrix construction."""

    def test_shape(self):
        """Incidence matrix has correct shape (n_edges, n_nodes)."""
        grid = Grid2D((5, 4))
        A = grid.incidence()
        assert A.shape == (31, 20)

    def test_row_sum_zero(self):
        """Each row of A sums to zero (one +1 and one -1)."""
        grid = Grid2D((5, 4))
        A = grid.incidence()
        row_sums = np.array(A.sum(axis=1)).ravel()
        assert_allclose(row_sums, 0.0)

    def test_two_nonzeros_per_row(self):
        """Each row has exactly two nonzero entries."""
        grid = Grid2D((5, 4))
        A = grid.incidence()
        nnz_per_row = np.diff(A.tocsr().indptr)
        assert np.all(nnz_per_row == 2)

    def test_values_are_pm1(self):
        """Incidence matrix entries are only -1 and +1."""
        grid = Grid2D((4, 3))
        A = grid.incidence()
        vals = np.unique(A.data)
        assert_allclose(sorted(vals), [-1.0, 1.0])

    def test_small_grid_manual(self):
        """Verify incidence matrix for a 3x2 grid by hand.

        Node layout (flat index):
            3  4  5      (y=1)
            0  1  2      (y=0)

        Horizontal edges (source → target):
            0: 0→1, 1: 1→2, 2: 3→4, 3: 4→5
        Vertical edges:
            4: 0→3, 5: 1→4, 6: 2→5
        """
        grid = Grid2D((3, 2))
        A = grid.incidence().toarray()

        assert A.shape == (7, 6)

        # Horizontal edge 0→1
        assert A[0, 0] == -1
        assert A[0, 1] == 1

        # Horizontal edge 1→2
        assert A[1, 1] == -1
        assert A[1, 2] == 1

        # Vertical edge 0→3
        assert A[4, 0] == -1
        assert A[4, 3] == 1

    def test_ATA_equals_graph_laplacian(self):
        """A^T A gives the combinatorial graph Laplacian."""
        grid = Grid2D((4, 3))
        A = grid.incidence()
        L = (A.T @ A).toarray()

        # Diagonal = node degree
        # Corner nodes: degree 2
        assert L[0, 0] == 2   # (0,0) corner
        # Edge nodes (non-corner boundary): degree 3
        assert L[1, 1] == 3   # (1,0) bottom edge
        # Interior nodes: degree 4
        assert L[5, 5] == 4   # (1,1) interior

        # Off-diagonal: -1 for adjacent nodes, 0 otherwise
        assert L[0, 1] == -1  # adjacent
        assert L[0, 4] == -1  # adjacent vertically
        assert L[0, 5] == 0   # not adjacent


# =============================================================================
# Region masks (formerly GridSelector)
# =============================================================================

class TestRegionMasks:
    """Tests for boundary and interior masks."""

    def test_boundary_interior_partition(self):
        """boundary and interior are complementary and cover all nodes."""
        grid = Grid2D((5, 4))
        assert np.all(grid.boundary | grid.interior)
        assert not np.any(grid.boundary & grid.interior)

    def test_boundary_count(self):
        """Boundary has 2*(nx + ny) - 4 nodes."""
        grid = Grid2D((5, 4))
        expected = 2 * (5 + 4) - 4
        assert grid.boundary.sum() == expected

    def test_interior_count(self):
        """Interior has (nx-2)*(ny-2) nodes."""
        grid = Grid2D((5, 4))
        expected = 3 * 2
        assert grid.interior.sum() == expected

    def test_left_mask(self):
        """Left mask selects x=0 nodes."""
        grid = Grid2D((4, 3))
        left_indices = np.where(grid.left)[0]
        for idx in left_indices:
            x, _ = grid.grid_coords(idx)
            assert x == 0
        assert grid.left.sum() == grid.ny

    def test_right_mask(self):
        """Right mask selects x=nx-1 nodes."""
        grid = Grid2D((4, 3))
        right_indices = np.where(grid.right)[0]
        for idx in right_indices:
            x, _ = grid.grid_coords(idx)
            assert x == grid.nx - 1
        assert grid.right.sum() == grid.ny

    def test_bottom_mask(self):
        """Bottom mask selects y=0 nodes."""
        grid = Grid2D((4, 3))
        bottom_indices = np.where(grid.bottom)[0]
        for idx in bottom_indices:
            _, y = grid.grid_coords(idx)
            assert y == 0
        assert grid.bottom.sum() == grid.nx

    def test_top_mask(self):
        """Top mask selects y=ny-1 nodes."""
        grid = Grid2D((4, 3))
        top_indices = np.where(grid.top)[0]
        for idx in top_indices:
            _, y = grid.grid_coords(idx)
            assert y == grid.ny - 1
        assert grid.top.sum() == grid.nx

    def test_corners(self):
        """Corners mask selects exactly the 4 corner nodes."""
        grid = Grid2D((5, 4))
        assert grid.corners.sum() == 4
        corner_indices = np.where(grid.corners)[0]
        corner_coords = [grid.grid_coords(i) for i in corner_indices]
        assert (0, 0) in corner_coords
        assert (4, 0) in corner_coords
        assert (0, 3) in corner_coords
        assert (4, 3) in corner_coords

    def test_2x2_all_boundary(self):
        """A 2x2 grid has all boundary nodes and no interior."""
        grid = Grid2D((2, 2))
        assert grid.boundary.sum() == 4
        assert grid.interior.sum() == 0


# =============================================================================
# Gradient operator
# =============================================================================

class TestGradient:
    """Tests for the gradient operator."""

    def test_gradient_shapes(self):
        """Gradient operators have correct shapes."""
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()
        assert Dx.shape == (grid.n_edges_x, grid.size)
        assert Dy.shape == (grid.n_edges_y, grid.size)

    def test_gradient_of_constant_is_zero(self):
        """Gradient of a constant field is zero."""
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()
        u = np.ones(grid.size) * 3.7
        assert_allclose(Dx @ u, 0.0, atol=1e-14)
        assert_allclose(Dy @ u, 0.0, atol=1e-14)

    def test_gradient_of_linear_x(self):
        """Gradient of f(x,y) = x has constant Dx and zero Dy."""
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()

        # f(x,y) = x
        f = np.array([x for _, _, x in
                       sorted(((y, x, x) for x, y, _ in grid.iter_points()))])
        # Recompute properly with flat index
        f = np.zeros(grid.size)
        for x, y, idx in grid.iter_points():
            f[idx] = x

        gx = Dx @ f
        gy = Dy @ f

        # All x-differences should be 1 (forward diff of linear x)
        assert_allclose(gx, 1.0)
        # All y-differences should be 0
        assert_allclose(gy, 0.0, atol=1e-14)

    def test_gradient_of_linear_y(self):
        """Gradient of f(x,y) = y has zero Dx and constant Dy."""
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()

        f = np.zeros(grid.size)
        for x, y, idx in grid.iter_points():
            f[idx] = y

        gx = Dx @ f
        gy = Dy @ f

        assert_allclose(gx, 0.0, atol=1e-14)
        assert_allclose(gy, 1.0)

    def test_gradient_is_submatrix_of_incidence(self):
        """Gradient operators are row slices of the incidence matrix."""
        grid = Grid2D((4, 3))
        A = grid.incidence()
        Dx, Dy = grid.gradient()

        A_dense = A.toarray()
        assert_allclose(Dx.toarray(), A_dense[:grid.n_edges_x, :])
        assert_allclose(Dy.toarray(), A_dense[grid.n_edges_x:, :])


# =============================================================================
# Laplacian operator
# =============================================================================

class TestLaplacian:
    """Tests for the Laplacian operator."""

    def test_laplacian_shape(self):
        """Laplacian is square with size n_nodes."""
        grid = Grid2D((5, 4))
        L = grid.laplacian()
        assert L.shape == (20, 20)

    def test_laplacian_symmetric(self):
        """Laplacian is symmetric."""
        grid = Grid2D((5, 4))
        L = grid.laplacian()
        diff = L - L.T
        assert_allclose(diff.toarray(), 0.0, atol=1e-14)

    def test_laplacian_row_sum_zero(self):
        """Rows of the Laplacian sum to zero."""
        grid = Grid2D((5, 4))
        L = grid.laplacian()
        row_sums = np.array(L.sum(axis=1)).ravel()
        assert_allclose(row_sums, 0.0, atol=1e-14)

    def test_laplacian_positive_semidefinite(self):
        """Laplacian eigenvalues are non-negative."""
        grid = Grid2D((4, 3))
        L = grid.laplacian().toarray()
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-12)

    def test_laplacian_nullspace(self):
        """Constant vector is in the nullspace of the Laplacian."""
        grid = Grid2D((5, 4))
        L = grid.laplacian()
        ones = np.ones(grid.size)
        assert_allclose(L @ ones, 0.0, atol=1e-14)

    def test_laplacian_diagonal_is_degree(self):
        """Diagonal of L equals node degree."""
        grid = Grid2D((4, 3))
        L = grid.laplacian().toarray()
        for x, y, idx in grid.iter_points():
            degree = 0
            if x > 0: degree += 1
            if x < grid.nx - 1: degree += 1
            if y > 0: degree += 1
            if y < grid.ny - 1: degree += 1
            assert L[idx, idx] == degree

    def test_laplacian_five_point_stencil(self):
        """Interior nodes have the standard 5-point stencil."""
        grid = Grid2D((5, 5))
        L = grid.laplacian().toarray()

        # Check an interior node (2,2) → idx = 2*5 + 2 = 12
        idx = grid.flat_index(2, 2)
        assert L[idx, idx] == 4
        assert L[idx, grid.flat_index(1, 2)] == -1
        assert L[idx, grid.flat_index(3, 2)] == -1
        assert L[idx, grid.flat_index(2, 1)] == -1
        assert L[idx, grid.flat_index(2, 3)] == -1

    def test_weighted_laplacian(self):
        """Weighted Laplacian L = A^T diag(w) A is correct."""
        grid = Grid2D((3, 3))
        A = grid.incidence()

        # Use random positive weights
        rng = np.random.default_rng(42)
        w = rng.uniform(0.5, 2.0, size=grid.n_edges)

        L_weighted = grid.laplacian(weights=w)
        W = sp.diags(w)
        L_manual = A.T @ W @ A

        assert_allclose(L_weighted.toarray(), L_manual.toarray(), atol=1e-12)

    def test_weighted_laplacian_symmetric(self):
        """Weighted Laplacian is symmetric."""
        grid = Grid2D((4, 3))
        rng = np.random.default_rng(123)
        w = rng.uniform(0.1, 5.0, size=grid.n_edges)
        L = grid.laplacian(weights=w)
        diff = L - L.T
        assert_allclose(diff.toarray(), 0.0, atol=1e-14)

    def test_weighted_laplacian_row_sum_zero(self):
        """Weighted Laplacian rows sum to zero."""
        grid = Grid2D((4, 3))
        rng = np.random.default_rng(99)
        w = rng.uniform(0.1, 5.0, size=grid.n_edges)
        L = grid.laplacian(weights=w)
        row_sums = np.array(L.sum(axis=1)).ravel()
        assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_unit_weights_equals_default(self):
        """Passing unit weights gives same result as no weights."""
        grid = Grid2D((4, 3))
        L_default = grid.laplacian()
        L_unit = grid.laplacian(weights=np.ones(grid.n_edges))
        assert_allclose(L_default.toarray(), L_unit.toarray(), atol=1e-14)


# =============================================================================
# Divergence operator
# =============================================================================

class TestDivergence:
    """Tests for the divergence operator."""

    def test_divergence_shape(self):
        """Divergence maps edges to nodes."""
        grid = Grid2D((5, 4))
        D = grid.divergence()
        assert D.shape == (grid.size, grid.n_edges)

    def test_divergence_is_negative_transpose_of_incidence(self):
        """Divergence equals -A^T."""
        grid = Grid2D((4, 3))
        A = grid.incidence()
        D = grid.divergence()
        assert_allclose(D.toarray(), -A.T.toarray(), atol=1e-14)

    def test_div_grad_equals_negative_laplacian(self):
        """div(grad(u)) = -L @ u for the graph Laplacian."""
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()
        A = grid.incidence()
        D = grid.divergence()
        L = grid.laplacian()

        # grad(u) gives an edge field via A
        # div(grad(u)) = -A^T @ A @ u = -L @ u
        rng = np.random.default_rng(42)
        u = rng.standard_normal(grid.size)

        grad_u = A @ u
        div_grad_u = D @ grad_u

        assert_allclose(div_grad_u, -L @ u, atol=1e-12)


# =============================================================================
# Curl operator
# =============================================================================

class TestCurl:
    """Tests for the 2D curl operator."""

    def test_curl_shape(self):
        """Curl maps edges to faces."""
        grid = Grid2D((5, 4))
        C = grid.curl()
        n_faces = (grid.nx - 1) * (grid.ny - 1)
        assert C.shape == (n_faces, grid.n_edges)

    def test_curl_of_gradient_is_zero(self):
        """curl(grad(u)) = 0 for any scalar field u."""
        grid = Grid2D((5, 4))
        A = grid.incidence()
        C = grid.curl()

        rng = np.random.default_rng(42)
        u = rng.standard_normal(grid.size)
        grad_u = A @ u

        assert_allclose(C @ grad_u, 0.0, atol=1e-12)

    def test_curl_four_entries_per_face(self):
        """Each face row has exactly 4 nonzero entries."""
        grid = Grid2D((4, 3))
        C = grid.curl()
        nnz_per_row = np.diff(C.tocsr().indptr)
        assert np.all(nnz_per_row == 4)

    def test_curl_irrotational_field(self):
        """Curl of a gradient field is zero (discrete exactness)."""
        grid = Grid2D((6, 5))
        A = grid.incidence()
        C = grid.curl()

        # f(x,y) = x^2 + y^2
        f = np.zeros(grid.size)
        for x, y, idx in grid.iter_points():
            f[idx] = x**2 + y**2

        grad_f = A @ f
        curl_grad_f = C @ grad_f
        assert_allclose(curl_grad_f, 0.0, atol=1e-12)


# =============================================================================
# Hodge star operator
# =============================================================================

class TestHodgeStar:
    """Tests for the Hodge star operator."""

    def test_hodge_star_shape(self):
        """Hodge star is 2n x 2n."""
        grid = Grid2D((5, 4))
        H = grid.hodge_star()
        assert H.shape == (40, 40)

    def test_hodge_star_rotation(self):
        """Hodge star rotates [u; v] to [-v; u]."""
        grid = Grid2D((3, 3))
        n = grid.size
        H = grid.hodge_star()

        u = np.arange(n, dtype=float)
        v = np.arange(n, 2 * n, dtype=float)
        uv = np.concatenate([u, v])

        result = H @ uv
        assert_allclose(result[:n], -v)
        assert_allclose(result[n:], u)

    def test_hodge_star_squared(self):
        """Applying Hodge star twice gives -identity."""
        grid = Grid2D((4, 3))
        H = grid.hodge_star()
        H2 = H @ H
        I = -sp.eye(2 * grid.size)
        assert_allclose(H2.toarray(), I.toarray(), atol=1e-14)


# =============================================================================
# Poisson equation (integration test)
# =============================================================================

class TestPoissonEquation:
    """Integration test: solve a discrete Poisson equation."""

    def test_poisson_dirichlet(self):
        """Solve -L u = f with Dirichlet BC on a small grid.

        Verifies that the Laplacian produces a solvable system with
        correct boundary handling.
        """
        grid = Grid2D((6, 6))
        L = grid.laplacian().toarray().astype(float)
        n = grid.size

        # RHS: f = 1 on interior, 0 on boundary
        f = np.zeros(n)
        f[grid.interior] = 1.0

        # Enforce Dirichlet BC: set boundary rows to identity
        L_bc = L.copy()
        L_bc[grid.boundary, :] = 0
        L_bc[grid.boundary, np.where(grid.boundary)[0]] = 1.0
        f[grid.boundary] = 0.0

        u = np.linalg.solve(L_bc, f)

        # Boundary values should be zero
        assert_allclose(u[grid.boundary], 0.0, atol=1e-14)
        # Interior values should be positive (source term is positive)
        assert np.all(u[grid.interior] > 0)
        # Solution should be symmetric (grid and source are symmetric)
        u_grid = u.reshape(grid.ny, grid.nx)
        assert_allclose(u_grid, u_grid[::-1, :], atol=1e-12)
        assert_allclose(u_grid, u_grid[:, ::-1], atol=1e-12)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
