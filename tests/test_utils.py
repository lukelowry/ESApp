"""
Unit tests for the esapp.utils module.

Covers all submodules:
- mesh.py: Grid2D, Mesh, extract_unique_edges, linear algebra functions
- misc.py: InjectionVector, timing
- b3d.py: B3D file format
- map.py: Plotting utilities

USAGE:
    pytest tests/test_utils.py -v
    pytest tests/test_utils.py -v --cov=esapp/utils --cov-report=term-missing
"""

import struct

import matplotlib
matplotlib.use('Agg')

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from pandas import DataFrame

from esapp.utils.mesh import (
    MU0,
    Grid2D,
    Mesh,
    extract_unique_edges,
    eigmax,
    hermitify,
    normlap,
    pathincidence,
    pathlap,
    sorteig,
    takagi,
)
from esapp.utils.misc import InjectionVector, timing
from esapp.utils.b3d import B3D
from esapp.utils.map import (
    border,
    darker_hsv_colormap,
    format_plot,
    plot_lines,
    plot_mesh,
    plot_tiles,
    plot_vecfield,
)


# =============================================================================
# Physical constants
# =============================================================================


def test_mu0_value():
    assert abs(MU0 - 1.256637e-6) < 1e-12


# =============================================================================
# extract_unique_edges
# =============================================================================


class TestExtractUniqueEdges:

    def test_single_triangle(self):
        edges = extract_unique_edges([[0, 1, 2]])
        expected = np.array([[0, 1], [0, 2], [1, 2]])
        assert_allclose(edges, expected)

    def test_shared_edge_deduplication(self):
        """Two triangles sharing edge (1,2) yield 5 unique edges."""
        edges = extract_unique_edges([[0, 1, 2], [1, 2, 3]])
        assert edges.shape == (5, 2)
        edge_set = {tuple(e) for e in edges}
        assert (1, 2) in edge_set  # shared edge appears once

    def test_quad_face(self):
        edges = extract_unique_edges([[0, 1, 2, 3]])
        assert edges.shape == (4, 2)

    def test_empty_faces(self):
        edges = extract_unique_edges([])
        assert len(edges) == 0


# =============================================================================
# Mesh class
# =============================================================================


class TestMesh:

    @pytest.fixture
    def tetrahedron(self):
        """Tetrahedron: 4 vertices, 4 triangular faces, 6 edges."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
        return Mesh(vertices=vertices, faces=faces)

    def test_get_xyz(self, tetrahedron):
        xyz = tetrahedron.get_xyz()
        assert xyz.shape == (4, 3)
        assert xyz.dtype == np.float64
        assert_allclose(xyz[1], [1, 0, 0])

    def test_incidence_matrix(self, tetrahedron):
        B = tetrahedron.get_incidence_matrix()
        assert B.shape == (4, 6)
        # Each column sums to zero (+1 and -1)
        assert_allclose(np.array(B.sum(axis=0)).ravel(), 0.0)

    def test_laplacian(self, tetrahedron):
        L = tetrahedron.to_laplacian()
        assert L.shape == (4, 4)
        assert_allclose((L - L.T).toarray(), 0.0)
        assert_allclose(np.array(L.sum(axis=1)).ravel(), 0.0)
        # Every vertex in a tetrahedron has degree 3
        assert_allclose(L.diagonal(), 3.0)

    def test_from_ply_ascii(self, tmp_path):
        ply_content = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "3 0 1 2\n"
        )
        ply_file = tmp_path / "test.ply"
        ply_file.write_text(ply_content)
        mesh = Mesh.from_ply(str(ply_file))
        assert len(mesh.vertices) == 3
        assert mesh.faces == [[0, 1, 2]]
        assert_allclose(mesh.vertices[0], (0.0, 0.0, 0.0))

    def test_from_ply_binary_little_endian(self, tmp_path):
        ply_file = tmp_path / "test_bin.ply"
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        with open(str(ply_file), 'wb') as f:
            f.write(header.encode('ascii'))
            for v in [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]:
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<3i', 0, 1, 2))

        mesh = Mesh.from_ply(str(ply_file))
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1

    def test_from_ply_binary_non_xyz_names(self, tmp_path):
        """Binary PLY with non-standard vertex property names (not x/y/z)."""
        ply_file = tmp_path / "test_names.ply"
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float a\n"
            "property float b\n"
            "property float c\n"
            "element face 0\n"
            "end_header\n"
        )
        with open(str(ply_file), 'wb') as f:
            f.write(header.encode('ascii'))
            for v in [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]:
                f.write(struct.pack('<3f', *v))

        mesh = Mesh.from_ply(str(ply_file))
        assert len(mesh.vertices) == 2
        assert_allclose(mesh.vertices[0], (1.0, 2.0, 3.0), atol=1e-5)

    def test_from_ply_truncated_header(self, tmp_path):
        """PLY file that ends before end_header (empty line triggers break)."""
        ply_content = "ply\nformat ascii 1.0\nelement vertex 0\n\n"
        ply_file = tmp_path / "trunc.ply"
        ply_file.write_bytes(ply_content.encode('ascii'))
        mesh = Mesh.from_ply(str(ply_file))
        assert len(mesh.vertices) == 0

    def test_from_ply_non_ascii_header_line(self, tmp_path):
        """PLY with non-ASCII bytes in header exercises 'if not parts' branch.

        Non-ASCII bytes survive strip() but are dropped by decode('ascii',
        errors='ignore'), resulting in an empty string that splits to [].
        """
        ply_file = tmp_path / "nonascii.ply"
        with open(str(ply_file), 'wb') as f:
            f.write(b"ply\n")
            f.write(b"format ascii 1.0\n")
            f.write(b"\x80\x81\n")  # non-ASCII, stripped to empty string
            f.write(b"element vertex 1\n")
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"element face 0\n")
            f.write(b"end_header\n")
            f.write(b"1.0 2.0 3.0\n")
        mesh = Mesh.from_ply(str(ply_file))
        assert len(mesh.vertices) == 1

    def test_from_ply_unsupported_format(self, tmp_path):
        ply_content = (
            "ply\n"
            "format binary_big_endian 1.0\n"
            "element vertex 0\n"
            "element face 0\n"
            "end_header\n"
        )
        ply_file = tmp_path / "bad.ply"
        ply_file.write_text(ply_content)
        with pytest.raises(ValueError, match="Unsupported PLY format"):
            Mesh.from_ply(str(ply_file))


# =============================================================================
# Grid2D
# =============================================================================


class TestGrid2D:

    def test_construction(self):
        grid = Grid2D((5, 4))
        assert grid.nx == 5 and grid.ny == 4
        assert grid.size == 20
        assert grid.shape == (5, 4)
        assert grid.n_edges_x == 16   # (5-1)*4
        assert grid.n_edges_y == 15   # 5*(4-1)
        assert grid.n_edges == 31

    def test_single_row_and_column(self):
        row = Grid2D((5, 1))
        assert row.n_edges_x == 4 and row.n_edges_y == 0

        col = Grid2D((1, 5))
        assert col.n_edges_x == 0 and col.n_edges_y == 4

    def test_flat_index_roundtrip(self):
        grid = Grid2D((7, 5))
        for x in range(grid.nx):
            for y in range(grid.ny):
                idx = grid.flat_index(x, y)
                assert grid.grid_coords(idx) == (x, y)

    def test_iter_points(self):
        grid = Grid2D((4, 3))
        points = list(grid.iter_points())
        assert len(points) == 12
        assert sorted(p[2] for p in points) == list(range(12))

    # --- Incidence matrix ---

    def test_incidence_structure(self):
        grid = Grid2D((5, 4))
        A = grid.incidence()
        assert A.shape == (31, 20)
        assert_allclose(np.array(A.sum(axis=1)).ravel(), 0.0)
        assert np.all(np.diff(A.tocsr().indptr) == 2)
        assert_allclose(sorted(np.unique(A.data)), [-1.0, 1.0])

    # --- Region masks ---

    def test_boundary_interior_partition(self):
        grid = Grid2D((5, 4))
        assert np.all(grid.boundary | grid.interior)
        assert not np.any(grid.boundary & grid.interior)
        assert grid.boundary.sum() == 2 * (5 + 4) - 4
        assert grid.interior.sum() == 3 * 2

    @pytest.mark.parametrize("mask_name,coord_idx,expected_val", [
        ("left", 0, 0),
        ("right", 0, 4),
        ("bottom", 1, 0),
        ("top", 1, 2),
    ])
    def test_direction_masks(self, mask_name, coord_idx, expected_val):
        grid = Grid2D((5, 3))
        mask = getattr(grid, mask_name)
        for idx in np.where(mask)[0]:
            assert grid.grid_coords(idx)[coord_idx] == expected_val

    def test_corners(self):
        grid = Grid2D((5, 4))
        assert grid.corners.sum() == 4

    def test_2x2_all_boundary(self):
        grid = Grid2D((2, 2))
        assert grid.boundary.sum() == 4
        assert grid.interior.sum() == 0

    # --- Gradient ---

    def test_gradient_of_constant_is_zero(self):
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()
        u = np.ones(grid.size) * 3.7
        assert_allclose(Dx @ u, 0.0, atol=1e-14)
        assert_allclose(Dy @ u, 0.0, atol=1e-14)

    def test_gradient_of_linear_fields(self):
        grid = Grid2D((5, 4))
        Dx, Dy = grid.gradient()
        fx = np.zeros(grid.size)
        fy = np.zeros(grid.size)
        for x, y, idx in grid.iter_points():
            fx[idx] = x
            fy[idx] = y

        assert_allclose(Dx @ fx, 1.0)
        assert_allclose(Dy @ fx, 0.0, atol=1e-14)
        assert_allclose(Dx @ fy, 0.0, atol=1e-14)
        assert_allclose(Dy @ fy, 1.0)

    # --- Laplacian ---

    def test_laplacian_properties(self):
        grid = Grid2D((5, 4))
        L = grid.laplacian()
        assert L.shape == (20, 20)
        # Symmetric
        assert_allclose((L - L.T).toarray(), 0.0, atol=1e-14)
        # Row sums zero
        assert_allclose(np.array(L.sum(axis=1)).ravel(), 0.0, atol=1e-14)
        # Positive semidefinite
        assert np.all(np.linalg.eigvalsh(L.toarray()) >= -1e-12)

    def test_laplacian_diagonal_is_degree(self):
        grid = Grid2D((4, 3))
        L = grid.laplacian().toarray()
        for x, y, idx in grid.iter_points():
            degree = sum([x > 0, x < grid.nx - 1, y > 0, y < grid.ny - 1])
            assert L[idx, idx] == degree

    def test_weighted_laplacian(self):
        grid = Grid2D((3, 3))
        A = grid.incidence()
        rng = np.random.default_rng(42)
        w = rng.uniform(0.5, 2.0, size=grid.n_edges)

        L = grid.laplacian(weights=w)
        L_manual = A.T @ sp.diags(w) @ A
        assert_allclose(L.toarray(), L_manual.toarray(), atol=1e-12)

        # Unit weights should equal default
        L_default = grid.laplacian()
        L_unit = grid.laplacian(weights=np.ones(grid.n_edges))
        assert_allclose(L_default.toarray(), L_unit.toarray(), atol=1e-14)

    # --- Divergence ---

    def test_divergence_is_neg_transpose(self):
        grid = Grid2D((4, 3))
        A = grid.incidence()
        D = grid.divergence()
        assert D.shape == (grid.size, grid.n_edges)
        assert_allclose(D.toarray(), -A.T.toarray(), atol=1e-14)

    def test_div_grad_equals_neg_laplacian(self):
        grid = Grid2D((5, 4))
        A = grid.incidence()
        D = grid.divergence()
        L = grid.laplacian()
        rng = np.random.default_rng(42)
        u = rng.standard_normal(grid.size)
        assert_allclose(D @ (A @ u), -L @ u, atol=1e-12)

    # --- Curl ---

    def test_curl_shape_and_structure(self):
        grid = Grid2D((5, 4))
        C = grid.curl()
        n_faces = (grid.nx - 1) * (grid.ny - 1)
        assert C.shape == (n_faces, grid.n_edges)
        # Each face has exactly 4 edges
        assert np.all(np.diff(C.tocsr().indptr) == 4)

    def test_curl_of_gradient_is_zero(self):
        grid = Grid2D((6, 5))
        A = grid.incidence()
        C = grid.curl()
        rng = np.random.default_rng(42)
        u = rng.standard_normal(grid.size)
        assert_allclose(C @ (A @ u), 0.0, atol=1e-12)

    def test_curl_degenerate_grid(self):
        """Single row has zero faces."""
        grid = Grid2D((5, 1))
        C = grid.curl()
        assert C.shape[0] == 0

    # --- Hodge star ---

    def test_hodge_star(self):
        grid = Grid2D((3, 3))
        n = grid.size
        H = grid.hodge_star()
        assert H.shape == (2 * n, 2 * n)

        # Rotation: [u; v] -> [-v; u]
        u = np.arange(n, dtype=float)
        v = np.arange(n, 2 * n, dtype=float)
        result = H @ np.concatenate([u, v])
        assert_allclose(result[:n], -v)
        assert_allclose(result[n:], u)

        # H^2 = -I
        assert_allclose((H @ H).toarray(), -sp.eye(2 * n).toarray(), atol=1e-14)

    # --- Poisson integration test ---

    def test_poisson_dirichlet(self):
        grid = Grid2D((6, 6))
        L = grid.laplacian().toarray().astype(float)
        n = grid.size

        f = np.zeros(n)
        f[grid.interior] = 1.0

        L_bc = L.copy()
        L_bc[grid.boundary, :] = 0
        L_bc[grid.boundary, np.where(grid.boundary)[0]] = 1.0
        f[grid.boundary] = 0.0

        u = np.linalg.solve(L_bc, f)
        assert_allclose(u[grid.boundary], 0.0, atol=1e-14)
        assert np.all(u[grid.interior] > 0)

        # Symmetry of solution
        u_grid = u.reshape(grid.ny, grid.nx)
        assert_allclose(u_grid, u_grid[::-1, :], atol=1e-12)
        assert_allclose(u_grid, u_grid[:, ::-1], atol=1e-12)


# =============================================================================
# Linear algebra functions
# =============================================================================


class TestTakagi:

    def test_diagonal_matrix(self):
        M = np.diag([2.0 + 0j, 3.0 + 0j])
        U, sigma = takagi(M)
        assert_allclose(sorted(sigma), [2.0, 3.0], atol=1e-10)

    def test_factorization_roundtrip(self):
        """M = U @ diag(sigma) @ U^T for complex symmetric M."""
        M = np.array([[1 + 1j, 2 + 0j], [2 + 0j, 1 - 1j]])
        U, sigma = takagi(M)
        reconstructed = U @ np.diag(sigma) @ U.T
        assert_allclose(reconstructed, M, atol=1e-10)

    def test_u_is_unitary(self):
        M = np.array([[1 + 1j, 2 + 0j], [2 + 0j, 1 - 1j]])
        U, _ = takagi(M)
        assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-10)


class TestEigmax:

    def test_known_eigenvalue(self):
        L = np.diag([1.0, 2.0, 5.0])
        assert abs(eigmax(L) - 5.0) < 1e-10

    def test_sparse_input(self):
        L = sp.diags([2, -1, -1], [0, 1, -1], shape=(10, 10)).tocsr()
        lmax = eigmax(L)
        dense_max = np.max(np.linalg.eigvalsh(L.toarray()))
        assert abs(lmax - dense_max) < 1e-10


class TestSorteig:

    def test_sorts_by_magnitude(self):
        vals = np.array([3.0, -5.0, 1.0, -2.0])
        vecs = np.eye(4)
        sorted_vals, sorted_vecs = sorteig(vals, vecs)
        assert_allclose(sorted_vals, [1.0, -2.0, 3.0, -5.0])
        # Eigenvectors follow the same reordering
        assert_allclose(sorted_vecs[:, 0], vecs[:, 2])


class TestPathlap:

    def test_path_graph(self):
        L = pathlap(4)
        assert L.shape == (4, 4)
        assert_allclose(L, L.T)
        assert_allclose(L.sum(axis=1), 0.0, atol=1e-14)
        assert L[0, 0] == 1
        assert L[1, 1] == 2
        assert L[-1, -1] == 1

    def test_cycle_graph(self):
        L = pathlap(4, periodic=True)
        assert_allclose(np.diag(L), 2.0)
        assert L[0, -1] == -1
        assert L[-1, 0] == -1
        assert_allclose(L.sum(axis=1), 0.0, atol=1e-14)


class TestPathincidence:

    def test_shape(self):
        B = pathincidence(4)
        assert B.shape == (4, 4)

    def test_cycle_wrapping(self):
        B = pathincidence(4, periodic=True)
        assert B[-1, 0] == -1

    def test_main_diagonal(self):
        B = pathincidence(4)
        assert_allclose(np.diag(B), 1.0)


class TestNormlap:

    def test_normalized_diagonal_is_one(self):
        L = pathlap(5)
        Ln = normlap(L)
        assert_allclose(np.diag(Ln), 1.0, atol=1e-14)

    def test_return_scaling_matrices(self):
        L = pathlap(5)
        Ln, D, Di = normlap(L, return_scaling=True)
        assert_allclose(np.diag(Ln), 1.0, atol=1e-14)
        # D @ Di = I
        assert_allclose((D @ Di).toarray(), np.eye(5), atol=1e-14)


class TestHermitify:

    def test_real_symmetric_unchanged(self):
        A = np.array([[1, 2], [2, 3]], dtype=float)
        H = hermitify(A)
        assert_allclose(H, H.conj().T)

    def test_complex_symmetric_to_hermitian(self):
        A = np.array([[1 + 1j, 2 + 3j], [2 + 3j, 4 - 1j]])
        H = hermitify(A)
        assert_allclose(H, H.conj().T, atol=1e-14)

    def test_sparse_input_returns_dense(self):
        A = sp.csr_matrix(np.array([[1 + 1j, 2], [2, 3 - 1j]]))
        H = hermitify(A)
        assert isinstance(H, np.ndarray)
        assert_allclose(H, H.conj().T, atol=1e-14)


# =============================================================================
# InjectionVector
# =============================================================================


class TestInjectionVector:

    @pytest.fixture
    def bus_df(self):
        return DataFrame({'BusNum': [1, 2, 3, 4, 5]})

    def test_initial_vector_zero(self, bus_df):
        inj = InjectionVector(bus_df)
        assert_allclose(inj.vec, 0.0)

    def test_supply_and_demand(self, bus_df):
        inj = InjectionVector(bus_df, losscomp=0.0)
        inj.supply(1, 2)
        inj.demand(4, 5)
        v = inj.vec
        assert np.all(v[inj.loaddf.index.isin([1, 2])] > 0)
        assert np.all(v[inj.loaddf.index.isin([4, 5])] < 0)

    def test_normalization_with_loss_comp(self, bus_df):
        inj = InjectionVector(bus_df, losscomp=0.05)
        inj.supply(1)
        inj.demand(3)
        v = inj.vec
        supply_total = v[v > 0].sum()
        demand_total = -v[v < 0].sum()
        assert abs(supply_total - 1.05 * demand_total) < 1e-12

    def test_zero_loss_comp(self, bus_df):
        inj = InjectionVector(bus_df, losscomp=0.0)
        inj.supply(1)
        inj.demand(5)
        v = inj.vec
        assert abs(v[v > 0].sum() + v[v < 0].sum()) < 1e-12

    def test_demand_only(self, bus_df):
        """Calling demand without supply exercises the supply_sum <= 0 branch."""
        inj = InjectionVector(bus_df, losscomp=0.05)
        inj.demand(3)
        v = inj.vec
        assert v[inj.loaddf.index.get_loc(3)] < 0

    def test_supply_only(self, bus_df):
        """Calling supply without demand exercises the demand_sum <= 0 branch."""
        inj = InjectionVector(bus_df, losscomp=0.05)
        inj.supply(1)
        v = inj.vec
        assert v[inj.loaddf.index.get_loc(1)] > 0


# =============================================================================
# timing decorator
# =============================================================================


class TestTiming:

    def test_preserves_return_value(self, capsys):
        @timing
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5
        assert "'add' took:" in capsys.readouterr().out

    def test_preserves_function_name(self):
        @timing
        def my_func():
            pass

        assert my_func.__name__ == 'my_func'


# =============================================================================
# B3D file format
# =============================================================================


class TestB3D:

    def test_default_construction(self):
        b = B3D()
        assert b.lat.shape == (4,)
        assert b.lon.shape == (4,)
        assert b.time.shape == (3,)
        assert b.ex.shape == (3, 4)
        assert b.ey.shape == (3, 4)

    def test_write_load_roundtrip(self, tmp_path):
        b = B3D()
        fpath = str(tmp_path / "test.b3d")
        b.write_b3d_file(fpath)

        b2 = B3D(fpath)
        assert_allclose(b2.lat, b.lat)
        assert_allclose(b2.lon, b.lon)
        assert_allclose(b2.time, b.time)
        assert_allclose(b2.ex, b.ex)
        assert_allclose(b2.ey, b.ey)
        assert b2.comment == b.comment

    def test_from_mesh_and_roundtrip(self, tmp_path):
        long = np.array([-85.0, -84.5])
        lat = np.array([30.5, 31.0])
        ex = np.ones((2, 2), dtype=np.float32) * 0.5
        ey = np.ones((2, 2), dtype=np.float32) * -0.3

        b = B3D.from_mesh(long, lat, ex, ey, comment="Test")
        assert b.comment == "Test"
        assert b.grid_dim == [2, 2]

        fpath = str(tmp_path / "mesh.b3d")
        b.write_b3d_file(fpath)
        b2 = B3D(fpath)
        assert_allclose(b2.ex, b.ex, atol=1e-6)
        assert_allclose(b2.ey, b.ey, atol=1e-6)

    def test_write_validation_lat_lon_mismatch(self, tmp_path):
        b = B3D()
        b.lon = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="lat and lon must have the same length"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_bad_lat_dtype(self, tmp_path):
        b = B3D()
        b.lat = b.lat.astype(np.float32)
        with pytest.raises(ValueError, match="lat must be a float64"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_bad_lon_dtype(self, tmp_path):
        b = B3D()
        b.lon = b.lon.astype(np.float32)
        with pytest.raises(ValueError, match="lon must be a float64"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_bad_time_dtype(self, tmp_path):
        b = B3D()
        b.time = b.time.astype(np.int32)
        with pytest.raises(ValueError, match="time must be a uint32"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_bad_ex_dtype(self, tmp_path):
        b = B3D()
        b.ex = b.ex.astype(np.float64)
        with pytest.raises(ValueError, match="ex must be a float32"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_bad_ey_dtype(self, tmp_path):
        b = B3D()
        b.ey = b.ey.astype(np.float64)
        with pytest.raises(ValueError, match="ey must be a float32"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_ex_shape_mismatch(self, tmp_path):
        b = B3D()
        b.ex = np.zeros((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="ex columns"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_ey_shape_mismatch(self, tmp_path):
        b = B3D()
        b.ey = np.zeros((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="ey columns"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_ex_rows_mismatch(self, tmp_path):
        b = B3D()
        b.ex = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="ex rows"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_write_validation_ey_rows_mismatch(self, tmp_path):
        b = B3D()
        b.ey = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="ey rows"):
            b.write_b3d_file(str(tmp_path / "bad.b3d"))

    def test_load_no_meta_strings(self, tmp_path):
        """B3D file with nmeta=0 triggers 'No comment' branch."""
        b = B3D()
        fpath = str(tmp_path / "test.b3d")
        b.write_b3d_file(fpath)

        # Patch the file: set nmeta=0 and remove meta string bytes.
        # Easier approach: build a raw file from scratch.
        fpath2 = str(tmp_path / "nometa.b3d")
        n = 1  # 1 location
        nt = 1  # 1 time step
        with open(fpath2, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280)  # code
            _w(4)      # version
            _w(0)      # nmeta = 0
            _w(2)      # float_channels
            _w(0)      # byte_channels
            _w(1)      # loc_format
            _w(n)      # n locations
            # Location data: lon, lat, z for each point
            loc = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
            f.write(loc.tobytes())
            _w(0)   # time_0
            _w(0)   # time_units
            _w(0)   # time_offset
            _w(0)   # time_step (variable)
            _w(nt)  # nt
            f.write(np.array([0], dtype=np.uint32).tobytes())
            # ex, ey interleaved: [ex0, ey0]
            f.write(np.zeros(2, dtype=np.float32).tobytes())

        b2 = B3D(fpath2)
        assert b2.comment == "No comment"

    def test_load_one_meta_string(self, tmp_path):
        """B3D with nmeta=1 skips grid_dim parsing (line 230->242 branch)."""
        fpath = str(tmp_path / "onemeta.b3d")
        n = 1
        nt = 1
        meta = "Only comment\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(1)  # nmeta = 1
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            f.write(np.zeros((n, 3), dtype=np.float64).tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            f.write(np.zeros(n * 2, dtype=np.float32).tobytes())

        b = B3D(fpath)
        assert b.comment == "Only comment"
        # grid_dim stays [0,0] since nmeta < 2, then product != n → [n, 1]
        assert b.grid_dim == [n, 1]

    def test_load_grid_dim_space_separated(self, tmp_path):
        """B3D with grid_dim meta using space-separated format."""
        fpath = str(tmp_path / "spacedim.b3d")
        n = 2
        nt = 1
        meta = "Test comment\x002 1\x00"  # space-separated grid_dim
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)  # nmeta = 2
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            loc = np.zeros((n, 3), dtype=np.float64)
            f.write(loc.tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            f.write(np.zeros(n * 2, dtype=np.float32).tobytes())

        b = B3D(fpath)
        assert b.grid_dim == [2, 1]

    def test_load_grid_dim_bad_format(self, tmp_path):
        """B3D with unparseable grid_dim falls back to [0, 0]."""
        fpath = str(tmp_path / "baddim.b3d")
        n = 2
        nt = 1
        meta = "Test\x00not_a_dim\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)  # nmeta = 2
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            loc = np.zeros((n, 3), dtype=np.float64)
            f.write(loc.tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            f.write(np.zeros(n * 2, dtype=np.float32).tobytes())

        b = B3D(fpath)
        # grid_dim falls back to [0,0], then product != n so becomes [n, 1]
        assert b.grid_dim == [n, 1]

    def test_load_grid_dim_wrong_length(self, tmp_path):
        """B3D with grid_dim having !=2 elements triggers ValueError fallback."""
        fpath = str(tmp_path / "wrongdim.b3d")
        n = 2
        nt = 1
        meta = "Test\x001, 2, 3\x00"  # 3 elements
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            loc = np.zeros((n, 3), dtype=np.float64)
            f.write(loc.tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            f.write(np.zeros(n * 2, dtype=np.float32).tobytes())

        b = B3D(fpath)
        assert b.grid_dim == [n, 1]

    def test_load_float_channels_too_few(self, tmp_path):
        """B3D with float_channels < 2 raises IOError."""
        fpath = str(tmp_path / "fewchan.b3d")
        meta = "Test\x00[2, 1]\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(1)  # float_channels = 1 (too few)
            _w(0); _w(1); _w(2)
            f.write(b'\x00' * 200)

        with pytest.raises(IOError, match="at least 2 float channels"):
            B3D(fpath)

    def test_load_bad_loc_format(self, tmp_path):
        """B3D with loc_format != 1 raises IOError."""
        fpath = str(tmp_path / "badloc.b3d")
        meta = "Test\x00[2, 1]\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(2); _w(0)
            _w(0)  # loc_format = 0 (unsupported)
            _w(2)
            f.write(b'\x00' * 200)

        with pytest.raises(IOError, match="Only location format 1 is supported"):
            B3D(fpath)

    def test_load_nonzero_time_step(self, tmp_path):
        """B3D with time_step != 0 raises IOError."""
        fpath = str(tmp_path / "fixedtime.b3d")
        n = 1
        meta = "Test\x00[1, 1]\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            f.write(np.zeros((n, 3), dtype=np.float64).tobytes())
            _w(0); _w(0); _w(0)
            _w(100)  # time_step != 0
            _w(1)
            f.write(b'\x00' * 200)

        with pytest.raises(IOError, match="variable time points"):
            B3D(fpath)

    def test_load_extra_channels(self, tmp_path):
        """B3D with extra float/byte channels uses the extraction loop."""
        fpath = str(tmp_path / "extrachan.b3d")
        n = 1
        nt = 1
        float_channels = 3  # 3 floats per point (ex, ey, extra)
        byte_channels = 1   # 1 extra byte per point
        meta = "Test\x00[1, 1]\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(float_channels); _w(byte_channels); _w(1); _w(n)
            f.write(np.zeros((n, 3), dtype=np.float64).tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            # Data: 3 floats + 1 byte per point per timestep
            ex_val = np.float32(1.5)
            ey_val = np.float32(-0.5)
            extra_val = np.float32(0.0)
            f.write(struct.pack('<fff', ex_val, ey_val, extra_val))
            f.write(b'\x00')  # 1 byte channel

        b = B3D(fpath)
        assert_allclose(b.ex[0, 0], 1.5, atol=1e-5)
        assert_allclose(b.ey[0, 0], -0.5, atol=1e-5)

    def test_load_grid_dim_product_mismatch(self, tmp_path):
        """When grid_dim product != n, grid_dim is reset to [n, 1]."""
        fpath = str(tmp_path / "mismatch.b3d")
        n = 3
        nt = 1
        # grid_dim says 2x2=4 but we have n=3 points
        meta = "Test\x00[2, 2]\x00"
        with open(fpath, "wb") as f:
            _w = lambda val: f.write(val.to_bytes(4, "little"))
            _w(34280); _w(4)
            _w(2)
            f.write(meta.encode('ascii'))
            _w(2); _w(0); _w(1); _w(n)
            f.write(np.zeros((n, 3), dtype=np.float64).tobytes())
            _w(0); _w(0); _w(0); _w(0); _w(nt)
            f.write(np.array([0], dtype=np.uint32).tobytes())
            f.write(np.zeros(n * 2, dtype=np.float32).tobytes())

        b = B3D(fpath)
        assert b.grid_dim == [3, 1]

    def test_load_invalid_code(self, tmp_path):
        fpath = str(tmp_path / "bad.b3d")
        with open(fpath, "wb") as f:
            f.write((0).to_bytes(4, "little"))
            f.write(b'\x00' * 100)
        with pytest.raises(IOError, match="Invalid B3D file"):
            B3D(fpath)

    def test_load_invalid_version(self, tmp_path):
        fpath = str(tmp_path / "badver.b3d")
        with open(fpath, "wb") as f:
            f.write((34280).to_bytes(4, "little"))
            f.write((99).to_bytes(4, "little"))
            f.write(b'\x00' * 100)
        with pytest.raises(IOError, match="Unsupported B3D version"):
            B3D(fpath)


# =============================================================================
# Map / plotting utilities
# =============================================================================


class TestMapUtilities:

    @pytest.fixture(autouse=True)
    def _close_plots(self):
        yield
        plt.close('all')

    def test_format_plot_with_all_options(self):
        fig, ax = plt.subplots()
        format_plot(
            ax,
            title="Test",
            xlabel="X",
            ylabel="Y",
            xlim=(0, 10),
            ylim=(-5, 5),
            xticksep=2,
            yticksep=1,
            grid=True,
        )
        assert ax.get_title() == "Test"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert ax.get_xlim() == (0, 10)

    def test_format_plot_no_grid(self):
        fig, ax = plt.subplots()
        format_plot(ax, grid=False)

    def test_format_plot_limits_without_tick_sep(self):
        """xlim/ylim without xticksep/yticksep exercises the falsy-ticksep branches."""
        fig, ax = plt.subplots()
        format_plot(ax, xlim=(0, 10), ylim=(-5, 5))
        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (-5, 5)

    def test_format_plot_minimal(self):
        """No optional args — should apply defaults without error."""
        fig, ax = plt.subplots()
        format_plot(ax)

    def test_darker_hsv_colormap(self):
        cmap = darker_hsv_colormap(0.5)
        assert isinstance(cmap, matplotlib.colors.ListedColormap)
        assert len(cmap.colors) == 256

    def test_border_texas(self):
        fig, ax = plt.subplots()
        border(ax, shape='Texas')
        assert len(ax.collections) >= 1

    def test_border_us(self):
        fig, ax = plt.subplots()
        border(ax, shape='US')
        assert len(ax.collections) >= 1

    def test_plot_lines(self):
        fig, ax = plt.subplots()
        lines_df = DataFrame({
            'Longitude': [-85.0, -84.0],
            'Longitude:1': [-84.5, -83.5],
            'Latitude': [30.0, 31.0],
            'Latitude:1': [30.5, 31.5],
        })
        plot_lines(ax, lines_df)
        assert len(ax.collections) >= 1

    def test_plot_vecfield(self):
        fig, ax = plt.subplots()
        X = np.array([0, 1, 0, 1], dtype=float)
        Y = np.array([0, 0, 1, 1], dtype=float)
        U = np.array([1, 0, -1, 0], dtype=float)
        V = np.array([0, 1, 0, -1], dtype=float)
        sm = plot_vecfield(ax, X, Y, U, V)
        assert isinstance(sm, matplotlib.cm.ScalarMappable)

    def test_plot_vecfield_with_nan(self):
        fig, ax = plt.subplots()
        X = np.array([0, 1], dtype=float)
        Y = np.array([0, 0], dtype=float)
        U = np.array([np.nan, 1], dtype=float)
        V = np.array([0, np.nan], dtype=float)
        sm = plot_vecfield(ax, X, Y, U, V)
        assert isinstance(sm, matplotlib.cm.ScalarMappable)

    def test_plot_vecfield_custom_cmap(self):
        fig, ax = plt.subplots()
        cmap = darker_hsv_colormap(0.3)
        sm = plot_vecfield(
            ax,
            np.array([0.0]),
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            cmap=cmap,
        )
        assert isinstance(sm, matplotlib.cm.ScalarMappable)

    def test_plot_mesh(self):
        fig, ax = plt.subplots()

        class MockGT:
            lines = DataFrame({
                'Longitude': [-85.0],
                'Longitude:1': [-84.5],
                'Latitude': [30.0],
                'Latitude:1': [30.5],
            })
            tile_info = (
                np.array([-85.0, -84.5, -84.0]),
                np.array([30.0, 30.5, 31.0]),
                0.5,
            )
            tile_ids = np.array([[0, 1, np.nan], [0, 1, np.nan]])

        plot_mesh(ax, MockGT())
        assert len(ax.collections) >= 1

    def test_plot_mesh_no_lines(self):
        fig, ax = plt.subplots()

        class MockGT:
            tile_info = (
                np.array([-85.0, -84.5]),
                np.array([30.0, 30.5]),
                0.5,
            )
            tile_ids = np.array([[0, np.nan], [0, np.nan]])

        plot_mesh(ax, MockGT(), include_lines=False)

    def test_plot_tiles_with_colors(self):
        fig, ax = plt.subplots()

        class MockGT:
            tile_info = (
                np.array([0, 1, 2]),
                np.array([0, 1, 2]),
                0.5,
            )

        colors = np.array([[1.0, 0.0], [0.0, 1.0]])
        plot_tiles(ax, MockGT(), colors=colors)
        assert len(ax.collections) >= 1

    def test_plot_tiles_no_colors(self):
        fig, ax = plt.subplots()

        class MockGT:
            tile_info = (
                np.array([0, 1, 2]),
                np.array([0, 1]),
                0.5,
            )

        plot_tiles(ax, MockGT())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
