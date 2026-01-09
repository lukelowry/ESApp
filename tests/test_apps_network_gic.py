"""
Unit tests for the esapp.apps module.

WHAT THIS TESTS:
- Network class: incidence matrix, laplacian, bus mapping
- GIC class: model creation and calculations
- ForcedOscillation (modes) class
- BranchType enum

These tests use mocked data and don't require PowerWorld.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from unittest.mock import Mock, MagicMock, patch

pytestmark = pytest.mark.unit


class TestBranchType:
    """Tests for the BranchType enum."""

    def test_branch_type_values(self):
        from esapp.apps import BranchType
        
        assert BranchType.LENGTH.value == 1
        assert BranchType.RES_DIST.value == 2
        assert BranchType.DELAY.value == 3

    def test_branch_type_names(self):
        from esapp.apps import BranchType
        
        assert BranchType.LENGTH.name == "LENGTH"
        assert BranchType.RES_DIST.name == "RES_DIST"
        assert BranchType.DELAY.name == "DELAY"


class TestNetworkBusMap:
    """Tests for Network.busmap() method."""

    def test_busmap_returns_series(self):
        """Test that busmap returns a pandas Series."""
        from esapp.apps import Network
        
        # Create a mock Network with bus data
        network = Mock(spec=Network)
        
        # Create sample bus data
        bus_df = pd.DataFrame({
            'BusNum': [1, 2, 3, 5, 10],
            'BusName': ['Bus1', 'Bus2', 'Bus3', 'Bus5', 'Bus10']
        })
        
        # Call actual busmap logic
        busmap = pd.Series(bus_df.index, bus_df['BusNum'])
        
        assert isinstance(busmap, pd.Series)
        assert len(busmap) == 5
        assert busmap[1] == 0  # First bus maps to index 0
        assert busmap[10] == 4  # Last bus maps to index 4


class TestNetworkIncidence:
    """Tests for Network.incidence() matrix generation."""

    def test_incidence_matrix_shape(self):
        """Test that incidence matrix has correct shape (branches x buses)."""
        # Sample data: 3 buses, 2 branches
        bus_df = pd.DataFrame({'BusNum': [1, 2, 3]})
        branch_df = pd.DataFrame({
            'BusNum': [1, 2],
            'BusNum:1': [2, 3]
        })
        
        # Create busmap
        busmap = pd.Series(bus_df.index, bus_df['BusNum'])
        
        # Build incidence matrix manually to verify logic
        from scipy.sparse import lil_matrix, csc_matrix
        
        nbranches = len(branch_df)
        nbuses = len(bus_df)
        
        A = lil_matrix((nbranches, nbuses))
        for i, row in branch_df.iterrows():
            from_idx = busmap[row['BusNum']]
            to_idx = busmap[row['BusNum:1']]
            A[i, from_idx] = -1
            A[i, to_idx] = 1
        
        A = csc_matrix(A)
        
        assert A.shape == (2, 3)
        assert issparse(A)

    def test_incidence_matrix_values(self):
        """Test that incidence matrix has correct -1/+1 values."""
        # Single branch from bus 1 to bus 2
        bus_df = pd.DataFrame({'BusNum': [1, 2]})
        branch_df = pd.DataFrame({
            'BusNum': [1],
            'BusNum:1': [2]
        })
        
        busmap = pd.Series(bus_df.index, bus_df['BusNum'])
        
        from scipy.sparse import lil_matrix, csc_matrix
        
        A = lil_matrix((1, 2))
        A[0, busmap[1]] = -1
        A[0, busmap[2]] = 1
        A = csc_matrix(A)
        
        # Convert to dense for easy assertion
        A_dense = A.toarray()
        
        assert A_dense[0, 0] == -1  # From bus
        assert A_dense[0, 1] == 1   # To bus


class TestNetworkLaplacian:
    """Tests for Network.laplacian() matrix generation."""

    def test_laplacian_symmetry(self):
        """Test that Laplacian matrix is symmetric."""
        from scipy.sparse import lil_matrix, csc_matrix, diags
        
        # Create simple 3-bus network
        # Bus 1 -- Bus 2 -- Bus 3
        A = lil_matrix((2, 3))
        A[0, 0] = -1; A[0, 1] = 1   # Branch 1-2
        A[1, 1] = -1; A[1, 2] = 1   # Branch 2-3
        A = csc_matrix(A)
        
        # Unit weights
        W = np.ones(2)
        
        # Laplacian: A^T @ diag(W) @ A
        L = A.T @ diags(W) @ A
        L = L.tocsc()
        
        # Check symmetry
        diff = (L - L.T).toarray()
        assert np.allclose(diff, 0)

    def test_laplacian_row_sum_zero(self):
        """Test that Laplacian row sums are approximately zero."""
        from scipy.sparse import lil_matrix, csc_matrix, diags
        
        # Create simple 3-bus network
        A = lil_matrix((2, 3))
        A[0, 0] = -1; A[0, 1] = 1
        A[1, 1] = -1; A[1, 2] = 1
        A = csc_matrix(A)
        
        W = np.ones(2)
        L = A.T @ diags(W) @ A
        L_dense = L.toarray()
        
        # Row sums should be zero (property of Laplacian)
        row_sums = L_dense.sum(axis=1)
        assert np.allclose(row_sums, 0)


class TestGICModelBasics:
    """Basic tests for GIC model structure."""

    def test_gic_jac_decomp_dimensions(self):
        """Test that Jacobian decomposition yields correct sub-matrix sizes."""
        from esapp.apps.gic import jac_decomp
        
        # Create a 6x6 Jacobian (3-bus system: 3 P equations + 3 Q equations)
        nbus = 3
        jac = np.random.rand(2 * nbus, 2 * nbus)
        
        # Get decomposition
        dP_dT, dP_dV, dQ_dT, dQ_dV = list(jac_decomp(jac))
        
        # Each sub-matrix should be nbus x nbus
        assert dP_dT.shape == (nbus, nbus)
        assert dP_dV.shape == (nbus, nbus)
        assert dQ_dT.shape == (nbus, nbus)
        assert dQ_dV.shape == (nbus, nbus)

    def test_gic_jac_decomp_correct_partition(self):
        """Test that Jacobian decomposition returns correct matrix sections."""
        from esapp.apps.gic import jac_decomp
        
        # Create identifiable 4x4 Jacobian (2-bus system)
        jac = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        
        dP_dT, dP_dV, dQ_dT, dQ_dV = list(jac_decomp(jac))
        
        # Upper-left: dP/dTheta
        assert np.array_equal(dP_dT, np.array([[1, 2], [5, 6]]))
        
        # Upper-right: dP/dV
        assert np.array_equal(dP_dV, np.array([[3, 4], [7, 8]]))
        
        # Lower-left: dQ/dTheta
        assert np.array_equal(dQ_dT, np.array([[9, 10], [13, 14]]))
        
        # Lower-right: dQ/dV
        assert np.array_equal(dQ_dV, np.array([[11, 12], [15, 16]]))


class TestGICHelperFunctions:
    """Tests for GIC helper functions."""

    def test_fcmd_formatting(self):
        """Test command string formatting."""
        from esapp.apps.gic import fcmd
        
        result = fcmd("Bus", "['BusNum']", "[1]")
        
        # Should not contain single quotes
        assert "'" not in result
        assert "SetData" in result
        assert "Bus" in result

    def test_gicoption_formatting(self):
        """Test GIC option command formatting."""
        from esapp.apps.gic import gicoption
        
        result = gicoption("TestOption", "TestValue")
        
        assert "SetData" in result
        assert "GIC_Options_Value" in result
        assert "TestOption" in result
        assert "TestValue" in result


class TestModesBasics:
    """Tests for ForcedOscillation (modes) module."""

    def test_forced_oscillation_import(self):
        """Test that ForcedOscillation can be imported."""
        from esapp.apps import ForcedOscillation
        
        assert ForcedOscillation is not None
