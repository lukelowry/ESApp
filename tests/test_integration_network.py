"""
Integration tests for the Network class (esapp.utils.network).

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test incidence matrix
construction, graph Laplacians (length, resistance-distance, delay weighted),
branch parameter calculations, and caching behavior.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

USAGE:
    pytest tests/test_integration_network.py -v
"""
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from esapp.utils import Network, BranchType
from esapp.components import Branch, Bus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]


@pytest.fixture(scope="module")
def net(saw_session):
    """Network instance connected to the session SAW."""
    n = Network()
    n.set_esa(saw_session)
    return n


# ---------------------------------------------------------------------------
# busmap
# ---------------------------------------------------------------------------

class TestBusmap:
    @pytest.mark.order(6000)
    def test_returns_series(self, net):
        bmap = net.busmap()
        assert isinstance(bmap, pd.Series)

    @pytest.mark.order(6010)
    def test_indices_are_sequential(self, net):
        bmap = net.busmap()
        assert list(bmap.values) == list(range(len(bmap)))

    @pytest.mark.order(6020)
    def test_indexed_by_bus_number(self, net):
        bmap = net.busmap()
        buses = net[Bus]
        assert set(bmap.index) == set(buses["BusNum"])


# ---------------------------------------------------------------------------
# incidence
# ---------------------------------------------------------------------------

class TestIncidence:
    @pytest.mark.order(6100)
    def test_sparse(self, net):
        A = net.incidence()
        assert issparse(A)

    @pytest.mark.order(6110)
    def test_shape(self, net):
        A = net.incidence()
        nbus = len(net[Bus])
        assert A.shape[1] == nbus
        assert A.shape[0] >= 1

    @pytest.mark.order(6120)
    def test_row_sums_zero(self, net):
        """Each row has exactly one +1 and one -1."""
        A = net.incidence().toarray()
        np.testing.assert_array_equal(A.sum(axis=1), 0)

    @pytest.mark.order(6130)
    def test_entries_are_pm1(self, net):
        A = net.incidence().toarray()
        unique = set(np.unique(A))
        assert unique <= {-1.0, 0.0, 1.0}

    @pytest.mark.order(6140)
    def test_cached(self, net):
        A1 = net.incidence()
        A2 = net.incidence(remake=False)
        assert A1 is A2

    @pytest.mark.order(6150)
    def test_remake_recomputes(self, net):
        A1 = net.incidence()
        A2 = net.incidence(remake=True)
        assert A1 is not A2
        np.testing.assert_array_equal(A1.toarray(), A2.toarray())


# ---------------------------------------------------------------------------
# ybranch / zmag / yshunt
# ---------------------------------------------------------------------------

class TestBranchParams:
    @pytest.mark.order(6200)
    def test_ybranch_returns_complex(self, net):
        Y = net.ybranch()
        assert np.iscomplexobj(Y)

    @pytest.mark.order(6210)
    def test_ybranch_impedance(self, net):
        Z = net.ybranch(asZ=True)
        assert np.iscomplexobj(Z)
        assert len(Z) >= 1

    @pytest.mark.order(6220)
    def test_ybranch_inverse_relation(self, net):
        Y = net.ybranch()
        Z = net.ybranch(asZ=True)
        np.testing.assert_allclose(np.abs(Y), 1 / np.abs(Z), rtol=1e-10)

    @pytest.mark.order(6230)
    def test_zmag(self, net):
        zmag = net.zmag()
        Z = net.ybranch(asZ=True)
        np.testing.assert_allclose(zmag, np.abs(Z), rtol=1e-10)

    @pytest.mark.order(6240)
    def test_yshunt_returns_complex(self, net):
        Ysh = net.yshunt()
        assert np.iscomplexobj(Ysh)
        assert len(Ysh) >= 1


# ---------------------------------------------------------------------------
# lengths
# ---------------------------------------------------------------------------

class TestLengths:
    @pytest.mark.order(6300)
    def test_returns_series(self, net):
        ell = net.lengths()
        assert isinstance(ell, pd.Series)

    @pytest.mark.order(6310)
    def test_all_positive(self, net):
        ell = net.lengths()
        assert (ell > 0).all()

    @pytest.mark.order(6320)
    def test_pseudo_lengths_all_positive(self, net):
        ell = net.lengths(longer_xfmr_lens=True, length_thresh_km=1.0)
        assert (ell > 0).all()

    @pytest.mark.order(6330)
    def test_length_matches_branch_count(self, net):
        ell = net.lengths()
        A = net.incidence()
        assert len(ell) == A.shape[0]


# ---------------------------------------------------------------------------
# gamma
# ---------------------------------------------------------------------------

class TestGamma:
    @pytest.mark.order(6400)
    def test_returns_complex(self, net):
        gam = net.gamma()
        assert np.iscomplexobj(gam)

    @pytest.mark.order(6410)
    def test_length_matches_branches(self, net):
        gam = net.gamma()
        branches = net[Branch]
        assert len(gam) >= len(branches)


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------

class TestLaplacian:
    @pytest.mark.order(6500)
    def test_shape_square(self, net):
        L = net.laplacian(BranchType.LENGTH)
        nbus = len(net[Bus])
        assert L.shape == (nbus, nbus)

    @pytest.mark.order(6510)
    def test_sparse(self, net):
        L = net.laplacian(BranchType.LENGTH)
        assert issparse(L)

    @pytest.mark.order(6520)
    def test_symmetric(self, net):
        L = net.laplacian(BranchType.LENGTH).toarray()
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    @pytest.mark.order(6530)
    def test_row_sums_zero(self, net):
        L = net.laplacian(BranchType.LENGTH).toarray()
        np.testing.assert_allclose(L.sum(axis=1), 0, atol=1e-10)

    @pytest.mark.order(6540)
    def test_positive_semidefinite(self, net):
        L = net.laplacian(BranchType.LENGTH).toarray()
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)

    @pytest.mark.order(6550)
    def test_resdist_symmetric(self, net):
        L = net.laplacian(BranchType.RES_DIST).toarray()
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    @pytest.mark.order(6560)
    def test_custom_weights(self, net):
        A = net.incidence()
        W = np.ones(A.shape[0])
        L = net.laplacian(W)
        assert L.shape == (A.shape[1], A.shape[1])
        np.testing.assert_allclose(L.toarray().sum(axis=1), 0, atol=1e-10)


# ---------------------------------------------------------------------------
# delay
# ---------------------------------------------------------------------------

class TestDelay:
    @pytest.mark.order(6600)
    def test_returns_array(self, net):
        beta = net.delay()
        assert isinstance(beta, np.ndarray)
        assert len(beta) > 0

    @pytest.mark.order(6610)
    def test_min_delay_floor(self, net):
        beta = net.delay(min_delay=0.5)
        assert np.all(beta >= 0.5)

    @pytest.mark.order(6620)
    def test_delay_laplacian(self, net):
        L = net.laplacian(BranchType.DELAY)
        nbus = len(net[Bus])
        assert L.shape == (nbus, nbus)
