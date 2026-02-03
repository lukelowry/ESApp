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
from esapp.workbench import PowerWorld

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]


@pytest.fixture(scope="module")
def net(saw_session):
    """Network instance connected to the session SAW."""
    pw = PowerWorld()
    pw.esa = saw_session
    return pw.network


class TestNetwork:

    @pytest.mark.order(6000)
    def test_busmap(self, net):
        bmap = net.busmap()
        assert isinstance(bmap, pd.Series)
        assert list(bmap.values) == list(range(len(bmap)))
        buses = net._pw[Bus]
        assert set(bmap.index) == set(buses["BusNum"])

    @pytest.mark.order(6100)
    def test_incidence(self, net):
        A = net.incidence()
        assert issparse(A)
        nbus = len(net._pw[Bus])
        assert A.shape[1] == nbus
        assert A.shape[0] >= 1

        Adense = A.toarray()
        np.testing.assert_array_equal(Adense.sum(axis=1), 0)
        assert set(np.unique(Adense)) <= {-1.0, 0.0, 1.0}

    @pytest.mark.order(6110)
    def test_incidence_caching(self, net):
        A1 = net.incidence()
        A2 = net.incidence(remake=False)
        assert A1 is A2

        A3 = net.incidence(remake=True)
        assert A1 is not A3
        np.testing.assert_array_equal(A1.toarray(), A3.toarray())

    @pytest.mark.order(6200)
    def test_branch_params(self, net):
        """ybranch, zmag, yshunt, lengths, gamma."""
        Y = net.ybranch()
        assert np.iscomplexobj(Y)

        Z = net.ybranch(asZ=True)
        assert np.iscomplexobj(Z)
        assert len(Z) >= 1
        np.testing.assert_allclose(np.abs(Y), 1 / np.abs(Z), rtol=1e-10)

        zmag = net.zmag()
        np.testing.assert_allclose(zmag, np.abs(Z), rtol=1e-10)

        Ysh = net.yshunt()
        assert np.iscomplexobj(Ysh)
        assert len(Ysh) >= 1

    @pytest.mark.order(6300)
    def test_lengths(self, net):
        ell = net.lengths()
        assert isinstance(ell, pd.Series)
        assert (ell > 0).all()
        assert len(ell) == net.incidence().shape[0]

        ell_pseudo = net.lengths(longer_xfmr_lens=True, length_thresh_km=1.0)
        assert (ell_pseudo > 0).all()

    @pytest.mark.order(6400)
    def test_gamma(self, net):
        gam = net.gamma()
        assert np.iscomplexobj(gam)

    @pytest.mark.order(6500)
    def test_laplacian(self, net):
        nbus = len(net._pw[Bus])

        L = net.laplacian(BranchType.LENGTH)
        assert L.shape == (nbus, nbus)
        assert issparse(L)

        Ldense = L.toarray()
        np.testing.assert_allclose(Ldense, Ldense.T, atol=1e-12)
        np.testing.assert_allclose(Ldense.sum(axis=1), 0, atol=1e-10)
        eigvals = np.linalg.eigvalsh(Ldense)
        assert np.all(eigvals >= -1e-10)

        Lr = net.laplacian(BranchType.RES_DIST).toarray()
        np.testing.assert_allclose(Lr, Lr.T, atol=1e-12)

        A = net.incidence()
        W = np.ones(A.shape[0])
        Lw = net.laplacian(W)
        assert Lw.shape == (A.shape[1], A.shape[1])
        np.testing.assert_allclose(Lw.toarray().sum(axis=1), 0, atol=1e-10)

    @pytest.mark.order(6600)
    def test_delay(self, net):
        beta = net.delay()
        assert isinstance(beta, np.ndarray)
        assert len(beta) > 0

        beta_floor = net.delay(min_delay=0.5)
        assert np.all(beta_floor >= 0.5)

        L = net.laplacian(BranchType.DELAY)
        nbus = len(net._pw[Bus])
        assert L.shape == (nbus, nbus)
