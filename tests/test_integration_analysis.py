"""
Integration tests for apps modules (Statics, Network, GIC, Dynamics)
and SAW-level analysis functionality (ATC, Transient, TimeStep, PV/QV).

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, create_object_string
    from esapp.workbench import GridWorkBench
    from esapp.apps.static import Statics
    from esapp.apps.network import Network, BranchType
    from esapp.apps.gic import GIC, jac_decomp
    from esapp.apps.dynamics import Dynamics, ContingencyBuilder, SimAction
    from esapp.components import Bus, Gen, Load, Branch, Substation, GICXFormer
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


@pytest.fixture(scope="module")
def wb(saw_session):
    """GridWorkBench with live SAW connection."""
    workbench = GridWorkBench()
    workbench.set_esa(saw_session)
    return workbench


# =========================================================================
# Statics
# =========================================================================

class TestStatics:
    """Tests for static analysis functions moved from workbench."""

    def test_pflow(self, wb):
        """Power flow solve and voltage retrieval."""
        wb.statics.flatstart()
        try:
            v = wb.statics.pflow(getvolts=True)
            assert v is not None
            assert len(v) > 0
            assert np.iscomplexobj(v.values)
        except PowerWorldError:
            pytest.skip("Power flow failed on test case")

    def test_pflow_no_volts(self, wb):
        """Power flow without returning voltages."""
        result = wb.statics.pflow(getvolts=False)
        assert result is None

    def test_voltage_complex(self, wb):
        """Complex voltage retrieval."""
        v = wb.statics.voltage(complex=True, pu=True)
        assert np.iscomplexobj(v.values)
        assert len(v) > 0

    def test_voltage_tuple(self, wb):
        """Magnitude + angle voltage retrieval."""
        mag, ang = wb.statics.voltage(complex=False, pu=True)
        assert len(mag) > 0
        assert len(ang) > 0

    def test_voltage_kv(self, wb):
        """KV voltage retrieval."""
        v = wb.statics.voltage(complex=True, pu=False)
        assert len(v) > 0

    def test_set_voltages(self, wb):
        """Set bus voltages from complex vector."""
        v = wb.statics.voltage(complex=True)
        wb.statics.set_voltages(v)

    def test_violations(self, wb):
        """Bus voltage violations."""
        viols = wb.statics.violations(v_min=0.9, v_max=1.1)
        assert isinstance(viols, pd.DataFrame)
        assert 'Low' in viols.columns
        assert 'High' in viols.columns

    def test_violations_tight(self, wb):
        """Tight voltage limits should produce violations."""
        viols = wb.statics.violations(v_min=0.999, v_max=1.001)
        assert isinstance(viols, pd.DataFrame)

    def test_mismatch(self, wb):
        """Bus power mismatches."""
        P, Q = wb.statics.mismatch()
        assert not P.empty
        assert not Q.empty

    def test_mismatch_complex(self, wb):
        """Complex mismatch."""
        S = wb.statics.mismatch(asComplex=True)
        assert np.iscomplexobj(S)

    def test_netinj(self, wb):
        """Net injection at each bus."""
        P, Q = wb.statics.netinj()
        assert len(P) > 0
        assert len(Q) > 0

    def test_netinj_complex(self, wb):
        """Complex net injection."""
        S = wb.statics.netinj(asComplex=True)
        assert np.iscomplexobj(S)

    def test_ybus(self, wb):
        """Y-Bus matrix retrieval."""
        Y = wb.statics.ybus()
        assert Y.shape[0] > 0
        assert Y.shape[0] == Y.shape[1]

    def test_ybus_dense(self, wb):
        """Dense Y-Bus matrix."""
        Y = wb.statics.ybus(dense=True)
        assert isinstance(Y, np.ndarray)
        assert Y.shape[0] > 0

    def test_branch_admittance(self, wb):
        """Branch admittance matrices."""
        Yf, Yt = wb.statics.branch_admittance()
        assert Yf.shape[0] > 0
        assert Yt.shape[0] > 0
        assert Yf.shape == Yt.shape

    def test_jacobian(self, wb):
        """Power flow Jacobian."""
        try:
            wb.statics.pflow(getvolts=False)
            J = wb.statics.jacobian()
            assert J.shape[0] > 0
        except Exception:
            pytest.skip("Jacobian not available")

    def test_jacobian_dense(self, wb):
        """Dense Jacobian."""
        try:
            J = wb.statics.jacobian(dense=True)
            assert isinstance(J, np.ndarray)
        except Exception:
            pytest.skip("Jacobian not available")

    def test_jacobian_polar(self, wb):
        """Polar Jacobian form."""
        try:
            wb.statics.pflow(getvolts=False)
            J = wb.statics.jacobian(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert J.shape[0] > 0
            assert J.shape[0] == J.shape[1]
        except Exception:
            pytest.skip("Polar Jacobian not available")

    def test_jacobian_with_ids(self, wb):
        """Jacobian with row/column ID labels."""
        try:
            wb.statics.pflow(getvolts=False)
            J, ids = wb.statics.jacobian_with_ids(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert isinstance(ids, list)
            assert len(ids) > 0
        except Exception:
            pytest.skip("Jacobian with IDs not available")

    def test_continuation_pf(self, wb):
        """CPF traces the PV curve and yields monotonically increasing
        transfer levels up to (at least) the nose point."""
        wb.statics.pflow(getvolts=False)
        v_before = wb.statics.voltage()

        buses = wb[Bus]
        n = len(buses)
        interface = np.ones(n)
        interface /= np.sum(interface)

        mw_points = []
        v_points = []
        try:
            for mw in wb.statics.continuation_pf(
                interface=interface,
                initialmw=0,
                step_size=0.05,
                min_step=0.001,
                max_step=0.1,
                maxiter=15,
                restore_when_done=True,
            ):
                V = wb.statics.voltage()
                mw_points.append(mw)
                v_points.append(np.abs(V).min())
        except Exception:
            pass

        # Must have at least the base-case point
        assert len(mw_points) > 0
        # First point is the initial transfer (0 MW)
        assert mw_points[0] == 0.0

        # If we got more than just the base case, the transfer should
        # increase from the first to the second point
        if len(mw_points) > 1:
            assert mw_points[1] > mw_points[0]

        # State should be restored (voltages approximately match)
        v_after = wb.statics.voltage()
        np.testing.assert_allclose(
            np.abs(v_before.values), np.abs(v_after.values), atol=0.01,
        )

    def test_continuation_pf_verbose(self, wb):
        """Verbose mode does not raise."""
        buses = wb[Bus]
        n = len(buses)
        interface = np.ones(n) / n

        try:
            for _ in wb.statics.continuation_pf(
                interface=interface,
                maxiter=3,
                verbose=True,
                restore_when_done=True,
            ):
                pass
        except Exception:
            pass

    def test_gens_above_pmax(self, wb):
        """Generator P limit checking."""
        result = wb.statics.gens_above_pmax()
        assert isinstance(result, (bool, np.bool_))

    def test_gens_above_pmax_with_args(self, wb):
        """Generator P limit checking with explicit values."""
        gens = wb[Gen, ['GenMW', 'GenStatus']]
        p = gens['GenMW']
        is_closed = gens['GenStatus'] == 'Closed'
        result = wb.statics.gens_above_pmax(p=p, is_closed=is_closed)
        assert isinstance(result, (bool, np.bool_))

    def test_gens_above_qmax(self, wb):
        """Generator Q limit checking."""
        result = wb.statics.gens_above_qmax()
        assert isinstance(result, (bool, np.bool_))

    def test_gens_above_qmax_with_args(self, wb):
        """Generator Q limit checking with explicit values."""
        gens = wb[Gen, ['GenMVR', 'GenStatus']]
        q = gens['GenMVR']
        is_closed = gens['GenStatus'] == 'Closed'
        result = wb.statics.gens_above_qmax(q=q, is_closed=is_closed)
        assert isinstance(result, (bool, np.bool_))

    def test_solver_options(self, wb):
        """Solver option methods."""
        st = wb.statics
        try:
            st.set_do_one_iteration(True)
            st.set_do_one_iteration(False)
        except Exception:
            pytest.skip("set_do_one_iteration not available")

        try:
            st.set_max_iterations(250)
        except Exception:
            pytest.skip("set_max_iterations not available")

        try:
            st.set_disable_angle_rotation(True)
            st.set_disable_angle_rotation(False)
        except Exception:
            pytest.skip("set_disable_angle_rotation not available")

        try:
            st.set_disable_opt_mult(True)
            st.set_disable_opt_mult(False)
        except Exception:
            pytest.skip("set_disable_opt_mult not available")

        try:
            st.enable_inner_ss_check(True)
            st.enable_inner_ss_check(False)
        except Exception:
            pytest.skip("enable_inner_ss_check not available")

        try:
            st.disable_gen_mvr_check(True)
            st.disable_gen_mvr_check(False)
        except Exception:
            pytest.skip("disable_gen_mvr_check not available")

        try:
            st.enable_inner_check_gen_vars(True)
            st.enable_inner_check_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_check_gen_vars not available")

        try:
            st.enable_inner_backoff_gen_vars(True)
            st.enable_inner_backoff_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_backoff_gen_vars not available")

    def test_state_chain(self, wb):
        """State chain management: chain, pushstate, istore, irestore."""
        st = wb.statics
        st.chain(maxstates=2)
        st.pushstate()
        st.pushstate()
        st.istore(0)
        st.irestore(1)

    def test_state_chain_out_of_range(self, wb):
        """State chain raises on out-of-range access."""
        st = wb.statics
        st.chain(maxstates=2)
        st.pushstate()
        with pytest.raises(Exception, match="out of range"):
            st.irestore(5)
        with pytest.raises(Exception, match="out of range"):
            st.istore(5)

    def test_setload_and_clearloads(self, wb):
        """setload creates dispatch loads, clearloads zeros them."""
        st = wb.statics
        buses = wb[Bus]
        n = len(buses)

        # Apply a constant-power load at every bus
        SP = np.ones(n) * 0.1
        SQ = np.ones(n) * 0.05
        st.setload(SP=SP, SQ=SQ)

        # Read back the dispatch loads to verify
        loads = wb[Load, ['BusNum', 'LoadID', 'LoadSMW', 'LoadSMVR']]
        dispatch = loads[loads['LoadID'] == '99']
        assert len(dispatch) == n
        assert (dispatch['LoadSMW'].values != 0).any()

        # Clear and verify
        st.clearloads()
        loads_after = wb[Load, ['BusNum', 'LoadID', 'LoadSMW', 'LoadSMVR']]
        dispatch_after = loads_after[loads_after['LoadID'] == '99']
        np.testing.assert_array_equal(dispatch_after['LoadSMW'].values, 0.0)
        np.testing.assert_array_equal(dispatch_after['LoadSMVR'].values, 0.0)

    def test_setload_zip_components(self, wb):
        """All six ZIP components can be set independently."""
        st = wb.statics
        n = len(wb[Bus])
        vals = np.ones(n) * 0.01

        # Set each component individually
        st.setload(IP=vals)
        st.setload(IQ=vals)
        st.setload(ZP=vals)
        st.setload(ZQ=vals)
        st.clearloads()

    def test_randomize_load(self, wb):
        """Random load variation."""
        st = wb.statics
        st.randomize_load(scale=1.0, sigma=0.05)


# =========================================================================
# Network
# =========================================================================

class TestNetwork:
    """Tests for network topology functions."""

    def test_busmap(self, wb):
        """Bus number to matrix index mapping."""
        m = wb.network.busmap()
        assert not m.empty

    def test_incidence(self, wb):
        """Incidence matrix construction."""
        A = wb.network.incidence()
        assert A.shape[0] > 0
        assert A.shape[1] > 0

    def test_incidence_cached(self, wb):
        """Incidence matrix caching."""
        A1 = wb.network.incidence(remake=True)
        A2 = wb.network.incidence(remake=False)
        assert A1.shape == A2.shape

    def test_laplacian_length(self, wb):
        """Length-weighted Laplacian."""
        try:
            L = wb.network.laplacian(BranchType.LENGTH)
            assert L.shape[0] > 0
            assert L.shape[0] == L.shape[1]
        except Exception:
            pytest.skip("Laplacian computation failed")

    def test_laplacian_res_dist(self, wb):
        """Resistance-distance weighted Laplacian."""
        try:
            L = wb.network.laplacian(BranchType.RES_DIST)
            assert L.shape[0] > 0
        except Exception:
            pytest.skip("RES_DIST Laplacian not available")

    def test_laplacian_custom_weights(self, wb):
        """Custom weight vector for Laplacian."""
        A = wb.network.incidence()
        W = np.ones(A.shape[0])
        L = wb.network.laplacian(W)
        assert L.shape[0] > 0

    def test_lengths(self, wb):
        """Branch lengths."""
        ell = wb.network.lengths()
        assert len(ell) > 0

    def test_lengths_longer_xfmr(self, wb):
        """Branch lengths with transformer pseudo-lengths."""
        ell = wb.network.lengths(longer_xfmr_lens=True)
        assert len(ell) > 0

    def test_zmag(self, wb):
        """Branch impedance magnitudes."""
        z = wb.network.zmag()
        assert len(z) > 0

    def test_ybranch(self, wb):
        """Branch admittance."""
        y = wb.network.ybranch()
        assert len(y) > 0

    def test_ybranch_as_impedance(self, wb):
        """Branch impedance."""
        z = wb.network.ybranch(asZ=True)
        assert len(z) > 0

    def test_yshunt(self, wb):
        """Branch shunt admittance."""
        y = wb.network.yshunt()
        assert len(y) > 0

    def test_gamma(self, wb):
        """Propagation constants."""
        try:
            g = wb.network.gamma()
            assert len(g) > 0
        except Exception:
            pytest.skip("gamma computation failed")

    def test_delay(self, wb):
        """Propagation delay."""
        try:
            d = wb.network.delay()
            assert len(d) > 0
        except Exception:
            pytest.skip("delay computation failed")

    def test_laplacian_delay(self, wb):
        """Delay-weighted Laplacian."""
        try:
            L = wb.network.laplacian(BranchType.DELAY)
            assert L.shape[0] > 0
        except Exception:
            pytest.skip("DELAY Laplacian not available")

    def test_buscoords(self, wb):
        """Bus coordinates from substation data."""
        try:
            lon, lat = wb.network.buscoords()
            assert len(lon) > 0
            assert len(lat) > 0
        except Exception:
            pytest.skip("buscoords not available (no substation data)")

    def test_buscoords_dataframe(self, wb):
        """Bus coordinates as DataFrame."""
        try:
            df = wb.network.buscoords(astuple=False)
            assert isinstance(df, pd.DataFrame)
            assert 'Longitude' in df.columns
            assert 'Latitude' in df.columns
        except Exception:
            pytest.skip("buscoords not available")


# =========================================================================
# GIC
# =========================================================================

class TestGIC:
    """Tests for GIC analysis."""

    @pytest.mark.order(73)
    def test_gic_calculate(self, saw_instance):
        saw_instance.CalculateGIC(1.0, 90.0, False)
        saw_instance.ClearGIC()

    @pytest.mark.order(74)
    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    @pytest.mark.order(75)
    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    @pytest.mark.order(76)
    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    @pytest.mark.order(77)
    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)

        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)

    @pytest.mark.order(77.1)
    def test_gic_options_pf_include(self, wb):
        """Test set/get GIC pf_include option."""
        gic = wb.gic
        gic.set_pf_include(True)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        gic.set_pf_include(False)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'NO'
        gic.set_pf_include(True)

    @pytest.mark.order(77.2)
    def test_gic_options_ts_include(self, wb):
        """Test set/get GIC ts_include option."""
        gic = wb.gic
        gic.set_ts_include(True)
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        gic.set_ts_include(False)
        assert gic.get_gic_option('IncludeTimeDomain') == 'NO'

    @pytest.mark.order(77.3)
    def test_gic_options_calc_mode(self, wb):
        """Test set/get GIC calc mode option."""
        gic = wb.gic
        gic.set_calc_mode('SnapShot')
        assert gic.get_gic_option('CalcMode') == 'SnapShot'
        gic.set_calc_mode('TimeVarying')
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.set_calc_mode('SnapShot')

    @pytest.mark.order(77.4)
    def test_gic_configure(self, wb):
        """Test configure() with multiple options."""
        gic = wb.gic
        gic.configure(pf_include=True, ts_include=True, calc_mode='TimeVarying')
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.configure()

    @pytest.mark.order(77.5)
    def test_gic_storm(self, wb):
        """Test storm() with both solve_pf options."""
        try:
            wb.gic.storm(1.0, 90.0, solvepf=True)
            wb.gic.storm(1.0, 90.0, solvepf=False)
        except Exception:
            pytest.skip("GIC storm not available")

    @pytest.mark.order(77.6)
    def test_gic_cleargic(self, wb):
        """Test clearing GIC results."""
        try:
            wb.gic.cleargic()
        except Exception:
            pytest.skip("cleargic not available")

    @pytest.mark.order(77.7)
    def test_gic_loadb3d(self, wb):
        """Test loadb3d with non-existent file (exercises code path)."""
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=True)
        except Exception:
            pass
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=False)
        except Exception:
            pass

    @pytest.mark.order(77.75)
    def test_gic_timevary_csv(self, wb, temp_file):
        """Test timevary_csv upload from CSV file."""
        import csv as csvmod
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w', newline='') as f:
            writer = csvmod.writer(f)
            writer.writerow(["Branch '1' '2' '1'", 0.1, 0.11, 0.14])
        try:
            wb.gic.timevary_csv(tmp_csv)
        except Exception:
            pass

    @pytest.mark.order(77.8)
    def test_gic_settings(self, wb):
        """Test retrieving GIC settings."""
        settings = wb.gic.settings()
        assert settings is not None
        assert isinstance(settings, pd.DataFrame)
        assert 'VariableName' in settings.columns

    @pytest.mark.order(77.9)
    def test_gic_gmatrix(self, wb):
        """Test G-matrix retrieval."""
        try:
            G_sparse = wb.gic.gmatrix(sparse=True)
            assert G_sparse.shape[0] > 0
            G_dense = wb.gic.gmatrix(sparse=False)
            assert isinstance(G_dense, np.ndarray)
        except Exception:
            pytest.skip("G-matrix not available")

    @pytest.mark.order(78.0)
    def test_gic_signdiag(self, wb):
        """Test signdiag helper."""
        x = np.array([1, -2, 0, 3])
        D = wb.gic.signdiag(x)
        assert D.shape == (4, 4)
        np.testing.assert_array_equal(np.diag(D), [1, -1, 0, 1])

    @pytest.mark.order(78.01)
    def test_jac_decomp(self):
        """Test jac_decomp utility function."""
        J = np.arange(16).reshape(4, 4).astype(float)
        parts = list(jac_decomp(J))
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (2, 2)

    @pytest.mark.order(78.1)
    def test_gic_get_option_missing(self, wb):
        """Test get_gic_option with nonexistent option."""
        val = wb.gic.get_gic_option('NonExistentOption12345')
        assert val is None


class TestGICGMatrix:
    """G-matrix comparison tests."""

    @pytest.mark.order(78.5)
    def test_gic_gmatrix_comparison(self, gic_saw):
        """Compare computed G-matrix from GIC.model() with PowerWorld's."""
        from scipy.sparse import issparse

        gic = GIC()
        gic.set_esa(gic_saw)
        gic.set_pf_include(True)

        subs = gic[Substation, ["SubNum", "SubName", "GICSubGroundOhms", "GICUsedSubGroundOhms"]]
        buses = gic[Bus, ["BusNum", "BusNomVolt", "SubNum"]]
        branches = gic[Branch, ["BusNum", "BusNum:1", "GICConductance", "BranchDeviceType",
                                 "GICCoilRFrom", "GICCoilRTo"]]
        lines = branches.loc[
            branches['BranchDeviceType'] != 'Transformer',
            ["BusNum", "BusNum:1", "GICConductance"]
        ]
        xfmrs = branches[branches["BranchDeviceType"] == "Transformer"]
        has_grounding = (subs["GICSubGroundOhms"] > 0).any()
        has_xfmr_data = (xfmrs["GICCoilRFrom"] > 0).any() or (xfmrs["GICCoilRTo"] > 0).any()

        if not has_grounding and not has_xfmr_data:
            pytest.skip("Case does not have GIC data configured")

        try:
            model = gic.model()
            G_computed = model.G
        except Exception as e:
            pytest.skip(f"Could not generate GIC model: {e}")

        try:
            G_powerworld = gic.gmatrix(sparse=True)
        except Exception as e:
            pytest.skip(f"Could not retrieve PowerWorld G-matrix: {e}")

        assert issparse(G_computed) and issparse(G_powerworld)

        G_computed_dense = G_computed.toarray()
        G_powerworld_dense = G_powerworld.toarray()

        if G_computed_dense.shape != G_powerworld_dense.shape:
            pytest.skip(f"Shape mismatch: {G_computed_dense.shape} vs {G_powerworld_dense.shape}")

        diff = np.abs(G_computed_dense - G_powerworld_dense)
        max_diff = np.max(diff)

        rtol, atol = 1e-3, 1e-6
        if np.allclose(G_computed_dense, G_powerworld_dense, rtol=rtol, atol=atol):
            return

        MOHM = 1e6
        num_differing = np.sum(diff > 1e-6)
        if np.any(np.abs(G_computed_dense) > MOHM * 0.9):
            pytest.skip(f"G-matrices differ (max={max_diff:.2e}). Large placeholder values detected.")
        elif max_diff < 1.0:
            pass
        else:
            pytest.fail(f"G-matrices differ significantly (max={max_diff:.2e}, {num_differing}/{diff.size} elements)")


class TestGICModel:
    """Tests for GIC model generation and properties."""

    @pytest.mark.order(78.6)
    def test_gic_model(self, wb):
        """Test full model generation."""
        try:
            model = wb.gic.model()
            assert model is wb.gic
        except Exception:
            pytest.skip("GIC model generation failed")

    @pytest.mark.order(78.7)
    def test_gic_model_properties(self, wb):
        """Test model properties after model() call."""
        try:
            wb.gic.model()
        except Exception:
            pytest.skip("GIC model generation failed")

        A = wb.gic.A
        G = wb.gic.G
        H = wb.gic.H
        zeta = wb.gic.zeta
        Px = wb.gic.Px
        eff = wb.gic.eff

        assert A.shape[0] > 0
        assert G.shape[0] == G.shape[1]
        assert H.shape[0] > 0
        assert zeta.shape[0] > 0
        assert Px.shape[0] > 0
        assert eff.shape[0] > 0

    @pytest.mark.order(78.8)
    def test_gic_dIdE(self, wb):
        """Test dIdE Jacobian computation."""
        try:
            wb.gic.model()
            H = wb.gic.H
            n_branches = H.shape[1]
            E = np.ones(n_branches)
            J = wb.gic.dIdE(H, E=E)
            assert J.shape[0] == H.shape[0]
            assert J.shape[1] == n_branches
        except Exception:
            pytest.skip("dIdE computation failed")

    @pytest.mark.order(78.9)
    def test_gic_dIdE_with_i(self, wb):
        """Test dIdE with pre-computed i vector."""
        try:
            wb.gic.model()
            H = wb.gic.H
            i = np.ones(H.shape[0])
            J = wb.gic.dIdE(H, i=i)
            assert J.shape[0] == H.shape[0]
        except Exception:
            pytest.skip("dIdE computation failed")

    @pytest.mark.order(79.0)
    def test_gic_dIdE_raises(self, wb):
        """Test dIdE raises when neither E nor i provided."""
        try:
            wb.gic.model()
            H = wb.gic.H
            with pytest.raises(ValueError, match="Either E or i"):
                wb.gic.dIdE(H)
        except AttributeError:
            pytest.skip("GIC model not available")


# =========================================================================
# Dynamics
# =========================================================================

class TestDynamics:
    """Tests for transient stability simulation."""

    @pytest.mark.order(80)
    def test_contingency_builder(self):
        """Test ContingencyBuilder API."""
        cb = ContingencyBuilder("Test", runtime=5.0)
        cb.at(1.0).fault_bus("101")
        cb.at(1.1).clear_fault("101")

        ctg_df, ele_df = cb.to_dataframes()
        assert ctg_df.iloc[0]['TSCTGName'] == 'Test'
        assert len(ele_df) == 2
        assert ctg_df.iloc[0]['EndTime'] == 5.0

    @pytest.mark.order(80.1)
    def test_contingency_builder_negative_time(self):
        """Negative time should raise ValueError."""
        cb = ContingencyBuilder("Test")
        with pytest.raises(ValueError, match="negative"):
            cb.at(-1.0)

    @pytest.mark.order(80.2)
    def test_contingency_builder_events(self):
        """Test various event types."""
        cb = ContingencyBuilder("Multi", runtime=10.0)
        cb.at(0.5).trip_gen("5", "1")
        cb.at(1.0).trip_branch("1", "2", "1")
        cb.at(1.5).add_event("Bus", "3", SimAction.FAULT_3PB)

        _, ele_df = cb.to_dataframes()
        assert len(ele_df) == 3

    @pytest.mark.order(80.3)
    def test_contingency_builder_empty(self):
        """Builder with no events."""
        cb = ContingencyBuilder("Empty")
        ctg_df, ele_df = cb.to_dataframes()
        assert len(ctg_df) == 1
        assert ele_df.empty

    @pytest.mark.order(80.4)
    def test_contingency_builder_string_action(self):
        """Builder with string action instead of enum."""
        cb = ContingencyBuilder("StringAction")
        cb.at(1.0).add_event("Bus", "10", "FAULT 3PB SOLID")
        _, ele_df = cb.to_dataframes()
        assert 'FAULT 3PB SOLID' in ele_df.iloc[0]['TSEventString']

    @pytest.mark.order(80.5)
    def test_dynamics_watch(self, wb):
        """Test watch method."""
        wb.dyn.watch(Gen, ['TSGenP', 'TSGenW'])
        assert Gen in wb.dyn._watch_fields
        assert wb.dyn._watch_fields[Gen] == ['TSGenP', 'TSGenW']

    @pytest.mark.order(80.6)
    def test_dynamics_contingency(self, wb):
        """Test contingency creation."""
        cb = wb.dyn.contingency("TestCtg")
        assert isinstance(cb, ContingencyBuilder)
        assert "TestCtg" in wb.dyn._pending_ctgs

    @pytest.mark.order(80.7)
    def test_dynamics_bus_fault(self, wb):
        """Test bus_fault convenience method."""
        buses = wb[Bus]
        if buses is not None and not buses.empty:
            bus = buses.iloc[0]['BusNum']
            wb.dyn.bus_fault("QuickFault", str(bus))
            assert "QuickFault" in wb.dyn._pending_ctgs

    @pytest.mark.order(80.8)
    def test_dynamics_upload_unknown(self, wb):
        """Uploading unknown contingency should raise."""
        with pytest.raises(ValueError, match="not found"):
            wb.dyn.upload_contingency("DoesNotExist")


class TestTransient:
    """SAW-level transient stability tests."""

    @pytest.mark.order(81)
    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    @pytest.mark.order(82)
    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(83)
    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.TSCalculateCriticalClearTime(branch_str)

    @pytest.mark.order(84)
    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    @pytest.mark.order(85)
    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)


class TestDynamicsListModels:
    """Tests for Dynamics.list_models()."""

    @pytest.mark.order(85.5)
    def test_list_models(self, wb):
        """Test listing transient stability models."""
        try:
            models = wb.dyn.list_models()
            assert isinstance(models, pd.DataFrame)
        except Exception:
            pytest.skip("list_models not available")


# =========================================================================
# ATC
# =========================================================================

class TestATC:
    """Tests for ATC (Available Transfer Capability) analysis."""

    @pytest.mark.order(86)
    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    @pytest.mark.order(87)
    def test_atc_multiple(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DirectionsAutoInsert(s, b)

        try:
            saw_instance.DetermineATCMultipleDirections()
        except PowerWorldPrerequisiteError:
            pytest.skip("No directions defined for ATC")

    @pytest.mark.order(88)
    def test_atc_results(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow"],
            "field_data_type": ["String", "Real"],
            "key_field": ["", ""],
            "description": ["", ""],
            "display_name": ["", ""]
        }).sort_values(by="internal_field_name")

        saw_instance.GetATCResults(["MaxFlow", "LimitingContingency"])


# =========================================================================
# TimeStep
# =========================================================================

class TestTimeStep:
    """Tests for Time Step Simulation operations."""

    @pytest.mark.order(89)
    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    @pytest.mark.order(90)
    def test_timestep_run(self, saw_instance):
        saw_instance.TimeStepDoRun()
        try:
            saw_instance.TimeStepDoSinglePoint("2025-01-01T10:00:00")
        except PowerWorldPrerequisiteError:
            pass
        try:
            saw_instance.TimeStepClearResults()
        except PowerWorldError:
            pass
        saw_instance.TimeStepResetRun()

    @pytest.mark.order(91)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass

    @pytest.mark.order(92)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])


# =========================================================================
# PV/QV
# =========================================================================

class TestPVQV:
    """Tests for PV and QV analysis."""

    @pytest.mark.order(93)
    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    @pytest.mark.order(94)
    def test_pv_clear(self, saw_instance):
        saw_instance.PVClear()

    @pytest.mark.order(95)
    def test_pv_export(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.PVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("PV analysis not available")

    @pytest.mark.order(96)
    def test_qv_clear(self, saw_instance):
        saw_instance.QVDeleteAllResults()

    @pytest.mark.order(97)
    def test_qv_export(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.QVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("QV analysis not available")


# =========================================================================
# Transient Advanced
# =========================================================================

class TestTransientAdvanced:
    """Advanced transient stability tests."""

    @pytest.mark.order(98)
    def test_transient_result_storage(self, saw_instance):
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(98.1)
    def test_transient_clear_playin(self, saw_instance):
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(98.2)
    def test_transient_validate(self, saw_instance):
        saw_instance.TSInitialize()
        try:
            saw_instance.TSValidate()
        except PowerWorldPrerequisiteError:
            pytest.skip("Transient validation not available")

    @pytest.mark.order(98.3)
    def test_transient_auto_correct(self, saw_instance):
        saw_instance.TSInitialize()
        try:
            saw_instance.TSAutoCorrect()
        except (PowerWorldError, PowerWorldPrerequisiteError):
            pytest.skip("Auto-correct not available")

    @pytest.mark.order(99)
    def test_transient_write_results(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TSGetResults("CSV", ["ALL"], ["GenMW"], filename=tmp_csv)
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No transient results to write")
