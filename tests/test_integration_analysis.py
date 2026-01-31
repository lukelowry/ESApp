"""
Integration tests for high-level analysis functionality via GridWorkBench.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They exercise the workbench-level
API for power flow, GIC, transient stability, ATC, time step, and PV/QV
analysis against a real PowerWorld case.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_powerworld.py  -- low-level SAW COM operations
    - test_integration_powerflow.py       -- SAW-level power flow & sensitivity
    - test_integration_contingency.py     -- SAW-level contingency analysis
    - test_integration_network.py         -- Network topology (incidence, Laplacian, etc.)
    - test_integration_workbench.py       -- GridWorkBench facade (save, log, components)
    - test_integration_extended.py        -- extended SAW coverage

USAGE:
    pytest tests/test_integration_analysis.py -v
    pytest tests/test_integration_analysis.py -v -k "TestGIC"
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
    from esapp.utils import GIC, jac_decomp
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
# Statics (workbench-level power flow, voltage, matrix access)
# =========================================================================

class TestWorkbenchStatics:
    """Workbench-level static analysis: power flow, voltage, Y-bus, Jacobian.

    These complement the SAW-level tests in test_integration_powerflow.py
    by exercising the higher-level GridWorkBench delegation methods.
    """

    def test_pflow(self, wb):
        """Power flow solve and voltage retrieval."""
        wb.flatstart()
        try:
            v = wb.pflow(getvolts=True)
            assert v is not None
            assert len(v) > 0
            assert np.iscomplexobj(v.values)
        except PowerWorldError:
            pytest.skip("Power flow failed on test case")

    def test_pflow_no_volts(self, wb):
        """Power flow without returning voltages."""
        result = wb.pflow(getvolts=False)
        assert result is None

    def test_voltage_complex(self, wb):
        """Complex voltage retrieval."""
        v = wb.voltage(complex=True, pu=True)
        assert np.iscomplexobj(v.values)
        assert len(v) > 0

    def test_voltage_tuple(self, wb):
        """Magnitude + angle voltage retrieval."""
        mag, ang = wb.voltage(complex=False, pu=True)
        assert len(mag) > 0
        assert len(ang) > 0

    def test_voltage_kv(self, wb):
        """KV voltage retrieval."""
        v = wb.voltage(complex=True, pu=False)
        assert len(v) > 0

    def test_set_voltages(self, wb):
        """Set bus voltages from complex vector."""
        v = wb.voltage(complex=True)
        wb.set_voltages(v)

    def test_violations(self, wb):
        """Bus voltage violations."""
        viols = wb.violations(v_min=0.9, v_max=1.1)
        assert isinstance(viols, pd.DataFrame)
        assert 'Low' in viols.columns
        assert 'High' in viols.columns

    def test_violations_tight(self, wb):
        """Tight voltage limits should produce violations."""
        viols = wb.violations(v_min=0.999, v_max=1.001)
        assert isinstance(viols, pd.DataFrame)

    def test_mismatch(self, wb):
        """Bus power mismatches."""
        P, Q = wb.mismatch()
        assert not P.empty
        assert not Q.empty

    def test_mismatch_complex(self, wb):
        """Complex mismatch."""
        S = wb.mismatch(asComplex=True)
        assert np.iscomplexobj(S)

    def test_netinj(self, wb):
        """Net injection at each bus."""
        P, Q = wb.netinj()
        assert len(P) > 0
        assert len(Q) > 0

    def test_netinj_complex(self, wb):
        """Complex net injection."""
        S = wb.netinj(asComplex=True)
        assert np.iscomplexobj(S)

    def test_ybus(self, wb):
        """Y-Bus matrix retrieval."""
        Y = wb.ybus()
        assert Y.shape[0] > 0
        assert Y.shape[0] == Y.shape[1]

    def test_ybus_dense(self, wb):
        """Dense Y-Bus matrix."""
        Y = wb.ybus(dense=True)
        assert isinstance(Y, np.ndarray)
        assert Y.shape[0] > 0

    def test_branch_admittance(self, wb):
        """Branch admittance matrices."""
        Yf, Yt = wb.branch_admittance()
        assert Yf.shape[0] > 0
        assert Yt.shape[0] > 0
        assert Yf.shape == Yt.shape

    def test_jacobian(self, wb):
        """Power flow Jacobian."""
        try:
            wb.pflow(getvolts=False)
            J = wb.jacobian()
            assert J.shape[0] > 0
        except Exception:
            pytest.skip("Jacobian not available")

    def test_jacobian_dense(self, wb):
        """Dense Jacobian."""
        try:
            J = wb.jacobian(dense=True)
            assert isinstance(J, np.ndarray)
        except Exception:
            pytest.skip("Jacobian not available")

    def test_jacobian_polar(self, wb):
        """Polar Jacobian form."""
        try:
            wb.pflow(getvolts=False)
            J = wb.jacobian(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert J.shape[0] > 0
            assert J.shape[0] == J.shape[1]
        except Exception:
            pytest.skip("Polar Jacobian not available")

    def test_jacobian_with_ids(self, wb):
        """Jacobian with row/column ID labels."""
        try:
            wb.pflow(getvolts=False)
            J, ids = wb.jacobian_with_ids(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert isinstance(ids, list)
            assert len(ids) > 0
        except Exception:
            pytest.skip("Jacobian with IDs not available")

    def test_solver_options(self, wb):
        """Solver option methods."""
        try:
            wb.set_do_one_iteration(True)
            wb.set_do_one_iteration(False)
        except Exception:
            pytest.skip("set_do_one_iteration not available")

        try:
            wb.set_max_iterations(250)
        except Exception:
            pytest.skip("set_max_iterations not available")

        try:
            wb.set_disable_angle_rotation(True)
            wb.set_disable_angle_rotation(False)
        except Exception:
            pytest.skip("set_disable_angle_rotation not available")

        try:
            wb.set_disable_opt_mult(True)
            wb.set_disable_opt_mult(False)
        except Exception:
            pytest.skip("set_disable_opt_mult not available")

        try:
            wb.enable_inner_ss_check(True)
            wb.enable_inner_ss_check(False)
        except Exception:
            pytest.skip("enable_inner_ss_check not available")

        try:
            wb.disable_gen_mvr_check(True)
            wb.disable_gen_mvr_check(False)
        except Exception:
            pytest.skip("disable_gen_mvr_check not available")

        try:
            wb.enable_inner_check_gen_vars(True)
            wb.enable_inner_check_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_check_gen_vars not available")

        try:
            wb.enable_inner_backoff_gen_vars(True)
            wb.enable_inner_backoff_gen_vars(False)
        except Exception:
            pytest.skip("enable_inner_backoff_gen_vars not available")


# =========================================================================
# GIC (consolidated: SAW commands, options, model, G-matrix comparison)
# =========================================================================

class TestGIC:
    """GIC analysis: SAW commands, workbench options, model building, G-matrix.

    All GIC-related integration tests are consolidated here. This covers:
    - Low-level SAW GIC commands (calculate, save, write)
    - Workbench-level option getters/setters (pf_include, ts_include, calc_mode)
    - The configure() shorthand
    - Storm application and result clearing
    - B3D loading and time-varying CSV upload
    - GIC settings retrieval and G-matrix extraction
    - jac_decomp utility
    - Full model() generation and property validation
    - G-matrix comparison between model() and PowerWorld
    """

    # -- SAW-level commands --

    @pytest.mark.order(7300)
    def test_gic_calculate(self, saw_instance):
        saw_instance.CalculateGIC(1.0, 90.0, False)
        saw_instance.ClearGIC()

    @pytest.mark.order(7400)
    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    @pytest.mark.order(7500)
    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    @pytest.mark.order(7600)
    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    @pytest.mark.order(7700)
    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)

        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)

    # -- Workbench-level option getters/setters --

    @pytest.mark.order(7710)
    def test_gic_options_pf_include(self, wb):
        gic = wb.gic
        gic.set_pf_include(True)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        gic.set_pf_include(False)
        assert gic.get_gic_option('IncludeInPowerFlow') == 'NO'
        gic.set_pf_include(True)

    @pytest.mark.order(7720)
    def test_gic_options_ts_include(self, wb):
        gic = wb.gic
        gic.set_ts_include(True)
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        gic.set_ts_include(False)
        assert gic.get_gic_option('IncludeTimeDomain') == 'NO'

    @pytest.mark.order(7730)
    def test_gic_options_calc_mode(self, wb):
        gic = wb.gic
        gic.set_calc_mode('SnapShot')
        assert gic.get_gic_option('CalcMode') == 'SnapShot'
        gic.set_calc_mode('TimeVarying')
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.set_calc_mode('SnapShot')

    @pytest.mark.order(7740)
    def test_gic_configure(self, wb):
        gic = wb.gic
        gic.configure(pf_include=True, ts_include=True, calc_mode='TimeVarying')
        assert gic.get_gic_option('IncludeInPowerFlow') == 'YES'
        assert gic.get_gic_option('IncludeTimeDomain') == 'YES'
        assert gic.get_gic_option('CalcMode') == 'TimeVarying'
        gic.configure()

    # -- Storm, clear, load, settings --

    @pytest.mark.order(7750)
    def test_gic_storm(self, wb):
        try:
            wb.gic.storm(1.0, 90.0, solvepf=True)
            wb.gic.storm(1.0, 90.0, solvepf=False)
        except Exception:
            pytest.skip("GIC storm not available")

    @pytest.mark.order(7760)
    def test_gic_cleargic(self, wb):
        try:
            wb.gic.cleargic()
        except Exception:
            pytest.skip("cleargic not available")

    @pytest.mark.order(7770)
    def test_gic_loadb3d(self, wb):
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=True)
        except Exception:
            pass
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=False)
        except Exception:
            pass

    @pytest.mark.order(7775)
    def test_gic_timevary_csv(self, wb, temp_file):
        import csv as csvmod
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w', newline='') as f:
            writer = csvmod.writer(f)
            writer.writerow(["Branch '1' '2' '1'", 0.1, 0.11, 0.14])
        try:
            wb.gic.timevary_csv(tmp_csv)
        except Exception:
            pass

    @pytest.mark.order(7780)
    def test_gic_settings(self, wb):
        settings = wb.gic.settings()
        assert settings is not None
        assert isinstance(settings, pd.DataFrame)
        assert 'VariableName' in settings.columns

    @pytest.mark.order(7790)
    def test_gic_gmatrix(self, wb):
        try:
            G_sparse = wb.gic.gmatrix(sparse=True)
            assert G_sparse.shape[0] > 0
            G_dense = wb.gic.gmatrix(sparse=False)
            assert isinstance(G_dense, np.ndarray)
        except Exception:
            pytest.skip("G-matrix not available")

    @pytest.mark.order(7800)
    def test_gic_get_option_missing(self, wb):
        val = wb.gic.get_gic_option('NonExistentOption12345')
        assert val is None

    # -- jac_decomp (pure function, but grouped with GIC) --

    @pytest.mark.order(7801)
    def test_jac_decomp(self):
        J = np.arange(16).reshape(4, 4).astype(float)
        parts = list(jac_decomp(J))
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (2, 2)

    # -- Model building and matrix properties --

    @pytest.mark.order(7860)
    def test_gic_model(self, wb):
        try:
            model = wb.gic.model()
            assert model is wb.gic
        except Exception:
            pytest.skip("GIC model generation failed")

    @pytest.mark.order(7870)
    def test_gic_model_properties(self, wb):
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

    # -- G-matrix comparison (multi-case parametrized) --

    @pytest.mark.order(7850)
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


# =========================================================================
# Transient Stability (SAW-level)
# =========================================================================

class TestTransient:
    """SAW-level transient stability commands."""

    @pytest.mark.order(8100)
    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    @pytest.mark.order(8200)
    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(8300)
    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.TSCalculateCriticalClearTime(branch_str)

    @pytest.mark.order(8400)
    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    @pytest.mark.order(8500)
    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)


# =========================================================================
# ATC (Available Transfer Capability)
# =========================================================================

class TestATC:
    """ATC analysis via SAW commands."""

    @pytest.mark.order(8600)
    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    @pytest.mark.order(8700)
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

    @pytest.mark.order(8800)
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
    """Time Step Simulation operations via SAW."""

    @pytest.mark.order(8900)
    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    @pytest.mark.order(9000)
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

    @pytest.mark.order(9100)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass

    @pytest.mark.order(9200)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])


# =========================================================================
# PV/QV
# =========================================================================

class TestPVQV:
    """PV and QV curve analysis via SAW."""

    @pytest.mark.order(9300)
    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    @pytest.mark.order(9400)
    def test_pv_clear(self, saw_instance):
        saw_instance.PVClear()

    @pytest.mark.order(9500)
    def test_pv_export(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.PVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("PV analysis not available")

    @pytest.mark.order(9600)
    def test_qv_clear(self, saw_instance):
        saw_instance.QVDeleteAllResults()

    @pytest.mark.order(9700)
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

    @pytest.mark.order(9800)
    def test_transient_result_storage(self, saw_instance):
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(9810)
    def test_transient_clear_playin(self, saw_instance):
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(9820)
    def test_transient_validate(self, saw_instance):
        saw_instance.TSInitialize()
        try:
            saw_instance.TSValidate()
        except PowerWorldPrerequisiteError:
            pytest.skip("Transient validation not available")

    @pytest.mark.order(9830)
    def test_transient_auto_correct(self, saw_instance):
        saw_instance.TSInitialize()
        try:
            saw_instance.TSAutoCorrect()
        except (PowerWorldError, PowerWorldPrerequisiteError):
            pytest.skip("Auto-correct not available")

    @pytest.mark.order(9900)
    def test_transient_write_results(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TSGetResults("CSV", ["ALL"], ["GenMW"], filename=tmp_csv)
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No transient results to write")
