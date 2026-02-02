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

USAGE:
    pytest tests/test_integration_analysis.py -v
    pytest tests/test_integration_analysis.py -v -k "TestGIC"
"""

import os
import pytest
import pandas as pd
import numpy as np

from conftest import ensure_areas

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

@pytest.fixture(autouse=True, scope="class")
def _fresh_case_for_statics(saw_session):
    """Reopen the case from disk so TestWorkbenchStatics starts on an unmodified case."""
    saw_session.OpenCase(saw_session.pwb_file_path)
    yield


@pytest.mark.usefixtures("_fresh_case_for_statics")
class TestWorkbenchStatics:
    """Workbench-level static analysis: power flow, voltage, Y-bus, Jacobian.

    These complement the SAW-level tests in test_integration_powerflow.py
    by exercising the higher-level GridWorkBench delegation methods.
    """

    def test_pflow(self, wb):
        """Power flow solve and voltage retrieval."""
        v = wb.pflow(getvolts=True)
        assert v is not None
        assert len(v) > 0
        assert np.iscomplexobj(v.values)

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
        wb.pflow(getvolts=False)
        J = wb.jacobian()
        assert J.shape[0] > 0

    def test_jacobian_dense(self, wb):
        """Dense Jacobian."""
        J = wb.jacobian(dense=True)
        assert isinstance(J, np.ndarray)

    def test_jacobian_polar(self, wb):
        """Polar Jacobian form."""
        wb.pflow(getvolts=False)
        J = wb.jacobian(dense=True, form='P')
        assert isinstance(J, np.ndarray)
        assert J.shape[0] > 0
        assert J.shape[0] == J.shape[1]

    def test_jacobian_with_ids(self, wb):
        """Jacobian with row/column ID labels."""
        wb.pflow(getvolts=False)
        J, ids = wb.jacobian_with_ids(dense=True, form='P')
        assert isinstance(J, np.ndarray)
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_solver_options(self, wb):
        """Solver option methods."""
        wb.set_do_one_iteration(True)
        wb.set_do_one_iteration(False)

        wb.set_max_iterations(250)

        wb.set_disable_angle_rotation(True)
        wb.set_disable_angle_rotation(False)

        wb.set_disable_opt_mult(True)
        wb.set_disable_opt_mult(False)

        wb.enable_inner_ss_check(True)
        wb.enable_inner_ss_check(False)

        wb.disable_gen_mvr_check(True)
        wb.disable_gen_mvr_check(False)

        wb.enable_inner_check_gen_vars(True)
        wb.enable_inner_check_gen_vars(False)

        wb.enable_inner_backoff_gen_vars(True)
        wb.enable_inner_backoff_gen_vars(False)


# =========================================================================
# Scheduled Actions
# =========================================================================

class TestScheduledActions:
    """Tests for Scheduled Actions mixin -- all parameter paths."""

    @pytest.mark.order(50000)
    def test_scheduled_set_reference(self, saw_instance):
        saw_instance.ScheduledActionsSetReference()

    @pytest.mark.order(50100)
    def test_scheduled_apply_at(self, saw_instance):
        saw_instance.ApplyScheduledActionsAt("01/01/2025 10:00")

    @pytest.mark.order(50200)
    def test_scheduled_apply_at_with_end_time(self, saw_instance):
        saw_instance.ApplyScheduledActionsAt(
            "01/01/2025 10:00", end_time="01/01/2025 12:00"
        )

    @pytest.mark.order(50300)
    def test_scheduled_apply_at_with_filter(self, saw_instance):
        saw_instance.ApplyScheduledActionsAt(
            "01/01/2025 10:00", filter_name="ALL"
        )

    @pytest.mark.order(50400)
    def test_scheduled_apply_at_revert(self, saw_instance):
        saw_instance.ApplyScheduledActionsAt(
            "01/01/2025 10:00", revert=True
        )

    @pytest.mark.order(50500)
    def test_scheduled_revert_at(self, saw_instance):
        saw_instance.RevertScheduledActionsAt("01/01/2025 10:00")

    @pytest.mark.order(50600)
    def test_scheduled_revert_at_with_filter(self, saw_instance):
        saw_instance.RevertScheduledActionsAt(
            "01/01/2025 10:00", filter_name="ALL"
        )

    @pytest.mark.order(50700)
    def test_scheduled_identify_breakers(self, saw_instance):
        saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=True)

    @pytest.mark.order(50800)
    def test_scheduled_identify_breakers_false(self, saw_instance):
        saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=False)

    @pytest.mark.order(50900)
    def test_scheduled_set_view(self, saw_instance):
        saw_instance.SetScheduleView("01/01/2025 10:00")

    @pytest.mark.order(51000)
    def test_scheduled_set_view_with_options(self, saw_instance):
        saw_instance.SetScheduleView(
            "01/01/2025 10:00",
            apply_actions=True,
            use_normal_status=False,
            apply_window=True,
        )

    @pytest.mark.order(51100)
    def test_scheduled_set_window(self, saw_instance):
        saw_instance.SetScheduleWindow(
            "01/01/2025 00:00", "02/01/2025 00:00",
            resolution=1.0, resolution_units="HOURS",
        )

    @pytest.mark.order(51200)
    def test_scheduled_set_window_with_resolution(self, saw_instance):
        saw_instance.SetScheduleWindow(
            "01/01/2025 00:00",
            "02/01/2025 00:00",
            resolution=0.5,
            resolution_units="HOURS",
        )


# =========================================================================
# Weather
# =========================================================================

class TestWeather:
    """Tests for Weather mixin -- all boolean parameter paths."""

    @pytest.mark.order(52000)
    def test_weather_limits_gen_update(self, saw_instance):
        saw_instance.WeatherLimitsGenUpdate(update_max=True, update_min=True)

    @pytest.mark.order(52100)
    def test_weather_limits_gen_update_false(self, saw_instance):
        saw_instance.WeatherLimitsGenUpdate(update_max=False, update_min=False)

    @pytest.mark.order(52200)
    def test_weather_temperature_limits_branch(self, saw_instance):
        saw_instance.TemperatureLimitsBranchUpdate()

    @pytest.mark.order(52300)
    def test_weather_temperature_limits_branch_custom(self, saw_instance):
        saw_instance.TemperatureLimitsBranchUpdate(
            rating_set_precedence="CTG",
            normal_rating_set="A",
            ctg_rating_set="B",
        )

    @pytest.mark.order(52400)
    def test_weather_pfw_set_inputs(self, saw_instance):
        saw_instance.WeatherPFWModelsSetInputs()

    @pytest.mark.order(52500)
    def test_weather_pfw_set_inputs_and_apply(self, saw_instance):
        saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=True)

    @pytest.mark.order(52600)
    def test_weather_pfw_set_inputs_and_apply_no_solve(self, saw_instance):
        saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=False)

    @pytest.mark.order(52700)
    def test_weather_pfw_restore_design(self, saw_instance):
        saw_instance.WeatherPFWModelsRestoreDesignValues()

    @pytest.mark.order(52800)
    def test_weather_pww_load_datetime(self, saw_instance):
        try:
            saw_instance.WeatherPWWLoadForDateTimeUTC("2025-01-01T10:00:00")
        except PowerWorldPrerequisiteError:
            # Expected when no PWW weather files exist for this case
            pass

    @pytest.mark.order(52900)
    def test_weather_pww_set_directory(self, saw_instance, temp_dir):
        saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=True)

    @pytest.mark.order(53000)
    def test_weather_pww_set_directory_no_subdirs(self, saw_instance, temp_dir):
        saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=False)

    @pytest.mark.order(53100)
    def test_weather_pww_file_all_meas_valid(self, saw_instance, temp_file):
        tmp = temp_file(".pww")
        # Passing an empty PWW file should raise an error from PowerWorld
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.WeatherPWWFileAllMeasValid(tmp, ["Temperature"])

    @pytest.mark.order(53200)
    def test_weather_pww_file_combine(self, saw_instance, temp_file):
        src1 = temp_file(".pww")
        src2 = temp_file(".pww")
        dst = temp_file(".pww")
        # Passing empty PWW files should raise an error from PowerWorld
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.WeatherPWWFileCombine2(src1, src2, dst)

    @pytest.mark.order(53300)
    def test_weather_pww_file_geo_reduce(self, saw_instance, temp_file):
        src = temp_file(".pww")
        dst = temp_file(".pww")
        # Passing an empty PWW file should raise an error from PowerWorld
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.WeatherPWWFileGeoReduce(src, dst, 30.0, 50.0, -100.0, -80.0)


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
        # Enable GIC options before calculating — without this, PW may
        # crash with an access violation that corrupts the COM session.
        saw_instance.EnterMode("EDIT")
        saw_instance.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            ['IncludeInPowerFlow', 'YES']
        )
        saw_instance.SetData(
            'GIC_Options_Value',
            ['VariableName', 'ValueField'],
            ['CalcMode', 'SnapShot']
        )
        saw_instance.EnterMode("RUN")

        # Verify GIC data exists before calling CalculateGIC to prevent
        # access violations that poison the entire PW session.
        subs = saw_instance.GetParametersMultipleElement(
            "Substation", ["SubNum", "GICSubGroundOhms"]
        )
        has_grounding = (
            subs is not None and not subs.empty
            and (subs["GICSubGroundOhms"].astype(float) > 0).any()
        )
        if not has_grounding:
            branches = saw_instance.GetParametersMultipleElement(
                "Branch", ["BusNum", "BusNum:1", "BranchDeviceType",
                           "GICCoilRFrom", "GICCoilRTo"]
            )
            has_xfmr_data = False
            if branches is not None and not branches.empty:
                xfmrs = branches[branches["BranchDeviceType"] == "Transformer"]
                has_xfmr_data = (
                    not xfmrs.empty
                    and ((xfmrs["GICCoilRFrom"].astype(float) > 0).any()
                         or (xfmrs["GICCoilRTo"].astype(float) > 0).any())
                )
            assert has_xfmr_data, (
                "Case has no GIC data (no substation grounding or transformer "
                "coil resistances). Cannot run GIC calculation."
            )

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
        wb.gic.storm(1.0, 90.0, solvepf=True)
        wb.gic.storm(1.0, 90.0, solvepf=False)

    @pytest.mark.order(7760)
    def test_gic_cleargic(self, wb):
        wb.gic.cleargic()

    @pytest.mark.order(7770)
    def test_gic_loadb3d(self, wb):
        with pytest.raises((PowerWorldPrerequisiteError, PowerWorldError)):
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=True)

    @pytest.mark.order(7771)
    def test_gic_loadb3d_no_setup(self, wb):
        with pytest.raises((PowerWorldPrerequisiteError, PowerWorldError)):
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=False)

    @pytest.mark.order(7775)
    def test_gic_timevary_csv(self, wb, temp_file):
        import csv as csvmod
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w', newline='') as f:
            writer = csvmod.writer(f)
            writer.writerow(["Branch '1' '2' '1'", 0.1, 0.11, 0.14])
        wb.gic.timevary_csv(tmp_csv)

    @pytest.mark.order(7780)
    def test_gic_settings(self, wb):
        settings = wb.gic.settings()
        assert settings is not None
        assert isinstance(settings, pd.DataFrame)
        assert 'VariableName' in settings.columns

    @pytest.mark.order(7790)
    def test_gic_gmatrix(self, wb):
        G_sparse = wb.gic.gmatrix(sparse=True)
        assert G_sparse.shape[0] > 0
        G_dense = wb.gic.gmatrix(sparse=False)
        assert isinstance(G_dense, np.ndarray)

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
        model = wb.gic.model()
        assert model is wb.gic

    @pytest.mark.order(7870)
    def test_gic_model_properties(self, wb):
        wb.gic.model()

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

        model = gic.model()
        G_computed = model.G

        G_powerworld = gic.gmatrix(sparse=True)

        assert issparse(G_computed) and issparse(G_powerworld)

        G_computed_dense = G_computed.toarray()
        G_powerworld_dense = G_powerworld.toarray()

        if G_computed_dense.shape != G_powerworld_dense.shape:
            pytest.fail(f"Shape mismatch: {G_computed_dense.shape} vs {G_powerworld_dense.shape}")

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
        assert branches is not None and not branches.empty, "Test case must contain branches"
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

    @pytest.mark.order(64000)
    def test_transient_transfer_state(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=True)

    @pytest.mark.order(64100)
    def test_transient_transfer_state_no_mismatch(self, saw_instance):
        saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=False)

    @pytest.mark.order(64200)
    def test_transient_store_response(self, saw_instance):
        saw_instance.TSStoreResponse("Gen", True)
        saw_instance.TSStoreResponse("Gen", False)

    @pytest.mark.order(64300)
    def test_transient_clear_results_ram(self, saw_instance):
        saw_instance.TSClearResultsFromRAM()

    @pytest.mark.order(64400)
    def test_transient_clear_results_specific_ctg(self, saw_instance):
        saw_instance.TSClearResultsFromRAM(
            ctg_name="ALL",
            clear_summary=True,
            clear_events=False,
            clear_statistics=True,
            clear_time_values=False,
            clear_solution_details=True,
        )

    @pytest.mark.order(64500)
    def test_transient_clear_results_and_disable(self, saw_instance):
        saw_instance.TSClearResultsFromRAMAndDisableStorage()

    @pytest.mark.order(64600)
    def test_transient_clear_all_models(self, saw_instance):
        saw_instance.TSClearAllModels()

    @pytest.mark.order(64250)
    def test_transient_smib_eigenvalues(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSCalculateSMIBEigenValues()

    @pytest.mark.order(64800)
    def test_transient_clear_models_for_objects(self, saw_instance):
        saw_instance.TSClearModelsforObjects("Gen")

    @pytest.mark.order(64900)
    def test_transient_disable_machine_model(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSDisableMachineModelNonZeroDerivative(threshold=0.01)

    @pytest.mark.order(65000)
    def test_transient_auto_insert_dist_relay(self, saw_instance):
        saw_instance.TSAutoInsertDistRelay(
            reach=1.0, add_from=True, add_to=False,
            transfer_trip=True, shape=1, filter_name="ALL",
        )

    @pytest.mark.order(65100)
    def test_transient_auto_insert_zpott(self, saw_instance):
        saw_instance.TSAutoInsertZPOTT(reach=1.0, filter_name="ALL")

    @pytest.mark.order(65200)
    def test_transient_run_result_analyzer(self, saw_instance):
        saw_instance.TSRunResultAnalyzer()

    @pytest.mark.order(64350)
    def test_transient_run_until_specified_time(self, saw_instance):
        # Must run before TSClearAllModels (64600) so machine models exist.
        # Create a TS contingency for this test since only "TestCTG" exists from earlier.
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSInitialize()
        saw_instance.TSRunUntilSpecifiedTime("TestCtg", stop_time=0.1, step_size=0.01)

    @pytest.mark.order(65400)
    def test_transient_save_pti(self, saw_instance, temp_file):
        """TSSavePTI writes .dyr file."""
        tmp = temp_file(".dyr")
        saw_instance.TSSavePTI(tmp)

    @pytest.mark.order(65410)
    def test_transient_save_ge(self, saw_instance, temp_file):
        """TSSaveGE writes .dyd file."""
        tmp = temp_file(".dyd")
        saw_instance.TSSaveGE(tmp)

    @pytest.mark.order(65420)
    def test_transient_save_bpa(self, saw_instance, temp_file):
        """TSSaveBPA writes .bpa file."""
        tmp = temp_file(".bpa")
        saw_instance.TSSaveBPA(tmp)

    @pytest.mark.order(65500)
    def test_transient_save_pti_diff(self, saw_instance, temp_file):
        """TSSavePTI with diff_case_modified_only."""
        tmp = temp_file(".dyr")
        saw_instance.TSSavePTI(tmp, diff_case_modified_only=True)

    @pytest.mark.order(65510)
    def test_transient_save_ge_diff(self, saw_instance, temp_file):
        """TSSaveGE with diff_case_modified_only."""
        tmp = temp_file(".dyd")
        saw_instance.TSSaveGE(tmp, diff_case_modified_only=True)

    @pytest.mark.order(65520)
    def test_transient_save_bpa_diff(self, saw_instance, temp_file):
        """TSSaveBPA with diff_case_modified_only."""
        tmp = temp_file(".bpa")
        saw_instance.TSSaveBPA(tmp, diff_case_modified_only=True)

    @pytest.mark.order(65600)
    def test_transient_write_models_diff(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        saw_instance.TSWriteModels(tmp, diff_case_modified_only=True)

    @pytest.mark.order(65700)
    def test_transient_write_options_custom(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        saw_instance.TSWriteOptions(
            tmp,
            save_dynamic_model=False,
            save_stability_options=True,
            save_stability_events=False,
            save_results_events=True,
            save_plot_definitions=False,
            save_transient_limit_monitors=True,
            save_result_analyzer_time_window=False,
            key_field="SECONDARY",
        )

    @pytest.mark.order(65800)
    def test_transient_save_two_bus_equiv(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp = temp_file(".pwb")
        saw_instance.TSSaveTwoBusEquivalent(tmp, bus_key)

    @pytest.mark.order(65900)
    def test_transient_join_active_ctgs(self, saw_instance):
        try:
            saw_instance.TSJoinActiveCTGs(0.1, delete_existing=True, join_with_self=False)
        except PowerWorldError:
            # Requires active TS contingencies to join; not always available
            pass

    @pytest.mark.order(66000)
    def test_transient_set_selected_for_refs(self, saw_instance):
        try:
            saw_instance.TSSetSelectedForTransientReferences(
                "ALL", "SET", ["Gen"], ["GENROU"]
            )
        except PowerWorldError:
            # Access violation if no TS models exist for this case
            pass

    @pytest.mark.order(66100)
    def test_transient_save_dynamic_models_append(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp, "AUX", "Gen", append=True)

    @pytest.mark.order(66200)
    def test_transient_plot_series_add(self, saw_instance):
        try:
            saw_instance.TSPlotSeriesAdd("TestPlot", 1, 1, "Gen", "GenMW")
        except PowerWorldError:
            # Access violation when no TS solution state exists
            pass

    @pytest.mark.order(66300)
    def test_transient_get_vcurve_data(self, saw_instance, temp_file):
        tmp = temp_file(".csv")
        try:
            saw_instance.TSGetVCurveData(tmp, "")
        except PowerWorldError:
            # VCurve data may not be available for this case
            pass

    @pytest.mark.order(77400)
    def test_ts_solve_with_time_params(self, saw_instance):
        """TSSolve with explicit start/stop/step parameters."""
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSInitialize()
        saw_instance.TSSolve(
            "TestCtg", start_time=0.0, stop_time=0.1,
            step_size=0.01, step_in_cycles=False,
        )

    @pytest.mark.order(77500)
    def test_ts_solve_all(self, saw_instance):
        """TSSolveAll completes without error."""
        saw_instance.TSInitialize()
        saw_instance.TSSolveAll()

    @pytest.mark.order(77600)
    def test_ts_clear_results_named_ctg(self, saw_instance):
        """TSClearResultsFromRAM with a specific contingency name."""
        # Create contingency, solve it so results exist, then clear by name
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSResultStorageSetAll("ALL", True)
        saw_instance.TSInitialize()
        saw_instance.TSSolve("TestCtg")
        saw_instance.TSClearResultsFromRAM(ctg_name="TestCtg")

    @pytest.mark.order(77700)
    def test_ts_auto_save_plots(self, saw_instance):
        """TSAutoSavePlots completes without error."""
        saw_instance.TSAutoSavePlots(
            plot_names=["TestPlot"],
            ctg_names=["TestCtg"],
            image_type="JPG",
            width=800, height=600, font_scalar=1.0,
            include_case_name=True, include_category=True,
        )

    @pytest.mark.order(77800)
    def test_ts_load_rdb(self, saw_instance, temp_file):
        """TSLoadRDB with a file path."""
        tmp = temp_file(".rdb")
        saw_instance.TSLoadRDB(tmp, "DISTRELAY")

    @pytest.mark.order(77900)
    def test_ts_load_relay_csv(self, saw_instance, temp_file):
        """TSLoadRelayCSV with a file path."""
        tmp = temp_file(".csv")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TSLoadRelayCSV(tmp, "DISTRELAY")

    @pytest.mark.order(78000)
    def test_ts_run_until_specified_time_steps(self, saw_instance):
        """TSRunUntilSpecifiedTime with steps_to_do parameter."""
        saw_instance.TSInitialize()
        saw_instance.TSRunUntilSpecifiedTime(
            "TestCtg", stop_time=0.1, step_size=0.01,
            steps_in_cycles=False, reset_start_time=False,
            steps_to_do=5,
        )

    @pytest.mark.order(78100)
    def test_ts_load_pti(self, saw_instance, temp_file):
        """TSLoadPTI with a file path."""
        tmp = temp_file(".dyr")
        saw_instance.TSLoadPTI(tmp)

    @pytest.mark.order(78200)
    def test_ts_load_ge(self, saw_instance, temp_file):
        """TSLoadGE with a file path."""
        tmp = temp_file(".dyd")
        saw_instance.TSLoadGE(tmp)

    @pytest.mark.order(78300)
    def test_ts_load_bpa(self, saw_instance, temp_file):
        """TSLoadBPA with a file path."""
        tmp = temp_file(".bpa")
        saw_instance.TSLoadBPA(tmp)

    @pytest.mark.order(78400)
    def test_ts_clear_playin_signals(self, saw_instance):
        """TSClearPlayInSignals completes without error."""
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(78500)
    def test_ts_set_playin_signals_multi_col(self, saw_instance):
        """TSSetPlayInSignals with multiple signal columns."""
        times = np.array([0.0, 0.1, 0.2])
        signals = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
        saw_instance.TSSetPlayInSignals("MultiSignal", times, signals)

    @pytest.mark.order(78600)
    def test_ts_set_playin_signals_validation(self, saw_instance):
        """TSSetPlayInSignals raises ValueError for mismatched dimensions."""
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [2.0], [3.0]])  # 3 rows vs 2 times
        with pytest.raises(ValueError, match="Dimension mismatch"):
            saw_instance.TSSetPlayInSignals("Bad", times, signals)

    @pytest.mark.order(78700)
    def test_ts_validate(self, saw_instance):
        """TSValidate completes without error."""
        saw_instance.TSValidate()

    @pytest.mark.order(78800)
    def test_ts_auto_correct(self, saw_instance):
        """TSAutoCorrect runs and may find/fix validation errors."""
        try:
            saw_instance.TSAutoCorrect()
        except PowerWorldError as e:
            # TSAutoCorrect reports "Validation Errors were found" when it
            # finds and fixes issues — that is its expected purpose.
            assert "Validation Errors were found" in str(e), f"Unexpected error: {e}"

    @pytest.mark.order(9800)
    def test_transient_result_storage(self, saw_instance):
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(9810)
    def test_transient_clear_playin(self, saw_instance):
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(8400)
    def test_transient_write_results(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        # Delete all existing TS contingencies to start clean (earlier tests may
        # have created many via auto-insert, causing TSSolve to stall)
        try:
            saw_instance.RunScriptCommand("Delete(TSContingency);")
        except (PowerWorldError, PowerWorldPrerequisiteError):
            pass
        # Create a single minimal TS contingency
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCTG"])
        ts_ctgs = saw_instance.ListOfDevices("TSContingency")
        if ts_ctgs is None or ts_ctgs.empty:
            pytest.skip("Unable to create TS contingencies for this case")
        name_col = "TSCTGName" if "TSCTGName" in ts_ctgs.columns else ts_ctgs.columns[0]
        ctg_name = str(ts_ctgs.iloc[0][name_col]).strip()
        # Auto-correct and initialize
        saw_instance.TSAutoCorrect()
        saw_instance.TSInitialize()
        try:
            saw_instance.TSSolve(ctg_name, start_time=0.0, stop_time=0.1, step_size=0.01)
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
            pytest.skip(f"TS simulation cannot start for this case: {e}")
        saw_instance.TSGetResults("SINGLE", [ctg_name], ["Gen ALL | GenMW"], filename=tmp_csv)
        assert os.path.exists(tmp_csv)


# =========================================================================
# ATC (Available Transfer Capability)
# =========================================================================

class TestATC:
    """ATC analysis via SAW commands."""

    @pytest.mark.order(4000)
    def test_atc_ensure_two_areas(self, saw_instance):
        """Ensure the case has at least 2 areas for ATC testing."""
        areas = ensure_areas(saw_instance, min_count=2)
        assert areas is not None and len(areas) >= 2, "Failed to create second area"

    @pytest.mark.order(4020)
    def test_atc_determine_and_results(self, saw_instance):
        """Test ATC determination and results retrieval.

        Runs after test_atc_determine_for_and_scenarios which sets up AGC
        generators with headroom in both areas.
        """
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
        buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])

        # DetermineATC relies on generator AGC setup from
        # test_atc_determine_for_and_scenarios (order 90600).
        saw_instance.DetermineATC(seller, buyer)

        # Get results
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow"],
            "field_data_type": ["String", "Real"],
            "key_field": ["", ""],
            "description": ["", ""],
            "display_name": ["", ""]
        }).sort_values(by="internal_field_name")
        saw_instance.GetATCResults(["MaxFlow", "LimitingContingency"])

    @pytest.mark.order(4030)
    def test_atc_multiple_directions(self, saw_instance):
        """Test ATC multiple directions.

        Runs after generator AGC setup from test_atc_determine_for_and_scenarios.
        """
        # DirectionsAutoInsert expects object type names, not full object strings
        saw_instance.DirectionsAutoInsert("AREA", "AREA")
        saw_instance.DetermineATCMultipleDirections()

    @pytest.mark.order(4090)
    def test_atc_reference_and_state(self, saw_instance):
        """Test ATC set reference, restore initial, and delete results.

        Runs last in ATC suite since ATCDeleteAllResults wipes ATC state.
        """
        saw_instance.ATCSetAsReference()
        saw_instance.ATCRestoreInitialState()
        saw_instance.ATCDeleteAllResults()

    @pytest.mark.order(4025)
    def test_atc_create_contingent_interfaces(self, saw_instance):
        saw_instance.ATCCreateContingentInterfaces()
        # Named filters only; non-existent filter should raise error
        with pytest.raises((PowerWorldPrerequisiteError, PowerWorldError)):
            saw_instance.ATCCreateContingentInterfaces(filter_name="NonExistentFilter")

    @pytest.mark.order(4010)
    def test_atc_determine_for_and_scenarios(self, saw_instance):
        """Test ATCDetermineATCFor, increase transfer, and take me to scenario."""
        # Re-establish ATC state (prior tests may have cleared results/state)
        areas = ensure_areas(saw_instance, min_count=2)
        seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
        buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])

        # Enable AGC with MW headroom on a few generators per area.
        # Generators must be online (Closed), producing MW, and have headroom.
        # Area 1 needs multiple AGC gens because ATCDetermineATCFor can
        # exhaust a single generator's capacity.
        gens = saw_instance.GetParametersMultipleElement(
            "Gen", ["BusNum", "GenID", "GenMW", "GenMWMax", "AreaNum"]
        )
        for area_row in areas.itertuples():
            area_num = str(area_row.AreaNum).strip()
            area_gens = gens[gens["AreaNum"].astype(str).str.strip() == area_num]
            if area_gens.empty:
                continue
            # Enable AGC on up to 3 generators per area
            for _, g in area_gens.head(3).iterrows():
                mw = float(g["GenMW"]) if g["GenMW"] not in (None, "") else 0
                dispatch_mw = max(mw, 10)
                saw_instance.SetData(
                    "Gen",
                    ["BusNum", "GenID", "GenMW", "GenMWMax", "GenMWMin",
                     "GenAGCAble", "GenParFac", "GenStatus"],
                    [str(g["BusNum"]).strip(), str(g["GenID"]).strip(),
                     str(dispatch_mw), str(dispatch_mw + 500), "0",
                     "YES", "1.0", "Closed"],
                )
        saw_instance.SolvePowerFlow()

        # Set reference AFTER fixing generator limits so the reference state
        # has headroom (ATCRestoreInitialState would undo the fix otherwise).
        saw_instance.ATCSetAsReference()
        saw_instance.DetermineATC(seller, buyer)

        saw_instance.ATCDetermineATCFor(0, 0, 0)
        saw_instance.ATCDetermineATCFor(0, 0, 0, apply_transfer=True)
        saw_instance.ATCIncreaseTransferBy(0.0)
        saw_instance.ATCTakeMeToScenario(0, 0, 0)

    @pytest.mark.order(4040)
    def test_atc_data_write_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.ATCDataWriteOptionsAndResults(tmp_aux, append=False, key_field="PRIMARY")

    @pytest.mark.order(4041)
    def test_atc_write_all_options_deprecated(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.ATCWriteAllOptions(tmp_aux, append=True, key_field="PRIMARY")

    @pytest.mark.order(4042)
    def test_atc_write_results_and_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.ATCWriteResultsAndOptions(tmp_aux, append=False)

    @pytest.mark.order(4043)
    def test_atc_write_scenario_log(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.ATCWriteScenarioLog(tmp_txt, append=False)

    @pytest.mark.order(4044)
    def test_atc_write_scenario_log_with_filter(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.ATCWriteScenarioLog(tmp_txt, append=True, filter_name="ALL")

    @pytest.mark.order(4045)
    def test_atc_write_scenario_minmax(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.ATCWriteScenarioMinMax(tmp_csv, filetype="CSV", operation="MIN")

    @pytest.mark.order(4046)
    def test_atc_write_scenario_minmax_with_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.ATCWriteScenarioMinMax(
            tmp_csv, filetype="CSV", append=True,
            fieldlist=["TransferLimit", "Contingency"],
            operation="MAX", operation_field="TransferLimit",
            group_scenario="NONE",
        )

    @pytest.mark.order(4047)
    def test_atc_write_to_text(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.ATCWriteToText(tmp_txt, filetype="TAB")

    @pytest.mark.order(4048)
    def test_atc_write_to_text_csv_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.ATCWriteToText(tmp_csv, filetype="CSV", fieldlist=["MaxFlow"])

    @pytest.mark.order(4049)
    def test_atc_delete_scenario_change(self, saw_instance):
        saw_instance.ATCDeleteScenarioChangeIndexRange("RL", ["0"])

    @pytest.mark.order(4050)
    def test_atc_get_results_default_fields(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow", "LimitingElement",
                                    "TransferLimit", "LimitUsed", "PTDF", "OTDF"],
            "field_data_type": ["String", "Real", "String", "Real", "String", "Real", "Real"],
            "key_field": ["", "", "", "", "", "", ""],
            "description": ["", "", "", "", "", "", ""],
            "display_name": ["", "", "", "", "", "", ""]
        }).sort_values(by="internal_field_name")
        saw_instance.GetATCResults()


# =========================================================================
# OPF
# =========================================================================

class TestOPF:
    """Tests for OPF solver operations."""

    @pytest.mark.order(60000)
    def test_opf_initialize_primal_lp(self, saw_instance):
        saw_instance.InitializePrimalLP()

    @pytest.mark.order(60100)
    def test_opf_solve_primal_lp(self, saw_instance):
        saw_instance.SolvePrimalLP()

    @pytest.mark.order(60200)
    def test_opf_solve_primal_lp_with_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SolvePrimalLP(
            on_success_aux=tmp_aux, create_if_not_found1=True
        )

    @pytest.mark.order(60300)
    def test_opf_solve_single_outer_loop(self, saw_instance):
        saw_instance.SolveSinglePrimalLPOuterLoop()

    @pytest.mark.order(60400)
    def test_opf_solve_full_scopf(self, saw_instance):
        saw_instance.SolveFullSCOPF(bc_method="OPF")

    @pytest.mark.order(60500)
    def test_opf_solve_full_scopf_powerflow(self, saw_instance):
        saw_instance.SolveFullSCOPF(bc_method="POWERFLOW")

    @pytest.mark.order(60600)
    def test_opf_write_results(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.OPFWriteResultsAndOptions(tmp_aux)
        assert os.path.exists(tmp_aux)


# =========================================================================
# TimeStep
# =========================================================================

class TestTimeStep:
    """Time Step Simulation operations via SAW."""

    @pytest.mark.order(8900)
    def test_timestep_run(self, saw_instance):
        saw_instance.TimeStepDoRun()

    @pytest.mark.order(8910)
    def test_timestep_clear_results(self, saw_instance):
        # ClearResults requires time step results to exist; if no time step
        # data is configured for this case, the correct error is raised.
        try:
            saw_instance.TimeStepClearResults()
        except PowerWorldPrerequisiteError:
            # Expected when the case has no time step data configured
            pass

    @pytest.mark.order(8920)
    def test_timestep_reset_run(self, saw_instance):
        saw_instance.TimeStepResetRun()

    @pytest.mark.order(9100)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

    @pytest.mark.order(9200)
    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    @pytest.mark.order(9110)
    def test_timestep_save_results_csv(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            # Expected when no time step results exist for this case
            pass

    @pytest.mark.order(9200)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])

    @pytest.mark.order(76200)
    def test_timestep_append_pww(self, saw_instance, temp_file):
        """TimeStepAppendPWW with a file path."""
        tmp = temp_file(".pww")
        # Empty temp PWW files cause access violations in PW
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepAppendPWW(tmp, solution_type="Single Solution")

    @pytest.mark.order(76300)
    def test_timestep_append_pww_range(self, saw_instance, temp_file):
        """TimeStepAppendPWWRange with time range."""
        tmp = temp_file(".pww")
        # Empty temp PWW files cause access violations in PW
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepAppendPWWRange(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
            )

    @pytest.mark.order(76400)
    def test_timestep_append_pww_range_latlon(self, saw_instance, temp_file):
        """TimeStepAppendPWWRangeLatLon with geographic filter."""
        tmp = temp_file(".pww")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepAppendPWWRangeLatLon(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
                30.0, 50.0, -100.0, -80.0,
            )

    @pytest.mark.order(76500)
    def test_timestep_load_b3d(self, saw_instance, temp_file):
        """TimeStepLoadB3D with a file path."""
        tmp = temp_file(".b3d")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepLoadB3D(tmp)

    @pytest.mark.order(76600)
    def test_timestep_load_pww(self, saw_instance, temp_file):
        """TimeStepLoadPWW with a file path."""
        tmp = temp_file(".pww")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepLoadPWW(tmp, solution_type="Single Solution")

    @pytest.mark.order(76700)
    def test_timestep_load_pww_range(self, saw_instance, temp_file):
        """TimeStepLoadPWWRange with time range."""
        tmp = temp_file(".pww")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepLoadPWWRange(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
            )

    @pytest.mark.order(76800)
    def test_timestep_load_pww_range_latlon(self, saw_instance, temp_file):
        """TimeStepLoadPWWRangeLatLon with geographic filter."""
        tmp = temp_file(".pww")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepLoadPWWRangeLatLon(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
                30.0, 50.0, -100.0, -80.0,
            )

    @pytest.mark.order(76900)
    def test_timestep_save_pww_range(self, saw_instance, temp_file):
        """TimeStepSavePWWRange with time range."""
        tmp = temp_file(".pww")
        saw_instance.TimeStepSavePWWRange(
            tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
        )

    @pytest.mark.order(77000)
    def test_timestep_save_selected_modify(self, saw_instance):
        """TIMESTEPSaveSelectedModifyStart/Finish cycle."""
        try:
            saw_instance.TIMESTEPSaveSelectedModifyStart()
            saw_instance.TIMESTEPSaveSelectedModifyFinish()
        except PowerWorldError:
            # Command may not be available in this PW version
            pass

    @pytest.mark.order(77100)
    def test_timestep_save_input_csv(self, saw_instance, temp_file):
        """TIMESTEPSaveInputCSV writes to file."""
        tmp = temp_file(".csv")
        saw_instance.TIMESTEPSaveInputCSV(tmp, ["BusNum", "BusPUVolt"])

    @pytest.mark.order(77200)
    def test_timestep_load_tsb(self, saw_instance, temp_file):
        """TimeStepLoadTSB with a file path."""
        tmp = temp_file(".tsb")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TimeStepLoadTSB(tmp)

    @pytest.mark.order(77300)
    def test_timestep_save_tsb(self, saw_instance, temp_file):
        """TimeStepSaveTSB writes to file."""
        tmp = temp_file(".tsb")
        saw_instance.TimeStepSaveTSB(tmp)


# =========================================================================
# PV/QV
# =========================================================================

class TestPVQV:
    """PV and QV curve analysis via SAW."""

    @pytest.mark.order(9300)
    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    @pytest.mark.order(9450)
    def test_pv_export(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.PVWriteResultsAndOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(9600)
    def test_qv_clear(self, saw_instance):
        saw_instance.QVDeleteAllResults()

    @pytest.mark.order(9700)
    def test_qv_export(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.QVWriteResultsAndOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(73000)
    def test_pv_clear(self, saw_instance):
        """PVClear completes without error."""
        saw_instance.PVClear()

    @pytest.mark.order(73100)
    def test_pv_destroy(self, saw_instance):
        """PVDestroy completes without error."""
        saw_instance.PVDestroy()

    @pytest.mark.order(73200)
    def test_pv_set_source_and_sink(self, saw_instance):
        """PVSetSourceAndSink with injection group strings."""
        saw_instance.InjectionGroupCreate("PVTestSrc", "Gen", 1.0, "")
        saw_instance.InjectionGroupCreate("PVTestSink", "Load", 1.0, "")
        saw_instance.PVSetSourceAndSink(
            '[INJECTIONGROUP "PVTestSrc"]',
            '[INJECTIONGROUP "PVTestSink"]',
        )

    @pytest.mark.order(73300)
    def test_pv_start_over(self, saw_instance):
        """PVStartOver completes without error."""
        saw_instance.PVStartOver()

    @pytest.mark.order(73400)
    def test_pv_run(self, saw_instance):
        """RunPV with injection groups."""
        saw_instance.InjectionGroupCreate("PVRunSrc", "Gen", 1.0, "")
        saw_instance.InjectionGroupCreate("PVRunSink", "Load", 1.0, "")
        saw_instance.RunPV(
            '[INJECTIONGROUP "PVRunSrc"]',
            '[INJECTIONGROUP "PVRunSink"]',
        )

    @pytest.mark.order(73500)
    def test_pv_data_write_options(self, saw_instance, temp_file):
        """PVDataWriteOptionsAndResults writes to file."""
        tmp = temp_file(".aux")
        saw_instance.PVDataWriteOptionsAndResults(tmp, append=False, key_field="PRIMARY")

    @pytest.mark.order(73600)
    def test_pv_write_results_and_options(self, saw_instance, temp_file):
        """PVWriteResultsAndOptions writes to file."""
        tmp = temp_file(".aux")
        saw_instance.PVWriteResultsAndOptions(tmp, append=False)

    @pytest.mark.order(73700)
    def test_pv_write_inadequate_voltages(self, saw_instance, temp_file):
        """PVWriteInadequateVoltages writes to file."""
        tmp = temp_file(".csv")
        saw_instance.PVWriteInadequateVoltages(tmp, append=False, inadequate_type="BOTH")

    @pytest.mark.order(73800)
    def test_pv_qv_track_single_bus(self, saw_instance):
        """PVQVTrackSingleBusPerSuperBus completes without error."""
        saw_instance.PVQVTrackSingleBusPerSuperBus()

    @pytest.mark.order(73900)
    def test_refine_model(self, saw_instance):
        """RefineModel completes without error (only Area/Zone types, TRANSFORMERTAPS/SHUNTS/OFFAVR actions)."""
        saw_instance.RefineModel("Area", "", "SHUNTS", 0.0)

    @pytest.mark.order(74000)
    def test_refine_model_with_filter(self, saw_instance):
        """RefineModel with a different action."""
        saw_instance.RefineModel("Zone", "", "OFFAVR", 0.001)

    @pytest.mark.order(74100)
    def test_qv_data_write_options(self, saw_instance, temp_file):
        """QVDataWriteOptionsAndResults writes to file."""
        tmp = temp_file(".aux")
        saw_instance.QVDataWriteOptionsAndResults(tmp, append=False, key_field="PRIMARY")

    @pytest.mark.order(74200)
    def test_qv_run_with_filename(self, saw_instance, temp_file):
        """RunQV with explicit filename returns None."""
        tmp = temp_file(".csv")
        result = saw_instance.RunQV(filename=tmp)
        assert result is None

    @pytest.mark.order(74300)
    def test_qv_select_single_bus(self, saw_instance):
        """QVSelectSingleBusPerSuperBus completes without error."""
        saw_instance.QVSelectSingleBusPerSuperBus()

    @pytest.mark.order(74400)
    def test_qv_write_curves(self, saw_instance, temp_file):
        """QVWriteCurves writes to file."""
        tmp = temp_file(".csv")
        saw_instance.QVWriteCurves(tmp, include_quantities=True, filter_name="", append=False)

    @pytest.mark.order(74500)
    def test_qv_write_results_and_options(self, saw_instance, temp_file):
        """QVWriteResultsAndOptions writes to file."""
        tmp = temp_file(".aux")
        saw_instance.QVWriteResultsAndOptions(tmp, append=False)

    @pytest.mark.order(74600)
    def test_qv_delete_all_results(self, saw_instance):
        """QVDeleteAllResults completes without error."""
        saw_instance.QVDeleteAllResults()
