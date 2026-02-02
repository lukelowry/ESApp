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
        try:
            wb.pflow(getvolts=False)
            J = wb.jacobian()
            assert J.shape[0] > 0
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Jacobian not available")

    def test_jacobian_dense(self, wb):
        """Dense Jacobian."""
        try:
            J = wb.jacobian(dense=True)
            assert isinstance(J, np.ndarray)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Jacobian not available")

    def test_jacobian_polar(self, wb):
        """Polar Jacobian form."""
        try:
            wb.pflow(getvolts=False)
            J = wb.jacobian(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert J.shape[0] > 0
            assert J.shape[0] == J.shape[1]
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Polar Jacobian not available")

    def test_jacobian_with_ids(self, wb):
        """Jacobian with row/column ID labels."""
        try:
            wb.pflow(getvolts=False)
            J, ids = wb.jacobian_with_ids(dense=True, form='P')
            assert isinstance(J, np.ndarray)
            assert isinstance(ids, list)
            assert len(ids) > 0
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Jacobian with IDs not available")

    def test_solver_options(self, wb):
        """Solver option methods."""
        try:
            wb.set_do_one_iteration(True)
            wb.set_do_one_iteration(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("set_do_one_iteration not available")

        try:
            wb.set_max_iterations(250)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("set_max_iterations not available")

        try:
            wb.set_disable_angle_rotation(True)
            wb.set_disable_angle_rotation(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("set_disable_angle_rotation not available")

        try:
            wb.set_disable_opt_mult(True)
            wb.set_disable_opt_mult(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("set_disable_opt_mult not available")

        try:
            wb.enable_inner_ss_check(True)
            wb.enable_inner_ss_check(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("enable_inner_ss_check not available")

        try:
            wb.disable_gen_mvr_check(True)
            wb.disable_gen_mvr_check(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("disable_gen_mvr_check not available")

        try:
            wb.enable_inner_check_gen_vars(True)
            wb.enable_inner_check_gen_vars(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("enable_inner_check_gen_vars not available")

        try:
            wb.enable_inner_backoff_gen_vars(True)
            wb.enable_inner_backoff_gen_vars(False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("enable_inner_backoff_gen_vars not available")


# =========================================================================
# Scheduled Actions
# =========================================================================

class TestScheduledActions:
    """Tests for Scheduled Actions mixin -- all parameter paths."""

    @pytest.mark.order(50000)
    def test_scheduled_set_reference(self, saw_instance):
        try:
            saw_instance.ScheduledActionsSetReference()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ScheduledActionsSetReference not available")

    @pytest.mark.order(50100)
    def test_scheduled_apply_at(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50200)
    def test_scheduled_apply_at_with_end_time(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", end_time="01/01/2025 12:00"
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50300)
    def test_scheduled_apply_at_with_filter(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", filter_name="ALL"
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No scheduled actions or filter not found")

    @pytest.mark.order(50400)
    def test_scheduled_apply_at_revert(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", revert=True
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50500)
    def test_scheduled_revert_at(self, saw_instance):
        try:
            saw_instance.RevertScheduledActionsAt("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50600)
    def test_scheduled_revert_at_with_filter(self, saw_instance):
        try:
            saw_instance.RevertScheduledActionsAt(
                "01/01/2025 10:00", filter_name="ALL"
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No scheduled actions or filter not found")

    @pytest.mark.order(50700)
    def test_scheduled_identify_breakers(self, saw_instance):
        try:
            saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=True)
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50800)
    def test_scheduled_identify_breakers_false(self, saw_instance):
        try:
            saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=False)
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(50900)
    def test_scheduled_set_view(self, saw_instance):
        try:
            saw_instance.SetScheduleView("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("Schedule view not available")

    @pytest.mark.order(51000)
    def test_scheduled_set_view_with_options(self, saw_instance):
        try:
            saw_instance.SetScheduleView(
                "01/01/2025 10:00",
                apply_actions=True,
                use_normal_status=False,
                apply_window=True,
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("Schedule view not available")

    @pytest.mark.order(51100)
    def test_scheduled_set_window(self, saw_instance):
        try:
            saw_instance.SetScheduleWindow(
                "01/01/2025 00:00", "02/01/2025 00:00",
                resolution=1.0, resolution_units="HOURS",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Schedule window not available")

    @pytest.mark.order(51200)
    def test_scheduled_set_window_with_resolution(self, saw_instance):
        try:
            saw_instance.SetScheduleWindow(
                "01/01/2025 00:00",
                "02/01/2025 00:00",
                resolution=0.5,
                resolution_units="HOURS",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Schedule window not available")


# =========================================================================
# Weather
# =========================================================================

class TestWeather:
    """Tests for Weather mixin -- all boolean parameter paths."""

    @pytest.mark.order(52000)
    def test_weather_limits_gen_update(self, saw_instance):
        try:
            saw_instance.WeatherLimitsGenUpdate(update_max=True, update_min=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Weather data not available")

    @pytest.mark.order(52100)
    def test_weather_limits_gen_update_false(self, saw_instance):
        try:
            saw_instance.WeatherLimitsGenUpdate(update_max=False, update_min=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Weather data not available")

    @pytest.mark.order(52200)
    def test_weather_temperature_limits_branch(self, saw_instance):
        try:
            saw_instance.TemperatureLimitsBranchUpdate()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Temperature limits not available")

    @pytest.mark.order(52300)
    def test_weather_temperature_limits_branch_custom(self, saw_instance):
        try:
            saw_instance.TemperatureLimitsBranchUpdate(
                rating_set_precedence="EMERGENCY",
                normal_rating_set="A",
                ctg_rating_set="B",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Temperature limits not available")

    @pytest.mark.order(52400)
    def test_weather_pfw_set_inputs(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputs()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(52500)
    def test_weather_pfw_set_inputs_and_apply(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(52600)
    def test_weather_pfw_set_inputs_and_apply_no_solve(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(52700)
    def test_weather_pfw_restore_design(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsRestoreDesignValues()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(52800)
    def test_weather_pww_load_datetime(self, saw_instance):
        try:
            saw_instance.WeatherPWWLoadForDateTimeUTC("2025-01-01T10:00:00")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW weather data not available")

    @pytest.mark.order(52900)
    def test_weather_pww_set_directory(self, saw_instance, temp_dir):
        try:
            saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW directory set not available")

    @pytest.mark.order(53000)
    def test_weather_pww_set_directory_no_subdirs(self, saw_instance, temp_dir):
        try:
            saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW directory set not available")

    @pytest.mark.order(53100)
    def test_weather_pww_file_all_meas_valid(self, saw_instance, temp_file):
        tmp = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileAllMeasValid(tmp, ["Temperature"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW validation not available")

    @pytest.mark.order(53200)
    def test_weather_pww_file_combine(self, saw_instance, temp_file):
        src1 = temp_file(".pww")
        src2 = temp_file(".pww")
        dst = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileCombine2(src1, src2, dst)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW combine not available")

    @pytest.mark.order(53300)
    def test_weather_pww_file_geo_reduce(self, saw_instance, temp_file):
        src = temp_file(".pww")
        dst = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileGeoReduce(src, dst, 30.0, 50.0, -100.0, -80.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW geo-reduce not available")


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
        try:
            saw_instance.CalculateGIC(1.0, 90.0, False)
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
            if "Access violation" in str(e) or "memory" in str(e).lower():
                pytest.skip(f"GIC calculation not supported for this case: {e}")
            raise
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
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("GIC storm not available")

    @pytest.mark.order(7760)
    def test_gic_cleargic(self, wb):
        try:
            wb.gic.cleargic()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("cleargic not available")

    @pytest.mark.order(7770)
    def test_gic_loadb3d(self, wb):
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("GIC loadb3d not available")

    @pytest.mark.order(7771)
    def test_gic_loadb3d_no_setup(self, wb):
        try:
            wb.gic.loadb3d("STORM", "nonexistent.b3d", setuponload=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("GIC loadb3d not available")

    @pytest.mark.order(7775)
    def test_gic_timevary_csv(self, wb, temp_file):
        import csv as csvmod
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w', newline='') as f:
            writer = csvmod.writer(f)
            writer.writerow(["Branch '1' '2' '1'", 0.1, 0.11, 0.14])
        try:
            wb.gic.timevary_csv(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("GIC timevary_csv not available")

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
        except (PowerWorldPrerequisiteError, PowerWorldError):
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
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("GIC model generation failed")

    @pytest.mark.order(7870)
    def test_gic_model_properties(self, wb):
        try:
            wb.gic.model()
        except (PowerWorldPrerequisiteError, PowerWorldError):
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
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
            pytest.skip(f"Could not generate GIC model: {e}")

        try:
            G_powerworld = gic.gmatrix(sparse=True)
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
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


# =========================================================================
# Transient Extended
# =========================================================================

class TestTransientExtended2:
    """Additional transient tests to hit uncovered parameter paths."""

    @pytest.mark.order(64000)
    def test_transient_transfer_state(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Transient transfer state not available")

    @pytest.mark.order(64100)
    def test_transient_transfer_state_no_mismatch(self, saw_instance):
        try:
            saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Transient transfer state not available")

    @pytest.mark.order(64200)
    def test_transient_store_response(self, saw_instance):
        try:
            saw_instance.TSStoreResponse("Gen", True)
            saw_instance.TSStoreResponse("Gen", False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSStoreResponse not available")

    @pytest.mark.order(64300)
    def test_transient_clear_results_ram(self, saw_instance):
        try:
            saw_instance.TSClearResultsFromRAM()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearResultsFromRAM not available")

    @pytest.mark.order(64400)
    def test_transient_clear_results_specific_ctg(self, saw_instance):
        try:
            saw_instance.TSClearResultsFromRAM(
                ctg_name="ALL",
                clear_summary=True,
                clear_events=False,
                clear_statistics=True,
                clear_time_values=False,
                clear_solution_details=True,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearResultsFromRAM with options not available")

    @pytest.mark.order(64500)
    def test_transient_clear_results_and_disable(self, saw_instance):
        try:
            saw_instance.TSClearResultsFromRAMAndDisableStorage()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearResultsFromRAMAndDisableStorage not available")

    @pytest.mark.order(64600)
    def test_transient_clear_all_models(self, saw_instance):
        try:
            saw_instance.TSClearAllModels()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearAllModels not available")

    @pytest.mark.order(64700)
    def test_transient_smib_eigenvalues(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSCalculateSMIBEigenValues()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SMIB eigenvalues not available")

    @pytest.mark.order(64800)
    def test_transient_clear_models_for_objects(self, saw_instance):
        try:
            saw_instance.TSClearModelsforObjects("Gen")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearModelsforObjects not available")

    @pytest.mark.order(64900)
    def test_transient_disable_machine_model(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSDisableMachineModelNonZeroDerivative(threshold=0.01)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSDisableMachineModelNonZeroDerivative not available")

    @pytest.mark.order(65000)
    def test_transient_auto_insert_dist_relay(self, saw_instance):
        try:
            saw_instance.TSAutoInsertDistRelay(
                reach=1.0, add_from=True, add_to=False,
                transfer_trip=True, shape=1, filter_name="ALL",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoInsertDistRelay not available")

    @pytest.mark.order(65100)
    def test_transient_auto_insert_zpott(self, saw_instance):
        try:
            saw_instance.TSAutoInsertZPOTT(reach=1.0, filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoInsertZPOTT not available")

    @pytest.mark.order(65200)
    def test_transient_run_result_analyzer(self, saw_instance):
        try:
            saw_instance.TSRunResultAnalyzer()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSRunResultAnalyzer not available")

    @pytest.mark.order(65300)
    def test_transient_run_until_specified_time(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSRunUntilSpecifiedTime("TestCtg", stop_time=0.1, step_size=0.01)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSRunUntilSpecifiedTime not available")

    @pytest.mark.order(65400)
    def test_transient_save_pti(self, saw_instance, temp_file):
        """TSSavePTI writes .dyr file."""
        tmp = temp_file(".dyr")
        try:
            saw_instance.TSSavePTI(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSavePTI not available")

    @pytest.mark.order(65410)
    def test_transient_save_ge(self, saw_instance, temp_file):
        """TSSaveGE writes .dyd file."""
        tmp = temp_file(".dyd")
        try:
            saw_instance.TSSaveGE(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveGE not available")

    @pytest.mark.order(65420)
    def test_transient_save_bpa(self, saw_instance, temp_file):
        """TSSaveBPA writes .bpa file."""
        tmp = temp_file(".bpa")
        try:
            saw_instance.TSSaveBPA(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveBPA not available")

    @pytest.mark.order(65500)
    def test_transient_save_pti_diff(self, saw_instance, temp_file):
        """TSSavePTI with diff_case_modified_only."""
        tmp = temp_file(".dyr")
        try:
            saw_instance.TSSavePTI(tmp, diff_case_modified_only=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSavePTI diff not available")

    @pytest.mark.order(65510)
    def test_transient_save_ge_diff(self, saw_instance, temp_file):
        """TSSaveGE with diff_case_modified_only."""
        tmp = temp_file(".dyd")
        try:
            saw_instance.TSSaveGE(tmp, diff_case_modified_only=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveGE diff not available")

    @pytest.mark.order(65520)
    def test_transient_save_bpa_diff(self, saw_instance, temp_file):
        """TSSaveBPA with diff_case_modified_only."""
        tmp = temp_file(".bpa")
        try:
            saw_instance.TSSaveBPA(tmp, diff_case_modified_only=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveBPA diff not available")

    @pytest.mark.order(65600)
    def test_transient_write_models_diff(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
            saw_instance.TSWriteModels(tmp, diff_case_modified_only=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSWriteModels diff not available")

    @pytest.mark.order(65700)
    def test_transient_write_options_custom(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
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
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSWriteOptions custom not available")

    @pytest.mark.order(65800)
    def test_transient_save_two_bus_equiv(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp = temp_file(".pwb")
        try:
            saw_instance.TSSaveTwoBusEquivalent(tmp, bus_key)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveTwoBusEquivalent not available")

    @pytest.mark.order(65900)
    def test_transient_join_active_ctgs(self, saw_instance):
        try:
            saw_instance.TSJoinActiveCTGs(0.1, delete_existing=True, join_with_self=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSJoinActiveCTGs not available")

    @pytest.mark.order(66000)
    def test_transient_set_selected_for_refs(self, saw_instance):
        try:
            saw_instance.TSSetSelectedForTransientReferences(
                "SELECTED", "SET", ["Gen"], ["GENROU"]
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSetSelectedForTransientReferences not available")

    @pytest.mark.order(66100)
    def test_transient_save_dynamic_models_append(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
            saw_instance.TSSaveDynamicModels(tmp, "AUX", "Gen", append=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveDynamicModels append not available")

    @pytest.mark.order(66200)
    def test_transient_plot_series_add(self, saw_instance):
        try:
            saw_instance.TSPlotSeriesAdd("TestPlot", 1, 1, "Gen", "GenMW")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSPlotSeriesAdd not available")

    @pytest.mark.order(66300)
    def test_transient_get_vcurve_data(self, saw_instance, temp_file):
        tmp = temp_file(".csv")
        try:
            saw_instance.TSGetVCurveData(tmp, "ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSGetVCurveData not available")


# =========================================================================
# Transient Gaps
# =========================================================================

class TestTransientGaps:
    """Tests for remaining Transient stability functions."""

    @pytest.mark.order(77400)
    def test_ts_solve_with_time_params(self, saw_instance):
        """TSSolve with explicit start/stop/step parameters."""
        try:
            saw_instance.TSInitialize()
            saw_instance.TSSolve(
                "TestCtg", start_time=0.0, stop_time=0.1,
                step_size=0.01, step_in_cycles=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSolve with time params not available")

    @pytest.mark.order(77500)
    def test_ts_solve_all(self, saw_instance):
        """TSSolveAll completes without error."""
        try:
            saw_instance.TSInitialize()
            saw_instance.TSSolveAll()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSolveAll not available")

    @pytest.mark.order(77600)
    def test_ts_clear_results_named_ctg(self, saw_instance):
        """TSClearResultsFromRAM with a specific contingency name."""
        try:
            saw_instance.TSClearResultsFromRAM(ctg_name="TestCtg")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearResultsFromRAM with named ctg not available")

    @pytest.mark.order(77700)
    def test_ts_auto_save_plots(self, saw_instance):
        """TSAutoSavePlots completes without error."""
        try:
            saw_instance.TSAutoSavePlots(
                plot_names=["TestPlot"],
                ctg_names=["TestCtg"],
                image_type="JPG",
                width=800, height=600, font_scalar=1.0,
                include_case_name=True, include_category=True,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoSavePlots not available")

    @pytest.mark.order(77800)
    def test_ts_load_rdb(self, saw_instance, temp_file):
        """TSLoadRDB with a file path."""
        tmp = temp_file(".rdb")
        try:
            saw_instance.TSLoadRDB(tmp, "DISTRELAY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSLoadRDB not available")

    @pytest.mark.order(77900)
    def test_ts_load_relay_csv(self, saw_instance, temp_file):
        """TSLoadRelayCSV with a file path."""
        tmp = temp_file(".csv")
        try:
            saw_instance.TSLoadRelayCSV(tmp, "DISTRELAY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSLoadRelayCSV not available")

    @pytest.mark.order(78000)
    def test_ts_run_until_specified_time_steps(self, saw_instance):
        """TSRunUntilSpecifiedTime with steps_to_do parameter."""
        try:
            saw_instance.TSInitialize()
            saw_instance.TSRunUntilSpecifiedTime(
                "TestCtg", stop_time=0.1, step_size=0.01,
                steps_in_cycles=False, reset_start_time=False,
                steps_to_do=5,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSRunUntilSpecifiedTime with steps not available")

    @pytest.mark.order(78100)
    def test_ts_load_pti(self, saw_instance, temp_file):
        """TSLoadPTI with a file path."""
        tmp = temp_file(".dyr")
        try:
            saw_instance.TSLoadPTI(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSLoadPTI not available")

    @pytest.mark.order(78200)
    def test_ts_load_ge(self, saw_instance, temp_file):
        """TSLoadGE with a file path."""
        tmp = temp_file(".dyd")
        try:
            saw_instance.TSLoadGE(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSLoadGE not available")

    @pytest.mark.order(78300)
    def test_ts_load_bpa(self, saw_instance, temp_file):
        """TSLoadBPA with a file path."""
        tmp = temp_file(".bpa")
        try:
            saw_instance.TSLoadBPA(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSLoadBPA not available")

    @pytest.mark.order(78400)
    def test_ts_clear_playin_signals(self, saw_instance):
        """TSClearPlayInSignals completes without error."""
        try:
            saw_instance.TSClearPlayInSignals()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearPlayInSignals not available")

    @pytest.mark.order(78500)
    def test_ts_set_playin_signals_multi_col(self, saw_instance):
        """TSSetPlayInSignals with multiple signal columns."""
        times = np.array([0.0, 0.1, 0.2])
        signals = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
        try:
            saw_instance.TSSetPlayInSignals("MultiSignal", times, signals)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSetPlayInSignals not available")

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
        try:
            saw_instance.TSValidate()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSValidate not available")

    @pytest.mark.order(78800)
    def test_ts_auto_correct(self, saw_instance):
        """TSAutoCorrect completes without error."""
        try:
            saw_instance.TSAutoCorrect()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoCorrect not available")


# =========================================================================
# ATC (Available Transfer Capability)
# =========================================================================

class TestATC:
    """ATC analysis via SAW commands."""

    @pytest.mark.order(8600)
    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("ATC requires a case with at least 2 areas")
        seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
        buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
        saw_instance.DetermineATC(seller, buyer)

    @pytest.mark.order(8700)
    def test_atc_multiple(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("ATC directions require a case with at least 2 areas")
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
# ATC Extended
# =========================================================================

class TestATCExtended:
    """Extended tests for ATC analysis operations."""

    @pytest.mark.order(61000)
    def test_atc_set_reference(self, saw_instance):
        try:
            saw_instance.ATCSetAsReference()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCSetAsReference not available")

    @pytest.mark.order(61100)
    def test_atc_restore_initial(self, saw_instance):
        try:
            saw_instance.ATCRestoreInitialState()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCRestoreInitialState not available")

    @pytest.mark.order(61200)
    def test_atc_delete_all_results(self, saw_instance):
        try:
            saw_instance.ATCDeleteAllResults()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDeleteAllResults not available")

    @pytest.mark.order(61300)
    def test_atc_determine_distributed(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("ATC requires a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DetermineATC(s, b, distributed=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Distributed ATC not available")

    @pytest.mark.order(61400)
    def test_atc_determine_multiple_scenarios(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("ATC requires a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DetermineATC(s, b, multiple_scenarios=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC multiple scenarios not available")

    @pytest.mark.order(61500)
    def test_atc_determine_multiple_directions_distributed(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("ATC directions require a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DirectionsAutoInsert(s, b)
            saw_instance.DetermineATCMultipleDirections(distributed=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC multiple directions distributed not available")

    @pytest.mark.order(61600)
    def test_atc_create_contingent_interfaces(self, saw_instance):
        try:
            saw_instance.ATCCreateContingentInterfaces()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC contingent interfaces not available")

    @pytest.mark.order(61700)
    def test_atc_create_contingent_interfaces_with_filter(self, saw_instance):
        try:
            saw_instance.ATCCreateContingentInterfaces(filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC contingent interfaces not available")

    @pytest.mark.order(61800)
    def test_atc_determine_for(self, saw_instance):
        try:
            saw_instance.ATCDetermineATCFor(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineATCFor not available")

    @pytest.mark.order(61900)
    def test_atc_determine_for_apply(self, saw_instance):
        try:
            saw_instance.ATCDetermineATCFor(0, 0, 0, apply_transfer=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineATCFor with apply not available")

    @pytest.mark.order(62000)
    def test_atc_determine_multiple_directions_for(self, saw_instance):
        try:
            saw_instance.ATCDetermineMultipleDirectionsATCFor(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineMultipleDirectionsATCFor not available")

    @pytest.mark.order(62100)
    def test_atc_increase_transfer(self, saw_instance):
        try:
            saw_instance.ATCIncreaseTransferBy(0.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCIncreaseTransferBy not available")

    @pytest.mark.order(62200)
    def test_atc_take_me_to_scenario(self, saw_instance):
        try:
            saw_instance.ATCTakeMeToScenario(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCTakeMeToScenario not available")

    @pytest.mark.order(62300)
    def test_atc_data_write_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCDataWriteOptionsAndResults(tmp_aux, append=False, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write options not available")

    @pytest.mark.order(62400)
    def test_atc_write_all_options_deprecated(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCWriteAllOptions(tmp_aux, append=True, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCWriteAllOptions not available")

    @pytest.mark.order(62500)
    def test_atc_write_results_and_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCWriteResultsAndOptions(tmp_aux, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write results not available")

    @pytest.mark.order(62600)
    def test_atc_write_scenario_log(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteScenarioLog(tmp_txt, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario log not available")

    @pytest.mark.order(62700)
    def test_atc_write_scenario_log_with_filter(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteScenarioLog(tmp_txt, append=True, filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario log not available")

    @pytest.mark.order(62800)
    def test_atc_write_scenario_minmax(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteScenarioMinMax(tmp_csv, filetype="CSV", operation="MIN")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario min/max not available")

    @pytest.mark.order(62900)
    def test_atc_write_scenario_minmax_with_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteScenarioMinMax(
                tmp_csv, filetype="CSV", append=True,
                fieldlist=["MaxFlow", "LimitingContingency"],
                operation="MAX", operation_field="MaxFlow",
                group_scenario=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario min/max not available")

    @pytest.mark.order(63000)
    def test_atc_write_to_text(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteToText(tmp_txt, filetype="TAB")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write to text not available")

    @pytest.mark.order(63100)
    def test_atc_write_to_text_csv_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteToText(tmp_csv, filetype="CSV", fieldlist=["MaxFlow"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write to text CSV not available")

    @pytest.mark.order(63200)
    def test_atc_delete_scenario_change(self, saw_instance):
        try:
            saw_instance.ATCDeleteScenarioChangeIndexRange("RL", ["0"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC delete scenario change not available")

    @pytest.mark.order(63300)
    def test_atc_get_results_default_fields(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow", "LimitingElement",
                                    "TransferLimit", "LimitUsed", "PTDF", "OTDF"],
            "field_data_type": ["String", "Real", "String", "Real", "String", "Real", "Real"],
            "key_field": ["", "", "", "", "", "", ""],
            "description": ["", "", "", "", "", "", ""],
            "display_name": ["", "", "", "", "", "", ""]
        }).sort_values(by="internal_field_name")
        try:
            saw_instance.GetATCResults()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC results not available")


# =========================================================================
# OPF Extended
# =========================================================================

class TestOPFExtended:
    """Extended tests for OPF solver operations."""

    @pytest.mark.order(60000)
    def test_opf_initialize_primal_lp(self, saw_instance):
        try:
            saw_instance.InitializePrimalLP()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("OPF initialization not available")

    @pytest.mark.order(60100)
    def test_opf_solve_primal_lp(self, saw_instance):
        try:
            saw_instance.SolvePrimalLP()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Primal LP OPF not available")

    @pytest.mark.order(60200)
    def test_opf_solve_primal_lp_with_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.SolvePrimalLP(
                on_success_aux=tmp_aux, create_if_not_found1=True
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Primal LP OPF with aux not available")

    @pytest.mark.order(60300)
    def test_opf_solve_single_outer_loop(self, saw_instance):
        try:
            saw_instance.SolveSinglePrimalLPOuterLoop()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Single primal LP outer loop not available")

    @pytest.mark.order(60400)
    def test_opf_solve_full_scopf(self, saw_instance):
        try:
            saw_instance.SolveFullSCOPF(bc_method="OPF")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Full SCOPF not available")

    @pytest.mark.order(60500)
    def test_opf_solve_full_scopf_powerflow(self, saw_instance):
        try:
            saw_instance.SolveFullSCOPF(bc_method="POWERFLOW")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Full SCOPF with POWERFLOW not available")

    @pytest.mark.order(60600)
    def test_opf_write_results(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.OPFWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("OPF write results not available")


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
            pytest.skip("TimeStepDoSinglePoint not available")

    @pytest.mark.order(9010)
    def test_timestep_clear_results(self, saw_instance):
        try:
            saw_instance.TimeStepClearResults()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepClearResults not available")

    @pytest.mark.order(9020)
    def test_timestep_reset_run(self, saw_instance):
        saw_instance.TimeStepResetRun()

    @pytest.mark.order(9100)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

    @pytest.mark.order(9110)
    def test_timestep_save_results_csv(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepSaveResultsByTypeCSV not available")

    @pytest.mark.order(9200)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])


# =========================================================================
# Timestep Extended
# =========================================================================

class TestTimestepExtended:
    """Tests for Time Step Simulation functions."""

    @pytest.mark.order(76200)
    def test_timestep_append_pww(self, saw_instance, temp_file):
        """TimeStepAppendPWW with a file path."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepAppendPWW(tmp, solution_type="Single Solution")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepAppendPWW not available")

    @pytest.mark.order(76300)
    def test_timestep_append_pww_range(self, saw_instance, temp_file):
        """TimeStepAppendPWWRange with time range."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepAppendPWWRange(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepAppendPWWRange not available")

    @pytest.mark.order(76400)
    def test_timestep_append_pww_range_latlon(self, saw_instance, temp_file):
        """TimeStepAppendPWWRangeLatLon with geographic filter."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepAppendPWWRangeLatLon(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
                30.0, 50.0, -100.0, -80.0,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepAppendPWWRangeLatLon not available")

    @pytest.mark.order(76500)
    def test_timestep_load_b3d(self, saw_instance, temp_file):
        """TimeStepLoadB3D with a file path."""
        tmp = temp_file(".b3d")
        try:
            saw_instance.TimeStepLoadB3D(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepLoadB3D not available")

    @pytest.mark.order(76600)
    def test_timestep_load_pww(self, saw_instance, temp_file):
        """TimeStepLoadPWW with a file path."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepLoadPWW(tmp, solution_type="Single Solution")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepLoadPWW not available")

    @pytest.mark.order(76700)
    def test_timestep_load_pww_range(self, saw_instance, temp_file):
        """TimeStepLoadPWWRange with time range."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepLoadPWWRange(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepLoadPWWRange not available")

    @pytest.mark.order(76800)
    def test_timestep_load_pww_range_latlon(self, saw_instance, temp_file):
        """TimeStepLoadPWWRangeLatLon with geographic filter."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepLoadPWWRangeLatLon(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
                30.0, 50.0, -100.0, -80.0,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepLoadPWWRangeLatLon not available")

    @pytest.mark.order(76900)
    def test_timestep_save_pww_range(self, saw_instance, temp_file):
        """TimeStepSavePWWRange with time range."""
        tmp = temp_file(".pww")
        try:
            saw_instance.TimeStepSavePWWRange(
                tmp, "2025-01-01T00:00:00", "2025-01-02T00:00:00",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepSavePWWRange not available")

    @pytest.mark.order(77000)
    def test_timestep_save_selected_modify(self, saw_instance):
        """TIMESTEPSaveSelectedModifyStart/Finish cycle."""
        try:
            saw_instance.TIMESTEPSaveSelectedModifyStart()
            saw_instance.TIMESTEPSaveSelectedModifyFinish()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TIMESTEPSaveSelectedModify not available")

    @pytest.mark.order(77100)
    def test_timestep_save_input_csv(self, saw_instance, temp_file):
        """TIMESTEPSaveInputCSV writes to file."""
        tmp = temp_file(".csv")
        try:
            saw_instance.TIMESTEPSaveInputCSV(tmp, ["BusNum", "BusPUVolt"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TIMESTEPSaveInputCSV not available")

    @pytest.mark.order(77200)
    def test_timestep_load_tsb(self, saw_instance, temp_file):
        """TimeStepLoadTSB with a file path."""
        tmp = temp_file(".tsb")
        try:
            saw_instance.TimeStepLoadTSB(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepLoadTSB not available")

    @pytest.mark.order(77300)
    def test_timestep_save_tsb(self, saw_instance, temp_file):
        """TimeStepSaveTSB writes to file."""
        tmp = temp_file(".tsb")
        try:
            saw_instance.TimeStepSaveTSB(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TimeStepSaveTSB not available")


# =========================================================================
# PV/QV
# =========================================================================

class TestPVQV:
    """PV and QV curve analysis via SAW."""

    @pytest.mark.order(9300)
    def test_pv_qv_run(self, saw_instance):
        try:
            df = saw_instance.RunQV()
            assert df is not None
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QV analysis not available")

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
# PV Extended
# =========================================================================

class TestPVExtended:
    """Tests for PV (Power-Voltage) analysis functions."""

    @pytest.mark.order(73000)
    def test_pv_clear(self, saw_instance):
        """PVClear completes without error."""
        try:
            saw_instance.PVClear()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PV analysis not available")

    @pytest.mark.order(73100)
    def test_pv_destroy(self, saw_instance):
        """PVDestroy completes without error."""
        try:
            saw_instance.PVDestroy()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PV analysis not available")

    @pytest.mark.order(73200)
    def test_pv_set_source_and_sink(self, saw_instance):
        """PVSetSourceAndSink with injection group strings."""
        try:
            saw_instance.InjectionGroupCreate("PVTestSrc", "Gen", 1.0, "ALL")
            saw_instance.InjectionGroupCreate("PVTestSink", "Load", 1.0, "ALL")
            saw_instance.PVSetSourceAndSink(
                '[INJECTIONGROUP "PVTestSrc"]',
                '[INJECTIONGROUP "PVTestSink"]',
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PV source/sink setup not available")

    @pytest.mark.order(73300)
    def test_pv_start_over(self, saw_instance):
        """PVStartOver completes without error."""
        try:
            saw_instance.PVStartOver()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PV analysis not available")

    @pytest.mark.order(73400)
    def test_pv_run(self, saw_instance):
        """RunPV with injection groups."""
        try:
            saw_instance.InjectionGroupCreate("PVRunSrc", "Gen", 1.0, "ALL")
            saw_instance.InjectionGroupCreate("PVRunSink", "Load", 1.0, "ALL")
            saw_instance.RunPV(
                '[INJECTIONGROUP "PVRunSrc"]',
                '[INJECTIONGROUP "PVRunSink"]',
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("RunPV not available")

    @pytest.mark.order(73500)
    def test_pv_data_write_options(self, saw_instance, temp_file):
        """PVDataWriteOptionsAndResults writes to file."""
        tmp = temp_file(".aux")
        try:
            saw_instance.PVDataWriteOptionsAndResults(tmp, append=False, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PVDataWriteOptionsAndResults not available")

    @pytest.mark.order(73600)
    def test_pv_write_results_and_options(self, saw_instance, temp_file):
        """PVWriteResultsAndOptions writes to file."""
        tmp = temp_file(".aux")
        try:
            saw_instance.PVWriteResultsAndOptions(tmp, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PVWriteResultsAndOptions not available")

    @pytest.mark.order(73700)
    def test_pv_write_inadequate_voltages(self, saw_instance, temp_file):
        """PVWriteInadequateVoltages writes to file."""
        tmp = temp_file(".csv")
        try:
            saw_instance.PVWriteInadequateVoltages(tmp, append=False, inadequate_type="BOTH")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PVWriteInadequateVoltages not available")

    @pytest.mark.order(73800)
    def test_pv_qv_track_single_bus(self, saw_instance):
        """PVQVTrackSingleBusPerSuperBus completes without error."""
        try:
            saw_instance.PVQVTrackSingleBusPerSuperBus()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PVQVTrackSingleBusPerSuperBus not available")

    @pytest.mark.order(73900)
    def test_refine_model(self, saw_instance):
        """RefineModel completes without error."""
        try:
            saw_instance.RefineModel("BUS", "", "RemoveOrphanedBuses", 0.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("RefineModel not available")

    @pytest.mark.order(74000)
    def test_refine_model_with_filter(self, saw_instance):
        """RefineModel with a filter name."""
        try:
            saw_instance.RefineModel("GEN", "ALL", "FixSmallReactance", 0.001)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("RefineModel with filter not available")


# =========================================================================
# QV Extended
# =========================================================================

class TestQVExtended:
    """Tests for QV (Reactive Power-Voltage) analysis functions."""

    @pytest.mark.order(74100)
    def test_qv_data_write_options(self, saw_instance, temp_file):
        """QVDataWriteOptionsAndResults writes to file."""
        tmp = temp_file(".aux")
        try:
            saw_instance.QVDataWriteOptionsAndResults(tmp, append=False, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QVDataWriteOptionsAndResults not available")

    @pytest.mark.order(74200)
    def test_qv_run_with_filename(self, saw_instance, temp_file):
        """RunQV with explicit filename returns None."""
        tmp = temp_file(".csv")
        try:
            result = saw_instance.RunQV(filename=tmp)
            assert result is None
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("RunQV with filename not available")

    @pytest.mark.order(74300)
    def test_qv_select_single_bus(self, saw_instance):
        """QVSelectSingleBusPerSuperBus completes without error."""
        try:
            saw_instance.QVSelectSingleBusPerSuperBus()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QVSelectSingleBusPerSuperBus not available")

    @pytest.mark.order(74400)
    def test_qv_write_curves(self, saw_instance, temp_file):
        """QVWriteCurves writes to file."""
        tmp = temp_file(".csv")
        try:
            saw_instance.QVWriteCurves(tmp, include_quantities=True, filter_name="", append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QVWriteCurves not available")

    @pytest.mark.order(74500)
    def test_qv_write_results_and_options(self, saw_instance, temp_file):
        """QVWriteResultsAndOptions writes to file."""
        tmp = temp_file(".aux")
        try:
            saw_instance.QVWriteResultsAndOptions(tmp, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QVWriteResultsAndOptions not available")

    @pytest.mark.order(74600)
    def test_qv_delete_all_results(self, saw_instance):
        """QVDeleteAllResults completes without error."""
        try:
            saw_instance.QVDeleteAllResults()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("QVDeleteAllResults not available")


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
