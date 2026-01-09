"""
Integration tests for Power Flow functionality against a live PowerWorld case.

WHAT THIS TESTS:
- Power flow solution execution and result validation
- Matrices (Ybus, Jacobian, Incidence)
- Sensitivity calculations (PTDF, LODF, shift factors)
- Diff case operations

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd

# Order markers for integration tests - powerflow tests run early (order 10-29)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, create_object_string
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


class TestPowerFlow:
    """Tests for power flow solution and related operations."""

    @pytest.mark.order(10)
    def test_powerflow_solve(self, saw_instance):
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(11)
    def test_powerflow_solve_retry(self, saw_instance):
        saw_instance.SolvePowerFlowWithRetry()

    @pytest.mark.order(12)
    def test_powerflow_clear_solution_aid(self, saw_instance):
        saw_instance.ClearPowerFlowSolutionAidValues()

    @pytest.mark.order(13)
    def test_powerflow_options(self, saw_instance):
        saw_instance.SetMVATolerance(0.1)
        saw_instance.SetDoOneIteration(False)
        saw_instance.SetInnerLoopCheckMVars(False)

    @pytest.mark.order(14)
    def test_powerflow_get_results(self, saw_instance):
        df = saw_instance.get_power_flow_results("Bus")
        assert df is not None
        assert "BusPUVolt" in df.columns

    @pytest.mark.order(15)
    def test_powerflow_min_pu_volt(self, saw_instance):
        v = saw_instance.GetMinPUVoltage()
        assert isinstance(v, float)

    @pytest.mark.order(16)
    def test_powerflow_mismatches(self, saw_instance):
        df = saw_instance.GetBusMismatches()
        assert df is not None

    @pytest.mark.order(17)
    def test_powerflow_update_islands(self, saw_instance):
        saw_instance.UpdateIslandsAndBusStatus()

    @pytest.mark.order(18)
    def test_powerflow_zero_mismatches(self, saw_instance):
        saw_instance.ZeroOutMismatches()

    @pytest.mark.order(19)
    def test_powerflow_estimate_voltages(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.EstimateVoltages("SELECTED")

    @pytest.mark.order(20)
    def test_powerflow_gen_force_ldc(self, saw_instance):
        saw_instance.GenForceLDC_RCC()

    @pytest.mark.order(21)
    def test_powerflow_save_gen_limit(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.SaveGenLimitStatusAction(tmp_txt)
        assert os.path.exists(tmp_txt)

    @pytest.mark.order(22)
    def test_powerflow_diff_case(self, saw_instance):
        saw_instance.DiffCaseSetAsBase()
        saw_instance.DiffCaseMode("DIFFERENCE")
        saw_instance.DiffCaseRefresh()
        saw_instance.DiffCaseClearBase()

    @pytest.mark.order(23)
    def test_powerflow_voltage_conditioning(self, saw_instance):
        saw_instance.VoltageConditioning()

    @pytest.mark.order(24)
    def test_powerflow_flat_start(self, saw_instance):
        saw_instance.ResetToFlatStart()
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(25)
    def test_powerflow_diff_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        tmp_epc = temp_file(".epc")
        saw_instance.DiffCaseWriteCompleteModel(tmp_aux)
        saw_instance.DiffCaseWriteBothEPC(tmp_epc, ge_file_type="GE21")
        saw_instance.DiffCaseWriteNewEPC(tmp_epc, ge_file_type="GE21")


class TestMatrices:
    """Tests for matrix extraction (Ybus, Jacobian, etc.)."""

    @pytest.mark.order(30)
    def test_matrix_ybus(self, saw_instance):
        ybus = saw_instance.get_ybus()
        assert ybus is not None

    @pytest.mark.order(31)
    def test_matrix_gmatrix(self, saw_instance):
        gmat = saw_instance.get_gmatrix()
        assert gmat is not None

    @pytest.mark.order(32)
    def test_matrix_jacobian(self, saw_instance):
        jac = saw_instance.get_jacobian()
        assert jac is not None

    @pytest.mark.order(33)
    def test_matrix_incidence(self, saw_instance):
        inc = saw_instance.get_incidence_matrix()
        assert inc is not None

    @pytest.mark.order(34)
    def test_matrix_branch_admittance(self, saw_instance):
        yf, yt = saw_instance.get_branch_admittance()
        assert yf is not None
        assert yt is not None

    @pytest.mark.order(35)
    def test_matrix_shunt_admittance(self, saw_instance):
        ysh = saw_instance.get_shunt_admittance()
        assert ysh is not None


class TestSensitivity:
    """Tests for sensitivity calculations (PTDF, LODF, shift factors)."""

    @pytest.mark.order(40)
    def test_sensitivity_volt_sense(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.CalculateVoltSense(bus_num)

    @pytest.mark.order(41)
    def test_sensitivity_flow_sense(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.CalculateFlowSense(branch_str, "MW")

    @pytest.mark.order(42)
    def test_sensitivity_ptdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.CalculatePTDF(seller, buyer)
            saw_instance.CalculateVoltToTransferSense(seller, buyer)

    @pytest.mark.order(43)
    def test_sensitivity_lodf(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.CalculateLODF(branch_str)

    @pytest.mark.order(44)
    def test_sensitivity_shift_factors(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"])
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if branches is not None and not branches.empty and areas is not None and not areas.empty:
            closed_branches = branches[branches["LineStatus"] == "Closed"]
            if not closed_branches.empty:
                b = closed_branches.iloc[0]
                branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
                area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
                saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)
                try:
                    saw_instance.CalculateShiftFactors(branch_str, "SELLER", area_str)
                except PowerWorldPrerequisiteError as e:
                    pytest.skip(f"Shift factors calculation failed: {e}")
            else:
                pytest.skip("No closed branches found for shift factors")

    @pytest.mark.order(45)
    def test_sensitivity_lodf_matrix(self, saw_instance):
        saw_instance.CalculateLODFMatrix("OUTAGES", "ALL", "ALL")

    @pytest.mark.order(36)
    def test_sensitivity_lodf_advanced(self, saw_instance, temp_file):
        """Test CalculateLODFAdvanced with full parameters."""
        tmp_csv = temp_file(".csv")
        try:
            # CalculateLODFAdvanced(include_phase_shifters, file_type, max_columns, min_lodf, 
            #                       number_format, decimal_points, only_increasing, filename)
            saw_instance.CalculateLODFAdvanced(
                include_phase_shifters=False,
                file_type="CSV",
                max_columns=100,
                min_lodf=0.01,
                number_format="DECIMAL",
                decimal_points=4,
                only_increasing=False,
                filename=tmp_csv
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("LODF Advanced not available")

    @pytest.mark.order(37)
    def test_sensitivity_lodf_screening(self, saw_instance):
        """Test CalculateLODFScreening for screening mode."""
        try:
            # CalculateLODFScreening with do_save_file=False to avoid file requirement
            saw_instance.CalculateLODFScreening(
                filter_process="ALL",
                filter_monitor="ALL",
                include_phase_shifters=False,
                include_open_lines=False,
                use_lodf_threshold=True,
                lodf_threshold=0.05,
                use_overload_threshold=False,
                overload_low=100.0,
                overload_high=200.0,
                do_save_file=False,
                file_location=""
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("LODF Screening not available")
        except PowerWorldError as e:
            if "LODF" in str(e) or "screening" in str(e).lower():
                pytest.skip("LODF Screening not available")
            raise

    @pytest.mark.order(38)
    def test_sensitivity_shift_factors_multiple(self, saw_instance):
        """Test CalculateShiftFactorsMultipleElement for multiple branches."""
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and not areas.empty:
            area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
            saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)
            try:
                # CalculateShiftFactorsMultipleElement(type_element, which_element, direction, transactor, method)
                # which_element must be SELECTED, OVERLOAD, or CTGOVERLOAD
                saw_instance.CalculateShiftFactorsMultipleElement("BRANCH", "SELECTED", "SELLER", area_str)
            except PowerWorldPrerequisiteError:
                pytest.skip("Shift factors multiple element not available")
            except PowerWorldError as e:
                if "SELECTED" in str(e) or "shift" in str(e).lower():
                    pytest.skip("No branches selected for shift factor calculation")
                raise

    @pytest.mark.order(39)
    def test_sensitivity_loss_sense(self, saw_instance):
        """Test CalculateLossSense for loss sensitivity."""
        try:
            # CalculateLossSense(function_type, area_ref, island_ref)
            # function_type can be AREA, ZONE, BUS, etc.
            saw_instance.CalculateLossSense("AREA", "NO", "EXISTING")
        except PowerWorldPrerequisiteError:
            pytest.skip("Loss sensitivity calculation not available")


class TestTopology:
    """Tests for topology analysis operations."""

    @pytest.mark.order(46)
    def test_topology_path_distance(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            df = saw_instance.DeterminePathDistance(bus_str)
            assert df is not None
            assert isinstance(df, pd.DataFrame)

    @pytest.mark.order(47)
    def test_topology_islands(self, saw_instance):
        df = saw_instance.DetermineBranchesThatCreateIslands()
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.order(48)
    def test_topology_shortest_path(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and len(buses) >= 2:
            start = create_object_string("Bus", buses.iloc[0]['BusNum'])
            end = create_object_string("Bus", buses.iloc[1]['BusNum'])
            df = saw_instance.DetermineShortestPath(start, end)
            assert df is not None


class TestPowerFlowAdvanced:
    """Additional power flow tests for DC solution and advanced features."""

    @pytest.mark.order(26)
    def test_powerflow_solve_dc(self, saw_instance):
        """Test DC power flow solution."""
        saw_instance.SolvePowerFlow("DC")
        # Verify solution was attempted (no exception means success)
        # Run AC again to restore state for subsequent tests
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(27)
    def test_powerflow_agc(self, saw_instance):
        """Test AGC-related generator participation factors."""
        # AGC calculation via participation factors  
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and not areas.empty:
            area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
            saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)

    @pytest.mark.order(28)
    def test_powerflow_results_structure(self, saw_instance):
        """Test power flow results have expected structure."""
        df = saw_instance.get_power_flow_results("Bus")
        assert df is not None
        assert not df.empty
        # Check essential columns exist
        expected_cols = ["BusNum", "BusPUVolt"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        # Verify voltage values are reasonable (0.5 to 1.5 pu)
        assert (df["BusPUVolt"] > 0.5).all(), "Voltage below 0.5 pu detected"
        assert (df["BusPUVolt"] < 1.5).all(), "Voltage above 1.5 pu detected"

    @pytest.mark.order(29)
    def test_powerflow_gen_results(self, saw_instance):
        """Test generator power flow results."""
        df = saw_instance.get_power_flow_results("Gen")
        if df is not None and not df.empty:
            assert "GenMW" in df.columns or "BusNum" in df.columns

