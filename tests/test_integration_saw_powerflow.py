"""
Integration tests for Power Flow, Matrices, Sensitivity, and Topology via SAW.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test power flow solution,
matrix extraction (Y-bus, Jacobian, incidence), sensitivity calculations
(PTDF, LODF, shift factors), topology analysis, and diff-case operations.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_core.py          -- base SAW operations, logging, I/O
    - test_integration_saw_modify.py        -- destructive modify, region, case actions
    - test_integration_saw_contingency.py   -- contingency and fault analysis
    - test_integration_saw_gic.py           -- GIC analysis
    - test_integration_saw_transient.py     -- transient stability
    - test_integration_saw_operations.py    -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py         -- GridWorkBench facade and statics
    - test_integration_network.py           -- Network topology

USAGE:
    pytest tests/test_integration_saw_powerflow.py -v
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
except ImportError:
    raise

from conftest import ensure_areas


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


class TestPowerFlow:
    """Tests for power flow solution and related operations."""

    @pytest.mark.order(1000)
    def test_powerflow_solve(self, saw_instance):
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(1200)
    def test_powerflow_clear_solution_aid(self, saw_instance):
        saw_instance.ClearPowerFlowSolutionAidValues()

    @pytest.mark.order(1300)
    def test_powerflow_options(self, saw_instance):
        saw_instance.SetMVATolerance(0.1)
        saw_instance.SetDoOneIteration(False)
        saw_instance.SetInnerLoopCheckMVars(False)

    @pytest.mark.order(1500)
    def test_powerflow_min_pu_volt(self, saw_instance):
        v = saw_instance.GetMinPUVoltage()
        assert isinstance(v, float)

    @pytest.mark.order(1700)
    def test_powerflow_update_islands(self, saw_instance):
        saw_instance.UpdateIslandsAndBusStatus()

    @pytest.mark.order(1800)
    def test_powerflow_zero_mismatches(self, saw_instance):
        saw_instance.ZeroOutMismatches()

    @pytest.mark.order(1900)
    def test_powerflow_estimate_voltages(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.EstimateVoltages("SELECTED")

    @pytest.mark.order(2000)
    def test_powerflow_gen_force_ldc(self, saw_instance):
        saw_instance.GenForceLDC_RCC()

    @pytest.mark.order(2100)
    def test_powerflow_save_gen_limit(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.SaveGenLimitStatusAction(tmp_txt)
        assert os.path.exists(tmp_txt)

    @pytest.mark.order(2200)
    def test_powerflow_diff_case(self, saw_instance):
        saw_instance.DiffCaseSetAsBase()
        saw_instance.DiffCaseMode("DIFFERENCE")
        saw_instance.DiffCaseRefresh()
        saw_instance.DiffCaseClearBase()

    @pytest.mark.order(2300)
    def test_powerflow_voltage_conditioning(self, saw_instance):
        saw_instance.VoltageConditioning()

    @pytest.mark.order(2400)
    def test_powerflow_flat_start(self, saw_instance):
        saw_instance.ResetToFlatStart()
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(2500)
    def test_powerflow_diff_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        tmp_epc = temp_file(".epc")
        saw_instance.DiffCaseWriteCompleteModel(tmp_aux)
        saw_instance.DiffCaseWriteBothEPC(tmp_epc, ge_file_type="GE21")
        saw_instance.DiffCaseWriteNewEPC(tmp_epc, ge_file_type="GE21")

    @pytest.mark.order(2600)
    def test_powerflow_solve_dc(self, saw_instance):
        """Test DC power flow solution."""
        saw_instance.SolvePowerFlow("DC")
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(2700)
    def test_powerflow_agc(self, saw_instance):
        """Test AGC-related generator participation factors."""
        areas = ensure_areas(saw_instance, 1)
        area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)

    @pytest.mark.order(71000)
    def test_powerflow_solve_with_method(self, saw_instance):
        """Test SolvePowerFlow with explicit method parameter."""
        saw_instance.SolvePowerFlow("RECTNEWT")
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(71100)
    def test_powerflow_condition_voltage_pockets(self, saw_instance):
        """Test VoltageConditioning."""
        saw_instance.VoltageConditioning()

    @pytest.mark.order(71200)
    def test_powerflow_diff_write_removed_epc(self, saw_instance, temp_file):
        """Test DiffCaseWriteRemovedEPC."""
        tmp_epc = temp_file(".epc")
        saw_instance.DiffCaseWriteRemovedEPC(tmp_epc)


class TestMatrices:
    """Tests for matrix extraction (Ybus, Jacobian, etc.)."""

    @pytest.mark.order(3000)
    def test_matrix_ybus(self, saw_instance):
        ybus = saw_instance.get_ybus()
        assert ybus is not None

    @pytest.mark.order(3100)
    def test_matrix_gmatrix(self, saw_instance):
        gmat = saw_instance.get_gmatrix()
        assert gmat is not None

    @pytest.mark.order(3200)
    def test_matrix_jacobian(self, saw_instance):
        jac = saw_instance.get_jacobian()
        assert jac is not None

    @pytest.mark.order(3300)
    def test_matrix_ybus_full_and_sparse(self, saw_instance):
        """get_ybus returns full dense or sparse matrix depending on flag."""
        from scipy.sparse import issparse

        sparse_ybus = saw_instance.get_ybus(full=False)
        assert issparse(sparse_ybus)

        full_ybus = saw_instance.get_ybus(full=True)
        assert not issparse(full_ybus)
        assert full_ybus.shape[0] == full_ybus.shape[1]

    @pytest.mark.order(3400)
    def test_matrix_gmatrix_full_and_sparse(self, saw_instance):
        """get_gmatrix returns both full and sparse modes."""
        from scipy.sparse import issparse

        sparse_g = saw_instance.get_gmatrix(full=False)
        assert issparse(sparse_g)

        full_g = saw_instance.get_gmatrix(full=True)
        assert not issparse(full_g)

    @pytest.mark.order(3500)
    def test_matrix_gmatrix_with_ids(self, saw_instance):
        """get_gmatrix_with_ids returns matrix and ID mapping."""
        matrix, ids = saw_instance.get_gmatrix_with_ids(full=True)
        assert matrix is not None
        assert isinstance(ids, list)
        assert len(ids) > 0

    @pytest.mark.order(3600)
    def test_matrix_jacobian_with_ids(self, saw_instance):
        """get_jacobian_with_ids returns matrix and ID mapping."""
        matrix, ids = saw_instance.get_jacobian_with_ids(full=True)
        assert matrix is not None
        assert isinstance(ids, list)
        assert len(ids) > 0

    @pytest.mark.order(3700)
    def test_matrix_save_ybus_and_parse(self, saw_instance):
        """SaveYbusInMatlabFormat saves and can be parsed back."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".m", delete=False)
        tmp.close()
        try:
            saw_instance.SaveYbusInMatlabFormat(tmp.name, include_voltages=False)
            assert os.path.exists(tmp.name)
            ybus = saw_instance.get_ybus(file=tmp.name, full=True)
            assert ybus is not None
            assert ybus.shape[0] > 0
        finally:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)


class TestSensitivity:
    """Tests for sensitivity calculations (PTDF, LODF, shift factors)."""

    @pytest.mark.order(4000)
    def test_sensitivity_volt_sense(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_num = buses.iloc[0]["BusNum"]
        saw_instance.CalculateVoltSense(bus_num)

    @pytest.mark.order(4100)
    def test_sensitivity_flow_sense(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.CalculateFlowSense(branch_str, "MW")

    @pytest.mark.order(4200)
    def test_sensitivity_ptdf(self, saw_instance):
        areas = ensure_areas(saw_instance, 2)
        seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
        buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
        saw_instance.CalculatePTDF(seller, buyer)
        saw_instance.CalculateVoltToTransferSense(seller, buyer)

    @pytest.mark.order(4300)
    def test_sensitivity_lodf(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.CalculateLODF(branch_str)

    @pytest.mark.order(4400)
    def test_sensitivity_shift_factors(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        areas = ensure_areas(saw_instance, 1)
        closed_branches = branches[branches["LineStatus"] == "Closed"]
        if closed_branches.empty:
            pytest.skip("No closed branches found for shift factors")
        b = closed_branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)
        saw_instance.CalculateShiftFactors(branch_str, "SELLER", area_str)

    @pytest.mark.order(4500)
    def test_sensitivity_lodf_matrix(self, saw_instance):
        saw_instance.CalculateLODFMatrix("OUTAGES", "ALL", "ALL")

    @pytest.mark.order(3601)
    def test_sensitivity_lodf_with_params(self, saw_instance, temp_file):
        """Test CalculateLODFAdvanced with full parameters."""
        tmp_csv = temp_file(".csv")
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

    @pytest.mark.order(3701)
    def test_sensitivity_lodf_screening(self, saw_instance):
        """Test CalculateLODFScreening for screening mode."""
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

    @pytest.mark.order(3800)
    def test_sensitivity_shift_factors_multiple(self, saw_instance):
        """Test CalculateShiftFactorsMultipleElement for multiple branches."""
        areas = ensure_areas(saw_instance, 1)
        area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)
        saw_instance.CalculateShiftFactorsMultipleElement("BRANCH", "SELECTED", "SELLER", area_str)

    @pytest.mark.order(3900)
    def test_sensitivity_loss_sense(self, saw_instance):
        """Test CalculateLossSense for loss sensitivity."""
        saw_instance.CalculateLossSense("AREA", "NO", "EXISTING")

    @pytest.mark.order(4900)
    def test_calculate_volt_to_transfer_sense(self, saw_instance):
        """CalculateVoltToTransferSense completes without error."""
        areas = ensure_areas(saw_instance, 2)
        area1 = str(areas.iloc[0]["AreaNum"]).strip()
        area2 = str(areas.iloc[1]["AreaNum"]).strip()

        gens = saw_instance.GetParametersMultipleElement(
            "Gen", ["BusNum", "GenID", "AreaNum", "GenAGCAble"]
        )
        a1_agc = gens[
            (gens["AreaNum"].astype(str).str.strip() == area1)
            & (gens["GenAGCAble"].astype(str).str.strip().str.upper() == "YES")
        ]
        a2_agc = gens[
            (gens["AreaNum"].astype(str).str.strip() == area2)
            & (gens["GenAGCAble"].astype(str).str.strip().str.upper() == "YES")
        ]
        if a1_agc.empty or a2_agc.empty:
            pytest.skip("Case needs AGC generators in both areas for transfer sensitivity")

        s = create_object_string("Area", area1)
        b = create_object_string("Area", area2)
        saw_instance.CalculateVoltToTransferSense(s, b, transfer_type="P", turn_off_avr=False)

    @pytest.mark.order(4950)
    def test_calculate_tap_sense(self, saw_instance):
        """CalculateTapSense completes without error."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty
        b = branches.iloc[0]
        saw_instance.SetData(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "Selected"],
            [str(b["BusNum"]).strip(), str(b["BusNum:1"]).strip(),
             str(b["LineCircuit"]).strip(), "YES"],
        )
        saw_instance.CalculateTapSense(filter_name="SELECTED")

    @pytest.mark.order(4960)
    def test_calculate_volt_self_sense(self, saw_instance):
        """CalculateVoltSelfSense completes without error."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty
        saw_instance.SetData(
            "Bus", ["BusNum", "Selected"],
            [str(buses.iloc[0]["BusNum"]).strip(), "YES"],
        )
        saw_instance.CalculateVoltSelfSense(filter_name="SELECTED")

    @pytest.mark.order(4970)
    def test_calculate_volt_sense_specific(self, saw_instance):
        """CalculateVoltSense for a specific bus."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        saw_instance.CalculateVoltSense(int(buses.iloc[0]["BusNum"]))

    @pytest.mark.order(4980)
    def test_set_sensitivities_at_oos_to_closest(self, saw_instance):
        """SetSensitivitiesAtOutOfServiceToClosest completes without error."""
        saw_instance.SetSensitivitiesAtOutOfServiceToClosest()

    @pytest.mark.order(4990)
    def test_calculate_lodf_with_enum(self, saw_instance):
        """CalculateLODF using LinearMethod enum."""
        from esapp.saw._enums import LinearMethod
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.CalculateLODF(branch_str, method=LinearMethod.DC)

    @pytest.mark.order(72000)
    def test_sensitivity_lodf_post_closure(self, saw_instance):
        """Test CalculateLODF with post_closure_lcdf='YES'."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.CalculateLODF(branch_str, post_closure_lcdf="YES")

    @pytest.mark.order(72100)
    def test_sensitivity_ptdf_multiple_directions(self, saw_instance):
        """Test CalculatePTDFMultipleDirections."""
        saw_instance.CalculatePTDFMultipleDirections()

    @pytest.mark.order(72200)
    def test_sensitivity_line_loading_replicator(self, saw_instance):
        """Test LineLoadingReplicatorCalculate."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.InjectionGroupCreate("TestLLR", "Gen", 1.0, "", append=False)
        ig_str = '[InjectionGroup "TestLLR"]'
        saw_instance.LineLoadingReplicatorCalculate(
            branch_str, ig_str, agc_only=False, desired_flow=100.0, implement=False,
        )

    @pytest.mark.order(72310)
    def test_sensitivity_line_loading_replicator_implement(self, saw_instance):
        """LineLoadingReplicatorImplement completes without error."""
        saw_instance.LineLoadingReplicatorImplement()


class TestTopology:
    """Tests for topology analysis operations."""

    @pytest.mark.order(4700)
    def test_topology_islands(self, saw_instance):
        df = saw_instance.DetermineBranchesThatCreateIslands()
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.order(4800)
    def test_topology_shortest_path(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and len(buses) >= 2, "Test case must contain at least 2 buses"
        start = create_object_string("Bus", buses.iloc[0]['BusNum'])
        end = create_object_string("Bus", buses.iloc[1]['BusNum'])
        df = saw_instance.DetermineShortestPath(start, end)
        assert df is not None

    @pytest.mark.order(72350)
    def test_topology_path_distance(self, saw_instance):
        """Test DeterminePathDistance."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.DeterminePathDistance(bus_key, BranchDistMeas="X")

    @pytest.mark.order(72400)
    def test_topology_set_bus_field_from_closest(self, saw_instance):
        """Test SetBusFieldFromClosest with all required args."""
        saw_instance.SetBusFieldFromClosest("BusName", "", "", "ALL", "Z")

    @pytest.mark.order(72500)
    def test_topology_save_consolidated_case(self, saw_instance, temp_file):
        """Test SaveConsolidatedCase."""
        tmp = temp_file(".pwb")
        saw_instance.SaveConsolidatedCase(tmp)

    @pytest.mark.order(75000)
    def test_do_facility_analysis(self, saw_instance, temp_file):
        """DoFacilityAnalysis with and without set_selected."""
        tmp = temp_file(".aux")
        try:
            saw_instance.DoFacilityAnalysis(tmp, set_selected=False)
            saw_instance.DoFacilityAnalysis(tmp, set_selected=True)
        except PowerWorldPrerequisiteError:
            pytest.skip("No Facility/External buses configured in test case")

    @pytest.mark.order(75200)
    def test_find_radial_bus_paths(self, saw_instance):
        """FindRadialBusPaths with and without ignore_status."""
        saw_instance.FindRadialBusPaths(
            ignore_status=False,
            treat_parallel_as_not_radial=False,
            bus_or_superbus="BUS",
        )
        saw_instance.FindRadialBusPaths(
            ignore_status=True,
            treat_parallel_as_not_radial=True,
            bus_or_superbus="BUS",
        )

    @pytest.mark.order(75400)
    def test_set_selected_from_network_cut(self, saw_instance):
        """SetSelectedFromNetworkCut completes without error."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        saw_instance.SetData(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "Selected"],
            [str(b["BusNum"]).strip(), str(b["BusNum:1"]).strip(),
             str(b["LineCircuit"]).strip(), "YES"],
        )
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.SetSelectedFromNetworkCut(
            set_how=True,
            bus_on_cut_side=bus_key,
            branch_filter="SELECTED",
            energized=True,
            num_tiers=0,
            initialize_selected=True,
            objects_to_select=["Bus", "Branch"],
        )

    @pytest.mark.order(75500)
    def test_create_new_areas_from_islands(self, saw_instance):
        """CreateNewAreasFromIslands completes without error."""
        saw_instance.CreateNewAreasFromIslands()

    @pytest.mark.order(75600)
    def test_expand_all_bus_topology(self, saw_instance):
        """ExpandAllBusTopology completes without error."""
        saw_instance.ExpandAllBusTopology()

    @pytest.mark.order(75700)
    def test_expand_bus_topology(self, saw_instance):
        """ExpandBusTopology for a specific bus."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        saw_instance.ExpandBusTopology(f"BUS {buses.iloc[0]['BusNum']}", "BREAKERANDAHALF")

    @pytest.mark.order(75800)
    def test_close_with_breakers(self, saw_instance):
        """CloseWithBreakers with simple and full object string."""
        gens = saw_instance.GetParametersMultipleElement("Gen", ["BusNum", "GenID"])
        assert gens is not None and not gens.empty, "Test case must contain generators"
        saw_instance.CloseWithBreakers("GEN", f"[{gens.iloc[0]['BusNum']} {gens.iloc[0]['GenID']}]")
        full_str = create_object_string("GEN", gens.iloc[0]["BusNum"], gens.iloc[0]["GenID"])
        saw_instance.CloseWithBreakers(
            "GEN", full_str,
            only_specified=True,
            switching_types=["Breaker", "Load Break Disconnect"],
            close_normally_closed=True,
        )

    @pytest.mark.order(76000)
    def test_open_with_breakers(self, saw_instance):
        """OpenWithBreakers with simple and full object string."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        saw_instance.OpenWithBreakers("BRANCH", f"[{b['BusNum']} {b['BusNum:1']} {b['LineCircuit']}]")
        full_str = create_object_string("BRANCH", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.OpenWithBreakers(
            "BRANCH", full_str,
            switching_types=["Breaker"],
            open_normally_open=True,
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
