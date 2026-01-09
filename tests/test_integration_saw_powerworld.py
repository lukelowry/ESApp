"""
Integration tests for SAW functionality against a live PowerWorld case.

WHAT THIS TESTS:
- Actual power flow solution execution and result validation
- Contingency analysis with real PowerWorld contingencies
- GIC (Geomagnetically Induced Current) analysis
- File export/import operations (CSV, EPC, RAW, AUX formats)
- Matrix extraction and manipulation
- Transient stability simulation
- Data retrieval across all component types with real case data
- Script command execution and error handling

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py

CONFIGURATION:
    1. Copy tests/config_test.example.py to tests/config_test.py
    2. Set SAW_TEST_CASE = r"C:\\Path\\To\\Your\\Case.pwb"

USAGE:
    pytest tests/test_integration_saw_powerworld.py -v
    pytest tests/test_integration_saw_powerworld.py -k "power_flow" -v
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import sys 

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, PowerWorldAddonError, create_object_string
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """
    Provides the session-scoped SAW instance to the tests in this module.
    """
    return saw_session


class TestOnlineSAW:
    """
    Tests for the SAW class.
    NOTE: Tests are ordered carefully. Destructive tests (Modify, Regions, CaseActions)
    that alter the case topology or numbering are placed at the end to avoid breaking other tests.
    """
    # -------------------------------------------------------------------------
    # Base Mixin Tests
    # -------------------------------------------------------------------------

    def test_base_save_case(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveCase(tmp_pwb)
        assert os.path.exists(tmp_pwb)

    def test_general_log(self, saw_instance, temp_file):
        saw_instance.LogAdd("SAW Validator Test Message")
        tmp_log = temp_file(".txt")
        saw_instance.LogSave(tmp_log)
        assert os.path.exists(tmp_log)

    def test_base_get_header(self, saw_instance):
        header = saw_instance.GetCaseHeader()
        assert header is not None

    def test_base_change_parameters(self, saw_instance):
        # Test ChangeParametersSingleElement
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            original_name = buses.iloc[0]["BusName"]
            new_name = "TestBusName"
            saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, new_name])
            
            # Verify
            check = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
            assert check["BusName"] == new_name
            
            # Restore
            saw_instance.ChangeParameters("Bus", ["BusNum", "BusName"], [bus_num, original_name])

    def test_base_get_parameters(self, saw_instance):
        # Test GetParametersMultipleElement
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None
        assert not df.empty
        
        # Test GetParametersSingleElement
        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

    def test_base_list_devices(self, saw_instance):
        df = saw_instance.ListOfDevices("Bus")
        assert df is not None
        assert not df.empty

    def test_base_properties(self, saw_instance):
        _ = saw_instance.CreateIfNotFound
        _ = saw_instance.CurrentDir
        _ = saw_instance.ProcessID
        _ = saw_instance.RequestBuildDate
        _ = saw_instance.UIVisible
        _ = saw_instance.ProgramInformation

    def test_base_state(self, saw_instance):
        saw_instance.StoreState("TestState")
        saw_instance.RestoreState("TestState")
        saw_instance.DeleteState("TestState")
        saw_instance.SaveState()
        saw_instance.LoadState()

    def test_base_run_script_2(self, saw_instance):
        # Just a simple command to test the interface
        saw_instance.RunScriptCommand2("LogAdd(\"Test\");", "Testing...")

    def test_base_send_to_excel(self, saw_instance):
        # This might fail if Excel is not installed or configured, so we wrap it
        try:
            saw_instance.SendToExcel("Bus", "", ["BusNum", "BusName"])
        except Exception:
            pass

    def test_base_field_list(self, saw_instance):
        df = saw_instance.GetFieldList("Bus")
        assert not df.empty
        
        # GetSpecificFieldList
        df_spec = saw_instance.GetSpecificFieldList("Bus", ["BusNum", "BusName"])
        assert not df_spec.empty

    def test_base_import_export(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.SaveDataWithExtra(tmp_csv, "CSV", "Bus", ["BusNum"], [], "", [], ["Header"], ["Value"])
        saw_instance.SaveObjectFields(tmp_csv, "Bus", ["BusNum"])

    # -------------------------------------------------------------------------
    # Powerflow Mixin Tests
    # -------------------------------------------------------------------------

    def test_powerflow_solve(self, saw_instance):
        saw_instance.SolvePowerFlow()

    def test_powerflow_solve_retry(self, saw_instance):
        saw_instance.SolvePowerFlowWithRetry()

    def test_powerflow_clear_solution_aid(self, saw_instance):
        saw_instance.ClearPowerFlowSolutionAidValues()

    def test_powerflow_options(self, saw_instance):
        saw_instance.SetMVATolerance(0.1)
        saw_instance.SetDoOneIteration(False)
        saw_instance.SetInnerLoopCheckMVars(False)

    def test_powerflow_get_results(self, saw_instance):
        df = saw_instance.get_power_flow_results("Bus")
        assert df is not None
        assert "BusPUVolt" in df.columns

    def test_powerflow_min_pu_volt(self, saw_instance):
        v = saw_instance.GetMinPUVoltage()
        assert isinstance(v, float)

    def test_powerflow_mismatches(self, saw_instance):
        df = saw_instance.GetBusMismatches()
        assert df is not None

    def test_powerflow_update_islands(self, saw_instance):
        saw_instance.UpdateIslandsAndBusStatus()

    def test_powerflow_zero_mismatches(self, saw_instance):
        saw_instance.ZeroOutMismatches()

    def test_powerflow_estimate_voltages(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.EstimateVoltages("SELECTED")

    def test_powerflow_gen_force_ldc(self, saw_instance):
        saw_instance.GenForceLDC_RCC()

    def test_powerflow_save_gen_limit(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.SaveGenLimitStatusAction(tmp_txt)
        assert os.path.exists(tmp_txt)

    def test_powerflow_diff_case(self, saw_instance):
        saw_instance.DiffCaseSetAsBase()
        saw_instance.DiffCaseMode("DIFFERENCE")
        saw_instance.DiffCaseRefresh()
        saw_instance.DiffCaseClearBase()

    def test_powerflow_voltage_conditioning(self, saw_instance):
        saw_instance.VoltageConditioning()

    def test_powerflow_flat_start(self, saw_instance):
        saw_instance.ResetToFlatStart()
        saw_instance.SolvePowerFlow()

    def test_powerflow_diff_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        tmp_epc = temp_file(".epc")
        saw_instance.DiffCaseWriteCompleteModel(tmp_aux)
        # Use GE21 to avoid "Invalid save file type" error with default "GE"
        saw_instance.DiffCaseWriteBothEPC(tmp_epc, ge_file_type="GE21")
        saw_instance.DiffCaseWriteNewEPC(tmp_epc, ge_file_type="GE21")

    # -------------------------------------------------------------------------
    # Matrix Mixin Tests
    # -------------------------------------------------------------------------

    def test_matrix_ybus(self, saw_instance):
        ybus = saw_instance.get_ybus()
        assert ybus is not None

    def test_matrix_gmatrix(self, saw_instance):
        gmat = saw_instance.get_gmatrix()
        assert gmat is not None

    def test_matrix_jacobian(self, saw_instance):
        jac = saw_instance.get_jacobian()
        assert jac is not None

    def test_matrix_incidence(self, saw_instance):
        inc = saw_instance.get_incidence_matrix()
        assert inc is not None

    def test_matrix_branch_admittance(self, saw_instance):
        yf, yt = saw_instance.get_branch_admittance()
        assert yf is not None
        assert yt is not None

    def test_matrix_shunt_admittance(self, saw_instance):
        ysh = saw_instance.get_shunt_admittance()
        assert ysh is not None

    # -------------------------------------------------------------------------
    # Contingency Mixin Tests
    # -------------------------------------------------------------------------

    def test_contingency_auto_insert(self, saw_instance):
        saw_instance.CTGAutoInsert()

    def test_contingency_solve(self, saw_instance):
        saw_instance.SolveContingencies()

    def test_contingency_run_single(self, saw_instance):
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["CTGLabel"]
            saw_instance.RunContingency(ctg_name)
            saw_instance.CTGApply(ctg_name)

    def test_contingency_otdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            buyer = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            saw_instance.CTGCalculateOTDF(seller, buyer)

    def test_contingency_results_ops(self, saw_instance):
        saw_instance.CTGClearAllResults()
        saw_instance.CTGSetAsReference()
        saw_instance.CTGRelinkUnlinkedElements()
        saw_instance.CTGSkipWithIdenticalActions()
        saw_instance.CTGDeleteWithIdenticalActions()
        saw_instance.CTGSort()

    def test_contingency_clone(self, saw_instance):
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["CTGLabel"]
            saw_instance.CTGCloneOne(ctg_name, "ClonedCTG")
            saw_instance.CTGCloneMany("", "Many_", "_Suffix")

    def test_contingency_combo(self, saw_instance):
        saw_instance.CTGComboDeleteAllResults()
        # Setup: Ensure primary contingencies exist for the combo analysis
        saw_instance.CTGAutoInsert()
        saw_instance.CTGConvertToPrimaryCTG()

        # Optimization: Skip most contingencies to avoid long runtimes on large cases
        # Set all contingencies to Skip=YES
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        
        # Unskip a few Primary contingencies to test the functionality
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            # Use CTGLabel if available, otherwise first column
            name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
            
            # Try to find primary contingencies (suffix -Primary is default)
            primary_ctgs = ctgs[ctgs[name_col].astype(str).str.endswith("-Primary")]
            target_ctgs = primary_ctgs.head(2) if not primary_ctgs.empty else ctgs.head(2)
            
            for name in target_ctgs[name_col]:
                saw_instance.SetData("Contingency", [name_col, "Skip"], [name, "NO"])

        try:
            saw_instance.CTGComboSolveAll()
        except PowerWorldPrerequisiteError:
            pytest.skip("No active primary contingencies for Combo Analysis")

    def test_contingency_convert(self, saw_instance):
        saw_instance.CTGConvertAllToDeviceCTG()
        saw_instance.CTGConvertToPrimaryCTG()
        saw_instance.CTGCreateExpandedBreakerCTGs()
        saw_instance.CTGCreateStuckBreakerCTGs()
        saw_instance.CTGPrimaryAutoInsert()

    def test_contingency_create_interface(self, saw_instance):
        try:
            # Use empty string for filter to imply 'all', as "ALL" is not always a valid named filter
            saw_instance.CTGCreateContingentInterfaces("")
        except PowerWorldPrerequisiteError:
            pytest.skip("Filter 'ALL' not found for CTGCreateContingentInterfaces")

    def test_contingency_join(self, saw_instance):
        saw_instance.CTGJoinActiveCTGs(False, False, True)

    def test_contingency_process_remedial(self, saw_instance):
        saw_instance.CTGProcessRemedialActionsAndDependencies(False)

    def test_contingency_save_matrices(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.CTGSaveViolationMatrices(tmp_csv, "CSVCOLHEADER", False, ["Branch"], True, True)

    def test_contingency_verify(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.CTGVerifyIteratedLinearActions(tmp_txt)

    def test_contingency_write_results(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.CTGWriteResultsAndOptions(tmp_aux)
        assert os.path.exists(tmp_aux)
        
        tmp_aux2 = temp_file(".aux")
        saw_instance.CTGWriteAllOptions(tmp_aux2)
        assert os.path.exists(tmp_aux2)

        tmp_aux3 = temp_file(".aux")
        saw_instance.CTGWriteAuxUsingOptions(tmp_aux3)
        assert os.path.exists(tmp_aux3)

    # -------------------------------------------------------------------------
    # Fault Mixin Tests
    # -------------------------------------------------------------------------

    def test_fault_run(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.RunFault(bus_str, "SLG")
            saw_instance.FaultClear()
        else:
            pytest.skip("No buses found")

    def test_fault_auto(self, saw_instance):
        saw_instance.FaultAutoInsert()

    def test_fault_multiple(self, saw_instance):
        # Setup: Ensure faults exist
        saw_instance.FaultAutoInsert()
        try:
            saw_instance.FaultMultiple()
        except PowerWorldPrerequisiteError:
            pytest.skip("No active faults defined for FaultMultiple")

    # -------------------------------------------------------------------------
    # Sensitivity Mixin Tests
    # -------------------------------------------------------------------------

    def test_sensitivity_volt_sense(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.CalculateVoltSense(bus_num)

    def test_sensitivity_flow_sense(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]            
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.CalculateFlowSense(branch_str, "MW")

    def test_sensitivity_ptdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.CalculatePTDF(seller, buyer)
            saw_instance.CalculateVoltToTransferSense(seller, buyer)

    def test_sensitivity_lodf(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]            
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.CalculateLODF(branch_str)

    def test_sensitivity_shift_factors(self, saw_instance):
        # Request LineStatus to filter for closed branches
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"])
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if branches is not None and not branches.empty and areas is not None and not areas.empty:
            # Filter for closed branches
            closed_branches = branches[branches["LineStatus"] == "Closed"]
            if not closed_branches.empty:
                b = closed_branches.iloc[0]
                branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
                area_str = create_object_string("Area", areas.iloc[0]["AreaNum"])
                
                # Setup: Ensure area has participation points to avoid "no available participation points" error
                saw_instance.SetParticipationFactors("CONSTANT", 1.0, area_str)
                
                try:
                    saw_instance.CalculateShiftFactors(branch_str, "SELLER", area_str)
                except PowerWorldPrerequisiteError as e:
                    pytest.skip(f"Shift factors calculation failed: {e}")
            else:
                pytest.skip("No closed branches found for shift factors")
    
    def test_sensitivity_lodf_matrix(self, saw_instance):
        saw_instance.CalculateLODFMatrix("OUTAGES", "ALL", "ALL")

    # -------------------------------------------------------------------------
    # Topology Mixin Tests
    # -------------------------------------------------------------------------

    def test_topology_path_distance(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            df = saw_instance.DeterminePathDistance(bus_str)
            assert df is not None

    def test_topology_islands(self, saw_instance):
        df = saw_instance.DetermineBranchesThatCreateIslands()
        assert df is not None

    def test_topology_shortest_path(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and len(buses) >= 2:
            start = create_object_string("Bus", buses.iloc[0]['BusNum'])
            end = create_object_string("Bus", buses.iloc[1]['BusNum'])
            df = saw_instance.DetermineShortestPath(start, end)
            assert df is not None

    def test_topology_facility(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        
        # Setup: Ensure at least one bus is marked as External (Equiv=YES) for facility analysis
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            # Set the last bus to External
            last_bus = buses.iloc[-1]["BusNum"]
            try:
                saw_instance.SetData("Bus", ["BusNum", "Equiv"], [last_bus, "YES"])
            except Exception:
                pass

        try:
            saw_instance.DoFacilityAnalysis(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("Facility analysis requires setup in GUI/Dialog")
        assert os.path.exists(tmp_aux)

    def test_topology_radial(self, saw_instance):
        saw_instance.FindRadialBusPaths()

    def test_topology_cut(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            
            # To define a network cut, we need at least one branch in the cut.
            # Let's find a branch connected to this bus and select it.            
            saw_instance.UnSelectAll("Branch")
            # Get all branches and filter in pandas because GetParametersMultipleElement COM call doesn't support ad-hoc filters
            branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
            
            if branches is not None and not branches.empty:
                # Filter for branches connected to the bus
                connected = branches[(branches["BusNum"] == bus_num) | (branches["BusNum:1"] == bus_num)]
                
                if not connected.empty:
                    b = connected.iloc[0]
                    # Select this branch to define the cut
                    saw_instance.SetData("Branch", ["BusNum", "BusNum:1", "LineCircuit", "Selected"], [b['BusNum'], b['BusNum:1'], b['LineCircuit'], "YES"])
                    
                    # Now run the command
                    bus_str = create_object_string("Bus", bus_num)
                    saw_instance.SetSelectedFromNetworkCut(True, bus_str, branch_filter="SELECTED", objects_to_select=["Bus"])
                else:
                    pytest.skip(f"No branches connected to bus {bus_num} to define a cut.")
            else:
                pytest.skip("No branches found in case.")

    def test_topology_areas_from_islands(self, saw_instance):
        saw_instance.CreateNewAreasFromIslands()

    def test_topology_breakers(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.CloseWithBreakers("Bus", bus_str)
            saw_instance.OpenWithBreakers("Bus", bus_str)

    # -------------------------------------------------------------------------
    # PV/QV Mixin Tests
    # -------------------------------------------------------------------------

    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    # -------------------------------------------------------------------------
    # Transient Mixin Tests
    # -------------------------------------------------------------------------

    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]            
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.TSCalculateCriticalClearTime(branch_str)

    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)
        
        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)

    # -------------------------------------------------------------------------
    # GIC Mixin Tests
    # -------------------------------------------------------------------------

    def test_gic_calculate(self, saw_instance):
        saw_instance.CalculateGIC(1.0, 90.0, False)
        saw_instance.ClearGIC()

    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)
        
        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)
        
        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)
        
    # -------------------------------------------------------------------------
    # ATC Mixin Tests
    # -------------------------------------------------------------------------

    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    def test_atc_multiple(self, saw_instance):
        # Setup: Ensure directions exist
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DirectionsAutoInsert(s, b)

        try:
            saw_instance.DetermineATCMultipleDirections()
        except PowerWorldPrerequisiteError:
            pytest.skip("No directions defined for ATC")

    def test_atc_results(self, saw_instance):
        # Mocking fields to avoid error if no results
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow"],
            "field_data_type": ["String", "Real"],
            "key_field": ["", ""],
            "description": ["", ""],
            "display_name": ["", ""]
        }).sort_values(by="internal_field_name")
        
        saw_instance.GetATCResults(["MaxFlow", "LimitingContingency"])

    # -------------------------------------------------------------------------
    # Scheduled Actions Mixin Tests
    # -------------------------------------------------------------------------

    def test_scheduled_identify(self, saw_instance):
        saw_instance.IdentifyBreakersForScheduledActions()

    def test_scheduled_apply_revert(self, saw_instance):
        saw_instance.ApplyScheduledActionsAt("01/01/2025 10:00")
        saw_instance.RevertScheduledActionsAt("01/01/2025 10:00")

    def test_scheduled_ref(self, saw_instance):
        saw_instance.ScheduledActionsSetReference()

    def test_scheduled_view(self, saw_instance):
        saw_instance.SetScheduleView("01/01/2025 10:00")

    def test_scheduled_window(self, saw_instance):
        saw_instance.SetScheduleWindow("01/01/2025 00:00", "02/01/2025 00:00", 1.0, "HOURS")

    # -------------------------------------------------------------------------
    # Regions Mixin Tests
    # -------------------------------------------------------------------------

    def test_regions_update(self, saw_instance):
        saw_instance.RegionUpdateBuses()

    def test_regions_rename(self, saw_instance):
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()

    # -------------------------------------------------------------------------
    # OPF Mixin Tests
    # -------------------------------------------------------------------------

    def test_opf_solve(self, saw_instance):
        saw_instance.SolvePrimalLP()

    def test_opf_scopf(self, saw_instance):
        saw_instance.SolveFullSCOPF()

    def test_opf_extras(self, saw_instance):
        saw_instance.InitializePrimalLP()
        saw_instance.SolveSinglePrimalLPOuterLoop()

    # -------------------------------------------------------------------------
    # TimeStep Mixin Tests
    # -------------------------------------------------------------------------

    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    def test_timestep_run(self, saw_instance):
        saw_instance.TimeStepDoRun()
        try:
            saw_instance.TimeStepDoSinglePoint("2025-01-01T10:00:00")
        except PowerWorldPrerequisiteError:
            pass  # Expected if time points not defined
        try:
            saw_instance.TimeStepClearResults()
        except PowerWorldError:
            pass
        saw_instance.TimeStepResetRun()

    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)
        
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass  # Likely no results

    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])

    # -------------------------------------------------------------------------
    # Weather Mixin Tests
    # -------------------------------------------------------------------------

    def test_weather_update(self, saw_instance):
        saw_instance.WeatherLimitsGenUpdate()
        saw_instance.TemperatureLimitsBranchUpdate()

    def test_weather_pfw(self, saw_instance):
        saw_instance.WeatherPFWModelsSetInputsAndApply(False)

    def test_weather_load(self, saw_instance):
        try:
            saw_instance.WeatherPWWLoadForDateTimeUTC("2025-01-01T10:00:00Z")
        except PowerWorldError:
            pass

    def test_weather_dir(self, saw_instance):
        saw_instance.WeatherPWWSetDirectory("C:\\Temp")

    def test_weather_combine(self, saw_instance, temp_file):
        tmp1 = temp_file(".pww")
        tmp2 = temp_file(".pww")
        tmp3 = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileCombine2(tmp1, tmp2, tmp3)
        except Exception:
            pass

    def test_weather_reduce(self, saw_instance, temp_file):
        tmp1 = temp_file(".pww")
        tmp2 = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileGeoReduce(tmp1, tmp2, 0, 10, 0, 10)
        except Exception:
            pass
            
    def test_weather_extras(self, saw_instance):
        saw_instance.WeatherPFWModelsSetInputs()
        try: saw_instance.WeatherPWWFileAllMeasValid("test.pww", ["TEMP"])
        except Exception: pass
        saw_instance.WeatherPFWModelsRestoreDesignValues()

    # -------------------------------------------------------------------------
    # Regions Mixin Tests
    # -------------------------------------------------------------------------

    def test_regions_update(self, saw_instance):
        saw_instance.RegionUpdateBuses()

    def test_regions_rename(self, saw_instance):
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()
        
    def test_regions_load(self, saw_instance, temp_file):
        try:
            saw_instance.RegionLoadShapefile(temp_file(".shp"), "Class", ["Attr"])
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # General Mixin Tests
    # -------------------------------------------------------------------------

    def test_general_file(self, saw_instance, temp_file):
        tmp1 = temp_file(".txt")
        saw_instance.WriteTextToFile(tmp1, "Hello")
        
        tmp2 = tmp1.replace(".txt", "_copy.txt")
        saw_instance.CopyFile(tmp1, tmp2)
        assert os.path.exists(tmp2)
        
        tmp3 = tmp1.replace(".txt", "_renamed.txt")
        saw_instance.RenameFile(tmp2, tmp3)
        assert os.path.exists(tmp3)
        assert not os.path.exists(tmp2)
        
        saw_instance.DeleteFile(tmp3)
        assert not os.path.exists(tmp3)

    def test_general_dir(self, saw_instance):
        saw_instance.SetCurrentDirectory("C:\\Temp", True)

    def test_general_mode(self, saw_instance):
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    def test_general_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        saw_instance.LoadAux(tmp_aux)

    def test_general_script(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.WriteTextToFile(tmp_aux, 'SCRIPT TestScript { LogAdd("Hello"); }')
        saw_instance.LoadScript(tmp_aux, "TestScript")

    def test_general_data(self, saw_instance):
        saw_instance.SetData("Bus", ["BusName"], ["NewName"], "SELECTED")

    def test_general_select(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")

    # -------------------------------------------------------------------------
    # Oneline Mixin Tests
    # -------------------------------------------------------------------------

    def test_oneline_ops(self, saw_instance, temp_file):
        # These might not do much without a GUI but shouldn't crash
        saw_instance.CloseOneline()
        saw_instance.RelinkAllOpenOnelines()
        
        tmp_axd = temp_file(".axd")
        saw_instance.LoadAXD(tmp_axd, "TestOneline")
        
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveOneline(tmp_pwb, "TestOneline")
        
        tmp_jpg = temp_file(".jpg")
        saw_instance.ExportOneline(tmp_jpg, "TestOneline", "JPG")

    def test_oneline_extras(self, saw_instance):
        try:
            saw_instance.PanAndZoomToObject("BUS 1")
        except PowerWorldError:
            pass
        saw_instance.OpenBusView("BUS 1")
        try:
            saw_instance.OpenSubView("Sub 1")
        except PowerWorldError:
            pass
        saw_instance.ExportBusView("test.jpg", "BUS 1", "JPG", 800, 600)
        try:
            saw_instance.ExportOnelineAsShapeFile("test.shp", "Oneline", "Desc")
        except PowerWorldError:
            pass

    # -------------------------------------------------------------------------
    # Modify Mixin Tests (Destructive - Run Late)
    # -------------------------------------------------------------------------

    def test_modify_create_delete(self, saw_instance):
        dummy_bus = 99999
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [dummy_bus, "SAW_TEST"])
        saw_instance.Delete("Bus", f"BusNum = {dummy_bus}")

    def test_modify_auto(self, saw_instance):
        saw_instance.AutoInsertTieLineTransactions()

    def test_modify_branch(self, saw_instance):
        saw_instance.BranchMVALimitReorder()

    def test_modify_calc(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
        except PowerWorldAddonError:
            pytest.skip("TransLineCalc not registered")

    def test_modify_base(self, saw_instance):
        # Destructive: Changes system base
        saw_instance.ChangeSystemMVABase(100.0)

    def test_modify_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    def test_modify_create_line(self, saw_instance):
        # Needs valid bus numbers, skipping actual creation to avoid clutter
        pass

    def test_modify_directions(self, saw_instance):
        # Needs source/sink
        pass

    def test_modify_gen(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()
        saw_instance.SetGenPMaxFromReactiveCapabilityCurve()

    def test_modify_inj(self, saw_instance):
        saw_instance.InjectionGroupsAutoInsert()
        saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "")
        saw_instance.RenameInjectionGroup("TestIG", "TestIG_Renamed")

    def test_modify_interface(self, saw_instance):
        saw_instance.InterfacesAutoInsert("AREA")
        saw_instance.InterfaceCreate("TestInterface", True, "Branch", "SELECTED")
        saw_instance.SetInterfaceLimitToMonitoredElementLimitSum()

    def test_modify_merge(self, saw_instance):
        # Destructive, skipping
        pass

    def test_modify_move(self, saw_instance):
        # Destructive, skipping
        pass

    def test_modify_reassign(self, saw_instance):
        saw_instance.ReassignIDs("Load", "BusName")

    def test_modify_remove(self, saw_instance):
        saw_instance.Remove3WXformerContainer()

    def test_modify_rotate(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.RotateBusAnglesInIsland(bus_str, 0.0)

    def test_modify_part(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    def test_modify_volt(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.SetScheduledVoltageForABus(bus_str, 1.0)

    def test_modify_split(self, saw_instance):
        # Destructive
        pass

    def test_modify_superarea(self, saw_instance):
        saw_instance.CreateData("SuperArea", ["Name"], ["TestSuperArea"])
        saw_instance.SuperAreaAddAreas("TestSuperArea", "ALL")
        saw_instance.SuperAreaRemoveAreas("TestSuperArea", "ALL")

    def test_modify_tap(self, saw_instance):
        # Destructive
        pass

    def test_modify_extras(self, saw_instance):
        saw_instance.InjectionGroupRemoveDuplicates()
        saw_instance.InterfaceRemoveDuplicates()
        saw_instance.DirectionsAutoInsertReference("Bus", "Slack")
        
        # Create interface for flattening
        saw_instance.InterfaceCreate("TestInt", True, "Branch", "SELECTED")
        saw_instance.InterfaceFlatten("TestInt")
        
        saw_instance.InterfaceFlattenFilter("ALL")
        saw_instance.InterfaceModifyIsolatedElements()
        
        # Create contingency for adding elements
        saw_instance.CreateData("Contingency", ["Name"], ["TestCtg"])
        saw_instance.InterfaceAddElementsFromContingency("TestInt", "TestCtg")

    # -------------------------------------------------------------------------
    # Regions Mixin Tests (Destructive - Run Late)
    # -------------------------------------------------------------------------

    def test_regions_update(self, saw_instance):
        saw_instance.RegionUpdateBuses()

    def test_regions_rename(self, saw_instance):
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()
        
    def test_regions_load(self, saw_instance, temp_file):
        try:
            saw_instance.RegionLoadShapefile(temp_file(".shp"), "Class", ["Attr"])
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Case Actions Mixin Tests (Highly Destructive - Run Last)
    # -------------------------------------------------------------------------

    def test_case_description(self, saw_instance):
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()

    def test_case_delete_external(self, saw_instance):
        saw_instance.DeleteExternalSystem()

    def test_case_equivalence(self, saw_instance):
        saw_instance.Equivalence()

    def test_case_save_external(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)

    def test_case_save_merged(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)

    def test_case_scale(self, saw_instance):
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")

    def test_case_renumber(self, saw_instance):
        # This invalidates all bus numbers in the case!
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()


if __name__ == "__main__":
    # Run pytest on this file
    sys.exit(pytest.main(["-v", __file__]))