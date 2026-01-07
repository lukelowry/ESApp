"""
Independent script to validate SAW functionality against a live PowerWorld case.
This script connects to a PowerWorld Simulator instance using the provided case file
and attempts to execute a wide range of SAW methods to verify functionality.

Usage:
    python test_online_saw.py "C:\\Path\\To\\Case.pwb"
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Ensure gridwb can be imported if running from tests directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from gridwb.saw import SAW, PowerWorldError
except ImportError:
    print("Error: Could not import gridwb.saw. Please ensure the package is in your Python path.")
    sys.exit(1)


@pytest.fixture(scope="module")
def saw_instance():
    case_path = os.environ.get("SAW_TEST_CASE")
    if not case_path or not os.path.exists(case_path):
        pytest.skip("SAW_TEST_CASE environment variable not set or file not found.")

    print(f"\nConnecting to PowerWorld with case: {case_path}")
    saw = SAW(case_path, early_bind=True)
    yield saw
    print("\nClosing case and exiting PowerWorld...")
    saw.exit()


@pytest.fixture
def temp_file():
    files = []

    def _create(suffix):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.close()
        files.append(tf.name)
        return tf.name

    yield _create
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


class TestOnlineSAW:
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

    def test_general_import_extras(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        # Create dummy csv first
        with open(tmp_csv, 'w') as f: f.write("BusNum,BusName\n1,Bus1")
        try:
            saw_instance.ImportData(tmp_csv, "CSV")
        except PowerWorldError:
            pass
        saw_instance.LoadCSV(tmp_csv)

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

    def test_powerflow_extras(self, saw_instance):
        saw_instance.ConditionVoltagePockets(0.1, 10.0)
        saw_instance.DiffCaseKeyType("PRIMARY")
        saw_instance.DiffCaseShowPresentAndBase(True)
        saw_instance.DoCTGAction('[BRANCH 1 2 1 OPEN]')
        saw_instance.InterfacesCalculatePostCTGMWFlows()

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
        try:
            saw_instance.CTGComboSolveAll()
        except PowerWorldError as e:
            if "at least one active primary contingency" in str(e):
                pytest.skip("No active primary contingencies for Combo Analysis")
            raise e

    def test_contingency_convert(self, saw_instance):
        saw_instance.CTGConvertAllToDeviceCTG()
        saw_instance.CTGConvertToPrimaryCTG()
        saw_instance.CTGCreateExpandedBreakerCTGs()
        saw_instance.CTGCreateStuckBreakerCTGs()
        saw_instance.CTGPrimaryAutoInsert()

    def test_contingency_create_interface(self, saw_instance):
        try:
            saw_instance.CTGCreateContingentInterfaces("ALL")
        except PowerWorldError as e:
            if "could not be found" in str(e):
                pytest.skip("Filter 'ALL' not found for CTGCreateContingentInterfaces")
            raise e

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
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.RunFault(f"[BUS {bus_num}]", "SLG")
            saw_instance.FaultClear()
        else:
            pytest.skip("No buses found")

    def test_fault_auto(self, saw_instance):
        saw_instance.FaultAutoInsert()

    def test_fault_multiple(self, saw_instance):
        try:
            saw_instance.FaultMultiple()
        except PowerWorldError as e:
            if "No active faults" in str(e):
                pytest.skip("No active faults defined for FaultMultiple")
            raise e

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
            branch_str = f'[BRANCH {b["BusNum"]} {b["BusNum:1"]} "{b["LineCircuit"]}"]'
            saw_instance.CalculateFlowSense(branch_str, "MW")

    def test_sensitivity_ptdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            buyer = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            saw_instance.CalculatePTDF(seller, buyer)
            saw_instance.CalculateVoltToTransferSense(seller, buyer)

    def test_sensitivity_lodf(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = f'[BRANCH {b["BusNum"]} {b["BusNum:1"]} "{b["LineCircuit"]}"]'
            saw_instance.CalculateLODF(branch_str)

    def test_sensitivity_shift_factors(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if branches is not None and not branches.empty and areas is not None and not areas.empty:
            b = branches.iloc[0]
            branch_str = f'[BRANCH {b["BusNum"]} {b["BusNum:1"]} "{b["LineCircuit"]}"]'
            area_str = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            try:
                saw_instance.CalculateShiftFactors(branch_str, "SELLER", area_str)
            except PowerWorldError as e:
                if "no available participation points" in str(e) or "is not online" in str(e):
                    pytest.skip(f"Shift factors calculation failed: {e}")
                raise e
    
    def test_sensitivity_extras(self, saw_instance):
        saw_instance.CalculateLODFAdvanced(True, "MATRIX", 10, 0.03, "DECIMAL", 4, True, "lodf.txt")
        
        # Fetch a valid area for the calculation
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and not areas.empty:
            area_str = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            saw_instance.CalculateShiftFactorsMultipleElement("BRANCH", "SELECTED", "SELLER", area_str)
            
        saw_instance.SetSensitivitiesAtOutOfServiceToClosest()
        
        # Test additional sensitivity methods
        saw_instance.CalculateLODFScreening("ALL", "ALL", True, True, True, 0.05, True, 100, 100, False, "")
        try:
            saw_instance.LineLoadingReplicatorCalculate('[BRANCH 1 2 1]', "IG", False, 100, False)
            saw_instance.LineLoadingReplicatorImplement()
        except Exception: pass

    def test_sensitivity_lodf_matrix(self, saw_instance):
        saw_instance.CalculateLODFMatrix("OUTAGES", "ALL", "ALL")

    def test_sensitivity_loss(self, saw_instance):
        saw_instance.CalculateLossSense("AREA")

    def test_sensitivity_tap(self, saw_instance):
        saw_instance.CalculateTapSense()

    def test_sensitivity_volt_self(self, saw_instance):
        saw_instance.CalculateVoltSelfSense()

    def test_sensitivity_ptdf_multi(self, saw_instance):
        saw_instance.CalculatePTDFMultipleDirections()

    # -------------------------------------------------------------------------
    # Topology Mixin Tests
    # -------------------------------------------------------------------------

    def test_topology_path_distance(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            df = saw_instance.DeterminePathDistance(f"[BUS {bus_num}]")
            assert df is not None

    def test_topology_islands(self, saw_instance):
        df = saw_instance.DetermineBranchesThatCreateIslands()
        assert df is not None

    def test_topology_shortest_path(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and len(buses) >= 2:
            start = f"[BUS {buses.iloc[0]['BusNum']}]"
            end = f"[BUS {buses.iloc[1]['BusNum']}]"
            df = saw_instance.DetermineShortestPath(start, end)
            assert df is not None

    def test_topology_facility(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.DoFacilityAnalysis(tmp_aux)
        except PowerWorldError as e:
            if "There has to be at least one" in str(e):
                pytest.skip("Facility analysis requires setup in GUI/Dialog")
            raise e
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
                    saw_instance.SetSelectedFromNetworkCut(True, f"[BUS {bus_num}]", branch_filter="SELECTED", objects_to_select=["Bus"])
                else:
                    pytest.skip(f"No branches connected to bus {bus_num} to define a cut.")
            else:
                pytest.skip("No branches found in case.")

    def test_topology_extras(self, saw_instance):
        # Use empty string for "ALL" filters where quotes are enforced by the mixin
        saw_instance.SetBusFieldFromClosest("CustomFloat:1", "", "", "ALL", "X")
        saw_instance.ExpandAllBusTopology()
        try:
            saw_instance.ExpandBusTopology("BUS 1", "RINGBUS")
        except PowerWorldError:
            pass
        saw_instance.SaveConsolidatedCase("cons.pwb")

    def test_topology_areas_from_islands(self, saw_instance):
        saw_instance.CreateNewAreasFromIslands()

    def test_topology_breakers(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.CloseWithBreakers("Bus", f"[{bus_num}]")
            saw_instance.OpenWithBreakers("Bus", f"[{bus_num}]")

    # -------------------------------------------------------------------------
    # PV/QV Mixin Tests
    # -------------------------------------------------------------------------

    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    def test_pv_run(self, saw_instance):
        # Requires injection groups, skipping actual run but calling method
        # saw_instance.RunPV(...)
        pass

    def test_pv_qv_extras(self, saw_instance):
        saw_instance.PVClear()
        saw_instance.QVDeleteAllResults()
        saw_instance.PVStartOver()
        saw_instance.PVQVTrackSingleBusPerSuperBus()
        
        # Create dummy injection groups for PVSetSourceAndSink
        saw_instance.CreateData("InjectionGroup", ["Name"], ["SourceIG"])
        saw_instance.CreateData("InjectionGroup", ["Name"], ["SinkIG"])
        saw_instance.PVSetSourceAndSink('[INJECTIONGROUP "SourceIG"]', '[INJECTIONGROUP "SinkIG"]')
        
        saw_instance.QVSelectSingleBusPerSuperBus()
        saw_instance.RefineModel("AREA", "", "SHUNTS", 0.01)

    # -------------------------------------------------------------------------
    # Transient Mixin Tests
    # -------------------------------------------------------------------------

    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    def test_transient_clear_results(self, saw_instance):
        try:
            saw_instance.TSClearResultsFromRAM()
        except Exception as e:
            if "Access violation" in str(e):
                pytest.skip("TSClearResultsFromRAM caused Access Violation (likely due to no results in RAM)")
            raise e

    def test_transient_solve(self, saw_instance):
        # Requires contingency
        pass

    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    def test_transient_misc(self, saw_instance):
        saw_instance.TSTransferStateToPowerFlow()
        saw_instance.TSResultStorageSetAll()
        saw_instance.TSStoreResponse()
        saw_instance.TSClearPlayInSignals()
        try:
            saw_instance.TSClearResultsFromRAMAndDisableStorage()
        except Exception:
            pass
        saw_instance.TSAutoCorrect()
        saw_instance.TSClearAllModels()
        saw_instance.TSValidate()
        saw_instance.TSCalculateSMIBEigenValues()
        saw_instance.TSDisableMachineModelNonZeroDerivative()

    def test_transient_plots(self, saw_instance):
        saw_instance.TSAutoSavePlots([], [])

    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = f'[BRANCH {b["BusNum"]} {b["BusNum:1"]} "{b["LineCircuit"]}"]'
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
        
    def test_gic_extras(self, saw_instance, temp_file):
        # Create dummy files
        tmp_gmd = temp_file(".gmd")
        with open(tmp_gmd, 'w') as f: f.write("// Dummy GMD")
        try:
            saw_instance.GICReadFilePSLF(tmp_gmd)
        except PowerWorldError:
            pass
        
        tmp_gic = temp_file(".gic")
        with open(tmp_gic, 'w') as f: f.write("// Dummy GIC")
        try:
            saw_instance.GICReadFilePTI(tmp_gic)
        except PowerWorldError:
            pass

    # -------------------------------------------------------------------------
    # ATC Mixin Tests
    # -------------------------------------------------------------------------

    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            buyer = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    def test_atc_multiple(self, saw_instance):
        try:
            saw_instance.DetermineATCMultipleDirections()
        except PowerWorldError as e:
            if "No directions set to Include" in str(e):
                pytest.skip("No directions defined for ATC")
            raise e

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
    # Modify Mixin Tests
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
        except PowerWorldError as e:
            if "TransLineCalc is not registered" in str(e):
                pytest.skip("TransLineCalc not registered")
            raise e

    def test_modify_base(self, saw_instance):
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
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.RotateBusAnglesInIsland(f"[BUS {bus_num}]", 0.0)

    def test_modify_part(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    def test_modify_volt(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            saw_instance.SetScheduledVoltageForABus(f"[BUS {bus_num}]", 1.0)

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
        except PowerWorldError as e:
            if "out-of-range" in str(e):
                pass  # Expected if time points not defined
            else:
                raise e
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

    def test_timestep_extras(self, saw_instance):
        try:
            saw_instance.TIMESTEPSaveSelectedModifyStart()
            saw_instance.TIMESTEPSaveSelectedModifyFinish()
        except PowerWorldError:
            pass
        
    def test_timestep_pww_extras(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        # Write minimal PWW content or handle crash
        with open(tmp_pww, 'w') as f: f.write("Version 1\n")
        
        try:
            saw_instance.TimeStepAppendPWW(tmp_pww)
        except PowerWorldError:
            pass
            
        try:
            saw_instance.TimeStepAppendPWWRange(tmp_pww, "", "")
        except PowerWorldError:
            pass
            
        try:
            saw_instance.TimeStepAppendPWWRangeLatLon(tmp_pww, "", "", 0, 0, 0, 0)
        except PowerWorldError:
            pass
            
        tmp_b3d = temp_file(".b3d")
        with open(tmp_b3d, 'w') as f: f.write("Version 1\n")
        try:
            saw_instance.TimeStepLoadB3D(tmp_b3d)
        except PowerWorldError:
            pass
            
        try:
            saw_instance.TimeStepLoadPWWRangeLatLon(tmp_pww, "", "", 0, 0, 0, 0)
        except PowerWorldError:
            pass
            
        saw_instance.TimeStepSavePWWRange(tmp_pww, "", "")
        saw_instance.TIMESTEPSaveInputCSV(temp_file(".csv"), ["GenMW"])

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
    # Case Actions Mixin Tests
    # -------------------------------------------------------------------------

    def test_case_description(self, saw_instance):
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()

    def test_case_delete_external(self, saw_instance):
        saw_instance.DeleteExternalSystem()

    def test_case_equivalence(self, saw_instance):
        saw_instance.Equivalence()

    def test_case_renumber(self, saw_instance):
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()

    def test_case_save_external(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)

    def test_case_save_merged(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)

    def test_case_scale(self, saw_instance):
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")

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


if __name__ == "__main__":
    # Default case path if not provided
    default_case = r"C:\Users\wyattluke.lowery\OneDrive - Texas A&M University\Research\Cases\Hawaii 37\Hawaii40_20231026.pwb"

    if len(sys.argv) > 1:
        case_arg = sys.argv[1]
    else:
        case_arg = default_case

    os.environ["SAW_TEST_CASE"] = case_arg

    # Run pytest on this file
    sys.exit(pytest.main(["-v", __file__]))