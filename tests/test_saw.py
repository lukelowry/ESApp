"""
Unit tests for the SAW class and its mixins.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from esapp.saw import SAW

def test_saw_initialization(saw_obj):
    """Test that the SAW object initializes correctly with the fixture."""
    assert saw_obj.pwb_file_path == "dummy.pwb"
    assert saw_obj._pwcom is not None

def test_open_case(saw_obj):
    """Test OpenCase calls the underlying COM method."""
    saw_obj.OpenCase("test_case.pwb")
    saw_obj._pwcom.OpenCase.assert_called_with("test_case.pwb")
    assert saw_obj.pwb_file_path == "test_case.pwb"

def test_save_case(saw_obj):
    """Test SaveCase calls the underlying COM method."""
    saw_obj.SaveCase("saved_case.pwb")
    # Check if SaveCase was called. 
    # convert_to_windows_path is used internally, so we check if the call argument contains the filename.
    saw_obj._pwcom.SaveCase.assert_called()
    args, _ = saw_obj._pwcom.SaveCase.call_args
    assert "saved_case.pwb" in args[0]

def test_run_script_command(saw_obj):
    """Test RunScriptCommand."""
    cmd = "SolvePowerFlow;"
    saw_obj.RunScriptCommand(cmd)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(cmd)

def test_solve_power_flow(saw_obj):
    """Test SolvePowerFlow mixin method."""
    saw_obj.SolvePowerFlow()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("SolvePowerFlow(RECTNEWT)")

def test_get_parameters_multiple_element(saw_obj):
    """Test retrieving parameters returns a DataFrame."""
    # Mock return: (Error, ListOfLists) where ListOfLists corresponds to columns.
    # We use BusNum and BusName which are set up in the conftest fixture's GetFieldList mock.
    saw_obj._pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["Bus1", "Bus2"]])
    
    df = saw_obj.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "BusNum" in df.columns
    assert "BusName" in df.columns
    # Check that BusNum is numeric (handled by clean_df_or_series -> _to_numeric)
    assert pd.api.types.is_numeric_dtype(df["BusNum"])

def test_change_parameters_single_element(saw_obj):
    """Test changing parameters."""
    saw_obj.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [1, "NewName"])
    saw_obj._pwcom.ChangeParametersSingleElement.assert_called()

def test_run_contingency(saw_obj):
    """Test RunContingency mixin."""
    saw_obj.RunContingency("MyCtg")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CTGSolve("MyCtg");')

def test_ts_solve(saw_obj):
    """Test TSSolve mixin."""
    saw_obj.TSSolve("MyCtg")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSSolve("MyCtg")')

def test_ts_get_contingency_results(saw_obj):
    """Test TSGetContingencyResults parsing."""
    # Mock return structure: (Error, MetaData, Data)
    # MetaData: List of lists (rows of metadata)
    # Data: List of rows (time steps)
    
    # MetaData columns: "ObjectType", "PrimaryKey", "SecondaryKey", "Label", "VariableName", "ColHeader"
    mock_meta = [
        ["Gen", "1", "", "", "GenMW", "MW"],
        ["Bus", "2", "", "", "BusPUVolt", "PU"]
    ]
    
    # Data: Time + 2 columns
    mock_data = [
        [0.0, 10.0, 1.0],
        [0.1, 10.1, 0.99]
    ]
    
    saw_obj._pwcom.TSGetContingencyResults.return_value = ("", mock_meta, mock_data)
    
    meta, data = saw_obj.TSGetContingencyResults("MyCtg", ["GenMW", "BusPUVolt"])
    
    assert isinstance(meta, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    assert "time" in data.columns
    assert len(data) == 2
    assert len(meta) == 2
    # Check that data is numeric
    assert pd.api.types.is_numeric_dtype(data["time"])

def test_topology_determine_path_distance(saw_obj):
    """Test DeterminePathDistance."""
    # This method calls RunScriptCommand and then GetParametersMultipleElement.
    # We need to mock GetParametersMultipleElement to return something valid for the dataframe construction.
    
    # Columns requested: KeyFields + [BusField]
    # KeyFields for Bus is BusNum and BusName (from conftest).
    # BusField defaults to CustomFloat:1.
    
    # Note: DeterminePathDistance sets pw_order=True temporarily.
    # This affects clean_df_or_series, skipping _clean_df.
    
    saw_obj._pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["Bus1", "Bus2"], [0.5, 1.5]]) # BusNum, BusName, CustomFloat:1
    
    df = saw_obj.DeterminePathDistance("1")
    
    saw_obj._pwcom.RunScriptCommand.assert_called()
    assert "BusNum" in df.columns
    assert "X" in df.columns # Default BranchDistMeas is "X"
    assert len(df) == 2

def test_oneline_open(saw_obj):
    """Test OpenOneLine."""
    saw_obj.OpenOneLine("test.axd")
    # Check if RunScriptCommand was called with expected string
    args, _ = saw_obj._pwcom.RunScriptCommand.call_args
    assert 'OpenOneline("test.axd"' in args[0]

def test_matrix_get_ybus(saw_obj):
    """Test get_ybus."""
    # get_ybus writes to a temp file and reads it.
    # We need to mock open() because get_ybus reads the file content.
    
    mock_mat_content = """
    Ybus = sparse(2,2)
    Ybus(1,1)=1.0+j*2.0
    Ybus(2,2)=1.0+j*2.0
    """
    
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = mock_mat_content
        mock_file.readline.return_value = "header"
        mock_open.return_value.__enter__.return_value = mock_file
        
        ybus = saw_obj.get_ybus()
        
        # Default is sparse matrix (csr_matrix)
        assert hasattr(ybus, "toarray")
        saw_obj._pwcom.RunScriptCommand.assert_called()

def test_close_case(saw_obj):
    """Test CloseCase."""
    saw_obj.CloseCase()
    saw_obj._pwcom.CloseCase.assert_called()

def test_get_case_header(saw_obj):
    """Test GetCaseHeader."""
    saw_obj.GetCaseHeader()
    saw_obj._pwcom.GetCaseHeader.assert_called()

def test_simauto_properties(saw_obj):
    """Test setting and getting SimAuto properties."""
    saw_obj.set_simauto_property("CreateIfNotFound", True)
    assert saw_obj._pwcom.CreateIfNotFound is True
    
    # Access properties to ensure they call the underlying COM object
    _ = saw_obj.CurrentDir
    _ = saw_obj.ProcessID
    _ = saw_obj.RequestBuildDate
    # UIVisible might log a warning if attribute missing, but should not crash
    _ = saw_obj.UIVisible

def test_enter_mode(saw_obj):
    """Test EnterMode."""
    saw_obj.EnterMode("EDIT")
    saw_obj._pwcom.RunScriptCommand.assert_called_with("EnterMode(EDIT);")

def test_state_management(saw_obj):
    """Test StoreState, RestoreState, DeleteState."""
    saw_obj.StoreState("State1")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('StoreState("State1");')
    
    saw_obj.RestoreState("State1")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('RestoreState(USER, "State1");')
    
    saw_obj.DeleteState("State1")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('DeleteState(USER, "State1");')

def test_logging(saw_obj):
    """Test LogAdd and LogClear."""
    saw_obj.LogAdd("Test Message")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('LogAdd("Test Message");')
    
    saw_obj.LogClear()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("LogClear;")

def test_matrix_branch_admittance(saw_obj):
    """Test get_branch_admittance calculation."""
    # Mock GetParametersMultipleElement to return dataframes for bus and branch
    with patch.object(saw_obj, 'GetParametersMultipleElement') as mock_get_params, \
         patch.object(saw_obj, 'get_key_field_list', return_value=["BusNum"]):
        
        def side_effect(ObjectType, ParamList, FilterName=""):
            if ObjectType.lower() == "bus":
                return pd.DataFrame({"BusNum": [1, 2]})
            elif ObjectType.lower() == "branch":
                return pd.DataFrame({
                    "BusNum": [1, 2],
                    "BusNum:1": [2, 1],
                    "LineR": [0.0, 0.0],
                    "LineX": [0.1, 0.1],
                    "LineC": [0.0, 0.0],
                    "LineTap": [1.0, 1.0],
                    "LinePhase": [0.0, 0.0]
                })
            return pd.DataFrame()
        
        mock_get_params.side_effect = side_effect
        
        Yf, Yt = saw_obj.get_branch_admittance()
        assert Yf.shape == (2, 2)
        assert Yt.shape == (2, 2)

def test_matrix_incidence(saw_obj):
    """Test get_incidence_matrix."""
    with patch.object(saw_obj, 'ListOfDevices') as mock_list_dev:
        mock_list_dev.side_effect = lambda obj, FilterName="": pd.DataFrame({
            "BusNum": [1, 2]
        }) if obj.lower() == "bus" else pd.DataFrame({
            "BusNum": [1, 2],
            "BusNum:1": [2, 1]
        })
        
        inc = saw_obj.get_incidence_matrix()
        assert inc.shape == (2, 2)

def test_matrix_jacobian(saw_obj):
    """Test get_jacobian."""
    mock_mat_content = "Jac = sparse(2,2)\nJac(1,1)=1.0\nJac(2,2)=1.0"
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = mock_mat_content
        mock_open.return_value.__enter__.return_value = mock_file
        
        jac = saw_obj.get_jacobian()
        assert hasattr(jac, "toarray")

def test_powerflow_extras(saw_obj):
    """Test additional PowerflowMixin methods."""
    saw_obj.ClearPowerFlowSolutionAidValues()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("ClearPowerFlowSolutionAidValues;")
    
    saw_obj.ResetToFlatStart()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("ResetToFlatStart();")
    
    saw_obj.SetMVATolerance(0.5)
    saw_obj._pwcom.ChangeParametersSingleElement.assert_called()
    
    saw_obj.SetDoOneIteration(True)
    saw_obj._pwcom.ChangeParametersSingleElement.assert_called()

def test_topology_extras(saw_obj):
    """Test additional TopologyMixin methods."""
    saw_obj.RenumberCase()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("RenumberCase;")
    
    saw_obj.RenumberBuses(5)
    saw_obj._pwcom.RunScriptCommand.assert_called_with("RenumberBuses(5);")

def test_transient_extras(saw_obj):
    """Test additional TransientMixin methods."""
    saw_obj.TSTransferStateToPowerFlow()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("TSTransferStateToPowerFlow(NO);")
    
    saw_obj.TSInitialize()
    saw_obj._pwcom.RunScriptCommand.assert_called()
    
    saw_obj.TSResultStorageSetAll("Gen", False)
    saw_obj._pwcom.RunScriptCommand.assert_called_with("TSResultStorageSetAll(Gen, NO)")
    
    saw_obj.TSSolveAll()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("TSSolveAll()")
    
    saw_obj.TSClearResultsFromRAM()
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_ts_set_play_in_signals(saw_obj):
    """Test TSSetPlayInSignals."""
    times = np.array([0.0, 0.1])
    signals = np.array([[1.0], [1.0]])
    saw_obj.TSSetPlayInSignals("TestSignal", times, signals)
    saw_obj._pwcom.ProcessAuxFile.assert_called()

def test_sensitivity_mixin(saw_obj):
    """Test SensitivityMixin methods."""
    saw_obj.CalculateFlowSense('[INTERFACE "Left-Right"]', 'MW')
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculateFlowSense([INTERFACE "Left-Right"], MW);')

    saw_obj.CalculatePTDF('[AREA "Top"]', '[BUS 7]', 'DCPS')
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculatePTDF([AREA "Top"], [BUS 7], DCPS);')

    saw_obj.CalculateLODF('[BRANCH 1 2 1]', 'DC')
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculateLODF([BRANCH 1 2 1], DC);')

    saw_obj.CalculateShiftFactors('[BRANCH 1 2 "1"]', 'SELLER', '[AREA "Top"]', 'DC')
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculateShiftFactors([BRANCH 1 2 "1"], SELLER, [AREA "Top"], DC);')

def test_solve_contingencies(saw_obj):
    saw_obj.SolveContingencies()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("CTGSolveAll(NO, YES);")

def test_fault_mixin(saw_obj):
    """Test FaultMixin methods."""
    saw_obj.RunFault('[BUS 1]', 'SLG', 0.001, 0.01)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('Fault([BUS 1], SLG, 0.001, 0.01);')

    saw_obj.RunFault('[BRANCH 1 2 1]', 'SLG', 0.0, 0.0, 50.0)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('Fault([BRANCH 1 2 1], 50.0, SLG, 0.0, 0.0);')

    saw_obj.FaultClear()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("FaultClear;")

    saw_obj.FaultAutoInsert()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("FaultAutoInsert;")

def test_sensitivity_extras(saw_obj):
    """Test extra SensitivityMixin methods."""
    saw_obj.CalculateLODFMatrix("OUTAGES", "ALL", "ALL")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculateLODFMatrix(OUTAGES, ALL, ALL, YES, DC, , YES);')

    saw_obj.CalculateVoltToTransferSense('[AREA "Top"]', '[AREA "Left"]', 'P', True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CalculateVoltToTransferSense([AREA "Top"], [AREA "Left"], P, YES);')

def test_topology_extras_2(saw_obj):
    """Test extra TopologyMixin methods."""
    saw_obj.DoFacilityAnalysis("cut.aux", True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('DoFacilityAnalysis("cut.aux", YES);')

    saw_obj.FindRadialBusPaths(True, False, "BUS")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('FindRadialBusPaths(YES, NO, BUS);')

    saw_obj.SetSelectedFromNetworkCut(True, "[BUS 1]", "SELECTED")
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_contingency_extras(saw_obj):
    """Test extra ContingencyMixin methods."""
    saw_obj.CTGAutoInsert()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("CTGAutoInsert;")

    saw_obj.CTGWriteResultsAndOptions("results.aux")
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_atc_mixin(saw_obj):
    """Test ATCMixin methods."""
    saw_obj.DetermineATC('[AREA "Top"]', '[AREA "Left"]', True, True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('ATCDetermine([AREA "Top"], [AREA "Left"], YES, YES);')

    saw_obj.DetermineATCMultipleDirections()
    saw_obj._pwcom.RunScriptCommand.assert_called_with('ATCDetermineMultipleDirections(NO, NO);')

    # Mock GetParametersMultipleElement for GetATCResults
    saw_obj._pwcom.GetParametersMultipleElement.return_value = ("", [[100], ["Ctg1"]])
    
    # Mock field list for TransferLimiter to avoid ValueError in identify_numeric_fields
    saw_obj._object_fields["transferlimiter"] = pd.DataFrame({
        "internal_field_name": ["LimitingContingency", "MaxFlow"],
        "field_data_type": ["String", "Real"],
        "key_field": ["", ""],
        "description": ["", ""],
        "display_name": ["", ""]
    }).sort_values(by="internal_field_name")
    
    df = saw_obj.GetATCResults(["MaxFlow", "LimitingContingency"])
    assert isinstance(df, pd.DataFrame)
    assert "MaxFlow" in df.columns

def test_gic_mixin(saw_obj):
    """Test GICMixin methods."""
    saw_obj.CalculateGIC(5.0, 90.0, True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICCalculate(5.0, 90.0, YES);')

    saw_obj.ClearGIC()
    saw_obj._pwcom.RunScriptCommand.assert_called_with("GICClear;")

def test_opf_mixin(saw_obj):
    """Test OPFMixin methods."""
    saw_obj.SolvePrimalLP()
    saw_obj._pwcom.RunScriptCommand.assert_called_with('SolvePrimalLP("", "", NO, NO);')

    saw_obj.SolveFullSCOPF()
    saw_obj._pwcom.RunScriptCommand.assert_called_with('SolveFullSCOPF(OPF, "", "", NO, NO);')

def test_pv_mixin(saw_obj):
    """Test PVMixin methods."""
    saw_obj.RunPV('[INJECTIONGROUP "Source"]', '[INJECTIONGROUP "Sink"]')
    saw_obj._pwcom.RunScriptCommand.assert_called_with('PVRun([INJECTIONGROUP "Source"], [INJECTIONGROUP "Sink"]);')

def test_qv_mixin(saw_obj):
    """Test QVMixin methods."""
    # Test with filename provided
    saw_obj.RunQV("results.csv")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('QVRun("results.csv", YES, NO);')

    # Test without filename (should use temp file and return DataFrame)
    # We need to mock open/read for the temp file part, but since we are mocking RunScriptCommand,
    # the file won't actually be created by PowerWorld.
    # We can mock the tempfile creation and existence check.
    with patch("tempfile.NamedTemporaryFile") as mock_temp, \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=100), \
         patch("pandas.read_csv", return_value=pd.DataFrame({"V": [1.0]})):
        
        df = saw_obj.RunQV()
        assert isinstance(df, pd.DataFrame)
        assert "V" in df.columns

def test_base_extras(saw_obj):
    """Test additional base methods."""
    saw_obj.LogSave("log.txt")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('LogSave("log.txt", NO);')

    saw_obj.SetCurrentDirectory("C:\\Temp")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('SetCurrentDirectory("C:\\Temp", NO);')

    saw_obj.SetData("Bus", ["Name"], ["NewName"], "SELECTED")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('SetData(Bus, [Name], [NewName], SELECTED);')

    saw_obj.CreateData("Bus", ["BusNum"], [99])
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CreateData(Bus, [BusNum], [99]);')

    saw_obj.Delete("Bus", "SELECTED")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('Delete(Bus, SELECTED);')

    saw_obj.SelectAll("Bus")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('SelectAll(Bus, );')

def test_transient_extras_2(saw_obj):
    """Test extra TransientMixin methods."""
    saw_obj.TSAutoInsertDistRelay(80, True, True, True, 3, "AREAZONE")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSAutoInsertDistRelay(80, YES, YES, YES, 3, "AREAZONE");')

    saw_obj.TSCalculateCriticalClearTime("[BRANCH 1 2 1]")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSCalculateCriticalClearTime([BRANCH 1 2 1]);')

def test_contingency_extras_2(saw_obj):
    """Test extra ContingencyMixin methods."""
    saw_obj.CTGCloneOne("Ctg1", "Ctg2", "Pre", "Suf", True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('CTGCloneOne("Ctg1", "Ctg2", "Pre", "Suf", YES);')

def test_gic_advanced(saw_obj):
    """Test advanced GIC methods from PDF."""
    # GICLoad3DEfield
    saw_obj.GICLoad3DEfield("B3D", "test.b3d", True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICLoad3DEfield(B3D, "test.b3d", YES);')

    # GICSaveGMatrix
    saw_obj.GICSaveGMatrix("gmatrix.mat", "gmatrix_ids.txt")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICSaveGMatrix("gmatrix.mat", "gmatrix_ids.txt");')

    # GICSetupTimeVaryingSeries
    saw_obj.GICSetupTimeVaryingSeries(0.0, 3600.0, 60.0)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICSetupTimeVaryingSeries(0.0, 3600.0, 60.0);')

    # GICTimeVaryingCalculate
    saw_obj.GICTimeVaryingCalculate(1800.0, True)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICTimeVaryingCalculate(1800.0, YES);')

    # GICWriteOptions
    saw_obj.GICWriteOptions("gic_opts.aux", "PRIMARY")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('GICWriteOptions("gic_opts.aux", PRIMARY);')

def test_transient_advanced(saw_obj):
    """Test advanced Transient Stability methods from PDF."""
    # TSAutoSavePlots
    saw_obj.TSAutoSavePlots(["Plot1"], ["Ctg1"], "JPG", 800, 600, 1, False, False)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSAutoSavePlots(["Plot1"], ["Ctg1"], JPG, 800, 600, 1, NO, NO);')

    # TSClearModelsforObjects
    saw_obj.TSClearModelsforObjects("Gen", "SELECTED")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSClearModelsforObjects(Gen, "SELECTED");')

    # TSJoinActiveCTGs
    saw_obj.TSJoinActiveCTGs(10.0, False, True, "", "Both")
    saw_obj._pwcom.RunScriptCommand.assert_called_with('TSJoinActiveCTGs(10.0, NO, YES, "", Both);')
