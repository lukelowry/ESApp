"""
Unit tests for the SAW class core methods and mixins.

WHAT THIS TESTS:
- Case file operations (open, save, close)
- Script command execution via RunScriptCommand
- Power flow solution commands (SolvePowerFlow, etc.)
- Contingency analysis commands (RunContingency, SolveContingencies)
- State management (StoreState, RestoreState, DeleteState)
- Mode switching (EnterMode)
- Logging and utility commands
- Command string formatting and validation

DEPENDENCIES: None (mocked COM interface, no PowerWorld required)

USAGE:
    pytest tests/test_saw_core_methods.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from esapp import SAW, grid

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

@pytest.mark.parametrize("method, args, expected_script", [
    ("RunScriptCommand", ("SolvePowerFlow;",), "SolvePowerFlow;"),
    ("SolvePowerFlow", (), "SolvePowerFlow(RECTNEWT)"),
    ("RunContingency", ("MyCtg",), 'CTGSolve("MyCtg");'),
    ("TSSolve", ("MyCtg",), 'TSSolve("MyCtg")'),
    ("EnterMode", ("EDIT",), "EnterMode(EDIT);"),
    ("StoreState", ("State1",), 'StoreState("State1");'),
    ("RestoreState", ("State1",), 'RestoreState(USER, "State1");'),
    ("DeleteState", ("State1",), 'DeleteState(USER, "State1");'),
    ("LogAdd", ("Test Message",), 'LogAdd("Test Message");'),
    ("LogClear", (), "LogClear;"),
    ("RenumberCase", (), "RenumberCase;"),
    ("RenumberBuses", (5,), "RenumberBuses(5);"),
    ("TSTransferStateToPowerFlow", (), "TSTransferStateToPowerFlow(NO);"),
    ("TSSolveAll", (), "TSSolveAll()"),
    ("SolveContingencies", (), "CTGSolveAll(NO, YES);"),
    ("FaultClear", (), "FaultClear;"),
    ("FaultAutoInsert", (), "FaultAutoInsert;"),
    ("CTGAutoInsert", (), "CTGAutoInsert;"),
    ("DetermineATCMultipleDirections", (), 'ATCDetermineMultipleDirections(NO, NO);'),
    ("ClearGIC", (), "GICClear;"),
    ("SolvePrimalLP", (), 'SolvePrimalLP("", "", NO, NO);'),
    ("SolveFullSCOPF", (), 'SolveFullSCOPF(OPF, "", "", NO, NO);'),
    ("RunPV", ('[INJECTIONGROUP "Source"]', '[INJECTIONGROUP "Sink"]'), 'PVRun([INJECTIONGROUP "Source"], [INJECTIONGROUP "Sink"]);'),
    ("RunQV", ("results.csv",), 'QVRun("results.csv", YES, NO);'),
    ("LogSave", ("log.txt",), 'LogSave("log.txt", NO);'),
    ("SetCurrentDirectory", ("C:\\Temp",), 'SetCurrentDirectory("C:\\Temp", NO);'),
    ("SetData", ("Bus", ["Name"], ["NewName"], "SELECTED"), 'SetData(Bus, [Name], [NewName], SELECTED);'),
    ("CreateData", ("Bus", ["BusNum"], [99]), 'CreateData(Bus, [BusNum], [99]);'),
    ("Delete", ("Bus", "SELECTED"), 'Delete(Bus, SELECTED);'),
    ("SelectAll", ("Bus",), 'SelectAll(Bus, );'),
    ("TSCalculateCriticalClearTime", ("[BRANCH 1 2 1]",), 'TSCalculateCriticalClearTime([BRANCH 1 2 1]);'),
    ("CTGCloneOne", ("Ctg1", "Ctg2", "Pre", "Suf", True), 'CTGCloneOne("Ctg1", "Ctg2", "Pre", "Suf", YES);'),
    ("GICSaveGMatrix", ("gmatrix.mat", "gmatrix_ids.txt"), 'GICSaveGMatrix("gmatrix.mat", "gmatrix_ids.txt");'),
    ("GICSetupTimeVaryingSeries", (0.0, 3600.0, 60.0), 'GICSetupTimeVaryingSeries(0.0, 3600.0, 60.0);'),
    ("GICTimeVaryingCalculate", (1800.0, True), 'GICTimeVaryingCalculate(1800.0, YES);'),
    ("GICWriteOptions", ("gic_opts.aux", "PRIMARY"), 'GICWriteOptions("gic_opts.aux", PRIMARY);'),
    ("TSClearModelsforObjects", ("Gen", "SELECTED"), 'TSClearModelsforObjects(Gen, "SELECTED");'),
    ("TSJoinActiveCTGs", (10.0, False, True, "", "Both"), 'TSJoinActiveCTGs(10.0, NO, YES, "", Both);'),
    ("CalculateFlowSense", ('[INTERFACE "Left-Right"]', 'MW'), 'CalculateFlowSense([INTERFACE "Left-Right"], MW);'),
    ("CalculatePTDF", ('[AREA "Top"]', '[BUS 7]', 'DCPS'), 'CalculatePTDF([AREA "Top"], [BUS 7], DCPS);'),
    ("CalculateLODF", ('[BRANCH 1 2 1]', 'DC'), 'CalculateLODF([BRANCH 1 2 1], DC);'),
    ("CalculateShiftFactors", ('[BRANCH 1 2 "1"]', 'SELLER', '[AREA "Top"]', 'DC'), 'CalculateShiftFactors([BRANCH 1 2 "1"], SELLER, [AREA "Top"], DC);'),
    ("RunFault", ('[BUS 1]', 'SLG', 0.001, 0.01), 'Fault([BUS 1], SLG, 0.001, 0.01);'),
    ("CalculateLODFMatrix", ("OUTAGES", "ALL", "ALL"), 'CalculateLODFMatrix(OUTAGES, ALL, ALL, YES, DC, , YES);'),
    ("CalculateVoltToTransferSense", ('[AREA "Top"]', '[AREA "Left"]', 'P', True), 'CalculateVoltToTransferSense([AREA "Top"], [AREA "Left"], P, YES);'),
    ("DoFacilityAnalysis", ("cut.aux", True), 'DoFacilityAnalysis("cut.aux", YES);'),
    ("FindRadialBusPaths", (True, False, "BUS"), 'FindRadialBusPaths(YES, NO, BUS);'),
    ("DetermineATC", ('[AREA "Top"]', '[AREA "Left"]', True, True), 'ATCDetermine([AREA "Top"], [AREA "Left"], YES, YES);'),
    ("CalculateGIC", (5.0, 90.0, True), 'GICCalculate(5.0, 90.0, YES);'),
    ("TSAutoInsertDistRelay", (80, True, True, True, 3, "AREAZONE"), 'TSAutoInsertDistRelay(80, YES, YES, YES, 3, "AREAZONE");'),
    ("GICLoad3DEfield", ("B3D", "test.b3d", True), 'GICLoad3DEfield(B3D, "test.b3d", YES);'),
    ("TSAutoSavePlots", (["Plot1"], ["Ctg1"], "JPG", 800, 600, 1, False, False), 'TSAutoSavePlots(["Plot1"], ["Ctg1"], JPG, 800, 600, 1, NO, NO);'),
])
def test_simple_script_commands(saw_obj, method, args, expected_script):
    """Parametrized test for simple wrapper methods that call RunScriptCommand."""
    getattr(saw_obj, method)(*args)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(expected_script)

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
        }) if obj.lower() == "branch" else pd.DataFrame()
        
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

def test_transient_extras(saw_obj):
    """Test additional TransientMixin methods."""
    saw_obj.TSInitialize()
    saw_obj._pwcom.RunScriptCommand.assert_called()
    
    saw_obj.TSResultStorageSetAll("Gen", False)
    saw_obj._pwcom.RunScriptCommand.assert_called_with("TSResultStorageSetAll(Gen, NO)")
    
    saw_obj.TSClearResultsFromRAM()
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_ts_set_play_in_signals(saw_obj):
    """Test TSSetPlayInSignals."""
    times = np.array([0.0, 0.1])
    signals = np.array([[1.0], [1.0]])
    saw_obj.TSSetPlayInSignals("TestSignal", times, signals)
    saw_obj._pwcom.ProcessAuxFile.assert_called()

def test_fault_mixin(saw_obj):
    """Test FaultMixin methods."""
    saw_obj.RunFault('[BRANCH 1 2 1]', 'SLG', 0.0, 0.0, 50.0)
    saw_obj._pwcom.RunScriptCommand.assert_called_with('Fault([BRANCH 1 2 1], 50.0, SLG, 0.0, 0.0);')

    saw_obj.SetSelectedFromNetworkCut(True, "[BUS 1]", "SELECTED")
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_contingency_extras(saw_obj):
    """Test extra ContingencyMixin methods."""
    saw_obj.CTGWriteResultsAndOptions("results.aux")
    saw_obj._pwcom.RunScriptCommand.assert_called()

def test_atc_mixin(saw_obj):
    """Test ATCMixin methods."""
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

def test_qv_mixin(saw_obj):
    """Test QVMixin methods."""
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
