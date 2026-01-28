"""
Unit tests for the SAW class core methods and mixins.

Tests script command formatting, data retrieval/transformation, and error handling
using a mocked COM interface (no PowerWorld required).

USAGE:
    pytest tests/test_saw_unit.py -v
"""
import os
import pytest
from unittest.mock import MagicMock, Mock, patch
import pandas as pd
import numpy as np
from esapp import SAW


# =============================================================================
# Script command formatting (parametrized)
# =============================================================================

@pytest.mark.parametrize("method, args, expected_script", [
    ("RunScriptCommand", ("SolvePowerFlow;",), "SolvePowerFlow;"),
    ("SolvePowerFlow", (), "SolvePowerFlow(RECTNEWT)"),
    ("EnterMode", ("EDIT",), "EnterMode(EDIT);"),
    ("StoreState", ("State1",), 'StoreState("State1");'),
    ("RestoreState", ("State1",), 'RestoreState(USER, "State1");'),
    ("DeleteState", ("State1",), 'DeleteState(USER, "State1");'),
    ("LogAdd", ("Test Message",), 'LogAdd("Test Message");'),
    ("LogClear", (), "LogClear;"),
    ("LogSave", ("log.txt",), 'LogSave("log.txt", NO);'),
    ("RenumberCase", (), "RenumberCase;"),
    ("RenumberBuses", (5,), "RenumberBuses(5);"),
    ("SetCurrentDirectory", ("C:\\Temp",), 'SetCurrentDirectory("C:\\Temp", NO);'),
    ("SetData", ("Bus", ["Name"], ["NewName"], "SELECTED"), 'SetData(Bus, [Name], [NewName], SELECTED);'),
    ("CreateData", ("Bus", ["BusNum"], [99]), 'CreateData(Bus, [BusNum], [99]);'),
    ("Delete", ("Bus", "SELECTED"), 'Delete(Bus, SELECTED);'),
    ("SelectAll", ("Bus",), 'SelectAll(Bus, );'),
    ("TSTransferStateToPowerFlow", (), "TSTransferStateToPowerFlow(NO);"),
    ("TSSolveAll", (), "TSSolveAll()"),
    ("TSSolve", ("MyCtg",), 'TSSolve("MyCtg", [0, 10, 0.25, YES])'),
    ("TSCalculateCriticalClearTime", ("[BRANCH 1 2 1]",), 'TSCalculateCriticalClearTime([BRANCH 1 2 1]);'),
    ("TSClearModelsforObjects", ("Gen", "SELECTED"), 'TSClearModelsforObjects(Gen, "SELECTED");'),
    ("TSJoinActiveCTGs", (10.0, False, True, "", "Both"), 'TSJoinActiveCTGs(10.0, NO, YES, "", Both);'),
    ("TSAutoInsertDistRelay", (80, True, True, True, 3, "AREAZONE"), 'TSAutoInsertDistRelay(80, YES, YES, YES, 3, "AREAZONE");'),
    ("TSAutoSavePlots", (["Plot1"], ["Ctg1"], "JPG", 800, 600, 1.0, False, False), 'TSAutoSavePlots(["Plot1"], ["Ctg1"], JPG, 800, 600, 1.0, NO, NO);'),
    ("TSResultStorageSetAll", ("Gen", False), "TSResultStorageSetAll(Gen, NO)"),
    ("SolveContingencies", (), "CTGSolveAll(NO, YES);"),
    ("RunContingency", ("MyCtg",), 'CTGSolve("MyCtg");'),
    ("CTGAutoInsert", (), "CTGAutoInsert;"),
    ("CTGCloneOne", ("Ctg1", "Ctg2", "Pre", "Suf", True), 'CTGCloneOne("Ctg1", "Ctg2", "Pre", "Suf", YES);'),
    ("FaultClear", (), "FaultClear;"),
    ("FaultAutoInsert", (), "FaultAutoInsert;"),
    ("RunFault", ('[BUS 1]', 'SLG', 0.001, 0.01), 'Fault([BUS 1], SLG, 0.001, 0.01);'),
    ("CalculateFlowSense", ('[INTERFACE "Left-Right"]', 'MW'), 'CalculateFlowSense([INTERFACE "Left-Right"], MW);'),
    ("CalculatePTDF", ('[AREA "Top"]', '[BUS 7]', 'DCPS'), 'CalculatePTDF([AREA "Top"], [BUS 7], DCPS);'),
    ("CalculateLODF", ('[BRANCH 1 2 1]', 'DC'), 'CalculateLODF([BRANCH 1 2 1], DC);'),
    ("CalculateShiftFactors", ('[BRANCH 1 2 "1"]', 'SELLER', '[AREA "Top"]', 'DC'), 'CalculateShiftFactors([BRANCH 1 2 "1"], SELLER, [AREA "Top"], DC);'),
    ("CalculateLODFMatrix", ("OUTAGES", "ALL", "ALL"), 'CalculateLODFMatrix(OUTAGES, ALL, ALL, YES, DC, , YES);'),
    ("CalculateVoltToTransferSense", ('[AREA "Top"]', '[AREA "Left"]', 'P', True), 'CalculateVoltToTransferSense([AREA "Top"], [AREA "Left"], P, YES);'),
    ("DoFacilityAnalysis", ("cut.aux", True), 'DoFacilityAnalysis("cut.aux", YES);'),
    ("FindRadialBusPaths", (True, False, "BUS"), 'FindRadialBusPaths(YES, NO, BUS);'),
    ("DetermineATC", ('[AREA "Top"]', '[AREA "Left"]', True, True), 'ATCDetermine([AREA "Top"], [AREA "Left"], YES, YES);'),
    ("DetermineATCMultipleDirections", (), 'ATCDetermineMultipleDirections(NO, NO);'),
    ("ClearGIC", (), "GICClear;"),
    ("CalculateGIC", (5.0, 90.0, True), 'GICCalculate(5.0, 90.0, YES);'),
    ("GICSaveGMatrix", ("gmatrix.mat", "gmatrix_ids.txt"), 'GICSaveGMatrix("gmatrix.mat", "gmatrix_ids.txt");'),
    ("GICSetupTimeVaryingSeries", (0.0, 3600.0, 60.0), 'GICSetupTimeVaryingSeries(0.0, 3600.0, 60.0);'),
    ("GICTimeVaryingCalculate", (1800.0, True), 'GICTimeVaryingCalculate(1800.0, YES);'),
    ("GICWriteOptions", ("gic_opts.aux", "PRIMARY"), 'GICWriteOptions("gic_opts.aux", PRIMARY);'),
    ("GICLoad3DEfield", ("B3D", "test.b3d", True), 'GICLoad3DEfield(B3D, "test.b3d", YES);'),
    ("SolvePrimalLP", (), 'SolvePrimalLP("", "", NO, NO);'),
    ("SolveFullSCOPF", (), 'SolveFullSCOPF(OPF, "", "", NO, NO);'),
    ("RunPV", ('[INJECTIONGROUP "Source"]', '[INJECTIONGROUP "Sink"]'), 'PVRun([INJECTIONGROUP "Source"], [INJECTIONGROUP "Sink"]);'),
    ("RunQV", ("results.csv",), 'QVRun("results.csv", YES, NO);'),
    ("AutoInsertTieLineTransactions", (), "AutoInsertTieLineTransactions;"),
    ("ChangeSystemMVABase", (100.0,), "ChangeSystemMVABase(100.0);"),
    ("ClearSmallIslands", (), "ClearSmallIslands;"),
    ("InitializeGenMvarLimits", (), "InitializeGenMvarLimits;"),
    ("InjectionGroupsAutoInsert", (), "InjectionGroupsAutoInsert;"),
    ("DirectionsAutoInsert", ('[AREA "Top"]', '[AREA "Bot"]', True, False), 'DirectionsAutoInsert([AREA "Top"], [AREA "Bot"], YES, NO);'),
    ("InterfacesAutoInsert", ("AREA", True, False, "", "AUTO"), 'InterfacesAutoInsert(AREA, YES, NO, "", AUTO);'),
    ("InterfaceFlatten", ("MyInterface",), 'InterfaceFlatten("MyInterface");'),
    ("InterfaceAddElementsFromContingency", ("Interface1", "Ctg1"), 'InterfaceAddElementsFromContingency("Interface1", "Ctg1");'),
    ("MergeLineTerminals", ("SELECTED",), "MergeLineTerminals(SELECTED);"),
    ("MergeMSLineSections", ("SELECTED",), "MergeMSLineSections(SELECTED);"),
    ("CaseDescriptionClear", (), "CaseDescriptionClear;"),
    ("CaseDescriptionSet", ("Test description", False), 'CaseDescriptionSet("Test description", NO);'),
    ("CaseDescriptionSet", ("Appended", True), 'CaseDescriptionSet("Appended", YES);'),
    ("DeleteExternalSystem", (), "DeleteExternalSystem;"),
    ("Equivalence", (), "Equivalence;"),
    ("NewCase", (), "NewCase;"),
    ("RenumberAreas", (0,), "RenumberAreas(0);"),
    ("RenumberSubs", (2,), "RenumberSubs(2);"),
    ("RenumberZones", (3,), "RenumberZones(3);"),
    ("CloseOneline", ("MyOneline",), 'CloseOneline("MyOneline")'),
    ("SaveOneline", ("out.pwb", "MyOneline", "PWB"), 'SaveOneline("out.pwb", "MyOneline", PWB);'),
    ("ExportOneline", ("out.jpg", "MyOneline", "JPG", "", "NO", "NO"), 'ExportOneline("out.jpg", "MyOneline", JPG, "", NO, NO);'),
    ("PVClear", (), "PVClear;"),
    ("PVDestroy", (), "PVDestroy;"),
    ("PVStartOver", (), "PVStartOver;"),
    ("PVSetSourceAndSink", ('[InjectionGroup "A"]', '[InjectionGroup "B"]'), 'PVSetSourceAndSink([InjectionGroup "A"], [InjectionGroup "B"]);'),
    ("PVQVTrackSingleBusPerSuperBus", (), "PVQVTrackSingleBusPerSuperBus;"),
    ("PVWriteResultsAndOptions", ("pv_results.aux", True), 'PVWriteResultsAndOptions("pv_results.aux", YES);'),
    ("PVWriteResultsAndOptions", ("pv_results.aux", False), 'PVWriteResultsAndOptions("pv_results.aux", NO);'),
    ("QVDeleteAllResults", (), "QVDeleteAllResults;"),
    ("QVSelectSingleBusPerSuperBus", (), "QVSelectSingleBusPerSuperBus;"),
    ("QVWriteResultsAndOptions", ("qv_results.aux", True), 'QVWriteResultsAndOptions("qv_results.aux", YES);'),
    ("QVWriteResultsAndOptions", ("qv_results.aux", False), 'QVWriteResultsAndOptions("qv_results.aux", NO);'),
    ("QVDataWriteOptionsAndResults", ("qv_data.aux", True, "PRIMARY"), 'QVDataWriteOptionsAndResults("qv_data.aux", YES, PRIMARY);'),
    ("ATCDeleteAllResults", (), "ATCDeleteAllResults;"),
    ("ATCRestoreInitialState", (), "ATCRestoreInitialState;"),
    ("ATCIncreaseTransferBy", (50.0,), "ATCIncreaseTransferBy(50.0);"),
    ("ATCDetermineATCFor", (0, 0, 0, False), "ATCDetermineATCFor(0, 0, 0, NO);"),
    ("ATCDetermineATCFor", (1, 2, 3, True), "ATCDetermineATCFor(1, 2, 3, YES);"),
    ("ATCDetermineMultipleDirectionsATCFor", (0, 0, 0), "ATCDetermineMultipleDirectionsATCFor(0, 0, 0);"),
    ("RegionRename", ("OldRegion", "NewRegion", True), 'RegionRename("OldRegion", "NewRegion", YES);'),
    ("RegionRename", ("OldRegion", "NewRegion", False), 'RegionRename("OldRegion", "NewRegion", NO);'),
    ("RegionRenameClass", ("OldClass", "NewClass", True, ""), 'RegionRenameClass("OldClass", "NewClass", YES, );'),
    ("TimeStepDeleteAll", (), "TimeStepDeleteAll;"),
    ("TimeStepResetRun", (), "TimeStepResetRun;"),
    ("TIMESTEPSaveSelectedModifyStart", (), "TIMESTEPSaveSelectedModifyStart;"),
    ("TIMESTEPSaveSelectedModifyFinish", (), "TIMESTEPSaveSelectedModifyFinish;"),
    ("TimeStepSavePWW", ("weather.pww",), 'TimeStepSavePWW("weather.pww");'),
    ("TimeStepLoadTSB", ("data.tsb",), 'TimeStepLoadTSB("data.tsb");'),
    ("TimeStepSaveTSB", ("output.tsb",), 'TimeStepSaveTSB("output.tsb");'),
    ("TimeStepAppendPWW", ("weather.pww", "Single Solution"), 'TimeStepAppendPWW("weather.pww", "Single Solution");'),
    ("TimeStepLoadPWW", ("weather.pww", "OPF"), 'TimeStepLoadPWW("weather.pww", "OPF");'),
    ("TimeStepDoSinglePoint", ("2025-01-01T00:00:00",), "TimeStepDoSinglePoint(2025-01-01T00:00:00);"),
    ("TimeStepLoadB3D", ("test.b3d", "GIC Only (No Power Flow)"), 'TimeStepLoadB3D("test.b3d", "GIC Only (No Power Flow)");'),
    ("UpdateIslandsAndBusStatus", (), "UpdateIslandsAndBusStatus;"),
    ("ZeroOutMismatches", ("BUSSHUNT",), "ZeroOutMismatches(BUSSHUNT);"),
    ("ZeroOutMismatches", ("LOAD",), "ZeroOutMismatches(LOAD);"),
    ("VoltageConditioning", (), "VoltageConditioning;"),
    ("DiffCaseClearBase", (), "DiffCaseClearBase;"),
    ("DiffCaseSetAsBase", (), "DiffCaseSetAsBase;"),
    ("DiffCaseKeyType", ("PRIMARY",), "DiffCaseKeyType(PRIMARY);"),
    ("DiffCaseShowPresentAndBase", (True,), "DiffCaseShowPresentAndBase(YES);"),
    ("DiffCaseShowPresentAndBase", (False,), "DiffCaseShowPresentAndBase(NO);"),
    ("DiffCaseMode", ("DIFFERENCE",), "DiffCaseMode(DIFFERENCE);"),
    ("DiffCaseRefresh", (), "DiffCaseRefresh;"),
    ("DoCTGAction", ("APPLY",), "DoCTGAction(APPLY);"),
    ("InterfacesCalculatePostCTGMWFlows", (), "InterfacesCalculatePostCTGMWFlows;"),
    ("GenForceLDC_RCC", ("MyFilter",), 'GenForceLDC_RCC("MyFilter");'),
    ("SaveGenLimitStatusAction", ("genlimits.txt",), 'SaveGenLimitStatusAction("genlimits.txt");'),
    ("CTGClearAllResults", (), "CTGClearAllResults;"),
    ("CTGSetAsReference", (), "CTGSetAsReference;"),
    ("CTGComboDeleteAllResults", (), "CTGComboDeleteAllResults;"),
    ("CTGCreateExpandedBreakerCTGs", (), "CTGCreateExpandedBreakerCTGs;"),
    ("CTGDeleteWithIdenticalActions", (), "CTGDeleteWithIdenticalActions;"),
    ("CTGPrimaryAutoInsert", (), "CTGPrimaryAutoInsert;"),
    ("CTGApply", ("Ctg1",), 'CTGApply("Ctg1");'),
    ("CTGProduceReport", ("ctg_report.txt",), 'CTGProduceReport("ctg_report.txt");'),
    ("CTGReadFilePSLF", ("contingencies.pslf",), 'CTGReadFilePSLF("contingencies.pslf");'),
    ("CTGCalculateOTDF", ('[AREA "Top"]', '[AREA "Bottom"]', "DC"), 'CTGCalculateOTDF([AREA "Top"], [AREA "Bottom"], DC);'),
    ("CTGCompareTwoListsofContingencyResults", ("List1", "List2"), "CTGCompareTwoListsofContingencyResults(List1, List2);"),
    ("CTGConvertAllToDeviceCTG", (False,), "CTGConvertAllToDeviceCTG(NO);"),
    ("CTGConvertAllToDeviceCTG", (True,), "CTGConvertAllToDeviceCTG(YES);"),
    ("CopyFile", ("old.txt", "new.txt"), 'CopyFile("old.txt", "new.txt");'),
    ("DeleteFile", ("todelete.txt",), 'DeleteFile("todelete.txt");'),
    ("RenameFile", ("old.txt", "new.txt"), 'RenameFile("old.txt", "new.txt");'),
    ("LogShow", (True,), "LogShow(YES);"),
    ("LogShow", (False,), "LogShow(NO);"),
    ("LogSave", ("log.txt", False), 'LogSave("log.txt", NO);'),
    ("LogSave", ("log.txt", True), 'LogSave("log.txt", YES);'),
    ("EnterMode", ("RUN",), "EnterMode(RUN);"),
    ("LoadCSV", ("data.csv", False), 'LoadCSV("data.csv", NO);'),
    ("LoadCSV", ("data.csv", True), 'LoadCSV("data.csv", YES);'),
    ("LoadScript", ("script.aux", "MyScript"), 'LoadScript("script.aux", "MyScript");'),
    ("Delete", ("Bus", "MyFilter"), 'Delete(Bus, "MyFilter");'),
    ("SelectAll", ("Gen", "MyFilter"), 'SelectAll(Gen, "MyFilter");'),
    ("UnSelectAll", ("Load", "MyFilter"), 'UnSelectAll(Load, "MyFilter");'),
    ("StopAuxFile", (), "StopAuxFile;"),
    ("LineLoadingReplicatorImplement", (), "LineLoadingReplicatorImplement;"),
    ("CalculateTapSense", ("MyFilter",), 'CalculateTapSense("MyFilter");'),
    ("CalculateVoltSelfSense", ("MyFilter",), 'CalculateVoltSelfSense("MyFilter");'),
    ("RelinkAllOpenOnelines", (), "RelinkAllOpenOnelines;"),
    ("TSAutoCorrect", (), "TSAutoCorrect;"),
    ("TSClearAllModels", (), "TSClearAllModels;"),
    ("TSValidate", (), "TSValidate;"),
    ("TSClearPlayInSignals", (), "DELETE(PLAYINSIGNAL);"),
    ("TSLoadPTI", ("dynamics.dyr",), 'TSLoadPTI("dynamics.dyr");'),
    ("TSLoadGE", ("dynamics.dyd",), 'TSLoadGE("dynamics.dyd");'),
    ("TSLoadBPA", ("dynamics.bpa",), 'TSLoadBPA("dynamics.bpa");'),
    ("TSCalculateSMIBEigenValues", (), "TSCalculateSMIBEigenValues;"),
    ("OPFWriteResultsAndOptions", ("opf_results.aux",), 'OPFWriteResultsAndOptions("opf_results.aux");'),
    ("GICReadFilePSLF", ("gic.gmd",), 'GICReadFilePSLF("gic.gmd");'),
    ("GICReadFilePTI", ("gic.gic",), 'GICReadFilePTI("gic.gic");'),
    ("GICTimeVaryingDeleteAllTimes", (), "GICTimeVaryingDeleteAllTimes;"),
    ("GICTimeVaryingElectricFieldsDeleteAllTimes", (), "GICTimeVaryingElectricFieldsDeleteAllTimes;"),
    ("GICTimeVaryingAddTime", (3600.0,), "GICTimeVaryingAddTime(3600.0);"),
    ("RegionUpdateBuses", (), "RegionUpdateBuses;"),
])
def test_script_commands(saw_obj, method, args, expected_script):
    """Verify each SAW wrapper method produces the correct PowerWorld script command."""
    getattr(saw_obj, method)(*args)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(expected_script)


# =============================================================================
# Data retrieval and transformation
# =============================================================================

def test_get_parameters_multiple_element(saw_obj):
    """GetParametersMultipleElement returns a properly structured DataFrame."""
    saw_obj._pwcom.GetParametersMultipleElement.return_value = ("", [[1, 2], ["Bus1", "Bus2"]])
    df = saw_obj.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "BusNum" in df.columns
    assert "BusName" in df.columns


def test_change_parameters_single_element(saw_obj):
    """ChangeParametersSingleElement calls the COM method."""
    saw_obj.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [1, "NewName"])
    saw_obj._pwcom.ChangeParametersSingleElement.assert_called()


def test_change_parameters_multiple_element(saw_obj):
    """ChangeParametersMultipleElement with nested list."""
    saw_obj._pwcom.ChangeParametersMultipleElement.return_value = ("",)
    saw_obj.ChangeParametersMultipleElement("Bus", ["BusNum", "BusName"], [[1, 2], ["Name1", "Name2"]])
    saw_obj._pwcom.ChangeParametersMultipleElement.assert_called()


def test_change_parameters_multiple_element_rect(saw_obj):
    """ChangeParametersMultipleElementRect with DataFrame."""
    df = pd.DataFrame({"BusNum": [1, 2], "BusName": ["A", "B"]})
    saw_obj.ChangeParametersMultipleElementRect("Bus", ["BusNum", "BusName"], df)
    saw_obj._pwcom.ChangeParametersMultipleElementRect.assert_called()


def test_change_parameters_multiple_element_flat_input_rejects_nested(saw_obj):
    """ChangeParametersMultipleElementFlatInput rejects nested lists."""
    from esapp.saw._exceptions import Error
    with pytest.raises(Error):
        saw_obj.ChangeParametersMultipleElementFlatInput("Bus", ["BusNum"], 2, [[1], [2]])


def test_get_params_rect_typed(saw_obj):
    """GetParamsRectTyped returns DataFrame."""
    saw_obj._pwcom.GetParamsRectTyped.return_value = ("", [[1, "A"], [2, "B"]])
    df = saw_obj.GetParamsRectTyped("Bus", ["BusNum", "BusName"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_get_params_rect_typed_empty(saw_obj):
    """GetParamsRectTyped returns None for empty result."""
    saw_obj._pwcom.GetParamsRectTyped.return_value = ("", None)
    result = saw_obj.GetParamsRectTyped("Bus", ["BusNum"])
    assert result is None


def test_ts_get_contingency_results(saw_obj):
    """TSGetContingencyResults parses metadata and data into DataFrames."""
    mock_meta = [
        ["Gen", "1", "", "", "GenMW", "MW"],
        ["Bus", "2", "", "", "BusPUVolt", "PU"]
    ]
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
    assert pd.api.types.is_numeric_dtype(data["time"])


# =============================================================================
# Matrix extraction
# =============================================================================

def test_matrix_get_ybus(saw_obj):
    """get_ybus parses MATLAB-style sparse matrix output."""
    mock_data_content = "Ybus=sparse(2,2);Ybus(1,1)=1.0+j*(2.0);Ybus(2,2)=1.0+j*(2.0);"
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = mock_data_content
        mock_file.readline.return_value = "header"
        mock_open.return_value.__enter__.return_value = mock_file
        ybus = saw_obj.get_ybus()
        assert hasattr(ybus, "toarray")


def test_matrix_jacobian(saw_obj):
    """get_jacobian parses MATLAB-style sparse matrix output."""
    mock_mat_content = "Jac=sparse(2,2);Jac(1,1)=1.0;Jac(2,2)=1.0;"
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = mock_mat_content
        mock_open.return_value.__enter__.return_value = mock_file
        jac = saw_obj.get_jacobian()
        assert hasattr(jac, "toarray")


# =============================================================================
# Internal data helpers
# =============================================================================

class TestDataTransformation:
    """Tests for _to_numeric and _replace_decimal_delimiter."""

    def test_to_numeric_dataframe_with_floats(self, saw_obj):
        df = pd.DataFrame({"A": ["1.5", "2.5"], "B": ["3.0", "4.0"]})
        result = saw_obj._to_numeric(df)
        assert pd.api.types.is_numeric_dtype(result["A"])
        assert result["A"].iloc[0] == 1.5

    def test_to_numeric_series(self, saw_obj):
        s = pd.Series(["1.0", "2.0", "3.0"])
        result = saw_obj._to_numeric(s)
        assert pd.api.types.is_numeric_dtype(result)

    def test_to_numeric_mixed_types(self, saw_obj):
        df = pd.DataFrame({"num": ["1", "2"], "text": ["a", "b"]})
        result = saw_obj._to_numeric(df)
        assert pd.api.types.is_numeric_dtype(result["num"])
        assert result["text"].iloc[0] == "a"

    def test_to_numeric_invalid_input(self, saw_obj):
        with pytest.raises(TypeError):
            saw_obj._to_numeric("not a dataframe or series")

    def test_to_numeric_with_locale_delimiter(self, saw_obj):
        saw_obj.decimal_delimiter = ","
        df = pd.DataFrame({"A": ["1,5", "2,5"]})
        result = saw_obj._to_numeric(df)
        assert result["A"].iloc[0] == 1.5
        saw_obj.decimal_delimiter = "."

    def test_replace_comma_delimiter(self, saw_obj):
        saw_obj.decimal_delimiter = ","
        s = pd.Series(["1,5", "2,5", "3,0"])
        result = saw_obj._replace_decimal_delimiter(s)
        assert result.iloc[0] == "1.5"
        saw_obj.decimal_delimiter = "."


class TestFieldMetadata:
    """Tests for GetFieldList caching."""

    def test_get_field_list_returns_dataframe(self, saw_obj):
        df = saw_obj.GetFieldList("Bus")
        assert isinstance(df, pd.DataFrame)
        assert "internal_field_name" in df.columns

    def test_get_field_list_caches_result(self, saw_obj):
        df1 = saw_obj.GetFieldList("Bus")
        saw_obj._pwcom.GetFieldList.reset_mock()
        df2 = saw_obj.GetFieldList("Bus")
        assert df2.equals(df1)


# =============================================================================
# Error handling
# =============================================================================

def test_run_script_command_error_raises(saw_obj):
    """RunScriptCommand raises PowerWorldError on non-empty error string."""
    from esapp.saw._exceptions import PowerWorldError
    saw_obj._pwcom.RunScriptCommand.return_value = ("Error: Something went wrong",)
    with pytest.raises(PowerWorldError):
        saw_obj.RunScriptCommand("BadCommand;")


def test_enter_mode_invalid(saw_obj):
    """EnterMode rejects invalid mode strings."""
    with pytest.raises(ValueError, match="Mode must be either"):
        saw_obj.EnterMode("INVALID")


# =============================================================================
# _helpers.py coverage
# =============================================================================

class TestDfToAux:
    """Tests for df_to_aux function — covers lines 25-53 of _helpers.py."""

    def test_df_to_aux_basic(self):
        """df_to_aux writes correct AUX format for a simple DataFrame."""
        from esapp.saw._helpers import df_to_aux
        import io

        df = pd.DataFrame({"BusNum": [1, 2], "BusName": ["Bus1", "Bus2"]})
        fp = io.StringIO()
        df_to_aux(fp, df, "Bus")
        content = fp.getvalue()

        assert "DATA (Bus, [BusNum,BusName])" in content
        assert "{" in content
        assert "}" in content
        assert "1" in content
        assert "Bus1" in content

    def test_df_to_aux_long_header_wraps(self):
        """df_to_aux wraps long headers across multiple lines."""
        from esapp.saw._helpers import df_to_aux
        import io

        # Create a DataFrame with many columns to force header wrapping
        cols = {f"VeryLongFieldName{i}": [i] for i in range(20)}
        df = pd.DataFrame(cols)
        fp = io.StringIO()
        df_to_aux(fp, df, "Branch")
        content = fp.getvalue()

        assert "DATA (Branch," in content
        assert "{" in content
        assert "}" in content


class TestHelperConversions:
    """Tests for helper conversion functions."""

    def test_convert_to_windows_path(self):
        from esapp.saw._helpers import convert_to_windows_path
        result = convert_to_windows_path("/tmp/test/file.pwb")
        assert "\\" in result or "/" not in result.replace("//", "")

    def test_create_object_string_single_key(self):
        from esapp.saw._helpers import create_object_string
        assert create_object_string("Bus", 1) == "[BUS 1]"

    def test_create_object_string_multiple_keys(self):
        from esapp.saw._helpers import create_object_string
        assert create_object_string("Branch", 1, 2, "1") == "[BRANCH 1 2 1]"


# =============================================================================
# case_actions.py coverage — AppendCase branches, Renumber methods
# =============================================================================

@pytest.mark.parametrize("method, args, expected_script", [
    # AppendCase PTI branch (lines 55-59)
    ("AppendCase", ("case.raw", "PTI"), 'AppendCase("case.raw", PTI, [NEAR, YES]);'),
    # AppendCase GE branch (lines 60-61)
    ("AppendCase", ("case.epc", "GE"), 'AppendCase("case.epc", GE, [MAINTAIN, 2.0, NO, YES]);'),
    # AppendCase default branch (lines 62-63)
    ("AppendCase", ("case.pwb", "PWB"), 'AppendCase("case.pwb", PWB);'),
    # AppendCase GE with custom params
    ("AppendCase", ("case.epc", "GE", "NEAR", False, "EQUIVALENCE", 3.0, True),
     'AppendCase("case.epc", GE, [EQUIVALENCE, 3.0, YES, NO]);'),
    # Renumber3WXFormerStarBuses (line 195)
    ("Renumber3WXFormerStarBuses", ("remap.txt",), 'Renumber3WXFormerStarBuses("remap.txt", BOTH);'),
    ("Renumber3WXFormerStarBuses", ("remap.txt", "COMMA"), 'Renumber3WXFormerStarBuses("remap.txt", COMMA);'),
    # RenumberMSLineDummyBuses (line 256)
    ("RenumberMSLineDummyBuses", ("remap.txt",), 'RenumberMSLineDummyBuses("remap.txt", BOTH);'),
    ("RenumberMSLineDummyBuses", ("remap.txt", "TAB"), 'RenumberMSLineDummyBuses("remap.txt", TAB);'),
    # SaveMergedFixedNumBusCase
    ("SaveMergedFixedNumBusCase", ("merged.pwb",), 'SaveMergedFixedNumBusCase("merged.pwb", PWB);'),
    # LoadEMS
    ("LoadEMS", ("data.hdb",), 'LoadEMS("data.hdb", AREVAHDB);'),
    # Scale
    ("Scale", ("LOAD", "FACTOR", [1.0], "SYSTEM"), "Scale(LOAD, FACTOR, [1.0], SYSTEM);"),
    # SaveExternalSystem with_ties=False (default)
    ("SaveExternalSystem", ("ext.pwb",), 'SaveExternalSystem("ext.pwb", PWB, NO);'),
])
def test_case_actions_extended(saw_obj, method, args, expected_script):
    """Verify case action methods produce correct script commands."""
    getattr(saw_obj, method)(*args)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(expected_script)


# =============================================================================
# base.py coverage — error handling, properties, edge cases
# =============================================================================

class TestSetSimautoProperty:
    """Tests for set_simauto_property error paths — covers lines 287-311."""

    def test_invalid_property_name(self, saw_obj):
        """Raises ValueError for unsupported property name."""
        with pytest.raises(ValueError, match="not currently supported"):
            saw_obj.set_simauto_property("InvalidProp", True)

    def test_invalid_property_type(self, saw_obj):
        """Raises ValueError for wrong property value type."""
        with pytest.raises(ValueError, match="invalid"):
            saw_obj.set_simauto_property("CreateIfNotFound", "not_a_bool")

    def test_invalid_current_dir(self, saw_obj):
        """Raises ValueError for non-existent CurrentDir path."""
        with pytest.raises(ValueError, match="not a valid path"):
            saw_obj.set_simauto_property("CurrentDir", "/nonexistent/path/xyz")

    def test_uivisible_attribute_error(self, saw_obj):
        """UIVisible gracefully handles AttributeError on old versions."""
        saw_obj._pwcom = MagicMock()
        # Make setting UIVisible raise AttributeError
        type(saw_obj._pwcom).UIVisible = property(
            fget=lambda self: False,
            fset=Mock(side_effect=AttributeError("no UIVisible")),
        )
        # Should not raise — just log a warning
        saw_obj.set_simauto_property("UIVisible", True)

    def test_non_uivisible_attribute_error_reraises(self, saw_obj):
        """Non-UIVisible AttributeError is re-raised."""
        saw_obj._pwcom = MagicMock()
        type(saw_obj._pwcom).CreateIfNotFound = property(
            fget=lambda self: False,
            fset=Mock(side_effect=AttributeError("oops")),
        )
        with pytest.raises(AttributeError):
            saw_obj.set_simauto_property("CreateIfNotFound", True)


class TestCallSimautoErrorHandling:
    """Tests for _call_simauto error handling — covers lines 1154-1188."""

    def test_invalid_function_name(self, saw_obj):
        """Raises AttributeError for non-existent SimAuto function."""
        del saw_obj._pwcom.NonExistentFunc
        with pytest.raises(AttributeError, match="not a valid SimAuto function"):
            saw_obj._call_simauto("NonExistentFunc")

    def test_com_exception_raises_com_error(self, saw_obj):
        """COM exceptions are wrapped in COMError."""
        from esapp.saw._exceptions import COMError
        saw_obj._pwcom.RunScriptCommand.side_effect = Exception("COM failure")
        with pytest.raises(COMError):
            saw_obj._call_simauto("RunScriptCommand", "BadCommand;")

    def test_output_minus_one_raises(self, saw_obj):
        """Return value of -1 raises PowerWorldError."""
        from esapp.saw._exceptions import PowerWorldError
        saw_obj._pwcom.GetSpecificFieldMaxNum.return_value = -1
        with pytest.raises(PowerWorldError, match="returned -1"):
            saw_obj._call_simauto("GetSpecificFieldMaxNum", "Bus", "CustomFloat")

    def test_output_integer_returned(self, saw_obj):
        """Non-negative integer return is passed through."""
        saw_obj._pwcom.GetSpecificFieldMaxNum.return_value = 5
        result = saw_obj._call_simauto("GetSpecificFieldMaxNum", "Bus", "CustomFloat")
        assert result == 5

    def test_no_data_returns_empty(self, saw_obj):
        """'No data' message returns empty tuple (no data fields)."""
        saw_obj._pwcom.GetParametersMultipleElement.return_value = ("No data returned",)
        result = saw_obj._call_simauto("GetParametersMultipleElement", "Bus", [], "")
        assert result == ()


class TestOpenCaseEdgeCases:
    """Tests for OpenCase edge cases — covers lines 909-910."""

    def test_open_case_none_filename_no_previous(self, saw_obj):
        """OpenCase with None FileName and no previous path raises TypeError."""
        saw_obj.pwb_file_path = None
        with pytest.raises(TypeError, match="FileName is required"):
            saw_obj.OpenCase(FileName=None)


class TestOpenCaseType:
    """Tests for OpenCaseType — covers lines 933-940."""

    def test_open_case_type_with_list_options(self, saw_obj):
        """OpenCaseType with list options."""
        saw_obj._pwcom.OpenCaseType.return_value = ("",)
        saw_obj.OpenCaseType("case.raw", "PTI", Options=["NEAR", "YES"])
        saw_obj._pwcom.OpenCaseType.assert_called()

    def test_open_case_type_with_string_options(self, saw_obj):
        """OpenCaseType with string options."""
        saw_obj._pwcom.OpenCaseType.return_value = ("",)
        saw_obj.OpenCaseType("case.raw", "PTI", Options="NEAR")
        saw_obj._pwcom.OpenCaseType.assert_called()

    def test_open_case_type_no_options(self, saw_obj):
        """OpenCaseType with no options."""
        saw_obj._pwcom.OpenCaseType.return_value = ("",)
        saw_obj.OpenCaseType("case.raw", "PTI")
        saw_obj._pwcom.OpenCaseType.assert_called()


class TestSaveCaseEdgeCases:
    """Tests for SaveCase edge cases — covers lines 1039-1042."""

    def test_save_case_no_filename_no_path_raises(self, saw_obj):
        """SaveCase without FileName and no pwb_file_path raises TypeError."""
        saw_obj.pwb_file_path = None
        with pytest.raises(TypeError, match="SaveCase was called without a FileName"):
            saw_obj.SaveCase()

    def test_save_case_no_filename_uses_existing_path(self, saw_obj):
        """SaveCase without FileName uses existing pwb_file_path."""
        saw_obj.pwb_file_path = "C:\\cases\\test.pwb"
        saw_obj._pwcom.SaveCase.return_value = ("",)
        saw_obj.SaveCase()
        saw_obj._pwcom.SaveCase.assert_called()


class TestGetCaseHeader:
    """Tests for GetCaseHeader — covers line 493-495."""

    def test_get_case_header_default_filename(self, saw_obj):
        """GetCaseHeader with no filename uses pwb_file_path."""
        saw_obj.pwb_file_path = "test.pwb"
        saw_obj._pwcom.GetCaseHeader.return_value = ("", ("Header line 1",))
        result = saw_obj.GetCaseHeader()
        saw_obj._pwcom.GetCaseHeader.assert_called_with("test.pwb")


class TestGetFieldListFallback:
    """Tests for GetFieldList column fallback — covers lines 543, 547."""

    def test_get_field_list_old_columns(self, saw_obj):
        """GetFieldList falls back to old column count."""
        saw_obj._object_fields = {}
        # 4-column data (old format)
        field_data = [
            ["*1*", "BusNum", "Integer", "Bus Number"],
            ["*2*", "BusName", "String", "Bus Name"],
        ]
        saw_obj._pwcom.GetFieldList.return_value = ("", field_data)
        df = saw_obj.GetFieldList("Bus")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_field_list_new_columns(self, saw_obj):
        """GetFieldList falls back to new (6-column) format."""
        saw_obj._object_fields = {}
        # 6-column data (new format)
        field_data = [
            ["*1*", "BusNum", "Integer", "Bus Number", "Bus Number", "YES"],
            ["*2*", "BusName", "String", "Bus Name", "Bus Name", "YES"],
        ]
        saw_obj._pwcom.GetFieldList.return_value = ("", field_data)
        df = saw_obj.GetFieldList("Bus")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


class TestGetParametersMultipleElementFlatOutput:
    """Tests for GetParametersMultipleElementFlatOutput — covers lines 700-710."""

    def test_flat_output_returns_data(self, saw_obj):
        """Returns data tuple when result is non-empty."""
        saw_obj._pwcom.GetParametersMultipleElementFlatOutput.return_value = ("", ("1", "Bus1", "2", "Bus2"))
        result = saw_obj.GetParametersMultipleElementFlatOutput("Bus", ["BusNum", "BusName"])
        assert result is not None

    def test_flat_output_empty_returns_none(self, saw_obj):
        """Returns None when result is empty."""
        saw_obj._pwcom.GetParametersMultipleElementFlatOutput.return_value = ("", ())
        result = saw_obj.GetParametersMultipleElementFlatOutput("Bus", ["BusNum"])
        assert result is None


class TestListOfDevices:
    """Tests for ListOfDevices — covers lines 814-832."""

    def test_list_of_devices_none_result(self, saw_obj):
        """Returns None when all output elements are None."""
        saw_obj._pwcom.ListOfDevices.return_value = ("", (None, None))
        result = saw_obj.ListOfDevices("Bus")
        assert result is None

    def test_list_of_devices_with_data(self, saw_obj):
        """Returns DataFrame when data is present."""
        saw_obj._pwcom.ListOfDevices.return_value = ("", ([1, 2], ["Bus1", "Bus2"]))
        result = saw_obj.ListOfDevices("Bus")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestListOfDevicesVariants:
    """Tests for ListOfDevicesAsVariantStrings and FlatOutput — covers lines 857, 882."""

    def test_list_of_devices_as_variant_strings(self, saw_obj):
        """ListOfDevicesAsVariantStrings returns tuple."""
        saw_obj._pwcom.ListOfDevicesAsVariantStrings.return_value = ("", ("1", "2"))
        result = saw_obj.ListOfDevicesAsVariantStrings("Bus")
        assert result is not None

    def test_list_of_devices_flat_output(self, saw_obj):
        """ListOfDevicesFlatOutput returns tuple."""
        saw_obj._pwcom.ListOfDevicesFlatOutput.return_value = ("", ("1", "2"))
        result = saw_obj.ListOfDevicesFlatOutput("Bus")
        assert result is not None


class TestGetSpecificFieldMaxNum:
    """Tests for GetSpecificFieldMaxNum — covers line 780."""

    def test_get_specific_field_max_num(self, saw_obj):
        """GetSpecificFieldMaxNum returns integer."""
        saw_obj._pwcom.GetSpecificFieldMaxNum.return_value = ("", 10)
        result = saw_obj.GetSpecificFieldMaxNum("Bus", "CustomFloat")
        assert result == 10


class TestProperties:
    """Tests for base.py properties — covers lines 1082-1105."""

    def test_create_if_not_found_property(self, saw_obj):
        """CreateIfNotFound property reads from COM."""
        saw_obj._pwcom.CreateIfNotFound = True
        assert saw_obj.CreateIfNotFound is True

    def test_current_dir_property(self, saw_obj):
        """CurrentDir property reads from COM."""
        saw_obj._pwcom.CurrentDir = "C:\\Test"
        assert saw_obj.CurrentDir == "C:\\Test"

    def test_process_id_property(self, saw_obj):
        """ProcessID property reads from COM."""
        saw_obj._pwcom.ProcessID = 1234
        assert saw_obj.ProcessID == 1234

    def test_request_build_date_property(self, saw_obj):
        """RequestBuildDate property reads from COM."""
        saw_obj._pwcom.RequestBuildDate = 20230101
        assert saw_obj.RequestBuildDate == 20230101

    def test_uivisible_property(self, saw_obj):
        """UIVisible property reads from COM."""
        saw_obj._pwcom.UIVisible = True
        assert saw_obj.UIVisible is True

    def test_uivisible_property_attribute_error(self, saw_obj):
        """UIVisible returns False when AttributeError occurs."""
        mock_pwcom = MagicMock()
        del mock_pwcom.UIVisible  # Make UIVisible raise AttributeError on access
        saw_obj._pwcom = mock_pwcom
        assert saw_obj.UIVisible is False

    def test_program_information_property(self, saw_obj):
        """ProgramInformation property processes datetime."""
        import datetime
        dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
        saw_obj._pwcom.ProgramInformation = [
            ["v23", "Build", dt],
            ["info1", "info2"],
        ]
        result = saw_obj.ProgramInformation
        assert isinstance(result, tuple)
        assert isinstance(result[0], tuple)



class TestExecAux:
    """Tests for exec_aux — covers line 1290."""

    def test_exec_aux_double_quotes(self, saw_obj):
        """exec_aux replaces single quotes with double quotes."""
        saw_obj._pwcom.ProcessAuxFile.return_value = ("",)
        with patch("os.unlink"):
            saw_obj.exec_aux("DATA ('Bus')", use_double_quotes=True)
            saw_obj._pwcom.ProcessAuxFile.assert_called()


class TestUpdateUI:
    """Tests for update_ui — covers line 1302."""

    def test_update_ui(self, saw_obj):
        """update_ui calls ProcessAuxFile with empty aux."""
        saw_obj._pwcom.ProcessAuxFile.return_value = ("",)
        saw_obj.update_ui()
        saw_obj._pwcom.ProcessAuxFile.assert_called()


class TestChangeParametersMultipleElementFlatInputValid:
    """Tests for valid path of ChangeParametersMultipleElementFlatInput — covers line 452."""

    def test_flat_input_valid(self, saw_obj):
        """ChangeParametersMultipleElementFlatInput succeeds with flat list."""
        saw_obj._pwcom.ChangeParametersMultipleElementFlatInput.return_value = ("",)
        saw_obj.ChangeParametersMultipleElementFlatInput(
            "Bus", ["BusNum", "BusName"], 2, [1, "A", 2, "B"]
        )
        saw_obj._pwcom.ChangeParametersMultipleElementFlatInput.assert_called()


class TestSendToExcel:
    """Tests for SendToExcel — covers line 1077."""

    def test_send_to_excel(self, saw_obj):
        """SendToExcel calls the COM method."""
        saw_obj._pwcom.SendToExcel.return_value = ("",)
        saw_obj.SendToExcel("Bus", "", ["BusNum", "BusName"])
        saw_obj._pwcom.SendToExcel.assert_called()


class TestToNumericLocale:
    """Tests for _to_numeric with locale Series — covers line 1250."""

    def test_to_numeric_series_with_locale(self, saw_obj):
        """_to_numeric handles locale delimiter for Series."""
        saw_obj.decimal_delimiter = ","
        s = pd.Series(["1,5", "2,5"])
        result = saw_obj._to_numeric(s)
        assert result.iloc[0] == 1.5
        saw_obj.decimal_delimiter = "."

    def test_replace_decimal_delimiter_non_string(self, saw_obj):
        """_replace_decimal_delimiter handles non-string data."""
        saw_obj.decimal_delimiter = ","
        s = pd.Series([1.5, 2.5])
        result = saw_obj._replace_decimal_delimiter(s)
        assert result.iloc[0] == 1.5
        saw_obj.decimal_delimiter = "."


# =============================================================================
# general.py — full coverage
# =============================================================================

@pytest.mark.parametrize("method, args, expected_script", [
    # WriteTextToFile (lines 90-91)
    ("WriteTextToFile", ("output.txt", "Hello World"), 'WriteTextToFile("output.txt", "Hello World");'),
    # WriteTextToFile with embedded quotes
    ("WriteTextToFile", ("output.txt", 'say "hi"'), 'WriteTextToFile("output.txt", "say ""hi""");'),
    # LoadAux (lines 295-296)
    ("LoadAux", ("data.aux",), 'LoadAux("data.aux", NO);'),
    ("LoadAux", ("data.aux", True), 'LoadAux("data.aux", YES);'),
    # ImportData (lines 321-322)
    ("ImportData", ("data.csv", "CSV"), 'ImportData("data.csv", CSV, 1, NO);'),
    ("ImportData", ("data.csv", "CSV", 2, True), 'ImportData("data.csv", CSV, 2, YES);'),
    # SaveData with defaults (lines 414-432)
    ("SaveData", ("out.csv", "CSV", "Bus", ["BusNum", "BusName"]),
     'SaveData("out.csv", CSV, Bus, [BusNum, BusName], [], , [], NO, YES);'),
    # SaveData with subdatalist
    ("SaveData", ("out.csv", "CSV", "Gen", ["BusNum"], ["BidCurve"]),
     'SaveData("out.csv", CSV, Gen, [BusNum], [BidCurve], , [], NO, YES);'),
    # SaveData with filter
    ("SaveData", ("out.csv", "CSV", "Bus", ["BusNum"], None, "MyFilter"),
     'SaveData("out.csv", CSV, Bus, [BusNum], [], "MyFilter", [], NO, YES);'),
    # SaveData with SELECTED filter (no quotes)
    ("SaveData", ("out.csv", "CSV", "Bus", ["BusNum"], None, "SELECTED"),
     'SaveData("out.csv", CSV, Bus, [BusNum], [], SELECTED, [], NO, YES);'),
    # SaveData with sortfieldlist
    ("SaveData", ("out.csv", "CSV", "Bus", ["BusNum"], None, "", ["BusNum"]),
     'SaveData("out.csv", CSV, Bus, [BusNum], [], , [BusNum], NO, YES);'),
    # SaveData with transpose and append
    ("SaveData", ("out.csv", "CSV", "Bus", ["BusNum"], None, "", None, True, False),
     'SaveData("out.csv", CSV, Bus, [BusNum], [], , [], YES, NO);'),
    # SaveDataWithExtra (lines 474-489)
    ("SaveDataWithExtra", ("out.csv", "CSV", "Bus", ["BusNum"]),
     'SaveDataWithExtra("out.csv", CSV, Bus, [BusNum], [], , [], [], [], NO, YES);'),
    # SaveDataWithExtra with all optional params
    ("SaveDataWithExtra", ("out.csv", "CSV", "Bus", ["BusNum"], ["SubField"], "MyFilter", ["BusNum"], ["Header1"], ["Value1"], True, False),
     'SaveDataWithExtra("out.csv", CSV, Bus, [BusNum], [SubField], "MyFilter", [BusNum], ["Header1"], ["Value1"], YES, NO);'),
    # SaveObjectFields (lines 643-644)
    ("SaveObjectFields", ("fields.txt", "Bus", ["BusNum", "BusName"]),
     'SaveObjectFields("fields.txt", Bus, [BusNum, BusName]);'),
    # SetData with custom filter
    ("SetData", ("Bus", ["BusName"], ["NewName"], "MyFilter"),
     'SetData(Bus, [BusName], [NewName], "MyFilter");'),
    # SetData with SELECTED filter
    ("SetData", ("Bus", ["BusName"], ["NewName"], "SELECTED"),
     'SetData(Bus, [BusName], [NewName], SELECTED);'),
    # SendToExcelAdvanced (lines 762-774)
    ("SendToExcelAdvanced", ("Bus", ["BusNum", "BusName"]),
     'SendtoExcel(Bus, [BusNum, BusName], , YES, "", "", [], [], [], YES, 0, 0);'),
    # SendToExcelAdvanced with all params
    ("SendToExcelAdvanced", ("Bus", ["BusNum"], "MyFilter", False, "Book1", "Sheet1", ["BusNum"], ["H1"], ["V1"], False, 1, 2),
     'SendtoExcel(Bus, [BusNum], "MyFilter", NO, "Book1", "Sheet1", [BusNum], ["H1"], ["V1"], NO, 1, 2);'),
    # LogAddDateTime (lines 813-816)
    ("LogAddDateTime", ("Timer",), 'LogAddDateTime("Timer", YES, YES, NO);'),
    ("LogAddDateTime", ("Timer", False, False, True), 'LogAddDateTime("Timer", NO, NO, YES);'),
    # LoadAuxDirectory (lines 854-858) — with filter
    ("LoadAuxDirectory", ("C:\\AuxFiles", "*.aux", True), 'LoadAuxDirectory("C:\\AuxFiles", "*.aux", YES);'),
    # LoadAuxDirectory — without filter
    ("LoadAuxDirectory", ("C:\\AuxFiles",), 'LoadAuxDirectory("C:\\AuxFiles", , NO);'),
    # LoadData (lines 884-885)
    ("LoadData", ("data.aux", "BusData"), 'LoadData("data.aux", BusData, NO);'),
    ("LoadData", ("data.aux", "BusData", True), 'LoadData("data.aux", BusData, YES);'),
])
def test_general_extended(saw_obj, method, args, expected_script):
    """Verify general mixin methods produce correct script commands."""
    getattr(saw_obj, method)(*args)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(expected_script)


class TestGetSubData:
    """Tests for GetSubData — covers lines 577-620."""

    @staticmethod
    def _write_tmp(content):
        """Write content to a real temp file (bypassing the mocked tempfile)."""
        import tempfile as real_tempfile
        path = os.path.join(real_tempfile.gettempdir(), f"test_subdata_{id(content)}.aux")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_get_sub_data_basic(self, saw_obj):
        """GetSubData parses AUX output with subdata sections."""
        aux_content = (
            'DATA (Gen, [BusNum, GenID])\n'
            '{\n'
            '1 "1"\n'
            '<SUBDATA BidCurve>\n'
            '10.0 50.0\n'
            '20.0 100.0\n'
            '</SUBDATA>\n'
            '2 "1"\n'
            '<SUBDATA BidCurve>\n'
            '15.0 75.0\n'
            '</SUBDATA>\n'
            '}\n'
        )
        tmp_name = self._write_tmp(aux_content)
        try:
            original_save = saw_obj.SaveData
            def mock_save(filename, *a, **kw):
                import shutil
                shutil.copy(tmp_name, filename)
            saw_obj.SaveData = mock_save

            result = saw_obj.GetSubData("Gen", ["BusNum", "GenID"], ["BidCurve"])
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "BidCurve" in result.columns
        finally:
            saw_obj.SaveData = original_save
            os.unlink(tmp_name)

    def test_get_sub_data_file_not_found(self, saw_obj):
        """GetSubData returns empty DataFrame when file doesn't exist."""
        original_save = saw_obj.SaveData
        saw_obj.SaveData = lambda filename, *a, **kw: None
        try:
            result = saw_obj.GetSubData("Bus", ["BusNum"], [])
            assert isinstance(result, pd.DataFrame)
            assert result.empty
        finally:
            saw_obj.SaveData = original_save

    def test_get_sub_data_no_data_match(self, saw_obj):
        """GetSubData returns empty DataFrame when AUX has no DATA block."""
        tmp_name = self._write_tmp("// just a comment\n")
        try:
            original_save = saw_obj.SaveData
            def mock_save(filename, *a, **kw):
                import shutil
                shutil.copy(tmp_name, filename)
            saw_obj.SaveData = mock_save

            result = saw_obj.GetSubData("Bus", ["BusNum"], [])
            assert isinstance(result, pd.DataFrame)
            assert result.empty
        finally:
            saw_obj.SaveData = original_save
            os.unlink(tmp_name)

    def test_get_sub_data_with_comments_and_blanks(self, saw_obj):
        """GetSubData ignores comments and blank lines."""
        aux_content = (
            'DATA (Bus, [BusNum, BusName])\n'
            '{\n'
            '// This is a comment\n'
            '\n'
            '1 "Bus1"\n'
            '}\n'
        )
        tmp_name = self._write_tmp(aux_content)
        try:
            original_save = saw_obj.SaveData
            def mock_save(filename, *a, **kw):
                import shutil
                shutil.copy(tmp_name, filename)
            saw_obj.SaveData = mock_save

            result = saw_obj.GetSubData("Bus", ["BusNum", "BusName"])
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
        finally:
            saw_obj.SaveData = original_save
            os.unlink(tmp_name)

    def test_get_sub_data_bracket_format(self, saw_obj):
        """GetSubData parses bracket-delimited subdata lines."""
        aux_content = (
            'DATA (Gen, [BusNum, GenID])\n'
            '{\n'
            '1 "1"\n'
            '<SUBDATA BidCurve>\n'
            '[10.0, 50.0]\n'
            '[20.0, 100.0]\n'
            '</SUBDATA>\n'
            '}\n'
        )
        tmp_name = self._write_tmp(aux_content)
        try:
            original_save = saw_obj.SaveData
            def mock_save(filename, *a, **kw):
                import shutil
                shutil.copy(tmp_name, filename)
            saw_obj.SaveData = mock_save

            result = saw_obj.GetSubData("Gen", ["BusNum", "GenID"], ["BidCurve"])
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert len(result.iloc[0]["BidCurve"]) == 2
        finally:
            saw_obj.SaveData = original_save
            os.unlink(tmp_name)


# =============================================================================
# transient.py — full coverage
# =============================================================================

@pytest.mark.parametrize("method, args, expected_script", [
    # TSInitialize (lines 121-124 — success path)
    ("TSInitialize", (), "TSInitialize()"),
    # TSStoreResponse (line 176)
    ("TSStoreResponse", ("Gen", False), "TSResultStorageSetAll(Gen, NO)"),
    ("TSStoreResponse", (), "TSResultStorageSetAll(ALL, YES)"),
    # TSClearResultsFromRAM — ALL (lines 193-201)
    ("TSClearResultsFromRAM", (), "TSClearResultsFromRAM(ALL,YES,YES,YES,YES,YES);"),
    # TSClearResultsFromRAM — named contingency (quoted)
    ("TSClearResultsFromRAM", ("MyCtg",), 'TSClearResultsFromRAM("MyCtg",YES,YES,YES,YES,YES);'),
    # TSClearResultsFromRAM — SELECTED
    ("TSClearResultsFromRAM", ("SELECTED",), "TSClearResultsFromRAM(SELECTED,YES,YES,YES,YES,YES);"),
    # TSClearResultsFromRAM — already quoted
    ("TSClearResultsFromRAM", ('"MyCtg"',), 'TSClearResultsFromRAM("MyCtg",YES,YES,YES,YES,YES);'),
    # TSClearResultsFromRAM — mixed flags
    ("TSClearResultsFromRAM", ("ALL", False, True, False, True, False),
     "TSClearResultsFromRAM(ALL,NO,YES,NO,YES,NO);"),
    # TSSetPlayInSignals is tested separately (needs numpy)
    # TSClearResultsFromRAMAndDisableStorage (lines 249-250)
    ("TSClearResultsFromRAMAndDisableStorage", (),
     "TSClearResultsFromRAM(ALL,YES,YES,YES,YES,YES);"),
    # TSWriteOptions (lines 277-287)
    ("TSWriteOptions", ("opts.aux",),
     'TSWriteOptions("opts.aux", [YES, YES, YES, YES, YES, YES, YES], PRIMARY);'),
    ("TSWriteOptions", ("opts.aux", False, False, False, False, False, False, False, "SECONDARY"),
     'TSWriteOptions("opts.aux", [NO, NO, NO, NO, NO, NO, NO], SECONDARY);'),
    # TSAutoInsertZPOTT (line 312)
    ("TSAutoInsertZPOTT", (80.0, "MyFilter"), 'TSAutoInsertZPOTT(80.0, "MyFilter");'),
    # TSDisableMachineModelNonZeroDerivative (line 348)
    ("TSDisableMachineModelNonZeroDerivative", (), "TSDisableMachineModelNonZeroDerivative(0.001);"),
    ("TSDisableMachineModelNonZeroDerivative", (0.01,), "TSDisableMachineModelNonZeroDerivative(0.01);"),
    # TSGetVCurveData (line 352)
    ("TSGetVCurveData", ("vcurve.csv", "AllGens"), 'TSGetVCurveData("vcurve.csv", "AllGens");'),
    # TSWriteResultsToCSV (lines 364-369)
    ("TSWriteResultsToCSV", ("results.csv", "PLOT", ["Ctg1"], ["BusPUVolt"]),
     'TSGetResults("results.csv", PLOT, ["Ctg1"], ["BusPUVolt"]);'),
    # TSWriteResultsToCSV with times
    ("TSWriteResultsToCSV", ("results.csv", "PLOT", ["Ctg1"], ["BusPUVolt"], 0.0, 10.0),
     'TSGetResults("results.csv", PLOT, ["Ctg1"], ["BusPUVolt"], 0.0, 10.0);'),
    # TSLoadRDB (line 381)
    ("TSLoadRDB", ("relay.rdb", "SEL421"), 'TSLoadRDB("relay.rdb", SEL421, "");'),
    ("TSLoadRDB", ("relay.rdb", "SEL421", "MyFilter"), 'TSLoadRDB("relay.rdb", SEL421, "MyFilter");'),
    # TSLoadRelayCSV (line 385)
    ("TSLoadRelayCSV", ("relay.csv", "SEL421"), 'TSLoadRelayCSV("relay.csv", SEL421, "");'),
    # TSPlotSeriesAdd (line 398)
    ("TSPlotSeriesAdd", ("Plot1", 1, 1, "Gen", "GenMW"),
     'TSPlotSeriesAdd("Plot1", 1, 1, Gen, GenMW, "", "");'),
    ("TSPlotSeriesAdd", ("Plot1", 1, 1, "Gen", "GenMW", "MyFilter", "color=red"),
     'TSPlotSeriesAdd("Plot1", 1, 1, Gen, GenMW, "MyFilter", "color=red");'),
    # TSRunResultAnalyzer (line 404)
    ("TSRunResultAnalyzer", (), 'TSRunResultAnalyzer("");'),
    ("TSRunResultAnalyzer", ("Ctg1",), 'TSRunResultAnalyzer("Ctg1");'),
    # TSRunUntilSpecifiedTime (lines 416-433) - uses defaults for step_size=0.25, steps_in_cycles=True
    ("TSRunUntilSpecifiedTime", ("Ctg1",), 'TSRunUntilSpecifiedTime("Ctg1", [, 0.25, YES, NO]);'),
    ("TSRunUntilSpecifiedTime", ("Ctg1", 10.0, 0.01, True, True, 5),
     'TSRunUntilSpecifiedTime("Ctg1", [10.0, 0.01, YES, YES, 5]);'),
    # TSSaveBPA (lines 437-438)
    ("TSSaveBPA", ("out.bpa",), 'TSSaveBPA("out.bpa", NO);'),
    ("TSSaveBPA", ("out.bpa", True), 'TSSaveBPA("out.bpa", YES);'),
    # TSSaveGE (lines 442-443)
    ("TSSaveGE", ("out.dyd",), 'TSSaveGE("out.dyd", NO);'),
    ("TSSaveGE", ("out.dyd", True), 'TSSaveGE("out.dyd", YES);'),
    # TSSavePTI (lines 447-448)
    ("TSSavePTI", ("out.dyr",), 'TSSavePTI("out.dyr", NO);'),
    ("TSSavePTI", ("out.dyr", True), 'TSSavePTI("out.dyr", YES);'),
    # TSSaveTwoBusEquivalent (line 452)
    ("TSSaveTwoBusEquivalent", ("twobus.pwb", "[BUS 1]"), 'TSSaveTwoBusEquivalent("twobus.pwb", [BUS 1]);'),
    # TSWriteModels (lines 456-457)
    ("TSWriteModels", ("models.aux",), 'TSWriteModels("models.aux", NO);'),
    ("TSWriteModels", ("models.aux", True), 'TSWriteModels("models.aux", YES);'),
    # TSSetSelectedForTransientReferences (lines 463-465)
    ("TSSetSelectedForTransientReferences", ("CUSTOMINTEGER", "SET", ["Gen", "Bus"], ["GENROU", "EXST1"]),
     "TSSetSelectedForTransientReferences(CUSTOMINTEGER, SET, [Gen, Bus], [GENROU, EXST1]);"),
    # TSSaveDynamicModels (lines 471-472)
    ("TSSaveDynamicModels", ("dyn.dyr", "PTI", "Gen"),
     'TSSaveDynamicModels("dyn.dyr", PTI, Gen, "", NO);'),
    ("TSSaveDynamicModels", ("dyn.dyr", "PTI", "Gen", "MyFilter", True),
     'TSSaveDynamicModels("dyn.dyr", PTI, Gen, "MyFilter", YES);'),
    # TSSolve with time params (new feature)
    ("TSSolve", ("GEN_TRIP", 0.0, 10.0, 0.01, False),
     'TSSolve("GEN_TRIP", [0.0, 10.0, 0.01, NO])'),
    ("TSSolve", ("GEN_TRIP", 0.0, 10.0, 0.01, True),
     'TSSolve("GEN_TRIP", [0.0, 10.0, 0.01, YES])'),
    # TSSolve with partial time params (uses defaults for step_size=0.25, step_in_cycles=True)
    ("TSSolve", ("GEN_TRIP", None, 10.0),
     'TSSolve("GEN_TRIP", [, 10.0, 0.25, YES])'),
])
def test_transient_extended(saw_obj, method, args, expected_script):
    """Verify transient mixin methods produce correct script commands."""
    getattr(saw_obj, method)(*args)
    saw_obj._pwcom.RunScriptCommand.assert_called_with(expected_script)


class TestTSSetPlayInSignals:
    """Tests for TSSetPlayInSignals — covers lines 220-241."""

    def test_set_play_in_signals(self, saw_obj):
        """TSSetPlayInSignals constructs correct AUX data."""
        times = np.array([0.0, 1.0, 2.0])
        signals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        saw_obj._pwcom.ProcessAuxFile.return_value = ("",)
        with patch("os.unlink"):
            saw_obj.TSSetPlayInSignals("Signal1", times, signals)
            saw_obj._pwcom.ProcessAuxFile.assert_called()

    def test_set_play_in_signals_dimension_mismatch(self, saw_obj):
        """TSSetPlayInSignals raises ValueError on dimension mismatch."""
        times = np.array([0.0, 1.0])
        signals = np.array([[1.0], [2.0], [3.0]])  # 3 rows vs 2 times
        with pytest.raises(ValueError, match="Dimension mismatch"):
            saw_obj.TSSetPlayInSignals("Signal1", times, signals)

    def test_set_play_in_signals_wrong_ndim(self, saw_obj):
        """TSSetPlayInSignals raises ValueError for wrong dimensions."""
        times = np.array([[0.0, 1.0]])  # 2D instead of 1D
        signals = np.array([[1.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            saw_obj.TSSetPlayInSignals("Signal1", times, signals)

    def test_set_play_in_signals_single_column(self, saw_obj):
        """TSSetPlayInSignals with single signal column."""
        times = np.array([0.0, 1.0])
        signals = np.array([[1.0], [2.0]])
        saw_obj._pwcom.ProcessAuxFile.return_value = ("",)
        with patch("os.unlink"):
            saw_obj.TSSetPlayInSignals("Signal1", times, signals)
            saw_obj._pwcom.ProcessAuxFile.assert_called()


class TestTSInitializeFailure:
    """Tests for TSInitialize failure path — covers lines 121-124."""

    def test_ts_initialize_failure_logs_warning(self, saw_obj):
        """TSInitialize catches exception and logs warning."""
        from esapp.saw._exceptions import PowerWorldError
        saw_obj._pwcom.RunScriptCommand.return_value = ("Error: TS not initialized",)
        # Should not raise — catches internally
        saw_obj.TSInitialize()


class TestTSGetContingencyResultsNotFound:
    """Tests for TSGetContingencyResults not-found path — covers line 63."""

    def test_returns_none_tuple_when_ctg_not_found(self, saw_obj):
        """TSGetContingencyResults returns (None, None) for missing contingency."""
        saw_obj._pwcom.TSGetContingencyResults.return_value = ("", None, (None,))
        meta, data = saw_obj.TSGetContingencyResults("NonExistent", ["BusPUVolt"])
        assert meta is None
        assert data is None
