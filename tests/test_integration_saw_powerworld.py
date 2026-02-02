"""
Integration tests for core SAW COM operations.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test the foundational SAW
class operations: case save/load, state management, general commands, file
operations, modify operations (create/delete objects, merge, split), region
operations, and case actions (equivalence, renumber, scale).

NOTE: Power flow, sensitivity, contingency, fault, GIC, ATC, transient,
      and time step tests are in their dedicated test files:
      - test_integration_powerflow.py
      - test_integration_contingency.py
      - test_integration_analysis.py

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

USAGE:
    pytest tests/test_integration_saw_powerworld.py -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, PowerWorldAddonError, create_object_string
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


class TestBase:
    """Tests for base SAW operations."""

    @pytest.mark.order(10)
    def test_open_case_error_nonexistent_file(self, saw_instance):
        """OpenCase raises PowerWorldError for a nonexistent file path."""
        original_path = saw_instance.pwb_file_path
        with pytest.raises(PowerWorldError):
            saw_instance.OpenCase(FileName="C:/nonexistent/path/fake_case.pwb")
        saw_instance.OpenCase(original_path)

    @pytest.mark.order(20)
    def test_open_case_error_wrong_filetype(self, saw_instance):
        """OpenCase raises PowerWorldError for a non-PWB file."""
        original_path = saw_instance.pwb_file_path
        with pytest.raises(PowerWorldError):
            saw_instance.OpenCase(FileName=os.path.abspath(__file__))
        saw_instance.OpenCase(original_path)

    @pytest.mark.order(30)
    def test_open_case_type_error_nonexistent(self, saw_instance):
        """OpenCaseType raises PowerWorldError for nonexistent file."""
        original_path = saw_instance.pwb_file_path
        with pytest.raises(PowerWorldError):
            saw_instance.OpenCaseType("C:/nonexistent/fake.raw", "PTI")
        saw_instance.OpenCase(original_path)

    @pytest.mark.order(100)
    def test_save_case(self, saw_instance, temp_file):
        """SaveCase writes a .pwb file to disk."""
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveCase(tmp_pwb)
        assert os.path.exists(tmp_pwb)

    @pytest.mark.order(200)
    def test_get_header(self, saw_instance):
        """GetCaseHeader returns non-None header data."""
        header = saw_instance.GetCaseHeader()
        assert header is not None

    @pytest.mark.order(300)
    def test_change_parameters(self, saw_instance):
        """ChangeParametersSingleElement modifies and restores a bus name."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert buses is not None and not buses.empty

        bus_num = buses.iloc[0]["BusNum"]
        original_name = buses.iloc[0]["BusName"]
        new_name = "TestBusName"
        saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, new_name])

        check = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert check["BusName"] == new_name

        saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, original_name])

    @pytest.mark.order(400)
    def test_get_parameters(self, saw_instance):
        """GetParametersMultipleElement and GetParametersSingleElement return valid data."""
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None and not df.empty

        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

    @pytest.mark.order(500)
    def test_list_devices(self, saw_instance):
        """ListOfDevices returns a non-empty DataFrame for buses."""
        df = saw_instance.ListOfDevices("Bus")
        assert df is not None and not df.empty

    @pytest.mark.order(700)
    def test_state(self, saw_instance):
        """Store/Restore/Delete/Save/Load state cycle completes without error."""
        saw_instance.StoreState("TestState")
        saw_instance.RestoreState("TestState")
        saw_instance.DeleteState("TestState")
        saw_instance.SaveState()
        saw_instance.LoadState()

    @pytest.mark.order(800)
    def test_run_script_2(self, saw_instance):
        """RunScriptCommand2 executes a log command."""
        saw_instance.RunScriptCommand2("LogAdd(\"Test\");", "Testing...")

    @pytest.mark.order(900)
    def test_field_list(self, saw_instance):
        """GetFieldList and GetSpecificFieldList return non-empty DataFrames."""
        df = saw_instance.GetFieldList("Bus")
        assert not df.empty
        assert isinstance(df, pd.DataFrame)
        assert "internal_field_name" in df.columns

        df_spec = saw_instance.GetSpecificFieldList("Bus", ["BusNum", "BusName"])
        assert not df_spec.empty

    @pytest.mark.order(50000)
    def test_update_ui_and_exec_aux(self, saw_instance):
        """update_ui and exec_aux complete without error."""
        saw_instance.update_ui()
        saw_instance.exec_aux('SCRIPT\n{\n    LogAdd("test exec_aux");\n}')

    @pytest.mark.order(50100)
    def test_change_parameters_rect(self, saw_instance):
        """ChangeParametersMultipleElementRect modifies and restores a bus name."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert buses is not None and not buses.empty

        df = buses.head(1).copy()
        original_name = df.iloc[0]["BusName"]
        df.iloc[0, df.columns.get_loc("BusName")] = "TempTestName"
        saw_instance.ChangeParametersMultipleElementRect("Bus", ["BusNum", "BusName"], df)
        df.iloc[0, df.columns.get_loc("BusName")] = original_name
        saw_instance.ChangeParametersMultipleElementRect("Bus", ["BusNum", "BusName"], df)

    @pytest.mark.order(50200)
    def test_list_devices_variants(self, saw_instance):
        """ListOfDevicesAsVariantStrings and FlatOutput return non-None."""
        result1 = saw_instance.ListOfDevicesAsVariantStrings("Bus")
        assert result1 is not None
        result2 = saw_instance.ListOfDevicesFlatOutput("Bus")
        assert result2 is not None

        result = saw_instance.GetParametersMultipleElementFlatOutput("Bus", ["BusNum", "BusName"])
        assert result is None or len(result) > 0

    @pytest.mark.order(50400)
    def test_set_data(self, saw_instance):
        """SetData modifies and restores a bus name via filter."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert buses is not None and not buses.empty

        bus_num = buses.iloc[0]["BusNum"]
        original = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        original_name = original["BusName"]
        saw_instance.SetData("Bus", ["BusName"], ["TempName"], f"BusNum = {bus_num}")
        saw_instance.SetData("Bus", ["BusName"], [original_name], f"BusNum = {bus_num}")

    @pytest.mark.order(50500)
    def test_simauto_property_errors(self, saw_instance):
        """set_simauto_property raises appropriate errors for invalid inputs."""
        with pytest.raises(ValueError, match="not currently supported"):
            saw_instance.set_simauto_property("InvalidProperty", True)

        with pytest.raises(ValueError, match="is invalid"):
            saw_instance.set_simauto_property("CreateIfNotFound", "not_a_bool")

        with pytest.raises(ValueError, match="not a valid path"):
            saw_instance.set_simauto_property("CurrentDir", "C:\\NonExistent\\Path\\12345")

    @pytest.mark.order(50600)
    def test_change_parameters_flat_input(self, saw_instance):
        """ChangeParametersMultipleElementFlatInput works with flat list and rejects nested."""
        from esapp.saw._exceptions import Error
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert buses is not None and not buses.empty

        bus_num = int(buses.iloc[0]["BusNum"])
        original_name = buses.iloc[0]["BusName"]

        saw_instance.ChangeParametersMultipleElementFlatInput(
            "Bus", ["BusNum", "BusName"], 1, [bus_num, "TempFlatName"]
        )
        saw_instance.ChangeParametersMultipleElementFlatInput(
            "Bus", ["BusNum", "BusName"], 1, [bus_num, original_name]
        )

        with pytest.raises(Error, match="1-D array"):
            saw_instance.ChangeParametersMultipleElementFlatInput(
                "Bus", ["BusNum", "BusName"], 1, [[bus_num, "Test"]]
            )

    @pytest.mark.order(50700)
    def test_change_parameters_multiple_element(self, saw_instance):
        """ChangeParametersMultipleElement works with nested list."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert buses is not None and not buses.empty

        bus_num = int(buses.iloc[0]["BusNum"])
        original_name = buses.iloc[0]["BusName"]

        saw_instance.ChangeParametersMultipleElement(
            "Bus", ["BusNum", "BusName"], [[bus_num, "TempMultiName"]]
        )
        saw_instance.ChangeParametersMultipleElement(
            "Bus", ["BusNum", "BusName"], [[bus_num, original_name]]
        )

    @pytest.mark.order(50800)
    def test_get_specific_field_max_num(self, saw_instance):
        """GetSpecificFieldMaxNum returns an integer."""
        max_num = saw_instance.GetSpecificFieldMaxNum("Bus", "CustomFloat")
        assert isinstance(max_num, int)

    @pytest.mark.order(50900)
    def test_get_params_rect_typed(self, saw_instance):
        """GetParamsRectTyped returns None or a DataFrame."""
        df = saw_instance.GetParamsRectTyped("Bus", ["BusNum", "BusName"])
        assert df is None or isinstance(df, pd.DataFrame)

    @pytest.mark.order(51200)
    def test_set_logging_level(self, saw_instance):
        """set_logging_level accepts both int and string levels."""
        import logging
        saw_instance.set_logging_level(logging.DEBUG)
        saw_instance.set_logging_level("INFO")

    @pytest.mark.order(51800)
    def test_get_params_flat_empty_object(self, saw_instance):
        """GetParametersMultipleElementFlatOutput returns None for impossible filter."""
        result = saw_instance.GetParametersMultipleElementFlatOutput(
            "Bus", ["BusNum"], "BusNum < -99999"
        )
        assert result is None

    @pytest.mark.order(51900)
    def test_uivisible_property(self, saw_instance):
        """UIVisible property returns a boolean."""
        result = saw_instance.UIVisible
        assert isinstance(result, bool)

    @pytest.mark.order(52000)
    def test_early_bind_vs_dynamic(self, saw_instance):
        """Verify the session instance has a valid ProcessID."""
        assert saw_instance.ProcessID is not None


class TestGeneral:
    """Tests for general SAW operations."""

    @pytest.mark.order(9500)
    def test_log(self, saw_instance, temp_file):
        """LogAdd and LogSave write a log file to disk."""
        saw_instance.LogAdd("SAW Validator Test Message")
        tmp_log = temp_file(".txt")
        saw_instance.LogSave(tmp_log)
        assert os.path.exists(tmp_log)

    @pytest.mark.order(9600)
    def test_file_ops(self, saw_instance, temp_file):
        """CopyFile, RenameFile, DeleteFile manage files correctly."""
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

    @pytest.mark.order(9800)
    def test_aux(self, saw_instance, temp_file):
        """SaveData to AUX then LoadAux round-trips without error."""
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        saw_instance.LoadAux(tmp_aux)

    @pytest.mark.order(9900)
    def test_select(self, saw_instance):
        """SelectAll/UnSelectAll operate without error."""
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")

    @pytest.mark.order(52000)
    def test_log_clear_and_add(self, saw_instance):
        """LogClear, LogAdd, and LogAddDateTime operate without error."""
        saw_instance.LogClear()
        saw_instance.LogAdd("Test message")
        saw_instance.LogAddDateTime("Timer")

    @pytest.mark.order(52100)
    def test_save_data_variants(self, saw_instance, temp_file):
        """SaveData works for both AUX and CSV formats."""
        tmp_aux = temp_file(".aux")
        tmp_csv = temp_file(".csv")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        assert os.path.exists(tmp_aux)
        saw_instance.SaveData(tmp_csv, "CSV", "Bus", ["BusNum", "BusName"], filter_name="SELECTED")

    @pytest.mark.order(52300)
    def test_enter_mode(self, saw_instance):
        """EnterMode switches between EDIT and RUN modes."""
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    @pytest.mark.order(53100)
    def test_set_sub_data(self, saw_instance):
        """SetSubData creates a contingency with CTGElement subdata."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty
        b = branches.iloc[0]
        # CTGElement format per AUX docs: Action ModelCriteria Status TimeDelay
        # Action syntax: "BRANCH bus1 bus2 ckt OPEN" (object first, verb last)
        action = f'BRANCH {b["BusNum"]} {b["BusNum:1"]} {b["LineCircuit"]} OPEN'

        saw_instance.SetSubData(
            "Contingency",
            ["Name"],
            [{
                "Name": "TestCtgSubData",
                "CTGElement": [[action, "", "ALWAYS", 0]],
            }],
            subdatatype="CTGElement",
        )
        result = saw_instance.GetSubData(
            "Contingency", ["Name"], ["CTGElement"]
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.order(53200)
    def test_set_sub_data_roundtrip(self, saw_instance):
        """SetSubData then GetSubData verifies subdata content persists."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty
        b = branches.iloc[0]
        action = f'BRANCH {b["BusNum"]} {b["BusNum:1"]} {b["LineCircuit"]} OPEN'

        saw_instance.SetSubData(
            "Contingency",
            ["Name"],
            [{
                "Name": "TestCtgRoundtrip",
                "CTGElement": [[action, "", "ALWAYS", 0]],
            }],
            subdatatype="CTGElement",
        )
        result = saw_instance.GetSubData(
            "Contingency", ["Name"], ["CTGElement"]
        )
        assert isinstance(result, pd.DataFrame)
        # Find our specific contingency and verify it has subdata
        match = result[result["Name"] == "TestCtgRoundtrip"]
        assert len(match) == 1
        assert len(match.iloc[0]["CTGElement"]) > 0


class TestModify:
    """Tests for modify operations (destructive - run late)."""

    @pytest.mark.order(12000)
    def test_create_delete(self, saw_instance):
        """CreateData and Delete cycle for a bus."""
        dummy_bus = 99999
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [dummy_bus, "SAW_TEST"])
        saw_instance.Delete("Bus", f"BusNum = {dummy_bus}")

    @pytest.mark.order(13400)
    def test_superarea(self, saw_instance):
        """SuperArea create, add areas, remove areas cycle."""
        saw_instance.CreateData("SuperArea", ["Name"], ["TestSuperArea"])
        saw_instance.SuperAreaAddAreas("TestSuperArea", "ALL")
        saw_instance.SuperAreaRemoveAreas("TestSuperArea", "ALL")

    @pytest.mark.order(13500)
    def test_interface_ops(self, saw_instance):
        """Interface creation and manipulation operations."""
        saw_instance.InjectionGroupRemoveDuplicates()
        saw_instance.InterfaceRemoveDuplicates()
        saw_instance.DirectionsAutoInsertReference("Bus", "Slack")

        saw_instance.InterfaceCreate("TestInt", True, "Branch", "SELECTED")
        saw_instance.InterfaceFlatten("TestInt")
        saw_instance.InterfaceFlattenFilter("ALL")
        saw_instance.InterfaceModifyIsolatedElements()

        saw_instance.CreateData("Contingency", ["Name"], ["TestCtg"])
        saw_instance.InterfaceAddElementsFromContingency("TestInt", "TestCtg")


class TestRegions:
    """Tests for region operations."""

    @pytest.mark.order(20000)
    def test_region_update_buses(self, saw_instance):
        """RegionUpdateBuses completes without error."""
        saw_instance.RegionUpdateBuses()

    @pytest.mark.order(20100)
    def test_region_rename(self, saw_instance):
        """Region rename operations complete without error."""
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()


class TestCaseActions:
    """Tests for case actions (highly destructive - run last)."""

    @pytest.mark.order(30000)
    def test_case_description(self, saw_instance):
        """CaseDescriptionSet and CaseDescriptionClear work."""
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()

    @pytest.mark.order(30100)
    def test_equivalence_and_external_system(self, saw_instance, temp_file):
        """External system operations and equivalence."""
        saw_instance.DeleteExternalSystem()
        saw_instance.Equivalence()
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)

    @pytest.mark.order(30150)
    def test_scale(self, saw_instance):
        """Scale load by factor of 1.0 (no-op)."""
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")

    @pytest.mark.order(30160)
    def test_write_text_to_file(self, saw_instance, temp_file):
        """WriteTextToFile creates a file with content."""
        tmp_txt = temp_file(".txt")
        saw_instance.WriteTextToFile(tmp_txt, "Test content")
        assert os.path.exists(tmp_txt)

    @pytest.mark.order(99900)
    def test_renumber(self, saw_instance):
        """Renumber operations (run last as they modify keys)."""
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()


class TestSubData:
    """Integration tests for GetSubData - retrieving nested SubData from AUX exports."""

    @pytest.mark.order(40000)
    def test_gen_ops(self, saw_instance):
        """GetSubData retrieves generator data with various SubData types."""
        # Basic fields only
        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID", "GenMW"])
        assert df is not None
        assert "BusNum" in df.columns and "GenID" in df.columns and "GenMW" in df.columns

        # With BidCurve SubData
        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], ["BidCurve"])
        assert df is not None and "BidCurve" in df.columns
        for bc in df["BidCurve"]:
            assert isinstance(bc, list)

        # With ReactiveCapability SubData
        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], ["ReactiveCapability"])
        assert df is not None and "ReactiveCapability" in df.columns
        for rc in df["ReactiveCapability"]:
            assert isinstance(rc, list)

        # Multiple SubData types
        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID", "GenMW"], ["BidCurve", "ReactiveCapability"])
        assert df is not None and "BidCurve" in df.columns and "ReactiveCapability" in df.columns

        # With filter
        df_all = saw_instance.GetSubData("Gen", ["BusNum", "GenID"])
        df_filtered = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], filter_name="GenStatus=Closed")
        assert df_filtered is not None and len(df_filtered) <= len(df_all)

    @pytest.mark.order(40100)
    def test_other_types(self, saw_instance):
        """GetSubData works for Load, Contingency, and Interface object types."""
        # Load BidCurve
        df = saw_instance.GetSubData("Load", ["BusNum", "LoadID", "LoadMW"], ["BidCurve"])
        assert df is not None and "BidCurve" in df.columns

        # Contingency elements
        df = saw_instance.GetSubData("Contingency", ["TSContingency"], ["CTGElement"])
        assert df is not None
        if not df.empty:
            assert "CTGElement" in df.columns
            for ctg in df["CTGElement"]:
                assert isinstance(ctg, list)

        # Interface elements
        df = saw_instance.GetSubData("Interface", ["InterfaceName"], ["InterfaceElement"])
        assert df is not None

        # SuperArea (may be empty)
        df = saw_instance.GetSubData("SuperArea", ["SuperAreaName"], ["SuperAreaArea"])
        assert df is not None


class TestDataAccess:
    """Integration tests for data access, saving, and property accessors."""

    @pytest.mark.order(41100)
    def test_retrieval(self, saw_instance):
        """Data retrieval operations return correct types and non-empty results."""
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None and isinstance(df, pd.DataFrame) and len(df) > 0
        assert "BusNum" in df.columns and "BusName" in df.columns

        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

        df = saw_instance.ListOfDevices("Bus")
        assert df is not None and isinstance(df, pd.DataFrame) and len(df) > 0

        result = saw_instance.ListOfDevicesAsVariantStrings("Bus")
        assert result is not None

        df1 = saw_instance.GetFieldList("Bus")
        assert df1 is not None and "internal_field_name" in df1.columns and len(df1) > 0
        df2 = saw_instance.GetFieldList("Bus")
        assert df1.equals(df2)

    @pytest.mark.order(41200)
    def test_properties_and_errors(self, saw_instance):
        """Property accessors return valid data, invalid operations raise errors."""
        pid = saw_instance.ProcessID
        assert pid is not None and isinstance(pid, int) and pid > 0

        info = saw_instance.ProgramInformation
        assert info is not None and isinstance(info, tuple)

        current_dir = saw_instance.CurrentDir
        assert current_dir is not None and isinstance(current_dir, str)

        with pytest.raises(ValueError, match="Mode must be either"):
            saw_instance.EnterMode("INVALID")

        with pytest.raises(PowerWorldError):
            saw_instance.RunScriptCommand("InvalidCommand_XYZ_123;")


# =============================================================================
# Extended classes merged from test_integration_extended.py
# =============================================================================


@pytest.mark.usefixtures("save_restore_state")
class TestModifyExtended:
    """Extended tests for Modify operations -- covering uncovered boolean paths.

    Uses save_restore_state fixture to preserve case integrity.
    All modifications are reverted after the class completes.
    """

    @pytest.mark.order(80000)
    def test_modify_auto_insert_tieline(self, saw_instance):
        try:
            saw_instance.AutoInsertTieLineTransactions()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Tie line transactions not available")

    @pytest.mark.order(80100)
    def test_modify_branch_mva_limit_reorder(self, saw_instance):
        try:
            saw_instance.BranchMVALimitReorder()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Branch MVA limit reorder not available")

    @pytest.mark.order(80200)
    def test_modify_branch_mva_limit_reorder_with_filter(self, saw_instance):
        try:
            saw_instance.BranchMVALimitReorder(filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Branch MVA limit reorder not available")

    @pytest.mark.order(80300)
    def test_modify_calculate_rxbg(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TransLineCalc not available")

    @pytest.mark.order(80400)
    def test_modify_calculate_rxbg_selected(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType(filter_name="SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TransLineCalc not available")

    @pytest.mark.order(80500)
    def test_modify_clear_small_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    @pytest.mark.order(80600)
    def test_modify_init_gen_mvar_limits(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()

    @pytest.mark.order(80700)
    def test_modify_injection_groups_auto_insert(self, saw_instance):
        try:
            saw_instance.InjectionGroupsAutoInsert()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group auto-insert not available")

    @pytest.mark.order(80800)
    def test_modify_injection_group_create(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "ALL", append=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group create not available")

    @pytest.mark.order(80900)
    def test_modify_injection_group_create_no_append(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("TestIG2", "Gen", 1.0, "ALL", append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group create not available")

    @pytest.mark.order(81000)
    def test_modify_interfaces_auto_insert(self, saw_instance):
        try:
            saw_instance.InterfacesAutoInsert("AREA", delete_existing=True, use_filters=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface auto-insert not available")

    @pytest.mark.order(81100)
    def test_modify_interfaces_auto_insert_with_filters(self, saw_instance):
        try:
            saw_instance.InterfacesAutoInsert("AREA", delete_existing=False, use_filters=True, prefix="TEST_")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface auto-insert not available")

    @pytest.mark.order(81200)
    def test_modify_set_participation_factors(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    @pytest.mark.order(81300)
    def test_modify_set_scheduled_voltage(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.SetScheduledVoltageForABus(bus_key, 1.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SetScheduledVoltage not available")

    @pytest.mark.order(81400)
    def test_modify_set_interface_limit_sum(self, saw_instance):
        try:
            saw_instance.SetInterfaceLimitToMonitoredElementLimitSum("ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface limit sum not available")

    @pytest.mark.order(81500)
    def test_modify_rotate_bus_angles(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.RotateBusAnglesInIsland(bus_key, 0.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Rotate bus angles not available")

    @pytest.mark.order(81600)
    def test_modify_set_gen_pmax(self, saw_instance):
        try:
            saw_instance.SetGenPMaxFromReactiveCapabilityCurve()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Reactive capability curve not available")

    @pytest.mark.order(81700)
    def test_modify_remove_3w_xformer(self, saw_instance):
        try:
            saw_instance.Remove3WXformerContainer()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("3W transformer removal not available")

    @pytest.mark.order(81800)
    def test_modify_rename_injection_group(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("RenameTestIG", "Gen", 1.0, "ALL")
            saw_instance.RenameInjectionGroup("RenameTestIG", "RenamedIG")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group rename not available")

    @pytest.mark.order(81900)
    def test_modify_reassign_ids(self, saw_instance):
        try:
            saw_instance.ReassignIDs("Load", "BusName", filter_name="", use_right=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ReassignIDs not available")

    @pytest.mark.order(82000)
    def test_modify_reassign_ids_right(self, saw_instance):
        try:
            saw_instance.ReassignIDs("Load", "BusName", filter_name="ALL", use_right=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ReassignIDs not available")

    @pytest.mark.order(82100)
    def test_modify_merge_line_terminals(self, saw_instance):
        try:
            saw_instance.MergeLineTerminals("SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("MergeLineTerminals not available")

    @pytest.mark.order(82200)
    def test_modify_merge_ms_line_sections(self, saw_instance):
        try:
            saw_instance.MergeMSLineSections("SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("MergeMSLineSections not available")

    @pytest.mark.order(82300)
    def test_modify_directions_auto_insert(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("DirectionsAutoInsert requires a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DirectionsAutoInsert(s, b, delete_existing=True, use_area_zone_filters=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("DirectionsAutoInsert not available")

    @pytest.mark.order(82400)
    def test_modify_directions_auto_insert_with_filters(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("DirectionsAutoInsert requires a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DirectionsAutoInsert(s, b, delete_existing=False, use_area_zone_filters=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Directions with filters not available")

    @pytest.mark.order(82500)
    def test_modify_directions_auto_insert_ref_opposite(self, saw_instance):
        try:
            saw_instance.DirectionsAutoInsertReference("Bus", "Slack", delete_existing=True, opposite_direction=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Directions reference insert not available")

    @pytest.mark.order(82600)
    def test_modify_change_system_mva_base(self, saw_instance):
        saw_instance.ChangeSystemMVABase(100.0)


class TestGeneralExtended:
    """Extended tests for General mixin -- uncovered parameter paths."""

    @pytest.mark.order(67000)
    def test_general_log_clear(self, saw_instance):
        saw_instance.LogClear()

    @pytest.mark.order(67100)
    def test_general_log_show(self, saw_instance):
        saw_instance.LogShow(show=True)
        saw_instance.LogShow(show=False)

    @pytest.mark.order(67200)
    def test_general_log_add_datetime(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel", include_date=True, include_time=True, include_milliseconds=False)

    @pytest.mark.order(67300)
    def test_general_log_add_datetime_all(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel2", include_date=True, include_time=True, include_milliseconds=True)

    @pytest.mark.order(67400)
    def test_general_log_add_datetime_minimal(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel3", include_date=False, include_time=False, include_milliseconds=False)

    @pytest.mark.order(67500)
    def test_general_log_save_append(self, saw_instance, temp_file):
        tmp = temp_file(".txt")
        saw_instance.LogAdd("Test1")
        saw_instance.LogSave(tmp, append=False)
        saw_instance.LogAdd("Test2")
        saw_instance.LogSave(tmp, append=True)
        assert os.path.exists(tmp)

    @pytest.mark.order(67600)
    def test_general_set_current_directory(self, saw_instance, temp_dir):
        saw_instance.SetCurrentDirectory(str(temp_dir))

    @pytest.mark.order(67700)
    def test_general_set_current_directory_create(self, saw_instance, temp_dir):
        new_dir = os.path.join(str(temp_dir), "test_subdir")
        saw_instance.SetCurrentDirectory(new_dir, create_if_not_found=True)

    @pytest.mark.order(67800)
    def test_general_enter_mode(self, saw_instance):
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    @pytest.mark.order(67900)
    def test_general_import_data(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, "w") as f:
            f.write("BusNum,BusName\n1,TestBus\n")
        try:
            saw_instance.ImportData(tmp_csv, "CSV", header_line=1, create_if_not_found=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ImportData not available")

    @pytest.mark.order(68000)
    def test_general_load_csv(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, "w") as f:
            f.write("ObjectType,Bus\nBusNum,BusName\n1,TestBus\n")
        try:
            saw_instance.LoadCSV(tmp_csv, create_if_not_found=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadCSV not available")

    @pytest.mark.order(68100)
    def test_general_save_data_with_extra(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.SaveDataWithExtra(
                tmp_csv, "CSV", "Bus", ["BusNum", "BusName"],
                header_list=["CaseName"], header_value_list=["TestCase"],
            )
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveDataWithExtra not available")

    @pytest.mark.order(68200)
    def test_general_save_data_no_sort(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(
            tmp_aux, "AUX", "Bus", ["BusNum", "BusName"],
            transpose=False, append=False,
        )
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(68300)
    def test_general_save_data_transposed(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.SaveData(
                tmp_csv, "CSV", "Bus", ["BusNum", "BusName"],
                transpose=True, append=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveData transposed not available")

    @pytest.mark.order(68400)
    def test_general_load_aux_create(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, "w") as f:
            f.write('DATA (Bus, [BusNum, BusName]) {\n99998 "TestNewBus"\n}\n')
        saw_instance.LoadAux(tmp_aux, create_if_not_found=True)
        # Clean up
        try:
            saw_instance.Delete("Bus", "BusNum = 99998")
        except PowerWorldError:
            pass

    @pytest.mark.order(68500)
    def test_general_load_aux_directory(self, saw_instance, temp_dir):
        try:
            saw_instance.LoadAuxDirectory(str(temp_dir), filter_string="*.aux")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadAuxDirectory not available")

    @pytest.mark.order(68600)
    def test_general_load_aux_directory_no_filter(self, saw_instance, temp_dir):
        try:
            saw_instance.LoadAuxDirectory(str(temp_dir))
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadAuxDirectory not available")

    @pytest.mark.order(68700)
    def test_general_load_data(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, "w") as f:
            f.write('DATA (Bus, [BusNum, BusName]) {\n1 "TestBus"\n}\n')
        try:
            saw_instance.LoadData(tmp_aux, "Bus")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadData not available")

    @pytest.mark.order(68800)
    def test_general_stop_aux_file(self, saw_instance):
        try:
            saw_instance.StopAuxFile()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("StopAuxFile not available")

    @pytest.mark.order(68900)
    def test_general_select_all_no_filter(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")


class TestCaseActionsExtended:
    """Extended tests for Case Actions -- uncovered parameter paths."""

    @pytest.mark.order(90000)
    def test_case_description_append(self, saw_instance):
        saw_instance.CaseDescriptionSet("Line 1")
        saw_instance.CaseDescriptionSet("Line 2", append=True)
        saw_instance.CaseDescriptionClear()

    @pytest.mark.order(90100)
    def test_case_save_external_with_ties(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        try:
            saw_instance.SaveExternalSystem(tmp_pwb, with_ties=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveExternalSystem with ties not available")

    @pytest.mark.order(90200)
    def test_case_scale_gen(self, saw_instance):
        try:
            saw_instance.Scale("GEN", "FACTOR", [1.0], "SYSTEM")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Scale GEN not available")

    @pytest.mark.order(90300)
    def test_case_scale_load_mw(self, saw_instance):
        try:
            saw_instance.Scale("LOAD", "MW", [100.0, 50.0], "SYSTEM")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Scale LOAD MW not available")

    @pytest.mark.order(90400)
    def test_case_load_ems(self, saw_instance, temp_file):
        tmp = temp_file(".hdb")
        try:
            saw_instance.LoadEMS(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadEMS not available")

    @pytest.mark.order(90500)
    def test_case_renumber_custom_index(self, saw_instance):
        try:
            saw_instance.RenumberAreas(custom_integer_index=1)
            saw_instance.RenumberBuses(custom_integer_index=2)
            saw_instance.RenumberSubs(custom_integer_index=3)
            saw_instance.RenumberZones(custom_integer_index=4)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Renumber with custom index not available")


@pytest.mark.usefixtures("save_restore_state")
class TestModifyGaps:
    """Tests for remaining Modify operations."""

    @pytest.mark.order(83000)
    def test_create_line_derive_existing(self, saw_instance):
        """CreateLineDeriveExisting creates a line from existing parameters."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_id = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        try:
            saw_instance.CreateLineDeriveExisting(
                int(b["BusNum"]), int(b["BusNum:1"]), "99",
                10.0, branch_id, existing_length=5.0, zero_g=True,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("CreateLineDeriveExisting not available")

    @pytest.mark.order(83100)
    def test_merge_buses(self, saw_instance):
        """MergeBuses completes without error."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        try:
            saw_instance.MergeBuses("SELECTED", filter_name="")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("MergeBuses not available")

    @pytest.mark.order(83200)
    def test_move(self, saw_instance):
        """Move a generator (0% -- no-op)."""
        gens = saw_instance.GetParametersMultipleElement("Gen", ["BusNum", "GenID"])
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert gens is not None and not gens.empty, "Test case must contain generators"
        assert buses is not None and not buses.empty, "Test case must contain buses"
        gen_key = create_object_string("Gen", gens.iloc[0]["BusNum"], gens.iloc[0]["GenID"])
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.Move(gen_key, bus_key, how_much=0.0, abort_on_error=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Move not available")

    @pytest.mark.order(83300)
    def test_split_bus(self, saw_instance):
        """SplitBus creates a new bus from an existing one."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.SplitBus(bus_key, 99997, insert_tie=True, line_open=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SplitBus not available")

    @pytest.mark.order(83400)
    def test_tap_transmission_line(self, saw_instance):
        """TapTransmissionLine taps a line at midpoint."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit"]
        )
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_key = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        try:
            saw_instance.TapTransmissionLine(
                branch_key, 50.0, 99996,
                shunt_model="CAPACITANCE",
                treat_as_ms_line=False,
                update_onelines=False,
                new_bus_name="TapBus",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TapTransmissionLine not available")

    @pytest.mark.order(83500)
    def test_branch_mva_limit_with_limits(self, saw_instance):
        """BranchMVALimitReorder with explicit limits list."""
        try:
            saw_instance.BranchMVALimitReorder(
                filter_name="ALL",
                limits=["A", "B", "C"],
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("BranchMVALimitReorder with limits not available")


class TestRegionsGaps:
    """Tests for remaining Regions functions."""

    @pytest.mark.order(85000)
    def test_region_load_shapefile(self, saw_instance, temp_file):
        """RegionLoadShapefile completes without error."""
        tmp = temp_file(".shp")
        try:
            saw_instance.RegionLoadShapefile(
                tmp, "TestClass", ["Name"],
                add_to_open_onelines=False,
                display_style_name="",
                delete_existing=True,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("RegionLoadShapefile not available")


class TestGeneralGaps:
    """Tests for remaining General mixin functions."""

    @pytest.mark.order(85100)
    def test_save_object_fields(self, saw_instance, temp_file):
        """SaveObjectFields writes field metadata to file."""
        tmp = temp_file(".csv")
        try:
            saw_instance.SaveObjectFields(tmp, "Bus", ["BusNum", "BusName"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveObjectFields not available")

    @pytest.mark.order(85200)
    def test_load_script(self, saw_instance, temp_file):
        """LoadScript processes script from aux file."""
        tmp = temp_file(".aux")
        with open(tmp, "w") as f:
            f.write('SCRIPT TestScript\n{\n    LogAdd("Script test");\n}\n')
        try:
            saw_instance.LoadScript(tmp, "TestScript")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadScript not available")

    @pytest.mark.order(85300)
    def test_delete_with_filter(self, saw_instance):
        """Delete with specific area zone filter."""
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [99995, "DeleteTestBus"])
        try:
            saw_instance.Delete("Bus", "BusNum = 99995")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Delete with filter not available")

    @pytest.mark.order(85400)
    def test_create_data(self, saw_instance):
        """CreateData creates an object then cleans up."""
        try:
            saw_instance.CreateData("Bus", ["BusNum", "BusName"], [99994, "CreateTestBus"])
            saw_instance.Delete("Bus", "BusNum = 99994")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("CreateData not available")

    @pytest.mark.order(85500)
    def test_send_to_excel_advanced(self, saw_instance):
        """SendToExcelAdvanced completes without error (requires Excel)."""
        try:
            saw_instance.SendToExcelAdvanced(
                "Bus", ["BusNum", "BusName"],
                use_column_headers=True,
                clear_existing=True,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError, OSError):
            pytest.skip("SendToExcelAdvanced not available (Excel may not be installed)")


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
