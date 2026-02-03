"""
Integration tests for core SAW COM operations.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test the foundational SAW
class operations: case open/save, parameter get/set, state management,
field lists, logging, file I/O, data import/export, and subdata retrieval.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_modify.py       -- destructive modify, region, case actions
    - test_integration_saw_powerflow.py    -- power flow, matrices, sensitivity, topology
    - test_integration_saw_contingency.py  -- contingency and fault analysis
    - test_integration_saw_gic.py          -- GIC analysis
    - test_integration_saw_transient.py    -- transient stability
    - test_integration_saw_operations.py   -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py        -- PowerWorld facade and statics
    - test_integration_network.py          -- Network topology

USAGE:
    pytest tests/test_integration_saw_core.py -v
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
    """Tests for base SAW operations: case open/save, parameters, state, fields."""

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
    """Tests for general SAW operations: logging, file I/O, data import/export."""

    @pytest.mark.order(9500)
    def test_log_operations(self, saw_instance, temp_file):
        """Log lifecycle: add, clear, show, datetime variants, save, save with append."""
        saw_instance.LogAdd("SAW Validator Test Message")
        tmp_log = temp_file(".txt")
        saw_instance.LogSave(tmp_log)
        assert os.path.exists(tmp_log)

        # Clear and re-add
        saw_instance.LogClear()
        saw_instance.LogAdd("Test message")
        saw_instance.LogAddDateTime("Timer")

        # Show toggle
        saw_instance.LogShow(show=True)
        saw_instance.LogShow(show=False)

        # DateTime variants
        saw_instance.LogAddDateTime("TestLabel", include_date=True, include_time=True, include_milliseconds=False)
        saw_instance.LogAddDateTime("TestLabel2", include_date=True, include_time=True, include_milliseconds=True)
        saw_instance.LogAddDateTime("TestLabel3", include_date=False, include_time=False, include_milliseconds=False)

        # Save with append
        tmp = temp_file(".txt")
        saw_instance.LogAdd("Test1")
        saw_instance.LogSave(tmp, append=False)
        saw_instance.LogAdd("Test2")
        saw_instance.LogSave(tmp, append=True)
        assert os.path.exists(tmp)

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

    @pytest.mark.order(52100)
    def test_save_data_variants(self, saw_instance, temp_file):
        """SaveData works for AUX, CSV, transposed, and non-sorted formats."""
        tmp_aux = temp_file(".aux")
        tmp_csv = temp_file(".csv")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        assert os.path.exists(tmp_aux)
        saw_instance.SaveData(tmp_csv, "CSV", "Bus", ["BusNum", "BusName"], filter_name="SELECTED")

        # Non-sorted AUX
        tmp_aux2 = temp_file(".aux")
        saw_instance.SaveData(
            tmp_aux2, "AUX", "Bus", ["BusNum", "BusName"],
            transpose=False, append=False,
        )
        assert os.path.exists(tmp_aux2)

        # Transposed CSV
        tmp_csv2 = temp_file(".csv")
        saw_instance.SaveData(
            tmp_csv2, "CSV", "Bus", ["BusNum", "BusName"],
            transpose=True, append=False,
        )

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
        match = result[result["Name"] == "TestCtgRoundtrip"]
        assert len(match) == 1
        assert len(match.iloc[0]["CTGElement"]) > 0

    @pytest.mark.order(67600)
    def test_set_current_directory(self, saw_instance, temp_dir):
        """SetCurrentDirectory with and without create_if_not_found."""
        saw_instance.SetCurrentDirectory(str(temp_dir))
        new_dir = os.path.join(str(temp_dir), "test_subdir")
        saw_instance.SetCurrentDirectory(new_dir, create_if_not_found=True)

    @pytest.mark.order(67900)
    def test_import_data(self, saw_instance, temp_file):
        """ImportData round-trips PTI format."""
        tmp = temp_file(".raw")
        saw_instance.SaveData(tmp, "PTI", "Bus", ["BusNum", "BusName"])
        saw_instance.ImportData(tmp, "PTI", header_line=1, create_if_not_found=True)

    @pytest.mark.order(68000)
    def test_load_csv(self, saw_instance, temp_file):
        """LoadCSV loads a simple CSV file."""
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, "w") as f:
            f.write("ObjectType,Bus\nBusNum,BusName\n1,TestBus\n")
        saw_instance.LoadCSV(tmp_csv, create_if_not_found=True)

    @pytest.mark.order(68100)
    def test_save_data_with_extra(self, saw_instance, temp_file):
        """SaveDataWithExtra writes CSV with header metadata."""
        tmp_csv = temp_file(".csv")
        saw_instance.SaveDataWithExtra(
            tmp_csv, "CSV", "Bus", ["BusNum", "BusName"],
            header_list=["CaseName"], header_value_list=["TestCase"],
        )
        assert os.path.exists(tmp_csv)

    @pytest.mark.order(68400)
    def test_load_aux_create(self, saw_instance, temp_file):
        """LoadAux with create_if_not_found creates new objects."""
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, "w") as f:
            f.write('DATA (Bus, [BusNum, BusName]) {\n99998 "TestNewBus"\n}\n')
        saw_instance.LoadAux(tmp_aux, create_if_not_found=True)
        try:
            saw_instance.Delete("Bus", "BusNum = 99998")
        except PowerWorldError:
            pass

    @pytest.mark.order(68500)
    def test_load_aux_directory(self, saw_instance, temp_dir):
        """LoadAuxDirectory with and without filter."""
        saw_instance.LoadAuxDirectory(str(temp_dir), filter_string="*.aux")
        saw_instance.LoadAuxDirectory(str(temp_dir))

    @pytest.mark.order(68700)
    def test_load_data(self, saw_instance, temp_file):
        """LoadData loads bus data from AUX."""
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, "w") as f:
            f.write('DATA (Bus, [BusNum, BusName]) {\n1 "TestBus"\n}\n')
        saw_instance.LoadData(tmp_aux, "Bus")

    @pytest.mark.order(68800)
    def test_stop_aux_file(self, saw_instance):
        """StopAuxFile completes without error."""
        saw_instance.StopAuxFile()

    @pytest.mark.order(68900)
    def test_select_all_no_filter(self, saw_instance):
        """SelectAll/UnSelectAll without filter."""
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")

    @pytest.mark.order(85100)
    def test_save_object_fields(self, saw_instance, temp_file):
        """SaveObjectFields writes field metadata to file."""
        tmp = temp_file(".csv")
        saw_instance.SaveObjectFields(tmp, "Bus", ["BusNum", "BusName"])

    @pytest.mark.order(85200)
    def test_load_script(self, saw_instance, temp_file):
        """LoadScript processes script from aux file."""
        tmp = temp_file(".aux")
        with open(tmp, "w") as f:
            f.write('SCRIPT TestScript\n{\n    LogAdd("Script test");\n}\n')
        saw_instance.LoadScript(tmp, "TestScript")

    @pytest.mark.order(85300)
    def test_delete_with_filter(self, saw_instance):
        """Delete with specific filter."""
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [99995, "DeleteTestBus"])
        saw_instance.Delete("Bus", "BusNum = 99995")

    @pytest.mark.order(85400)
    def test_create_data(self, saw_instance):
        """CreateData creates an object then cleans up."""
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [99994, "CreateTestBus"])
        saw_instance.Delete("Bus", "BusNum = 99994")

    @pytest.mark.order(85500)
    def test_send_to_excel(self, saw_instance):
        """SendtoExcel completes without error (requires Excel)."""
        try:
            saw_instance.SendtoExcel(
                "Bus", ["BusNum", "BusName"],
                workbook="TestWorkbook",
                worksheet="Sheet1",
                use_column_headers=True,
                clear_existing=True,
            )
        except PowerWorldError as e:
            msg = str(e).lower()
            if "excel" in msg or "workbook" in msg or "saveas" in msg:
                pytest.skip("Excel not available or workbook error")
            raise


class TestSubData:
    """Integration tests for GetSubData - retrieving nested SubData from AUX exports."""

    @pytest.mark.order(40000)
    def test_gen_ops(self, saw_instance):
        """GetSubData retrieves generator data with various SubData types."""
        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID", "GenMW"])
        assert df is not None
        assert "BusNum" in df.columns and "GenID" in df.columns and "GenMW" in df.columns

        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], ["BidCurve"])
        assert df is not None and "BidCurve" in df.columns
        for bc in df["BidCurve"]:
            assert isinstance(bc, list)

        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], ["ReactiveCapability"])
        assert df is not None and "ReactiveCapability" in df.columns
        for rc in df["ReactiveCapability"]:
            assert isinstance(rc, list)

        df = saw_instance.GetSubData("Gen", ["BusNum", "GenID", "GenMW"], ["BidCurve", "ReactiveCapability"])
        assert df is not None and "BidCurve" in df.columns and "ReactiveCapability" in df.columns

        df_all = saw_instance.GetSubData("Gen", ["BusNum", "GenID"])
        df_filtered = saw_instance.GetSubData("Gen", ["BusNum", "GenID"], filter_name="GenStatus=Closed")
        assert df_filtered is not None and len(df_filtered) <= len(df_all)

    @pytest.mark.order(40100)
    def test_other_types(self, saw_instance):
        """GetSubData works for Load, Contingency, and Interface object types."""
        df = saw_instance.GetSubData("Load", ["BusNum", "LoadID", "LoadMW"], ["BidCurve"])
        assert df is not None and "BidCurve" in df.columns

        df = saw_instance.GetSubData("Contingency", ["TSContingency"], ["CTGElement"])
        assert df is not None
        if not df.empty:
            assert "CTGElement" in df.columns
            for ctg in df["CTGElement"]:
                assert isinstance(ctg, list)

        df = saw_instance.GetSubData("Interface", ["InterfaceName"], ["InterfaceElement"])
        assert df is not None

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


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
