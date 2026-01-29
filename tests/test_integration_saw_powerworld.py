"""
Integration tests for SAW base, general, oneline, modify, regions, and case actions.

WHAT THIS TESTS:
- Base SAW operations (save, load, properties, state)
- General commands (file ops, modes, scripts)
- Oneline diagram operations
- Modify operations (create/delete objects, merge, split)
- Regions operations
- Case actions (equivalence, renumber, scale)

NOTE: Power flow, matrices, sensitivity, contingency, fault, GIC, ATC, transient,
      and time step tests are in their dedicated test files:
      - test_integration_powerflow.py
      - test_integration_contingency.py
      - test_integration_analysis.py

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py

USAGE:
    pytest tests/test_integration_saw_powerworld.py -v
"""

import os
import sys
import pytest
import pandas as pd
import tempfile as tf

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

    @pytest.mark.order(1)
    def test_save_case(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveCase(tmp_pwb)
        assert os.path.exists(tmp_pwb)

        # Also test saving to original path (may fail if on network/OneDrive)
        original_path = saw_instance.pwb_file_path
        assert original_path is not None
        try:
            saw_instance.SaveCase()
            assert os.path.exists(original_path)
        except PowerWorldError:
            pass

    @pytest.mark.order(2)
    def test_get_header(self, saw_instance):
        header = saw_instance.GetCaseHeader()
        assert header is not None

    @pytest.mark.order(3)
    def test_change_parameters(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            original_name = buses.iloc[0]["BusName"]
            new_name = "TestBusName"
            saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, new_name])

            check = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
            assert check["BusName"] == new_name

            saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, original_name])

    @pytest.mark.order(4)
    def test_get_parameters(self, saw_instance):
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None
        assert not df.empty

        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

    @pytest.mark.order(5)
    def test_list_devices(self, saw_instance):
        df = saw_instance.ListOfDevices("Bus")
        assert df is not None
        assert not df.empty

    @pytest.mark.order(7)
    def test_state(self, saw_instance):
        saw_instance.StoreState("TestState")
        saw_instance.RestoreState("TestState")
        saw_instance.DeleteState("TestState")
        saw_instance.SaveState()
        saw_instance.LoadState()

    @pytest.mark.order(8)
    def test_run_script_2(self, saw_instance):
        saw_instance.RunScriptCommand2("LogAdd(\"Test\");", "Testing...")

    @pytest.mark.order(9)
    def test_field_list(self, saw_instance):
        df = saw_instance.GetFieldList("Bus")
        assert not df.empty

        df_spec = saw_instance.GetSpecificFieldList("Bus", ["BusNum", "BusName"])
        assert not df_spec.empty

    @pytest.mark.order(500)
    def test_update_ui_and_exec_aux(self, saw_instance):
        """Test update_ui and exec_aux operations."""
        saw_instance.update_ui()

        aux_content = '''SCRIPT
{
    LogAdd("test exec_aux");
}'''
        try:
            saw_instance.exec_aux(aux_content)
        except PowerWorldError:
            pass

    @pytest.mark.order(501)
    def test_change_parameters_rect(self, saw_instance):
        """Test ChangeParametersMultipleElementRect."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            df = buses.head(1).copy()
            original_name = df.iloc[0]["BusName"]
            df.iloc[0, df.columns.get_loc("BusName")] = "TempTestName"
            saw_instance.ChangeParametersMultipleElementRect("Bus", ["BusNum", "BusName"], df)
            df.iloc[0, df.columns.get_loc("BusName")] = original_name
            saw_instance.ChangeParametersMultipleElementRect("Bus", ["BusNum", "BusName"], df)

    @pytest.mark.order(502)
    def test_list_devices_variants(self, saw_instance):
        """Test ListOfDevices variants."""
        result1 = saw_instance.ListOfDevicesAsVariantStrings("Bus")
        assert result1 is not None
        result2 = saw_instance.ListOfDevicesFlatOutput("Bus")
        assert result2 is not None

        result = saw_instance.GetParametersMultipleElementFlatOutput("Bus", ["BusNum", "BusName"])
        assert result is None or len(result) > 0

    @pytest.mark.order(503)
    def test_send_to_excel(self, saw_instance):
        """Test SendToExcel operation."""
        try:
            saw_instance.SendToExcel("Bus", "", ["BusNum", "BusName"])
        except (PowerWorldError, Exception):
            pass

    @pytest.mark.order(504)
    def test_set_data(self, saw_instance):
        """Test SetData operation."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            original = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
            original_name = original["BusName"]
            saw_instance.SetData("Bus", ["BusName"], ["TempName"], f"BusNum = {bus_num}")
            saw_instance.SetData("Bus", ["BusName"], [original_name], f"BusNum = {bus_num}")

    @pytest.mark.order(505)
    def test_simauto_property_errors(self, saw_instance):
        """Test set_simauto_property validation errors."""
        # Invalid property name
        with pytest.raises(ValueError, match="not currently supported"):
            saw_instance.set_simauto_property("InvalidProperty", True)

        # Invalid property type
        with pytest.raises(ValueError, match="is invalid"):
            saw_instance.set_simauto_property("CreateIfNotFound", "not_a_bool")

        # Invalid CurrentDir path
        with pytest.raises(ValueError, match="not a valid path"):
            saw_instance.set_simauto_property("CurrentDir", "C:\\NonExistent\\Path\\12345")

    @pytest.mark.order(506)
    def test_change_parameters_flat_input(self, saw_instance):
        """Test ChangeParametersMultipleElementFlatInput."""
        from esapp.saw._exceptions import Error
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = int(buses.iloc[0]["BusNum"])
            original_name = buses.iloc[0]["BusName"]

            # Test with flat list (should work)
            saw_instance.ChangeParametersMultipleElementFlatInput(
                "Bus", ["BusNum", "BusName"], 1, [bus_num, "TempFlatName"]
            )
            # Restore original
            saw_instance.ChangeParametersMultipleElementFlatInput(
                "Bus", ["BusNum", "BusName"], 1, [bus_num, original_name]
            )

            # Test error case with nested list
            with pytest.raises(Error, match="1-D array"):
                saw_instance.ChangeParametersMultipleElementFlatInput(
                    "Bus", ["BusNum", "BusName"], 1, [[bus_num, "Test"]]
                )

    @pytest.mark.order(507)
    def test_change_parameters_multiple_element(self, saw_instance):
        """Test ChangeParametersMultipleElement with nested list."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = int(buses.iloc[0]["BusNum"])
            original_name = buses.iloc[0]["BusName"]

            saw_instance.ChangeParametersMultipleElement(
                "Bus", ["BusNum", "BusName"], [[bus_num, "TempMultiName"]]
            )
            saw_instance.ChangeParametersMultipleElement(
                "Bus", ["BusNum", "BusName"], [[bus_num, original_name]]
            )

    @pytest.mark.order(508)
    def test_get_specific_field_max_num(self, saw_instance):
        """Test GetSpecificFieldMaxNum operation."""
        max_num = saw_instance.GetSpecificFieldMaxNum("Bus", "CustomFloat")
        assert isinstance(max_num, int)

    @pytest.mark.order(509)
    def test_get_params_rect_typed(self, saw_instance):
        """Test GetParamsRectTyped operation."""
        df = saw_instance.GetParamsRectTyped("Bus", ["BusNum", "BusName"])
        assert df is None or isinstance(df, pd.DataFrame)

    @pytest.mark.order(510)
    def test_open_case_type(self, saw_instance):
        """Test OpenCaseType operation."""
        # Use existing case file path
        pwb_path = saw_instance.pwb_file_path

        # Test OpenCaseType with PWB format
        try:
            saw_instance.OpenCaseType(pwb_path, "PWB")
        except PowerWorldError:
            pass

        # Reopen original case
        saw_instance.OpenCase()

    @pytest.mark.order(511)
    def test_exec_aux_double_quotes(self, saw_instance):
        """Test exec_aux with double quote replacement."""
        aux_content = "SCRIPT { LogAdd('test double quotes'); }"
        try:
            saw_instance.exec_aux(aux_content, use_double_quotes=True)
        except PowerWorldError:
            pass

    @pytest.mark.order(512)
    def test_set_logging_level(self, saw_instance):
        """Test set_logging_level operation."""
        import logging
        saw_instance.set_logging_level(logging.DEBUG)
        saw_instance.set_logging_level("INFO")


class TestGeneral:
    """Tests for general SAW operations."""

    @pytest.mark.order(95)
    def test_log(self, saw_instance, temp_file):
        saw_instance.LogAdd("SAW Validator Test Message")
        tmp_log = temp_file(".txt")
        saw_instance.LogSave(tmp_log)
        assert os.path.exists(tmp_log)

    @pytest.mark.order(96)
    def test_file_ops(self, saw_instance, temp_file):
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

    @pytest.mark.order(98)
    def test_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        saw_instance.LoadAux(tmp_aux)

    @pytest.mark.order(99)
    def test_select(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")

    @pytest.mark.order(520)
    def test_log_clear_and_show(self, saw_instance):
        """Test log clear, add, datetime, and show operations."""
        saw_instance.LogClear()
        saw_instance.LogAdd("Test message")
        saw_instance.LogAddDateTime("Timer")
        try:
            saw_instance.LogShow(True)
            saw_instance.LogShow(False)
        except PowerWorldError:
            pass

    @pytest.mark.order(521)
    def test_save_data_variants(self, saw_instance, temp_file):
        """Test SaveData variants."""
        tmp_aux = temp_file(".aux")
        tmp_csv = temp_file(".csv")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        assert os.path.exists(tmp_aux)
        saw_instance.SaveData(tmp_csv, "CSV", "Bus", ["BusNum", "BusName"], filter_name="SELECTED")

    @pytest.mark.order(522)
    def test_save_object_fields(self, saw_instance, temp_file):
        """Test SaveObjectFields operation."""
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.SaveObjectFields(tmp_txt, "Bus", ["BusNum", "BusName"])
        except PowerWorldError:
            pass

    @pytest.mark.order(523)
    def test_enter_mode(self, saw_instance):
        """Test EnterMode operations."""
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    @pytest.mark.order(524)
    def test_set_current_directory(self, saw_instance):
        """Test SetCurrentDirectory operation."""
        temp_dir = tf.gettempdir()
        try:
            saw_instance.SetCurrentDirectory(temp_dir)
        except PowerWorldError:
            pass

    @pytest.mark.order(525)
    def test_load_script(self, saw_instance, temp_file):
        """Test LoadScript operation."""
        tmp_script = temp_file(".pws")
        with open(tmp_script, 'w') as f:
            f.write('LogAdd("Test");')
        try:
            saw_instance.LoadScript(tmp_script)
        except PowerWorldError:
            pass

    @pytest.mark.order(526)
    def test_stop_aux_file(self, saw_instance):
        """Test StopAuxFile operation."""
        try:
            saw_instance.StopAuxFile()
        except PowerWorldError:
            pass

    @pytest.mark.order(527)
    def test_import_data(self, saw_instance, temp_file):
        """Test ImportData operation."""
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w') as f:
            f.write("BusNum,BusName\n1,TestBus\n")
        try:
            saw_instance.ImportData(tmp_csv, "CSV", 1, False)
        except PowerWorldError:
            pass

    @pytest.mark.order(528)
    def test_load_data(self, saw_instance, temp_file):
        """Test LoadData operation."""
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        try:
            saw_instance.LoadData(tmp_aux, "Bus", False)
        except PowerWorldError:
            pass

    @pytest.mark.order(529)
    def test_save_data_with_extra(self, saw_instance, temp_file):
        """Test SaveDataWithExtra operation."""
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.SaveDataWithExtra(
                tmp_csv, "CSV", "Bus", ["BusNum", "BusName"],
                header_list=["CaseName"], header_value_list=["TestCase"]
            )
        except PowerWorldError:
            pass

    @pytest.mark.order(530)
    def test_send_to_excel_advanced(self, saw_instance):
        """Test SendToExcelAdvanced operation."""
        try:
            saw_instance.SendToExcelAdvanced(
                "Bus", ["BusNum", "BusName"], "",
                use_column_headers=True, workbook="", worksheet=""
            )
        except (PowerWorldError, Exception):
            pass

    @pytest.mark.order(531)
    def test_set_sub_data(self, saw_instance):
        """Test SetSubData operation."""
        try:
            saw_instance.SetSubData(
                "Contingency",
                ["Name"],
                [{"Name": "TestCtgSubData"}]
            )
        except PowerWorldError:
            pass


class TestModify:
    """Tests for modify operations (destructive - run late, order 100-199)."""

    @pytest.mark.order(120)
    def test_create_delete(self, saw_instance):
        dummy_bus = 99999
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [dummy_bus, "SAW_TEST"])
        saw_instance.Delete("Bus", f"BusNum = {dummy_bus}")

    @pytest.mark.order(134)
    def test_superarea(self, saw_instance):
        saw_instance.CreateData("SuperArea", ["Name"], ["TestSuperArea"])
        saw_instance.SuperAreaAddAreas("TestSuperArea", "ALL")
        saw_instance.SuperAreaRemoveAreas("TestSuperArea", "ALL")

    @pytest.mark.order(135)
    def test_interface_ops(self, saw_instance):
        saw_instance.InjectionGroupRemoveDuplicates()
        saw_instance.InterfaceRemoveDuplicates()
        saw_instance.DirectionsAutoInsertReference("Bus", "Slack")

        saw_instance.InterfaceCreate("TestInt", True, "Branch", "SELECTED")
        saw_instance.InterfaceFlatten("TestInt")
        saw_instance.InterfaceFlattenFilter("ALL")
        saw_instance.InterfaceModifyIsolatedElements()

        saw_instance.CreateData("Contingency", ["Name"], ["TestCtg"])
        saw_instance.InterfaceAddElementsFromContingency("TestInt", "TestCtg")

    @pytest.mark.order(510)
    def test_refine_model(self, saw_instance):
        """Test RefineModel operation."""
        try:
            saw_instance.RefineModel("Gen", "", "FIX", 0.01)
        except PowerWorldError:
            pass

    @pytest.mark.order(511)
    def test_participation_factors(self, saw_instance):
        """Test SetParticipationFactors operation."""
        gens = saw_instance.ListOfDevices("Gen")
        if gens is not None and not gens.empty:
            try:
                saw_instance.SetParticipationFactors("PROPORTIONAL", 1.0, "ALL")
            except PowerWorldError:
                pass

    @pytest.mark.order(512)
    def test_merge_buses(self, saw_instance):
        """Test MergeBuses operation."""
        buses = saw_instance.ListOfDevices("Bus")
        if buses is not None and len(buses) >= 2:
            bus_num = buses.iloc[0]["BusNum"]
            try:
                saw_instance.MergeBuses(f"[BUS {bus_num}]", "SELECTED")
            except PowerWorldError:
                pass

    @pytest.mark.order(513)
    def test_gen_pmax_reactive(self, saw_instance):
        """Test SetGenPMaxFromReactiveCapabilityCurve operation."""
        try:
            saw_instance.SetGenPMaxFromReactiveCapabilityCurve()
        except PowerWorldError:
            pass

    @pytest.mark.order(514)
    def test_branch_mva_limit_reorder(self, saw_instance):
        """Test BranchMVALimitReorder operation."""
        try:
            saw_instance.BranchMVALimitReorder()
            saw_instance.BranchMVALimitReorder("", ["A", "B", "C"])
        except PowerWorldError:
            pass

    @pytest.mark.order(515)
    def test_create_line_derive_existing(self, saw_instance):
        """Test CreateLineDeriveExisting operation."""
        branches = saw_instance.ListOfDevices("Branch")
        if branches is not None and not branches.empty:
            try:
                saw_instance.CreateLineDeriveExisting(
                    99998, 99999, "1", 10.0, "[BRANCH 1 2 1]", 5.0, True
                )
            except PowerWorldError:
                pass

    @pytest.mark.order(516)
    def test_directions_auto_insert(self, saw_instance):
        """Test DirectionsAutoInsert operation."""
        try:
            saw_instance.DirectionsAutoInsert('[BUS 1]', '[BUS 2]', True, False)
        except PowerWorldError:
            pass

    @pytest.mark.order(517)
    def test_injection_group_create(self, saw_instance):
        """Test InjectionGroupCreate operation."""
        try:
            saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "", True)
        except PowerWorldError:
            pass

    @pytest.mark.order(518)
    def test_interfaces_auto_insert(self, saw_instance):
        """Test InterfacesAutoInsert operation."""
        try:
            saw_instance.InterfacesAutoInsert("AREA", True, False, "Test_", "AUTO")
        except PowerWorldError:
            pass

    @pytest.mark.order(519)
    def test_move(self, saw_instance):
        """Test Move operation."""
        gens = saw_instance.ListOfDevices("Gen")
        buses = saw_instance.ListOfDevices("Bus")
        if gens is not None and not gens.empty and buses is not None and len(buses) >= 2:
            try:
                gen_bus = gens.iloc[0]["BusNum"]
                dest_bus = buses.iloc[1]["BusNum"]
                saw_instance.Move(f"[GEN {gen_bus}]", f"[BUS {dest_bus}]", 0.0, True)
            except PowerWorldError:
                pass

    @pytest.mark.order(520)
    def test_reassign_ids(self, saw_instance):
        """Test ReassignIDs operation."""
        try:
            saw_instance.ReassignIDs("Load", "BusName", "", False)
        except PowerWorldError:
            pass

    @pytest.mark.order(521)
    def test_rename_injection_group(self, saw_instance):
        """Test RenameInjectionGroup operation."""
        try:
            saw_instance.RenameInjectionGroup("OldIG", "NewIG")
        except PowerWorldError:
            pass

    @pytest.mark.order(522)
    def test_split_bus(self, saw_instance):
        """Test SplitBus operation."""
        buses = saw_instance.ListOfDevices("Bus")
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            try:
                saw_instance.SplitBus(f"[BUS {bus_num}]", 99997, True, False, "Line")
            except PowerWorldError:
                pass

    @pytest.mark.order(523)
    def test_tap_transmission_line(self, saw_instance):
        """Test TapTransmissionLine operation."""
        branches = saw_instance.ListOfDevices("Branch")
        if branches is not None and not branches.empty:
            try:
                saw_instance.TapTransmissionLine("[BRANCH 1 2 1]", 50.0, 99996)
            except PowerWorldError:
                pass


class TestRegions:
    """Tests for regions operations (destructive - run late, order 200-299)."""

    @pytest.mark.order(200)
    def test_region_update_buses(self, saw_instance):
        """Test RegionUpdateBuses operation."""
        saw_instance.RegionUpdateBuses()

    @pytest.mark.order(201)
    def test_region_rename(self, saw_instance):
        """Test RegionRename operation."""
        saw_instance.RegionRename("OldRegion", "NewRegion")

    @pytest.mark.order(202)
    def test_region_rename_class(self, saw_instance):
        """Test RegionRenameClass operation."""
        saw_instance.RegionRenameClass("OldClass", "NewClass")

    @pytest.mark.order(203)
    def test_region_rename_proper1(self, saw_instance):
        """Test RegionRenameProper1 operation."""
        saw_instance.RegionRenameProper1("OldP1", "NewP1")

    @pytest.mark.order(204)
    def test_region_rename_proper2(self, saw_instance):
        """Test RegionRenameProper2 operation."""
        saw_instance.RegionRenameProper2("OldP2", "NewP2")

    @pytest.mark.order(205)
    def test_region_rename_proper3(self, saw_instance):
        """Test RegionRenameProper3 operation."""
        saw_instance.RegionRenameProper3("OldP3", "NewP3")

    @pytest.mark.order(206)
    def test_region_rename_proper12_flip(self, saw_instance):
        """Test RegionRenameProper12Flip operation."""
        saw_instance.RegionRenameProper12Flip()

    @pytest.mark.order(207)
    def test_region_load_shapefile(self, saw_instance, temp_file):
        """Test RegionLoadShapefile operation."""
        # Create a dummy shapefile path (won't exist, but tests the call)
        tmp_shp = temp_file(".shp")
        try:
            saw_instance.RegionLoadShapefile(tmp_shp, "TestClass", ["Name"], False, "", False)
        except PowerWorldError:
            pass


class TestCaseActions:
    """Tests for case actions (highly destructive - run last, order 300+)."""

    @pytest.mark.order(300)
    def test_all_ops(self, saw_instance, temp_file):
        """Test all case action operations together."""
        # Case description
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()

        # External system
        saw_instance.DeleteExternalSystem()
        saw_instance.Equivalence()
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)

        # Scaling
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")

        # LoadAuxDirectory
        temp_dir = tf.gettempdir()
        try:
            saw_instance.LoadAuxDirectory(temp_dir, "*.aux")
        except PowerWorldError:
            pass

        # WriteTextToFile
        tmp_txt = temp_file(".txt")
        saw_instance.WriteTextToFile(tmp_txt, "Test content")
        assert os.path.exists(tmp_txt)

        # LoadCSV
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, 'w') as f:
            f.write("BusNum,BusName\n1,Test\n")
        try:
            saw_instance.LoadCSV(tmp_csv)
        except PowerWorldError:
            pass

    @pytest.mark.order(302)
    def test_new_case(self, saw_instance):
        """Test NewCase operation."""
        # Store original path to reopen
        original_path = saw_instance.pwb_file_path
        try:
            saw_instance.NewCase()
        except PowerWorldError:
            pass
        # Reopen original case
        saw_instance.OpenCase(original_path)

    @pytest.mark.order(303)
    def test_renumber_files(self, saw_instance, temp_file):
        """Test renumbering file operations."""
        tmp_csv = temp_file(".csv")
        # Create a simple renumbering file
        with open(tmp_csv, 'w') as f:
            f.write("OldBus,NewBus\n1,100001\n")

        try:
            saw_instance.Renumber3WXFormerStarBuses(tmp_csv, "COMMA")
        except PowerWorldError:
            pass

        try:
            saw_instance.RenumberMSLineDummyBuses(tmp_csv, "COMMA")
        except PowerWorldError:
            pass

    @pytest.mark.order(999)
    def test_renumber(self, saw_instance):
        """Test renumbering operations (run last as they modify keys)."""
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()


class TestSubData:
    """Integration tests for GetSubData - retrieving nested SubData from AUX exports."""

    @pytest.mark.order(400)
    def test_gen_ops(self, saw_instance):
        """Test GetSubData with various generator configurations."""
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

    @pytest.mark.order(401)
    def test_other_types(self, saw_instance):
        """Test GetSubData with other object types."""
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

        # Bus marginal costs (may cause access violation in some PowerWorld versions)
        try:
            df = saw_instance.GetSubData("Bus", ["BusNum", "BusName"], ["MWMarginalCostValues"])
            assert df is not None and "MWMarginalCostValues" in df.columns
        except PowerWorldError:
            pass


class TestDataAccess:
    """Integration tests for data access, saving, and property accessors."""

    @pytest.mark.order(411)
    def test_retrieval(self, saw_instance):
        """Test data retrieval operations."""
        # GetParametersMultipleElement
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None and isinstance(df, pd.DataFrame) and len(df) > 0
        assert "BusNum" in df.columns and "BusName" in df.columns

        # GetParametersSingleElement
        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

        # ListOfDevices
        df = saw_instance.ListOfDevices("Bus")
        assert df is not None and isinstance(df, pd.DataFrame) and len(df) > 0

        # ListOfDevicesAsVariantStrings
        result = saw_instance.ListOfDevicesAsVariantStrings("Bus")
        assert result is not None

        # GetFieldList and caching
        df1 = saw_instance.GetFieldList("Bus")
        assert df1 is not None and "internal_field_name" in df1.columns and len(df1) > 0
        df2 = saw_instance.GetFieldList("Bus")
        assert df1.equals(df2)

    @pytest.mark.order(412)
    def test_properties_and_errors(self, saw_instance):
        """Test property accessors and error handling."""
        # ProcessID
        pid = saw_instance.ProcessID
        assert pid is not None and isinstance(pid, int) and pid > 0

        # ProgramInformation
        info = saw_instance.ProgramInformation
        assert info is not None and isinstance(info, tuple)

        # CurrentDir
        current_dir = saw_instance.CurrentDir
        assert current_dir is not None and isinstance(current_dir, str)

        # Invalid mode
        with pytest.raises(ValueError, match="Mode must be either"):
            saw_instance.EnterMode("INVALID")

        # Invalid script command
        with pytest.raises(PowerWorldError):
            saw_instance.RunScriptCommand("InvalidCommand_XYZ_123;")


class TestTopology:
    """Tests for topology analysis operations."""

    @pytest.mark.order(450)
    def test_all_ops(self, saw_instance, temp_file):
        """Test all topology operations together."""
        # Basic topology operations
        try:
            saw_instance.CreateNewAreasFromIslands()
        except PowerWorldError:
            pass
        try:
            saw_instance.ExpandAllBusTopology()
        except PowerWorldError:
            pass
        try:
            saw_instance.FindRadialBusPaths()
            saw_instance.FindRadialBusPaths(ignore_status=True, treat_parallel_as_not_radial=True, bus_or_superbus="SUPERBUS")
        except PowerWorldError:
            pass

        # Bus-specific topology
        buses = saw_instance.ListOfDevices("Bus")
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            try:
                saw_instance.ExpandBusTopology(f"[BUS {bus_num}]", "FULL")
            except PowerWorldError:
                pass
            try:
                saw_instance.SetSelectedFromNetworkCut(set_how=True, bus_on_cut_side=f"[BUS {bus_num}]", energized=True, num_tiers=1)
            except PowerWorldError:
                pass

        # Facility analysis
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.DoFacilityAnalysis(tmp_aux, set_selected=True)
        except PowerWorldError:
            pass

        # Consolidated case save
        tmp_pwb = temp_file(".pwb")
        try:
            saw_instance.SaveConsolidatedCase(tmp_pwb, "PWB", "Number", False, False)
        except PowerWorldError:
            pass

        # Breaker operations
        try:
            saw_instance.CloseWithBreakers("Branch", "SELECTED", False, None, False)
        except PowerWorldError:
            pass
        try:
            saw_instance.OpenWithBreakers("Branch", "SELECTED", None, False)
        except PowerWorldError:
            pass


class TestPVQV:
    """Tests for PV and QV analysis operations."""

    @pytest.mark.order(460)
    def test_all_ops(self, saw_instance, temp_file):
        """Test all PV and QV operations together."""
        tmp_aux = temp_file(".aux")
        tmp_csv = temp_file(".csv")

        # PV basic operations
        try:
            saw_instance.PVClear()
            saw_instance.PVDestroy()
            saw_instance.PVStartOver()
            saw_instance.PVQVTrackSingleBusPerSuperBus()
            saw_instance.PVSetSourceAndSink('[AREA "1"]', '[AREA "1"]')
        except PowerWorldError:
            pass

        # PV write operations
        try:
            saw_instance.PVWriteResultsAndOptions(tmp_aux, append=False)
            saw_instance.PVDataWriteOptionsAndResults(tmp_aux, append=True, key_field="PRIMARY")
            saw_instance.PVWriteInadequateVoltages(tmp_csv, append=False, inadequate_type="LOW")
        except PowerWorldError:
            pass

        # QV basic operations
        try:
            saw_instance.QVDeleteAllResults()
            saw_instance.QVSelectSingleBusPerSuperBus()
        except PowerWorldError:
            pass

        # QV write operations
        try:
            saw_instance.QVWriteResultsAndOptions(tmp_aux, append=False)
            saw_instance.QVDataWriteOptionsAndResults(tmp_aux, append=True, key_field="PRIMARY")
            saw_instance.QVWriteCurves(tmp_csv, include_quantities=True, filter_name="", append=False)
        except PowerWorldError:
            pass

        # RunQV
        try:
            result = saw_instance.RunQV()
            assert result is None or isinstance(result, pd.DataFrame)
        except PowerWorldError:
            pass


class TestTimestep:
    """Tests for timestep simulation operations."""

    @pytest.mark.order(470)
    def test_timestep_basic(self, saw_instance):
        """Test basic timestep operations."""
        try:
            saw_instance.TimeStepDeleteAll()
            saw_instance.TimeStepResetRun()
        except PowerWorldError:
            pass

    @pytest.mark.order(471)
    def test_timestep_modify(self, saw_instance):
        """Test timestep modify operations."""
        try:
            saw_instance.TIMESTEPSaveSelectedModifyStart()
            saw_instance.TIMESTEPSaveSelectedModifyFinish()
        except PowerWorldError:
            pass

    @pytest.mark.order(472)
    def test_timestep_save_pww(self, saw_instance, temp_file):
        """Test TimeStepSavePWW operation."""
        tmp_pww = temp_file(".pww")
        try:
            saw_instance.TimeStepSavePWW(tmp_pww)
        except PowerWorldError:
            pass

    @pytest.mark.order(473)
    def test_timestep_save_tsb(self, saw_instance, temp_file):
        """Test TimeStepSaveTSB operation."""
        tmp_tsb = temp_file(".tsb")
        try:
            saw_instance.TimeStepSaveTSB(tmp_tsb)
        except PowerWorldError:
            pass

    @pytest.mark.order(474)
    def test_timestep_save_csv(self, saw_instance, temp_file):
        """Test timestep CSV save operations."""
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass
        try:
            saw_instance.TIMESTEPSaveInputCSV(tmp_csv, ["GenMW", "GenMvar"])
        except PowerWorldError:
            pass

    @pytest.mark.order(475)
    def test_timestep_single_point(self, saw_instance):
        """Test TimeStepDoSinglePoint operation."""
        try:
            saw_instance.TimeStepDoSinglePoint("2025-01-01T00:00:00")
        except PowerWorldError:
            pass

    @pytest.mark.order(476)
    def test_timestep_do_run(self, saw_instance):
        """Test TimeStepDoRun operation."""
        try:
            saw_instance.TimeStepDoRun()
            saw_instance.TimeStepDoRun("2025-01-01T00:00:00", "2025-01-01T01:00:00")
        except PowerWorldError:
            pass

    @pytest.mark.order(477)
    def test_timestep_clear_results(self, saw_instance):
        """Test TimeStepClearResults operation."""
        try:
            saw_instance.TimeStepClearResults()
            saw_instance.TimeStepClearResults("2025-01-01T00:00:00", "2025-01-01T01:00:00")
        except PowerWorldError:
            pass

    @pytest.mark.order(478)
    def test_timestep_append_pww(self, saw_instance, temp_file):
        """Test TimeStepAppendPWW operations."""
        tmp_pww = temp_file(".pww")
        try:
            saw_instance.TimeStepAppendPWW(tmp_pww)
            saw_instance.TimeStepAppendPWWRange(tmp_pww, "0", "100", "Single Solution")
        except PowerWorldError:
            pass

    @pytest.mark.order(479)
    def test_timestep_load_operations(self, saw_instance, temp_file):
        """Test timestep load operations."""
        tmp_pww = temp_file(".pww")
        tmp_tsb = temp_file(".tsb")
        tmp_b3d = temp_file(".b3d")
        try:
            saw_instance.TimeStepLoadPWWRange(tmp_pww, "0", "100", "Single Solution")
        except PowerWorldError:
            pass
        try:
            saw_instance.TimeStepLoadB3D(tmp_b3d)
        except PowerWorldError:
            pass
        try:
            saw_instance.TimeStepLoadTSB(tmp_tsb)
        except PowerWorldError:
            pass
        try:
            saw_instance.TimeStepLoadPWW(tmp_pww, "Single Solution")
        except PowerWorldError:
            pass

    @pytest.mark.order(480)
    def test_timestep_save_fields(self, saw_instance):
        """Test TimeStepSaveFieldsSet and Clear operations."""
        try:
            saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW", "GenMvar"], "ALL")
            saw_instance.TimeStepSaveFieldsClear(["Gen"])
            saw_instance.TimeStepSaveFieldsClear()
        except PowerWorldError:
            pass

    @pytest.mark.order(481)
    def test_timestep_lat_lon_operations(self, saw_instance, temp_file):
        """Test timestep lat/lon operations."""
        tmp_pww = temp_file(".pww")
        try:
            saw_instance.TimeStepAppendPWWRangeLatLon(tmp_pww, "0", "100", 30.0, 40.0, -100.0, -90.0)
        except PowerWorldError:
            pass
        try:
            saw_instance.TimeStepLoadPWWRangeLatLon(tmp_pww, "0", "100", 30.0, 40.0, -100.0, -90.0)
        except PowerWorldError:
            pass

    @pytest.mark.order(482)
    def test_timestep_save_pww_range(self, saw_instance, temp_file):
        """Test TimeStepSavePWWRange operation."""
        tmp_pww = temp_file(".pww")
        try:
            saw_instance.TimeStepSavePWWRange(tmp_pww, "0", "100")
        except PowerWorldError:
            pass


class TestPowerflow:
    """Tests for powerflow operations."""

    @pytest.mark.order(480)
    def test_all_ops(self, saw_instance):
        """Test all powerflow operations together."""
        # Solution aid operations
        try:
            saw_instance.ClearPowerFlowSolutionAidValues()
        except PowerWorldError:
            pass
        try:
            saw_instance.ResetToFlatStart()
        except PowerWorldError:
            pass
        try:
            saw_instance.VoltageConditioning()
        except PowerWorldError:
            pass

        # ZeroOutMismatches variants
        try:
            saw_instance.ZeroOutMismatches()
            saw_instance.ZeroOutMismatches("GEN")
            saw_instance.ZeroOutMismatches("LOAD")
        except PowerWorldError:
            pass

        # Voltage estimation and conditioning
        try:
            saw_instance.EstimateVoltages("")
        except PowerWorldError:
            pass
        try:
            saw_instance.ConditionVoltagePockets(0.8, 30.0)
        except PowerWorldError:
            pass

        # DiffCase operations
        try:
            saw_instance.DiffCaseSetAsBase()
            saw_instance.DiffCaseKeyType("PRIMARY")
            saw_instance.DiffCaseShowPresentAndBase(True)
            saw_instance.DiffCaseMode("DIFFERENCE")
            saw_instance.DiffCaseRefresh()
            saw_instance.DiffCaseClearBase()
        except PowerWorldError:
            pass

        # Solve variants
        try:
            saw_instance.SolvePowerFlow("RECTNEWT")
            saw_instance.SolvePowerFlow("DC")
        except PowerWorldError:
            pass


class TestSensitivity:
    """Tests for sensitivity analysis operations."""

    @pytest.mark.order(550)
    def test_tap_sense(self, saw_instance):
        """Test CalculateTapSense operation."""
        try:
            saw_instance.CalculateTapSense()
            saw_instance.CalculateTapSense("SELECTED")
        except PowerWorldError:
            pass

    @pytest.mark.order(551)
    def test_volt_self_sense(self, saw_instance):
        """Test CalculateVoltSelfSense operation."""
        try:
            saw_instance.CalculateVoltSelfSense()
            saw_instance.CalculateVoltSelfSense("SELECTED")
        except PowerWorldError:
            pass

    @pytest.mark.order(552)
    def test_volt_sense(self, saw_instance):
        """Test CalculateVoltSense operation."""
        buses = saw_instance.ListOfDevices("Bus")
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            try:
                saw_instance.CalculateVoltSense(bus_num)
            except PowerWorldError:
                pass

    @pytest.mark.order(553)
    def test_loss_sense(self, saw_instance):
        """Test CalculateLossSense operation."""
        try:
            saw_instance.CalculateLossSense("MW")
            saw_instance.CalculateLossSense("SUBS", "NO", "EXISTING")
        except PowerWorldError:
            pass

    @pytest.mark.order(554)
    def test_ptdf_multiple_directions(self, saw_instance):
        """Test CalculatePTDFMultipleDirections operation."""
        try:
            saw_instance.CalculatePTDFMultipleDirections()
            saw_instance.CalculatePTDFMultipleDirections(False, True, "DCPS")
        except PowerWorldError:
            pass

    @pytest.mark.order(555)
    def test_out_of_service_sensitivities(self, saw_instance):
        """Test SetSensitivitiesAtOutOfServiceToClosest operation."""
        try:
            saw_instance.SetSensitivitiesAtOutOfServiceToClosest()
        except PowerWorldError:
            pass

    @pytest.mark.order(556)
    def test_flow_sense(self, saw_instance):
        """Test CalculateFlowSense operation."""
        try:
            saw_instance.CalculateFlowSense("[BRANCH 1 2 1]", "MW")
        except PowerWorldError:
            pass

    @pytest.mark.order(557)
    def test_ptdf(self, saw_instance):
        """Test CalculatePTDF operation."""
        try:
            saw_instance.CalculatePTDF('[BUS 1]', '[BUS 2]', "DC")
        except PowerWorldError:
            pass

    @pytest.mark.order(558)
    def test_lodf(self, saw_instance):
        """Test CalculateLODF operation."""
        try:
            saw_instance.CalculateLODF("[BRANCH 1 2 1]", "DC")
            saw_instance.CalculateLODF("[BRANCH 1 2 1]", "DCPS", "YES")
        except PowerWorldError:
            pass

    @pytest.mark.order(559)
    def test_lodf_advanced(self, saw_instance, temp_file):
        """Test CalculateLODFAdvanced operation."""
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.CalculateLODFAdvanced(True, "CSV", 100, 0.01, "DECIMAL", 4, False, tmp_csv, True)
        except PowerWorldError:
            pass

    @pytest.mark.order(560)
    def test_lodf_screening(self, saw_instance, temp_file):
        """Test CalculateLODFScreening operation."""
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.CalculateLODFScreening(
                "ALL", "ALL", True, False, True, 0.1, True, 90.0, 110.0, True, tmp_csv
            )
        except PowerWorldError:
            pass

    @pytest.mark.order(561)
    def test_shift_factors(self, saw_instance):
        """Test CalculateShiftFactors operations."""
        try:
            saw_instance.CalculateShiftFactors("[BRANCH 1 2 1]", "BUYER", '[BUS 1]', "DC")
        except PowerWorldError:
            pass
        try:
            saw_instance.CalculateShiftFactorsMultipleElement("BRANCH", "ALL", "BUYER", '[BUS 1]', "DC")
        except PowerWorldError:
            pass

    @pytest.mark.order(562)
    def test_lodf_matrix(self, saw_instance):
        """Test CalculateLODFMatrix operation."""
        try:
            saw_instance.CalculateLODFMatrix("OUTAGES", "ALL", "ALL", True, "DC", "", True)
        except PowerWorldError:
            pass

    @pytest.mark.order(563)
    def test_volt_to_transfer_sense(self, saw_instance):
        """Test CalculateVoltToTransferSense operation."""
        try:
            saw_instance.CalculateVoltToTransferSense('[BUS 1]', '[BUS 2]', "P", False)
        except PowerWorldError:
            pass

    @pytest.mark.order(564)
    def test_line_loading_replicator(self, saw_instance):
        """Test LineLoadingReplicator operations."""
        try:
            saw_instance.LineLoadingReplicatorCalculate(
                "[BRANCH 1 2 1]", '[INJECTIONGROUP "Test"]', False, 100.0, False, "DC"
            )
        except PowerWorldError:
            pass
        try:
            saw_instance.LineLoadingReplicatorImplement()
        except PowerWorldError:
            pass


class TestTransient:
    """Tests for transient stability operations."""

    @pytest.mark.order(600)
    def test_ts_basic(self, saw_instance, temp_file):
        """Test basic transient operations."""
        tmp_aux = temp_file(".aux")
        tmp_dyr = temp_file(".dyr")

        try:
            saw_instance.TSAutoCorrect()
        except PowerWorldError:
            pass
        try:
            saw_instance.TSValidate()
        except PowerWorldError:
            pass
        try:
            saw_instance.TSTransferStateToPowerFlow()
            saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=True)
        except PowerWorldError:
            pass

    @pytest.mark.order(601)
    def test_ts_initialize(self, saw_instance):
        """Test TSInitialize operation."""
        try:
            saw_instance.TSInitialize()
        except PowerWorldError:
            pass

    @pytest.mark.order(602)
    def test_ts_result_storage(self, saw_instance):
        """Test TSResultStorageSetAll and TSStoreResponse operations."""
        try:
            saw_instance.TSResultStorageSetAll("ALL", True)
            saw_instance.TSResultStorageSetAll("Gen", False)
            saw_instance.TSStoreResponse()
            saw_instance.TSStoreResponse("Gen", False)
        except PowerWorldError:
            pass

    @pytest.mark.order(603)
    def test_ts_clear_results(self, saw_instance):
        """Test TSClearResultsFromRAM operations."""
        try:
            saw_instance.TSClearResultsFromRAM()
            saw_instance.TSClearResultsFromRAM("ALL", True, True, True, True, True)
            saw_instance.TSClearResultsFromRAM("TestCtg")
            saw_instance.TSClearResultsFromRAMAndDisableStorage()
        except PowerWorldError:
            pass

    @pytest.mark.order(604)
    def test_ts_write_operations(self, saw_instance, temp_file):
        """Test TS write operations."""
        tmp_aux = temp_file(".aux")
        tmp_dyr = temp_file(".dyr")
        tmp_bpa = temp_file(".dat")

        try:
            saw_instance.TSWriteOptions(tmp_aux)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSWriteModels(tmp_aux)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSSavePTI(tmp_dyr)
            saw_instance.TSSavePTI(tmp_dyr, diff_case_modified_only=True)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSSaveGE(tmp_dyr)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSSaveBPA(tmp_bpa)
        except PowerWorldError:
            pass

    @pytest.mark.order(605)
    def test_ts_load_operations(self, saw_instance, temp_file):
        """Test TS load operations."""
        tmp_dyr = temp_file(".dyr")
        try:
            saw_instance.TSLoadPTI(tmp_dyr)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSLoadGE(tmp_dyr)
        except PowerWorldError:
            pass
        try:
            saw_instance.TSLoadBPA(tmp_dyr)
        except PowerWorldError:
            pass

    @pytest.mark.order(606)
    def test_ts_solve(self, saw_instance):
        """Test TSSolve operations."""
        ctgs = saw_instance.ListOfDevices("TSContingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["TSContingency"] if "TSContingency" in ctgs.columns else ctgs.iloc[0][ctgs.columns[0]]
            try:
                saw_instance.TSSolve(ctg_name)
                saw_instance.TSSolve(ctg_name, start_time=0.0, stop_time=1.0, step_size=0.01)
            except PowerWorldError:
                pass
        try:
            saw_instance.TSSolveAll()
        except PowerWorldError:
            pass

    @pytest.mark.order(607)
    def test_ts_clear_models(self, saw_instance):
        """Test TSClearAllModels and TSClearModelsforObjects operations."""
        try:
            saw_instance.TSClearModelsforObjects("Gen", "")
        except PowerWorldError:
            pass

    @pytest.mark.order(608)
    def test_ts_auto_insert_relay(self, saw_instance):
        """Test TSAutoInsertDistRelay and TSAutoInsertZPOTT operations."""
        try:
            saw_instance.TSAutoInsertDistRelay(0.8, True, True, False, 1, "")
        except PowerWorldError:
            pass
        try:
            saw_instance.TSAutoInsertZPOTT(0.8, "")
        except PowerWorldError:
            pass

    @pytest.mark.order(609)
    def test_ts_calculate_operations(self, saw_instance):
        """Test TS calculation operations."""
        try:
            saw_instance.TSCalculateCriticalClearTime("[BRANCH 1 2 1]")
        except PowerWorldError:
            pass
        try:
            saw_instance.TSCalculateSMIBEigenValues()
        except PowerWorldError:
            pass

    @pytest.mark.order(610)
    def test_ts_playin_signals(self, saw_instance):
        """Test PlayIn signal operations."""
        import numpy as np
        try:
            saw_instance.TSClearPlayInSignals()
        except PowerWorldError:
            pass

        times = np.array([0.0, 0.5, 1.0])
        signals = np.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])
        try:
            saw_instance.TSSetPlayInSignals("TestSignal", times, signals)
        except PowerWorldError:
            pass

    @pytest.mark.order(611)
    def test_ts_join_ctgs(self, saw_instance, temp_file):
        """Test TSJoinActiveCTGs operation."""
        tmp_file = temp_file(".aux")
        try:
            saw_instance.TSJoinActiveCTGs(0.0, False, False, tmp_file, "Both")
        except PowerWorldError:
            pass

    @pytest.mark.order(612)
    def test_ts_plot_series(self, saw_instance):
        """Test TSPlotSeriesAdd operation."""
        try:
            saw_instance.TSPlotSeriesAdd("TestPlot", 1, 1, "Gen", "GenMW", "", "")
        except PowerWorldError:
            pass

    @pytest.mark.order(613)
    def test_ts_run_result_analyzer(self, saw_instance):
        """Test TSRunResultAnalyzer operation."""
        try:
            saw_instance.TSRunResultAnalyzer()
        except PowerWorldError:
            pass

    @pytest.mark.order(614)
    def test_ts_run_until_specified_time(self, saw_instance):
        """Test TSRunUntilSpecifiedTime operation."""
        ctgs = saw_instance.ListOfDevices("TSContingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["TSContingency"] if "TSContingency" in ctgs.columns else ctgs.iloc[0][ctgs.columns[0]]
            try:
                saw_instance.TSRunUntilSpecifiedTime(ctg_name, stop_time=1.0)
            except PowerWorldError:
                pass

    @pytest.mark.order(615)
    def test_ts_save_dynamic_models(self, saw_instance, temp_file):
        """Test TSSaveDynamicModels operation."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.TSSaveDynamicModels(tmp_aux, "AUX", "Gen", "", False)
        except PowerWorldError:
            pass

    @pytest.mark.order(616)
    def test_ts_get_vcurve_data(self, saw_instance, temp_file):
        """Test TSGetVCurveData operation."""
        tmp_file = temp_file(".csv")
        try:
            saw_instance.TSGetVCurveData(tmp_file, "")
        except PowerWorldError:
            pass

    @pytest.mark.order(617)
    def test_ts_disable_machine_model(self, saw_instance):
        """Test TSDisableMachineModelNonZeroDerivative operation."""
        try:
            saw_instance.TSDisableMachineModelNonZeroDerivative(0.001)
        except PowerWorldError:
            pass

    @pytest.mark.order(618)
    def test_ts_set_selected_for_references(self, saw_instance):
        """Test TSSetSelectedForTransientReferences operation."""
        try:
            saw_instance.TSSetSelectedForTransientReferences("Selected", "YES", ["Gen"], ["GENROU"])
        except PowerWorldError:
            pass

    @pytest.mark.order(619)
    def test_ts_save_two_bus_equivalent(self, saw_instance, temp_file):
        """Test TSSaveTwoBusEquivalent operation."""
        tmp_pwb = temp_file(".pwb")
        buses = saw_instance.ListOfDevices("Bus")
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            try:
                saw_instance.TSSaveTwoBusEquivalent(tmp_pwb, f"[BUS {bus_num}]")
            except PowerWorldError:
                pass

    @pytest.mark.order(620)
    def test_ts_auto_save_plots(self, saw_instance):
        """Test TSAutoSavePlots operation."""
        try:
            saw_instance.TSAutoSavePlots(["Plot1"], ["Ctg1"], "JPG", 800, 600)
        except PowerWorldError:
            pass

    @pytest.mark.order(621)
    def test_ts_load_relay_files(self, saw_instance, temp_file):
        """Test TSLoadRDB and TSLoadRelayCSV operations."""
        tmp_rdb = temp_file(".rdb")
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TSLoadRDB(tmp_rdb, "DIST", "")
        except PowerWorldError:
            pass
        try:
            saw_instance.TSLoadRelayCSV(tmp_csv, "DIST", "")
        except PowerWorldError:
            pass



if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
