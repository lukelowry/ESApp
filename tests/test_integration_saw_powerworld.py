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


class TestBaseSAW:
    """Tests for base SAW operations (order 1-9)."""

    @pytest.mark.order(1)
    def test_base_save_case(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveCase(tmp_pwb)
        assert os.path.exists(tmp_pwb)

    @pytest.mark.order(2)
    def test_base_get_header(self, saw_instance):
        header = saw_instance.GetCaseHeader()
        assert header is not None

    @pytest.mark.order(3)
    def test_base_change_parameters(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        if buses is not None and not buses.empty:
            bus_num = buses.iloc[0]["BusNum"]
            original_name = buses.iloc[0]["BusName"]
            new_name = "TestBusName"
            saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, new_name])
            
            check = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
            assert check["BusName"] == new_name
            
            saw_instance.ChangeParameters("Bus", ["BusNum", "BusName"], [bus_num, original_name])

    @pytest.mark.order(4)
    def test_base_get_parameters(self, saw_instance):
        df = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusName"])
        assert df is not None
        assert not df.empty
        
        bus_num = df.iloc[0]["BusNum"]
        s = saw_instance.GetParametersSingleElement("Bus", ["BusNum", "BusName"], [bus_num, ""])
        assert isinstance(s, pd.Series)

    @pytest.mark.order(5)
    def test_base_list_devices(self, saw_instance):
        df = saw_instance.ListOfDevices("Bus")
        assert df is not None
        assert not df.empty

    @pytest.mark.order(6)
    def test_base_properties(self, saw_instance):
        _ = saw_instance.CreateIfNotFound
        _ = saw_instance.CurrentDir
        _ = saw_instance.ProcessID
        _ = saw_instance.RequestBuildDate
        _ = saw_instance.UIVisible
        _ = saw_instance.ProgramInformation

    @pytest.mark.order(7)
    def test_base_state(self, saw_instance):
        saw_instance.StoreState("TestState")
        saw_instance.RestoreState("TestState")
        saw_instance.DeleteState("TestState")
        saw_instance.SaveState()
        saw_instance.LoadState()

    @pytest.mark.order(8)
    def test_base_run_script_2(self, saw_instance):
        saw_instance.RunScriptCommand2("LogAdd(\"Test\");", "Testing...")

    @pytest.mark.order(9)
    def test_base_field_list(self, saw_instance):
        df = saw_instance.GetFieldList("Bus")
        assert not df.empty
        
        df_spec = saw_instance.GetSpecificFieldList("Bus", ["BusNum", "BusName"])
        assert not df_spec.empty


class TestGeneralSAW:
    """Tests for general SAW operations."""

    @pytest.mark.order(95)
    def test_general_log(self, saw_instance, temp_file):
        saw_instance.LogAdd("SAW Validator Test Message")
        tmp_log = temp_file(".txt")
        saw_instance.LogSave(tmp_log)
        assert os.path.exists(tmp_log)

    @pytest.mark.order(96)
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

    @pytest.mark.order(97)
    def test_general_mode(self, saw_instance):
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    @pytest.mark.order(98)
    def test_general_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(tmp_aux, "AUX", "Bus", ["BusNum", "BusName"])
        saw_instance.LoadAux(tmp_aux)

    @pytest.mark.order(99)
    def test_general_select(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")


class TestOnelineSAW:
    """Tests for oneline diagram operations."""

    @pytest.mark.order(110)
    def test_oneline_ops(self, saw_instance, temp_file):
        saw_instance.CloseOneline()
        saw_instance.RelinkAllOpenOnelines()
        
        tmp_axd = temp_file(".axd")
        saw_instance.LoadAXD(tmp_axd, "TestOneline")


class TestModifySAW:
    """Tests for modify operations (destructive - run late, order 100-199)."""

    @pytest.mark.order(120)
    def test_modify_create_delete(self, saw_instance):
        dummy_bus = 99999
        saw_instance.CreateData("Bus", ["BusNum", "BusName"], [dummy_bus, "SAW_TEST"])
        saw_instance.Delete("Bus", f"BusNum = {dummy_bus}")

    @pytest.mark.order(121)
    def test_modify_auto(self, saw_instance):
        saw_instance.AutoInsertTieLineTransactions()

    @pytest.mark.order(122)
    def test_modify_branch(self, saw_instance):
        saw_instance.BranchMVALimitReorder()

    @pytest.mark.order(123)
    def test_modify_calc(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
        except PowerWorldAddonError:
            pytest.skip("TransLineCalc not registered")

    @pytest.mark.order(124)
    def test_modify_base(self, saw_instance):
        saw_instance.ChangeSystemMVABase(100.0)

    @pytest.mark.order(125)
    def test_modify_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    @pytest.mark.order(126)
    def test_modify_gen(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()
        saw_instance.SetGenPMaxFromReactiveCapabilityCurve()

    @pytest.mark.order(127)
    def test_modify_inj(self, saw_instance):
        saw_instance.InjectionGroupsAutoInsert()
        saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "")
        saw_instance.RenameInjectionGroup("TestIG", "TestIG_Renamed")

    @pytest.mark.order(128)
    def test_modify_interface(self, saw_instance):
        saw_instance.InterfacesAutoInsert("AREA")
        saw_instance.InterfaceCreate("TestInterface", True, "Branch", "SELECTED")
        saw_instance.SetInterfaceLimitToMonitoredElementLimitSum()

    @pytest.mark.order(129)
    def test_modify_reassign(self, saw_instance):
        saw_instance.ReassignIDs("Load", "BusName")

    @pytest.mark.order(130)
    def test_modify_remove(self, saw_instance):
        saw_instance.Remove3WXformerContainer()

    @pytest.mark.order(131)
    def test_modify_rotate(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.RotateBusAnglesInIsland(bus_str, 0.0)

    @pytest.mark.order(132)
    def test_modify_part(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    @pytest.mark.order(133)
    def test_modify_volt(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.SetScheduledVoltageForABus(bus_str, 1.0)

    @pytest.mark.order(134)
    def test_modify_superarea(self, saw_instance):
        saw_instance.CreateData("SuperArea", ["Name"], ["TestSuperArea"])
        saw_instance.SuperAreaAddAreas("TestSuperArea", "ALL")
        saw_instance.SuperAreaRemoveAreas("TestSuperArea", "ALL")

    @pytest.mark.order(135)
    def test_modify_extras(self, saw_instance):
        saw_instance.InjectionGroupRemoveDuplicates()
        saw_instance.InterfaceRemoveDuplicates()
        saw_instance.DirectionsAutoInsertReference("Bus", "Slack")
        
        saw_instance.InterfaceCreate("TestInt", True, "Branch", "SELECTED")
        saw_instance.InterfaceFlatten("TestInt")
        saw_instance.InterfaceFlattenFilter("ALL")
        saw_instance.InterfaceModifyIsolatedElements()
        
        saw_instance.CreateData("Contingency", ["Name"], ["TestCtg"])
        saw_instance.InterfaceAddElementsFromContingency("TestInt", "TestCtg")


class TestRegionsSAW:
    """Tests for regions operations (destructive - run late, order 200-299)."""

    @pytest.mark.order(200)
    def test_regions_update(self, saw_instance):
        saw_instance.RegionUpdateBuses()

    @pytest.mark.order(201)
    def test_regions_rename(self, saw_instance):
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()


class TestCaseActionsSAW:
    """Tests for case actions (highly destructive - run last, order 300+)."""

    @pytest.mark.order(300)
    def test_case_description(self, saw_instance):
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()

    @pytest.mark.order(301)
    def test_case_delete_external(self, saw_instance):
        saw_instance.DeleteExternalSystem()

    @pytest.mark.order(302)
    def test_case_equivalence(self, saw_instance):
        saw_instance.Equivalence()

    @pytest.mark.order(303)
    def test_case_save_external(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)

    @pytest.mark.order(304)
    def test_case_save_merged(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)

    @pytest.mark.order(305)
    def test_case_scale(self, saw_instance):
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")

    @pytest.mark.order(999)
    def test_case_renumber(self, saw_instance):
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
