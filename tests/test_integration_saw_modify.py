"""
Integration tests for SAW modify, region, and case action operations.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test destructive modify operations
(create/delete objects, merge, split, topology changes), region operations,
and case-level actions (equivalence, renumber, scale, description).

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_core.py          -- base SAW operations, logging, I/O
    - test_integration_saw_powerflow.py     -- power flow, matrices, sensitivity, topology
    - test_integration_saw_contingency.py   -- contingency and fault analysis
    - test_integration_saw_gic.py           -- GIC analysis
    - test_integration_saw_transient.py     -- transient stability
    - test_integration_saw_operations.py    -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py         -- PowerWorld facade and statics
    - test_integration_network.py           -- Network topology

USAGE:
    pytest tests/test_integration_saw_modify.py -v
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


@pytest.mark.usefixtures("save_restore_state")
class TestModify:
    """Tests for modify operations (destructive - run late)."""

    @pytest.mark.order(12000)
    def test_create_delete(self, saw_instance):
        """CreateData and Delete cycle for a bus."""
        dummy_bus = 99999
        saw_instance.CreateData(
            "Bus",
            ["BusNum", "BusName", "BusNomVolt"],
            [dummy_bus, "SAW_TEST", 115]
        )
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

    @pytest.mark.order(14000)
    def test_create_line_derive_existing(self, saw_instance):
        """CreateLineDeriveExisting creates a line from existing parameters."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit", "BranchDeviceType"]
        )
        assert branches is not None and not branches.empty, "Test case must contain branches"
        lines = branches[branches["BranchDeviceType"] == "Line"]
        if lines.empty:
            pytest.skip("No line branches available for CreateLineDeriveExisting test")
        b = lines.iloc[0]
        branch_id = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.SaveState()
        try:
            saw_instance.CreateLineDeriveExisting(
                int(b["BusNum"]), int(b["BusNum:1"]), "99",
                10.0, branch_id, existing_length=5.0, zero_g=True,
            )
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(14100)
    def test_merge_buses(self, saw_instance):
        """MergeBuses completes without error."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_num = str(buses.iloc[0]["BusNum"]).strip()
        bus_str = create_object_string("Bus", bus_num)
        saw_instance.SaveState()
        try:
            saw_instance.SetData("Bus", ["BusNum", "Selected"], [bus_num, "YES"])
            saw_instance.MergeBuses(bus_str, filter_name="SELECTED")
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(14200)
    def test_move(self, saw_instance):
        """Move a switched shunt (0% -- no-op)."""
        shunts = saw_instance.GetParametersMultipleElement("Shunt", ["BusNum", "ShuntID"])
        if shunts is None or shunts.empty:
            pytest.skip("No switched shunts found for Move test")
        shunt_key = create_object_string("Shunt", shunts.iloc[0]["BusNum"], shunts.iloc[0]["ShuntID"])
        bus_key = create_object_string("Bus", shunts.iloc[0]["BusNum"])
        saw_instance.SaveState()
        try:
            saw_instance.Move(shunt_key, bus_key, how_much=0.0, abort_on_error=True)
        except PowerWorldError as e:
            if "Unknown object" in str(e) or "not supported" in str(e).lower():
                pytest.skip(f"Move not supported for this object type on this case: {e}")
            raise
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(14300)
    def test_split_bus(self, saw_instance):
        """SplitBus creates a new bus from an existing one."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.SaveState()
        try:
            saw_instance.SplitBus(bus_key, 99997, insert_tie=True, line_open=False)
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(14400)
    def test_tap_transmission_line(self, saw_instance):
        """TapTransmissionLine taps a line at midpoint (lines only)."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit", "BranchDeviceType"]
        )
        assert branches is not None and not branches.empty, "Test case must contain branches"
        lines = branches[branches["BranchDeviceType"] == "Line"]
        if lines.empty:
            pytest.skip("No line branches available for TapTransmissionLine test")
        b = lines.iloc[0]
        branch_key = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.SaveState()
        try:
            saw_instance.TapTransmissionLine(
                branch_key, 50.0, 99996,
                shunt_model="CAPACITANCE",
                treat_as_ms_line=False,
                update_onelines=False,
                new_bus_name="TapBus",
            )
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(14500)
    def test_branch_mva_limit_with_limits(self, saw_instance):
        """BranchMVALimitReorder with explicit limits list."""
        saw_instance.SaveState()
        try:
            saw_instance.BranchMVALimitReorder(
                filter_name="ALL",
                limits=["A", "B", "C"],
            )
        finally:
            saw_instance.LoadState()

    @pytest.mark.order(80000)
    def test_modify_auto_insert_tieline(self, saw_instance):
        saw_instance.AutoInsertTieLineTransactions()

    @pytest.mark.order(80100)
    def test_modify_branch_mva_limit_reorder(self, saw_instance):
        saw_instance.BranchMVALimitReorder()

    @pytest.mark.order(80200)
    def test_modify_branch_mva_limit_reorder_with_filter(self, saw_instance):
        saw_instance.BranchMVALimitReorder(filter_name="ALL")

    @pytest.mark.order(80300)
    def test_modify_calculate_rxbg(self, saw_instance):
        """CalculateRXBGFromLengthConfigCondType with and without filter."""
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
            saw_instance.CalculateRXBGFromLengthConfigCondType(filter_name="SELECTED")
        except PowerWorldAddonError:
            pytest.skip("TransLineCalc add-on not registered")

    @pytest.mark.order(80500)
    def test_modify_clear_small_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    @pytest.mark.order(80600)
    def test_modify_init_gen_mvar_limits(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()

    @pytest.mark.order(80700)
    def test_modify_injection_groups_auto_insert(self, saw_instance):
        saw_instance.InjectionGroupsAutoInsert()

    @pytest.mark.order(80800)
    def test_modify_injection_group_create(self, saw_instance):
        saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "", append=True)

    @pytest.mark.order(80900)
    def test_modify_injection_group_create_no_append(self, saw_instance):
        saw_instance.InjectionGroupCreate("TestIG2", "Gen", 1.0, "", append=False)

    @pytest.mark.order(81000)
    def test_modify_interfaces_auto_insert(self, saw_instance):
        """InterfacesAutoInsert with and without filters."""
        saw_instance.InterfacesAutoInsert("AREA", delete_existing=True, use_filters=False)
        saw_instance.InterfacesAutoInsert("AREA", delete_existing=False, use_filters=True, prefix="TEST_")

    @pytest.mark.order(81200)
    def test_modify_set_participation_factors(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    @pytest.mark.order(81300)
    def test_modify_set_scheduled_voltage(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.SetScheduledVoltageForABus(bus_key, 1.0)

    @pytest.mark.order(81400)
    def test_modify_set_interface_limit_sum(self, saw_instance):
        saw_instance.SetInterfaceLimitToMonitoredElementLimitSum("ALL")

    @pytest.mark.order(81500)
    def test_modify_rotate_bus_angles(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.RotateBusAnglesInIsland(bus_key, 0.0)

    @pytest.mark.order(81600)
    def test_modify_set_gen_pmax(self, saw_instance):
        saw_instance.SetGenPMaxFromReactiveCapabilityCurve()

    @pytest.mark.order(81700)
    def test_modify_remove_3w_xformer(self, saw_instance):
        saw_instance.Remove3WXformerContainer()

    @pytest.mark.order(81800)
    def test_modify_rename_injection_group(self, saw_instance):
        saw_instance.InjectionGroupCreate("RenameTestIG", "Gen", 1.0, "")
        saw_instance.RenameInjectionGroup("RenameTestIG", "RenamedIG")

    @pytest.mark.order(81900)
    def test_modify_reassign_ids(self, saw_instance):
        """ReassignIDs with and without use_right."""
        saw_instance.ReassignIDs("Load", "BusName", filter_name="", use_right=False)
        saw_instance.ReassignIDs("Load", "BusName", filter_name="ALL", use_right=True)

    @pytest.mark.order(82100)
    @pytest.mark.skip(reason="MergeLineTerminals causes PW access violation that kills COM server")
    def test_modify_merge_line_terminals(self, saw_instance):
        saw_instance.MergeLineTerminals("SELECTED")

    @pytest.mark.order(82200)
    @pytest.mark.skip(reason="MergeMSLineSections causes PW access violation that kills COM server")
    def test_modify_merge_ms_line_sections(self, saw_instance):
        saw_instance.MergeMSLineSections("SELECTED")

    @pytest.mark.order(82300)
    def test_modify_directions_auto_insert(self, saw_instance):
        """DirectionsAutoInsert with multiple parameter combinations."""
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("DirectionsAutoInsert requires a case with at least 2 areas")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        saw_instance.DirectionsAutoInsert(s, b, delete_existing=True, use_area_zone_filters=False)
        saw_instance.DirectionsAutoInsert(s, b, delete_existing=False, use_area_zone_filters=True)
        saw_instance.DirectionsAutoInsertReference("Bus", "Slack", delete_existing=True, opposite_direction=True)

    @pytest.mark.order(82600)
    def test_modify_change_system_mva_base(self, saw_instance):
        saw_instance.ChangeSystemMVABase(100.0)


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

    @pytest.mark.order(85000)
    def test_region_load_shapefile(self, saw_instance, temp_file):
        """RegionLoadShapefile completes without error."""
        tmp = temp_file(".shp")
        saw_instance.RegionLoadShapefile(
            tmp, "TestClass", ["Name"],
            add_to_open_onelines=False,
            display_style_name="",
            delete_existing=True,
        )


class TestCaseActions:
    """Tests for case actions (highly destructive - run last)."""

    @pytest.mark.order(30000)
    def test_case_description(self, saw_instance):
        """CaseDescriptionSet, append, and clear."""
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()
        saw_instance.CaseDescriptionSet("Line 1")
        saw_instance.CaseDescriptionSet("Line 2", append=True)
        saw_instance.CaseDescriptionClear()

    @pytest.mark.order(30100)
    def test_equivalence_and_external_system(self, saw_instance, temp_file):
        """External system operations, equivalence, and save with ties."""
        saw_instance.DeleteExternalSystem()
        saw_instance.Equivalence()
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)
        saw_instance.SaveExternalSystem(tmp_pwb, with_ties=True)

    @pytest.mark.order(30150)
    def test_scale(self, saw_instance):
        """Scale load, gen, and load MW."""
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")
        saw_instance.Scale("GEN", "FACTOR", [1.0], "SYSTEM")
        saw_instance.Scale("LOAD", "MW", [100.0, 50.0], "SYSTEM")

    @pytest.mark.order(30160)
    def test_write_text_to_file(self, saw_instance, temp_file):
        """WriteTextToFile creates a file with content."""
        tmp_txt = temp_file(".txt")
        saw_instance.WriteTextToFile(tmp_txt, "Test content")
        assert os.path.exists(tmp_txt)

    @pytest.mark.order(90400)
    def test_case_load_ems(self, saw_instance, temp_file):
        tmp = temp_file(".hdb")
        with pytest.raises(PowerWorldError):
            saw_instance.LoadEMS(tmp)

    @pytest.mark.order(99900)
    def test_renumber(self, saw_instance):
        """Renumber operations including custom index (run last as they modify keys)."""
        saw_instance.RenumberAreas()
        saw_instance.RenumberBuses()
        saw_instance.RenumberSubs()
        saw_instance.RenumberZones()
        saw_instance.RenumberCase()
        saw_instance.RenumberAreas(custom_integer_index=1)
        saw_instance.RenumberBuses(custom_integer_index=2)
        saw_instance.RenumberSubs(custom_integer_index=3)
        saw_instance.RenumberZones(custom_integer_index=4)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
