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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

from esapp.saw import PowerWorldError, PowerWorldAddonError, create_object_string
from conftest import ensure_areas


@pytest.fixture
def saw_instance(live_case):
    return live_case


class TestModify:
    """Modify operations on a fresh case for each test."""

    def test_create_delete(self, saw_instance):
        """CreateData and Delete cycle for a bus."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "AreaNum", "ZoneNum"])
        dummy_bus = int(buses["BusNum"].astype(int).max()) + 1
        saw_instance.CreateData(
            "Bus",
            ["BusNum", "BusName", "BusNomVolt", "AreaNum", "ZoneNum"],
            [dummy_bus, "SAW_TEST", 115, int(buses.iloc[0]["AreaNum"]), int(buses.iloc[0]["ZoneNum"])],
        )
        created = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert set(created["BusNum"].astype(int)) == set(buses["BusNum"].astype(int)) | {dummy_bus}
        saw_instance.UnSelectAll("Bus")
        saw_instance.ChangeParametersSingleElement("Bus", ["BusNum", "Selected"], [dummy_bus, "YES"])
        saw_instance.Delete("Bus", "SELECTED")
        remaining = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert set(remaining["BusNum"].astype(int)) == set(buses["BusNum"].astype(int))

    def test_superarea(self, saw_instance):
        """SuperArea create, add areas, remove areas cycle."""
        saw_instance.CreateData("SuperArea", ["Name"], ["TestSuperArea"])
        saw_instance.SuperAreaAddAreas("TestSuperArea", "ALL")
        saw_instance.SuperAreaRemoveAreas("TestSuperArea", "ALL")

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
        saw_instance.CreateLineDeriveExisting(
            int(b["BusNum"]), int(b["BusNum:1"]), "99",
            10.0, branch_id, existing_length=5.0, zero_g=True,
        )

    def test_merge_buses(self, saw_instance):
        """MergeBuses completes without error."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_num = str(buses.iloc[0]["BusNum"]).strip()
        bus_str = create_object_string("Bus", bus_num)
        saw_instance.SetData("Bus", ["BusNum", "Selected"], [bus_num, "YES"])
        saw_instance.MergeBuses(bus_str, filter_name="SELECTED")

    @pytest.mark.parametrize("percent", [25.0, 100.0], ids=["partial", "whole"])
    def test_move(self, radial_case, percent):
        """Transfer real load and verify its location and conserved MW/Mvar."""
        saw_instance = radial_case
        source, target = 1, 2
        fields = ["BusNum", "LoadID", "LoadMW", "LoadMVR"]
        saw_instance.CreateData(
            "Load", ["BusNum", "LoadID", "LoadSMW", "LoadSMVR", "LoadStatus"],
            [source, "1", 40.0, 16.0, "Closed"],
        )
        before = saw_instance.GetParametersMultipleElement("Load", fields).astype(
            {"BusNum": int, "LoadMW": float, "LoadMVR": float}
        )
        initial = before[before["BusNum"] == source]
        assert len(initial) == 1, "Transfer load was not created"
        assert initial.iloc[0]["LoadMW"] == pytest.approx(40.0)
        assert initial.iloc[0]["LoadMVR"] == pytest.approx(16.0)

        saw_instance.Move(
            create_object_string("Load", source, "1"),
            f"[{target} 1]", how_much=percent,
        )

        loads = saw_instance.GetParametersMultipleElement("Load", fields).astype(
            {"BusNum": int, "LoadMW": float, "LoadMVR": float}
        )
        moved = loads[loads["BusNum"] == target]
        assert len(moved) == 1
        assert moved.iloc[0]["LoadID"].strip() == "1"
        fraction = percent / 100.0
        assert moved.iloc[0]["LoadMW"] == pytest.approx(40.0 * fraction)
        assert moved.iloc[0]["LoadMVR"] == pytest.approx(16.0 * fraction)
        remaining = loads[loads["BusNum"] == source]
        if percent == 100.0:
            assert remaining.empty
        else:
            assert len(remaining) == 1
            assert remaining.iloc[0]["LoadMW"] == pytest.approx(40.0 * (1 - fraction))
            assert remaining.iloc[0]["LoadMVR"] == pytest.approx(16.0 * (1 - fraction))

    def test_split_bus(self, saw_instance):
        """SplitBus creates a new bus from an existing one."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.SplitBus(bus_key, 99997, insert_tie=True, line_open=False)

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
        saw_instance.TapTransmissionLine(
            branch_key, 50.0, 99996,
            shunt_model="CAPACITANCE",
            treat_as_ms_line=False,
            update_onelines=False,
            new_bus_name="TapBus",
        )

    def test_branch_mva_limit_with_limits(self, saw_instance):
        """BranchMVALimitReorder with explicit limits list."""
        saw_instance.BranchMVALimitReorder(filter_name="ALL", limits=["A", "B", "C"])

    def test_modify_auto_insert_tieline(self, saw_instance):
        saw_instance.AutoInsertTieLineTransactions()

    def test_modify_branch_mva_limit_reorder(self, saw_instance):
        saw_instance.BranchMVALimitReorder()

    def test_modify_branch_mva_limit_reorder_with_filter(self, saw_instance):
        saw_instance.BranchMVALimitReorder(filter_name="ALL")

    def test_modify_calculate_rxbg(self, saw_instance):
        """CalculateRXBGFromLengthConfigCondType with and without filter."""
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
            saw_instance.CalculateRXBGFromLengthConfigCondType(filter_name="SELECTED")
        except PowerWorldAddonError:
            pytest.skip("TransLineCalc add-on not registered")

    def test_modify_clear_small_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    def test_modify_init_gen_mvar_limits(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()

    def test_modify_injection_groups_auto_insert(self, saw_instance):
        saw_instance.InjectionGroupsAutoInsert()

    def test_modify_injection_group_create(self, saw_instance):
        saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "", append=True)

    def test_modify_injection_group_create_no_append(self, saw_instance):
        saw_instance.InjectionGroupCreate("TestIG2", "Gen", 1.0, "", append=False)

    def test_modify_interfaces_auto_insert(self, saw_instance):
        """InterfacesAutoInsert with and without filters."""
        saw_instance.InterfacesAutoInsert("AREA", delete_existing=True, use_filters=False)
        saw_instance.InterfacesAutoInsert("AREA", delete_existing=False, use_filters=True, prefix="TEST_")

    def test_modify_set_participation_factors(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    def test_modify_set_scheduled_voltage(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.SetScheduledVoltageForABus(bus_key, 1.0)

    def test_modify_set_interface_limit_sum(self, saw_instance):
        saw_instance.SetInterfaceLimitToMonitoredElementLimitSum("ALL")

    def test_modify_rotate_bus_angles(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.RotateBusAnglesInIsland(bus_key, 0.0)

    def test_modify_set_gen_pmax(self, saw_instance):
        saw_instance.SetGenPMaxFromReactiveCapabilityCurve()

    def test_modify_remove_3w_xformer(self, saw_instance):
        saw_instance.Remove3WXformerContainer()

    def test_modify_rename_injection_group(self, saw_instance):
        saw_instance.InjectionGroupCreate("RenameTestIG", "Gen", 1.0, "")
        saw_instance.RenameInjectionGroup("RenameTestIG", "RenamedIG")

    def test_modify_reassign_ids(self, saw_instance):
        """ReassignIDs with and without use_right."""
        saw_instance.ReassignIDs("Load", "BusName", filter_name="", use_right=False)
        saw_instance.ReassignIDs("Load", "BusName", filter_name="ALL", use_right=True)

    def test_modify_merge_line_terminals(self, radial_case):
        radial_case.SelectAll("Branch")
        radial_case.MergeLineTerminals("SELECTED")
        buses = radial_case.GetParametersMultipleElement("Bus", ["BusNum"])
        assert len(buses) == 1
        assert radial_case.ListOfDevices("Branch") is None

    def test_modify_merge_ms_line_sections(self, radial_case):
        radial_case.TapTransmissionLine(
            "[BRANCH 1 2 1]", 50.0, 3, treat_as_ms_line=True, update_onelines=False,
        )
        assert len(radial_case.ListOfDevices("Bus")) == 3
        assert len(radial_case.ListOfDevices("Branch")) == 2
        assert len(radial_case.ListOfDevices("MultiSectionLine")) == 1
        radial_case.SelectAll("MultiSectionLine")

        radial_case.MergeMSLineSections("SELECTED")

        buses = radial_case.GetParametersMultipleElement("Bus", ["BusNum"])
        assert set(buses["BusNum"].astype(int)) == {1, 2}
        assert radial_case.ListOfDevices("MultiSectionLine") is None
        branches = radial_case.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineR", "LineX"]
        )
        assert len(branches) == 1
        assert tuple(branches.iloc[0][["BusNum", "BusNum:1"]].astype(int)) == (1, 2)
        assert float(branches.iloc[0]["LineR"]) == pytest.approx(0.02)
        assert float(branches.iloc[0]["LineX"]) == pytest.approx(0.2)

    @pytest.mark.parametrize("delete_existing,use_filters", [(True, False), (False, True)])
    def test_modify_directions_auto_insert(self, saw_instance, delete_existing, use_filters):
        areas = ensure_areas(saw_instance)
        assert len(areas) >= 2
        saw_instance.Delete("Direction")
        saw_instance.DirectionsAutoInsert(
            "AREA", "AREA", delete_existing=delete_existing,
            use_area_zone_filters=use_filters,
        )
        directions = saw_instance.GetParametersMultipleElement(
            "Direction", ["DirName", "DirSource", "DirSink"]
        )
        assert directions is not None and not directions.empty
        assert len(directions) == len(areas) * (len(areas) - 1) // 2
        assert (directions["DirSource"] != directions["DirSink"]).all()
        expected = {f"AREA '{int(number)}'" for number in areas["AreaNum"]}
        actual = set(directions["DirSource"].str.upper()) | set(directions["DirSink"].str.upper())
        assert actual == expected

    def test_modify_change_system_mva_base(self, saw_instance):
        saw_instance.ChangeSystemMVABase(100.0)


class TestRegions:
    """Tests for region operations."""

    def test_region_update_buses(self, saw_instance):
        """RegionUpdateBuses completes without error."""
        saw_instance.RegionUpdateBuses()

    def test_region_rename(self, saw_instance):
        """Region rename operations complete without error."""
        saw_instance.RegionRename("OldRegion", "NewRegion")
        saw_instance.RegionRenameClass("OldClass", "NewClass")
        saw_instance.RegionRenameProper1("OldP1", "NewP1")
        saw_instance.RegionRenameProper2("OldP2", "NewP2")
        saw_instance.RegionRenameProper3("OldP3", "NewP3")
        saw_instance.RegionRenameProper12Flip()

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
    """Case actions on disposable cases."""

    def test_case_description(self, saw_instance):
        """CaseDescriptionSet, append, and clear."""
        saw_instance.CaseDescriptionSet("Test Description")
        saw_instance.CaseDescriptionClear()
        saw_instance.CaseDescriptionSet("Line 1")
        saw_instance.CaseDescriptionSet("Line 2", append=True)
        saw_instance.CaseDescriptionClear()

    def test_equivalence_and_external_system(self, saw_instance, temp_file):
        """External system operations, equivalence, and save with ties."""
        saw_instance.DeleteExternalSystem()
        saw_instance.Equivalence()
        tmp_pwb = temp_file(".pwb")
        saw_instance.SaveExternalSystem(tmp_pwb)
        saw_instance.SaveMergedFixedNumBusCase(tmp_pwb)
        saw_instance.SaveExternalSystem(tmp_pwb, with_ties=True)

    def test_scale(self, saw_instance):
        """Scale load, gen, and load MW."""
        saw_instance.Scale("LOAD", "FACTOR", [1.0], "SYSTEM")
        saw_instance.Scale("GEN", "FACTOR", [1.0], "SYSTEM")
        saw_instance.Scale("LOAD", "MW", [100.0, 50.0], "SYSTEM")

    def test_write_text_to_file(self, saw_instance, temp_file):
        """WriteTextToFile creates a file with content."""
        tmp_txt = temp_file(".txt")
        saw_instance.WriteTextToFile(tmp_txt, "Test content")
        assert os.path.exists(tmp_txt)

    def test_case_load_ems(self, saw_instance, temp_file):
        tmp = temp_file(".hdb")
        with pytest.raises(PowerWorldError):
            saw_instance.LoadEMS(tmp)

    def test_renumber(self, saw_instance):
        """Renumber operations including custom index."""
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
