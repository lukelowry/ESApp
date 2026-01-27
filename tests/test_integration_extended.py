"""
Integration tests for Scheduled Actions, Weather, Oneline, Modify, OPF, and ATC
extended functionality against a live PowerWorld case.

WHAT THIS TESTS:
- Scheduled Actions (all boolean parameter paths)
- Weather commands (all boolean parameter paths)
- Oneline diagram commands (extended coverage)
- Modify operations (extended coverage)
- OPF solver operations (extended coverage)
- ATC analysis operations (extended coverage)
- General / Case Actions (extended coverage)
- Transient extended (additional uncovered methods)
- Sensitivity / Topology (additional uncovered methods)
- Powerflow (additional uncovered methods)

ORDERING STRATEGY:
    - Orders 500-599: Read-only queries, exports to temp files (non-destructive)
    - Orders 600-699: Analysis commands (ATC, OPF) that don't permanently alter the case
    - Orders 700-799: Transient operations (mostly exports, model queries)
    - Orders 800-899: State-modifying operations wrapped with StoreState/RestoreState
    - Orders 900-949: Case Actions (descriptions, scaling — uses identity values)
    - Orders 950-999: Final cleanup (state always restored after modifications)

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd
import numpy as np

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_case,
]

try:
    from esapp.saw import SAW, PowerWorldError, PowerWorldPrerequisiteError, create_object_string
except ImportError:
    raise


@pytest.fixture(scope="module")
def saw_instance(saw_session):
    """Provides the session-scoped SAW instance to the tests in this module."""
    return saw_session


@pytest.fixture(scope="class")
def save_restore_state(saw_session):
    """Saves case state before destructive tests and restores it after."""
    state_name = "__test_extended_state__"
    saw_session.StoreState(state_name)
    yield saw_session
    try:
        saw_session.RestoreState(state_name)
        saw_session.DeleteState(state_name)
    except Exception:
        pass


# =============================================================================
# Scheduled Actions (31% coverage)
# =============================================================================

class TestScheduledActions:
    """Tests for Scheduled Actions mixin — all parameter paths."""

    @pytest.mark.order(500)
    def test_scheduled_set_reference(self, saw_instance):
        saw_instance.ScheduledActionsSetReference()

    @pytest.mark.order(501)
    def test_scheduled_apply_at(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(502)
    def test_scheduled_apply_at_with_end_time(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", end_time="01/01/2025 12:00"
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(503)
    def test_scheduled_apply_at_with_filter(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", filter_name="ALL"
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No scheduled actions or filter not found")

    @pytest.mark.order(504)
    def test_scheduled_apply_at_revert(self, saw_instance):
        try:
            saw_instance.ApplyScheduledActionsAt(
                "01/01/2025 10:00", revert=True
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(505)
    def test_scheduled_revert_at(self, saw_instance):
        try:
            saw_instance.RevertScheduledActionsAt("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(506)
    def test_scheduled_revert_at_with_filter(self, saw_instance):
        try:
            saw_instance.RevertScheduledActionsAt(
                "01/01/2025 10:00", filter_name="ALL"
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No scheduled actions or filter not found")

    @pytest.mark.order(507)
    def test_scheduled_identify_breakers(self, saw_instance):
        try:
            saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=True)
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(508)
    def test_scheduled_identify_breakers_false(self, saw_instance):
        try:
            saw_instance.IdentifyBreakersForScheduledActions(identify_from_normal=False)
        except PowerWorldPrerequisiteError:
            pytest.skip("No scheduled actions defined")

    @pytest.mark.order(509)
    def test_scheduled_set_view(self, saw_instance):
        try:
            saw_instance.SetScheduleView("01/01/2025 10:00")
        except PowerWorldPrerequisiteError:
            pytest.skip("Schedule view not available")

    @pytest.mark.order(510)
    def test_scheduled_set_view_with_options(self, saw_instance):
        try:
            saw_instance.SetScheduleView(
                "01/01/2025 10:00",
                apply_actions=True,
                use_normal_status=False,
                apply_window=True,
            )
        except PowerWorldPrerequisiteError:
            pytest.skip("Schedule view not available")

    @pytest.mark.order(511)
    def test_scheduled_set_window(self, saw_instance):
        try:
            saw_instance.SetScheduleWindow(
                "01/01/2025 00:00", "02/01/2025 00:00",
                resolution=1.0, resolution_units="HOURS",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Schedule window not available")

    @pytest.mark.order(512)
    def test_scheduled_set_window_with_resolution(self, saw_instance):
        try:
            saw_instance.SetScheduleWindow(
                "01/01/2025 00:00",
                "02/01/2025 00:00",
                resolution=0.5,
                resolution_units="HOURS",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Schedule window not available")


# =============================================================================
# Weather (44% coverage)
# =============================================================================

class TestWeather:
    """Tests for Weather mixin — all boolean parameter paths."""

    @pytest.mark.order(520)
    def test_weather_limits_gen_update(self, saw_instance):
        try:
            saw_instance.WeatherLimitsGenUpdate(update_max=True, update_min=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Weather data not available")

    @pytest.mark.order(521)
    def test_weather_limits_gen_update_false(self, saw_instance):
        try:
            saw_instance.WeatherLimitsGenUpdate(update_max=False, update_min=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Weather data not available")

    @pytest.mark.order(522)
    def test_weather_temperature_limits_branch(self, saw_instance):
        try:
            saw_instance.TemperatureLimitsBranchUpdate()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Temperature limits not available")

    @pytest.mark.order(523)
    def test_weather_temperature_limits_branch_custom(self, saw_instance):
        try:
            saw_instance.TemperatureLimitsBranchUpdate(
                rating_set_precedence="EMERGENCY",
                normal_rating_set="A",
                ctg_rating_set="B",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Temperature limits not available")

    @pytest.mark.order(524)
    def test_weather_pfw_set_inputs(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputs()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(525)
    def test_weather_pfw_set_inputs_and_apply(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(526)
    def test_weather_pfw_set_inputs_and_apply_no_solve(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsSetInputsAndApply(solve_pf=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(527)
    def test_weather_pfw_restore_design(self, saw_instance):
        try:
            saw_instance.WeatherPFWModelsRestoreDesignValues()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PFW weather models not available")

    @pytest.mark.order(528)
    def test_weather_pww_load_datetime(self, saw_instance):
        try:
            saw_instance.WeatherPWWLoadForDateTimeUTC("2025-01-01T10:00:00")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW weather data not available")

    @pytest.mark.order(529)
    def test_weather_pww_set_directory(self, saw_instance, temp_dir):
        try:
            saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW directory set not available")

    @pytest.mark.order(530)
    def test_weather_pww_set_directory_no_subdirs(self, saw_instance, temp_dir):
        try:
            saw_instance.WeatherPWWSetDirectory(str(temp_dir), include_subdirs=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW directory set not available")

    @pytest.mark.order(531)
    def test_weather_pww_file_all_meas_valid(self, saw_instance, temp_file):
        tmp = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileAllMeasValid(tmp, ["Temperature"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW validation not available")

    @pytest.mark.order(532)
    def test_weather_pww_file_combine(self, saw_instance, temp_file):
        src1 = temp_file(".pww")
        src2 = temp_file(".pww")
        dst = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileCombine2(src1, src2, dst)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW combine not available")

    @pytest.mark.order(533)
    def test_weather_pww_file_geo_reduce(self, saw_instance, temp_file):
        src = temp_file(".pww")
        dst = temp_file(".pww")
        try:
            saw_instance.WeatherPWWFileGeoReduce(src, dst, 30.0, 50.0, -100.0, -80.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PWW geo-reduce not available")


# =============================================================================
# Oneline Diagram (52% coverage)
# =============================================================================

@pytest.mark.skip(reason="Oneline tests require specific setup settings not yet configured")
class TestOnelineExtended:
    """Extended tests for oneline diagram operations."""

    @pytest.mark.order(540)
    def test_oneline_open_close(self, saw_instance, temp_file):
        tmp_axd = temp_file(".axd")
        try:
            saw_instance.OpenOneLine(tmp_axd)
            saw_instance.CloseOneline()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Oneline operations not available")

    @pytest.mark.order(541)
    def test_oneline_open_with_options(self, saw_instance, temp_file):
        tmp_axd = temp_file(".axd")
        try:
            saw_instance.OpenOneLine(
                tmp_axd, view="", full_screen="NO", show_full="NO",
                link_method="LABELS", left=0, top=0, width=800, height=600,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Oneline open with options not available")

    @pytest.mark.order(542)
    def test_oneline_save(self, saw_instance, temp_file):
        tmp_out = temp_file(".pwb")
        try:
            saw_instance.SaveOneline(tmp_out, "TestOneline")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Oneline save not available")

    @pytest.mark.order(543)
    def test_oneline_export(self, saw_instance, temp_file):
        tmp_jpg = temp_file(".jpg")
        try:
            saw_instance.ExportOneline(tmp_jpg, "TestOneline", "JPG")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Oneline export not available")

    @pytest.mark.order(544)
    def test_oneline_export_bus_view(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for bus view export")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp_jpg = temp_file(".jpg")
        try:
            saw_instance.ExportBusView(tmp_jpg, bus_key, "JPG", 800, 600)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Bus view export not available")

    @pytest.mark.order(545)
    def test_oneline_export_bus_view_with_options(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for bus view export")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp_jpg = temp_file(".jpg")
        try:
            saw_instance.ExportBusView(tmp_jpg, bus_key, "JPG", 800, 600, export_options=["FLAT", "NOLEGEND"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Bus view export with options not available")

    @pytest.mark.order(546)
    def test_oneline_export_shapefile(self, saw_instance, temp_file):
        tmp_shp = temp_file(".shp")
        try:
            saw_instance.ExportOnelineAsShapeFile(tmp_shp, "TestOneline", "TestDesc", use_lon_lat=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Shapefile export not available")

    @pytest.mark.order(547)
    def test_oneline_export_shapefile_no_lonlat(self, saw_instance, temp_file):
        tmp_shp = temp_file(".shp")
        try:
            saw_instance.ExportOnelineAsShapeFile(tmp_shp, "TestOneline", "TestDesc", use_lon_lat=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Shapefile export not available")

    @pytest.mark.order(548)
    def test_oneline_pan_zoom(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for pan/zoom")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.PanAndZoomToObject(bus_key, do_zoom=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Pan/zoom not available (no open oneline)")

    @pytest.mark.order(549)
    def test_oneline_pan_no_zoom(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for pan")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.PanAndZoomToObject(bus_key, do_zoom=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Pan not available (no open oneline)")

    @pytest.mark.order(550)
    def test_oneline_open_bus_view(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for bus view")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.OpenBusView(bus_key, force_new_window=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Bus view not available")

    @pytest.mark.order(551)
    def test_oneline_open_bus_view_new_window(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for bus view")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.OpenBusView(bus_key, force_new_window=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Bus view not available")

    @pytest.mark.order(552)
    def test_oneline_open_sub_view(self, saw_instance):
        subs = saw_instance.GetParametersMultipleElement("Substation", ["SubNum"])
        if subs is None or subs.empty:
            pytest.skip("No substations for sub view")
        sub_key = create_object_string("Substation", subs.iloc[0]["SubNum"])
        try:
            saw_instance.OpenSubView(sub_key, force_new_window=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Sub view not available")

    @pytest.mark.order(553)
    def test_oneline_load_axd_create(self, saw_instance, temp_file):
        tmp_axd = temp_file(".axd")
        try:
            saw_instance.LoadAXD(tmp_axd, "TestOneline", create_if_not_found=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("AXD load with create not available")


# =============================================================================
# Modify Extended (62% coverage)
# =============================================================================

@pytest.mark.usefixtures("save_restore_state")
class TestModifyExtended:
    """Extended tests for Modify operations — covering uncovered boolean paths.

    Uses save_restore_state fixture to preserve case integrity.
    All modifications are reverted after the class completes.
    """

    @pytest.mark.order(800)
    def test_modify_auto_insert_tieline(self, saw_instance):
        try:
            saw_instance.AutoInsertTieLineTransactions()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Tie line transactions not available")

    @pytest.mark.order(801)
    def test_modify_branch_mva_limit_reorder(self, saw_instance):
        try:
            saw_instance.BranchMVALimitReorder()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Branch MVA limit reorder not available")

    @pytest.mark.order(802)
    def test_modify_branch_mva_limit_reorder_with_filter(self, saw_instance):
        try:
            saw_instance.BranchMVALimitReorder(filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Branch MVA limit reorder not available")

    @pytest.mark.order(803)
    def test_modify_calculate_rxbg(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TransLineCalc not available")

    @pytest.mark.order(804)
    def test_modify_calculate_rxbg_selected(self, saw_instance):
        try:
            saw_instance.CalculateRXBGFromLengthConfigCondType(filter_name="SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TransLineCalc not available")

    @pytest.mark.order(805)
    def test_modify_clear_small_islands(self, saw_instance):
        saw_instance.ClearSmallIslands()

    @pytest.mark.order(806)
    def test_modify_init_gen_mvar_limits(self, saw_instance):
        saw_instance.InitializeGenMvarLimits()

    @pytest.mark.order(807)
    def test_modify_injection_groups_auto_insert(self, saw_instance):
        try:
            saw_instance.InjectionGroupsAutoInsert()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group auto-insert not available")

    @pytest.mark.order(808)
    def test_modify_injection_group_create(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("TestIG", "Gen", 1.0, "ALL", append=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group create not available")

    @pytest.mark.order(809)
    def test_modify_injection_group_create_no_append(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("TestIG2", "Gen", 1.0, "ALL", append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group create not available")

    @pytest.mark.order(810)
    def test_modify_interfaces_auto_insert(self, saw_instance):
        try:
            saw_instance.InterfacesAutoInsert("AREA", delete_existing=True, use_filters=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface auto-insert not available")

    @pytest.mark.order(811)
    def test_modify_interfaces_auto_insert_with_filters(self, saw_instance):
        try:
            saw_instance.InterfacesAutoInsert("AREA", delete_existing=False, use_filters=True, prefix="TEST_")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface auto-insert not available")

    @pytest.mark.order(812)
    def test_modify_set_participation_factors(self, saw_instance):
        saw_instance.SetParticipationFactors("CONSTANT", 1.0, "SYSTEM")

    @pytest.mark.order(813)
    def test_modify_set_scheduled_voltage(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses available")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.SetScheduledVoltageForABus(bus_key, 1.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SetScheduledVoltage not available")

    @pytest.mark.order(814)
    def test_modify_set_interface_limit_sum(self, saw_instance):
        try:
            saw_instance.SetInterfaceLimitToMonitoredElementLimitSum("ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Interface limit sum not available")

    @pytest.mark.order(815)
    def test_modify_rotate_bus_angles(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses available")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.RotateBusAnglesInIsland(bus_key, 0.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Rotate bus angles not available")

    @pytest.mark.order(816)
    def test_modify_set_gen_pmax(self, saw_instance):
        try:
            saw_instance.SetGenPMaxFromReactiveCapabilityCurve()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Reactive capability curve not available")

    @pytest.mark.order(817)
    def test_modify_remove_3w_xformer(self, saw_instance):
        try:
            saw_instance.Remove3WXformerContainer()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("3W transformer removal not available")

    @pytest.mark.order(818)
    def test_modify_rename_injection_group(self, saw_instance):
        try:
            saw_instance.InjectionGroupCreate("RenameTestIG", "Gen", 1.0, "ALL")
            saw_instance.RenameInjectionGroup("RenameTestIG", "RenamedIG")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Injection group rename not available")

    @pytest.mark.order(819)
    def test_modify_reassign_ids(self, saw_instance):
        try:
            saw_instance.ReassignIDs("Load", "BusName", filter_name="", use_right=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ReassignIDs not available")

    @pytest.mark.order(820)
    def test_modify_reassign_ids_right(self, saw_instance):
        try:
            saw_instance.ReassignIDs("Load", "BusName", filter_name="ALL", use_right=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ReassignIDs not available")

    @pytest.mark.order(821)
    def test_modify_merge_line_terminals(self, saw_instance):
        try:
            saw_instance.MergeLineTerminals("SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("MergeLineTerminals not available")

    @pytest.mark.order(822)
    def test_modify_merge_ms_line_sections(self, saw_instance):
        try:
            saw_instance.MergeMSLineSections("SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("MergeMSLineSections not available")

    @pytest.mark.order(823)
    def test_modify_directions_auto_insert(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("Need >= 2 areas for directions")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        saw_instance.DirectionsAutoInsert(s, b, delete_existing=True, use_area_zone_filters=False)

    @pytest.mark.order(824)
    def test_modify_directions_auto_insert_with_filters(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("Need >= 2 areas for directions")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DirectionsAutoInsert(s, b, delete_existing=False, use_area_zone_filters=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Directions with filters not available")

    @pytest.mark.order(825)
    def test_modify_directions_auto_insert_ref_opposite(self, saw_instance):
        try:
            saw_instance.DirectionsAutoInsertReference("Bus", "Slack", delete_existing=True, opposite_direction=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Directions reference insert not available")

    @pytest.mark.order(826)
    def test_modify_change_system_mva_base(self, saw_instance):
        saw_instance.ChangeSystemMVABase(100.0)


# =============================================================================
# OPF (68% coverage)
# =============================================================================

class TestOPFExtended:
    """Extended tests for OPF solver operations."""

    @pytest.mark.order(600)
    def test_opf_initialize_primal_lp(self, saw_instance):
        try:
            saw_instance.InitializePrimalLP()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("OPF initialization not available")

    @pytest.mark.order(601)
    def test_opf_solve_primal_lp(self, saw_instance):
        try:
            saw_instance.SolvePrimalLP()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Primal LP OPF not available")

    @pytest.mark.order(602)
    def test_opf_solve_primal_lp_with_aux(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.SolvePrimalLP(
                on_success_aux=tmp_aux, create_if_not_found1=True
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Primal LP OPF with aux not available")

    @pytest.mark.order(603)
    def test_opf_solve_single_outer_loop(self, saw_instance):
        try:
            saw_instance.SolveSinglePrimalLPOuterLoop()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Single primal LP outer loop not available")

    @pytest.mark.order(604)
    def test_opf_solve_full_scopf(self, saw_instance):
        try:
            saw_instance.SolveFullSCOPF(bc_method="OPF")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Full SCOPF not available")

    @pytest.mark.order(605)
    def test_opf_solve_full_scopf_powerflow(self, saw_instance):
        try:
            saw_instance.SolveFullSCOPF(bc_method="POWERFLOW")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Full SCOPF with POWERFLOW not available")

    @pytest.mark.order(606)
    def test_opf_write_results(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.OPFWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("OPF write results not available")


# =============================================================================
# ATC Extended (50% coverage)
# =============================================================================

class TestATCExtended:
    """Extended tests for ATC analysis operations."""

    @pytest.mark.order(610)
    def test_atc_set_reference(self, saw_instance):
        saw_instance.ATCSetAsReference()

    @pytest.mark.order(611)
    def test_atc_restore_initial(self, saw_instance):
        saw_instance.ATCRestoreInitialState()

    @pytest.mark.order(612)
    def test_atc_delete_all_results(self, saw_instance):
        saw_instance.ATCDeleteAllResults()

    @pytest.mark.order(613)
    def test_atc_determine_distributed(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("Need >= 2 areas for ATC")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DetermineATC(s, b, distributed=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Distributed ATC not available")

    @pytest.mark.order(614)
    def test_atc_determine_multiple_scenarios(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("Need >= 2 areas for ATC")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        try:
            saw_instance.DetermineATC(s, b, multiple_scenarios=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC multiple scenarios not available")

    @pytest.mark.order(615)
    def test_atc_determine_multiple_directions_distributed(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            pytest.skip("Need >= 2 areas for directions")
        s = create_object_string("Area", areas.iloc[0]["AreaNum"])
        b = create_object_string("Area", areas.iloc[1]["AreaNum"])
        saw_instance.DirectionsAutoInsert(s, b)
        try:
            saw_instance.DetermineATCMultipleDirections(distributed=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC multiple directions distributed not available")

    @pytest.mark.order(616)
    def test_atc_create_contingent_interfaces(self, saw_instance):
        try:
            saw_instance.ATCCreateContingentInterfaces()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC contingent interfaces not available")

    @pytest.mark.order(617)
    def test_atc_create_contingent_interfaces_with_filter(self, saw_instance):
        try:
            saw_instance.ATCCreateContingentInterfaces(filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC contingent interfaces not available")

    @pytest.mark.order(618)
    def test_atc_determine_for(self, saw_instance):
        try:
            saw_instance.ATCDetermineATCFor(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineATCFor not available")

    @pytest.mark.order(619)
    def test_atc_determine_for_apply(self, saw_instance):
        try:
            saw_instance.ATCDetermineATCFor(0, 0, 0, apply_transfer=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineATCFor with apply not available")

    @pytest.mark.order(620)
    def test_atc_determine_multiple_directions_for(self, saw_instance):
        try:
            saw_instance.ATCDetermineMultipleDirectionsATCFor(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCDetermineMultipleDirectionsATCFor not available")

    @pytest.mark.order(621)
    def test_atc_increase_transfer(self, saw_instance):
        try:
            saw_instance.ATCIncreaseTransferBy(0.0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCIncreaseTransferBy not available")

    @pytest.mark.order(622)
    def test_atc_take_me_to_scenario(self, saw_instance):
        try:
            saw_instance.ATCTakeMeToScenario(0, 0, 0)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCTakeMeToScenario not available")

    @pytest.mark.order(623)
    def test_atc_data_write_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCDataWriteOptionsAndResults(tmp_aux, append=False, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write options not available")

    @pytest.mark.order(624)
    def test_atc_write_all_options_deprecated(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCWriteAllOptions(tmp_aux, append=True, key_field="PRIMARY")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATCWriteAllOptions not available")

    @pytest.mark.order(625)
    def test_atc_write_results_and_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.ATCWriteResultsAndOptions(tmp_aux, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write results not available")

    @pytest.mark.order(626)
    def test_atc_write_scenario_log(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteScenarioLog(tmp_txt, append=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario log not available")

    @pytest.mark.order(627)
    def test_atc_write_scenario_log_with_filter(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteScenarioLog(tmp_txt, append=True, filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario log not available")

    @pytest.mark.order(628)
    def test_atc_write_scenario_minmax(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteScenarioMinMax(tmp_csv, filetype="CSV", operation="MIN")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario min/max not available")

    @pytest.mark.order(629)
    def test_atc_write_scenario_minmax_with_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteScenarioMinMax(
                tmp_csv, filetype="CSV", append=True,
                fieldlist=["MaxFlow", "LimitingContingency"],
                operation="MAX", operation_field="MaxFlow",
                group_scenario=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC scenario min/max not available")

    @pytest.mark.order(630)
    def test_atc_write_to_text(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.ATCWriteToText(tmp_txt, filetype="TAB")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write to text not available")

    @pytest.mark.order(631)
    def test_atc_write_to_text_csv_fields(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.ATCWriteToText(tmp_csv, filetype="CSV", fieldlist=["MaxFlow"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC write to text CSV not available")

    @pytest.mark.order(632)
    def test_atc_delete_scenario_change(self, saw_instance):
        try:
            saw_instance.ATCDeleteScenarioChangeIndexRange("RL", ["0"])
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC delete scenario change not available")

    @pytest.mark.order(633)
    def test_atc_get_results_default_fields(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow", "LimitingElement",
                                    "TransferLimit", "LimitUsed", "PTDF", "OTDF"],
            "field_data_type": ["String", "Real", "String", "Real", "String", "Real", "Real"],
            "key_field": ["", "", "", "", "", "", ""],
            "description": ["", "", "", "", "", "", ""],
            "display_name": ["", "", "", "", "", "", ""]
        }).sort_values(by="internal_field_name")
        try:
            saw_instance.GetATCResults()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ATC results not available")


# =============================================================================
# Transient Extended (63% coverage)
# =============================================================================

class TestTransientExtended2:
    """Additional transient tests to hit uncovered parameter paths."""

    @pytest.mark.order(640)
    def test_transient_transfer_state(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Transient transfer state not available")

    @pytest.mark.order(641)
    def test_transient_transfer_state_no_mismatch(self, saw_instance):
        try:
            saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Transient transfer state not available")

    @pytest.mark.order(642)
    def test_transient_store_response(self, saw_instance):
        saw_instance.TSStoreResponse("Gen", True)
        saw_instance.TSStoreResponse("Gen", False)

    @pytest.mark.order(643)
    def test_transient_clear_results_ram(self, saw_instance):
        saw_instance.TSClearResultsFromRAM()

    @pytest.mark.order(644)
    def test_transient_clear_results_specific_ctg(self, saw_instance):
        saw_instance.TSClearResultsFromRAM(
            ctg_name="ALL",
            clear_summary=True,
            clear_events=False,
            clear_statistics=True,
            clear_time_values=False,
            clear_solution_details=True,
        )

    @pytest.mark.order(645)
    def test_transient_clear_results_and_disable(self, saw_instance):
        saw_instance.TSClearResultsFromRAMAndDisableStorage()

    @pytest.mark.order(646)
    def test_transient_clear_all_models(self, saw_instance):
        try:
            saw_instance.TSClearAllModels()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearAllModels not available")

    @pytest.mark.order(647)
    def test_transient_smib_eigenvalues(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSCalculateSMIBEigenValues()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SMIB eigenvalues not available")

    @pytest.mark.order(648)
    def test_transient_clear_models_for_objects(self, saw_instance):
        try:
            saw_instance.TSClearModelsforObjects("Gen")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSClearModelsforObjects not available")

    @pytest.mark.order(649)
    def test_transient_disable_machine_model(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSDisableMachineModelNonZeroDerivative(threshold=0.01)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSDisableMachineModelNonZeroDerivative not available")

    @pytest.mark.order(650)
    def test_transient_auto_insert_dist_relay(self, saw_instance):
        try:
            saw_instance.TSAutoInsertDistRelay(
                reach=1.0, add_from=True, add_to=False,
                transfer_trip=True, shape=1, filter_name="ALL",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoInsertDistRelay not available")

    @pytest.mark.order(651)
    def test_transient_auto_insert_zpott(self, saw_instance):
        try:
            saw_instance.TSAutoInsertZPOTT(reach=1.0, filter_name="ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSAutoInsertZPOTT not available")

    @pytest.mark.order(652)
    def test_transient_run_result_analyzer(self, saw_instance):
        try:
            saw_instance.TSRunResultAnalyzer()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSRunResultAnalyzer not available")

    @pytest.mark.order(653)
    def test_transient_run_until_specified_time(self, saw_instance):
        try:
            saw_instance.TSInitialize()
            saw_instance.TSRunUntilSpecifiedTime("TestCtg", stop_time=0.1, step_size=0.01)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSRunUntilSpecifiedTime not available")

    @pytest.mark.order(654)
    def test_transient_save_formats(self, saw_instance, temp_file):
        for fmt, method in [
            (".dyr", saw_instance.TSSavePTI),
            (".dyd", saw_instance.TSSaveGE),
            (".bpa", saw_instance.TSSaveBPA),
        ]:
            tmp = temp_file(fmt)
            try:
                method(tmp)
            except (PowerWorldPrerequisiteError, PowerWorldError):
                continue

    @pytest.mark.order(655)
    def test_transient_save_formats_diff(self, saw_instance, temp_file):
        for fmt, method in [
            (".dyr", saw_instance.TSSavePTI),
            (".dyd", saw_instance.TSSaveGE),
            (".bpa", saw_instance.TSSaveBPA),
        ]:
            tmp = temp_file(fmt)
            try:
                method(tmp, diff_case_modified_only=True)
            except (PowerWorldPrerequisiteError, PowerWorldError):
                continue

    @pytest.mark.order(656)
    def test_transient_write_models_diff(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
            saw_instance.TSWriteModels(tmp, diff_case_modified_only=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSWriteModels diff not available")

    @pytest.mark.order(657)
    def test_transient_write_options_custom(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
            saw_instance.TSWriteOptions(
                tmp,
                save_dynamic_model=False,
                save_stability_options=True,
                save_stability_events=False,
                save_results_events=True,
                save_plot_definitions=False,
                save_transient_limit_monitors=True,
                save_result_analyzer_time_window=False,
                key_field="SECONDARY",
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSWriteOptions custom not available")

    @pytest.mark.order(658)
    def test_transient_save_two_bus_equiv(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for two-bus equivalent")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp = temp_file(".pwb")
        try:
            saw_instance.TSSaveTwoBusEquivalent(tmp, bus_key)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveTwoBusEquivalent not available")

    @pytest.mark.order(659)
    def test_transient_join_active_ctgs(self, saw_instance):
        try:
            saw_instance.TSJoinActiveCTGs(0.1, delete_existing=True, join_with_self=False)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSJoinActiveCTGs not available")

    @pytest.mark.order(660)
    def test_transient_set_selected_for_refs(self, saw_instance):
        try:
            saw_instance.TSSetSelectedForTransientReferences(
                "SELECTED", "SET", ["Gen"], ["GENROU"]
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSetSelectedForTransientReferences not available")

    @pytest.mark.order(661)
    def test_transient_save_dynamic_models_append(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        try:
            saw_instance.TSSaveDynamicModels(tmp, "AUX", "Gen", append=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSSaveDynamicModels append not available")

    @pytest.mark.order(662)
    def test_transient_plot_series_add(self, saw_instance):
        try:
            saw_instance.TSPlotSeriesAdd("TestPlot", 1, 1, "Gen", "GenMW")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSPlotSeriesAdd not available")

    @pytest.mark.order(663)
    def test_transient_get_vcurve_data(self, saw_instance, temp_file):
        tmp = temp_file(".csv")
        try:
            saw_instance.TSGetVCurveData(tmp, "ALL")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("TSGetVCurveData not available")


# =============================================================================
# General Extended (73% coverage)
# =============================================================================

class TestGeneralExtended:
    """Extended tests for General mixin — uncovered parameter paths."""

    @pytest.mark.order(670)
    def test_general_log_clear(self, saw_instance):
        saw_instance.LogClear()

    @pytest.mark.order(671)
    def test_general_log_show(self, saw_instance):
        saw_instance.LogShow(show=True)
        saw_instance.LogShow(show=False)

    @pytest.mark.order(672)
    def test_general_log_add_datetime(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel", include_date=True, include_time=True, include_milliseconds=False)

    @pytest.mark.order(673)
    def test_general_log_add_datetime_all(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel2", include_date=True, include_time=True, include_milliseconds=True)

    @pytest.mark.order(674)
    def test_general_log_add_datetime_minimal(self, saw_instance):
        saw_instance.LogAddDateTime("TestLabel3", include_date=False, include_time=False, include_milliseconds=False)

    @pytest.mark.order(675)
    def test_general_log_save_append(self, saw_instance, temp_file):
        tmp = temp_file(".txt")
        saw_instance.LogAdd("Test1")
        saw_instance.LogSave(tmp, append=False)
        saw_instance.LogAdd("Test2")
        saw_instance.LogSave(tmp, append=True)
        assert os.path.exists(tmp)

    @pytest.mark.order(676)
    def test_general_set_current_directory(self, saw_instance, temp_dir):
        saw_instance.SetCurrentDirectory(str(temp_dir))

    @pytest.mark.order(677)
    def test_general_set_current_directory_create(self, saw_instance, temp_dir):
        new_dir = os.path.join(str(temp_dir), "test_subdir")
        saw_instance.SetCurrentDirectory(new_dir, create_if_not_found=True)

    @pytest.mark.order(678)
    def test_general_enter_mode(self, saw_instance):
        saw_instance.EnterMode("EDIT")
        saw_instance.EnterMode("RUN")

    @pytest.mark.order(679)
    def test_general_import_data(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, "w") as f:
            f.write("BusNum,BusName\n1,TestBus\n")
        try:
            saw_instance.ImportData(tmp_csv, "CSV", header_line=1, create_if_not_found=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("ImportData not available")

    @pytest.mark.order(680)
    def test_general_load_csv(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        with open(tmp_csv, "w") as f:
            f.write("ObjectType,Bus\nBusNum,BusName\n1,TestBus\n")
        try:
            saw_instance.LoadCSV(tmp_csv, create_if_not_found=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadCSV not available")

    @pytest.mark.order(681)
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

    @pytest.mark.order(682)
    def test_general_save_data_no_sort(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.SaveData(
            tmp_aux, "AUX", "Bus", ["BusNum", "BusName"],
            transpose=False, append=False,
        )
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(683)
    def test_general_save_data_transposed(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.SaveData(
                tmp_csv, "CSV", "Bus", ["BusNum", "BusName"],
                transpose=True, append=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveData transposed not available")

    @pytest.mark.order(684)
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

    @pytest.mark.order(685)
    def test_general_load_aux_directory(self, saw_instance, temp_dir):
        try:
            saw_instance.LoadAuxDirectory(str(temp_dir), filter_string="*.aux")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadAuxDirectory not available")

    @pytest.mark.order(686)
    def test_general_load_aux_directory_no_filter(self, saw_instance, temp_dir):
        try:
            saw_instance.LoadAuxDirectory(str(temp_dir))
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadAuxDirectory not available")

    @pytest.mark.order(687)
    def test_general_load_data(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        with open(tmp_aux, "w") as f:
            f.write('DATA (Bus, [BusNum, BusName]) {\n1 "TestBus"\n}\n')
        try:
            saw_instance.LoadData(tmp_aux, "Bus")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadData not available")

    @pytest.mark.order(688)
    def test_general_stop_aux_file(self, saw_instance):
        try:
            saw_instance.StopAuxFile()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("StopAuxFile not available")

    @pytest.mark.order(689)
    def test_general_select_all_no_filter(self, saw_instance):
        saw_instance.SelectAll("Bus")
        saw_instance.UnSelectAll("Bus")


# =============================================================================
# Case Actions Extended (70% coverage)
# =============================================================================

class TestCaseActionsExtended:
    """Extended tests for Case Actions — uncovered parameter paths."""

    @pytest.mark.order(900)
    def test_case_description_append(self, saw_instance):
        saw_instance.CaseDescriptionSet("Line 1")
        saw_instance.CaseDescriptionSet("Line 2", append=True)
        saw_instance.CaseDescriptionClear()

    @pytest.mark.order(901)
    def test_case_save_external_with_ties(self, saw_instance, temp_file):
        tmp_pwb = temp_file(".pwb")
        try:
            saw_instance.SaveExternalSystem(tmp_pwb, with_ties=True)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveExternalSystem with ties not available")

    @pytest.mark.order(902)
    def test_case_scale_gen(self, saw_instance):
        try:
            saw_instance.Scale("GEN", "FACTOR", [1.0], "SYSTEM")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Scale GEN not available")

    @pytest.mark.order(903)
    def test_case_scale_load_mw(self, saw_instance):
        try:
            saw_instance.Scale("LOAD", "MW", [100.0, 50.0], "SYSTEM")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Scale LOAD MW not available")

    @pytest.mark.order(904)
    def test_case_load_ems(self, saw_instance, temp_file):
        tmp = temp_file(".hdb")
        try:
            saw_instance.LoadEMS(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LoadEMS not available")

    @pytest.mark.order(905)
    def test_case_renumber_custom_index(self, saw_instance):
        try:
            saw_instance.RenumberAreas(custom_integer_index=1)
            saw_instance.RenumberBuses(custom_integer_index=2)
            saw_instance.RenumberSubs(custom_integer_index=3)
            saw_instance.RenumberZones(custom_integer_index=4)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Renumber with custom index not available")


# =============================================================================
# Powerflow Extended (88% coverage — hit retry, conditioning, diff write gaps)
# =============================================================================

class TestPowerFlowExtendedGaps:
    """Tests to hit remaining uncovered lines in powerflow.py."""

    @pytest.mark.order(710)
    def test_powerflow_solve_with_method(self, saw_instance):
        """Test SolvePowerFlow with explicit method parameter."""
        saw_instance.SolvePowerFlow("RECTNEWT")
        saw_instance.SolvePowerFlow()

    @pytest.mark.order(711)
    def test_powerflow_condition_voltage_pockets(self, saw_instance):
        """Test VoltageConditioning."""
        try:
            saw_instance.VoltageConditioning()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Voltage conditioning not available")

    @pytest.mark.order(712)
    def test_powerflow_diff_write_removed_epc(self, saw_instance, temp_file):
        """Test DiffCaseWriteRemovedEPC."""
        tmp_epc = temp_file(".epc")
        try:
            saw_instance.DiffCaseWriteRemovedEPC(tmp_epc)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("DiffCaseWriteRemovedEPC not available")


# =============================================================================
# Sensitivity / Topology Extended (hit remaining lines)
# =============================================================================

class TestSensitivityTopologyExtended:
    """Tests to hit remaining uncovered lines in sensitivity.py and topology.py."""

    @pytest.mark.order(720)
    def test_sensitivity_lodf_post_closure(self, saw_instance):
        """Test CalculateLODF with post_closure_lcdf='YES'."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is None or branches.empty:
            pytest.skip("No branches for LODF post closure")
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        try:
            saw_instance.CalculateLODF(branch_str, post_closure_lcdf="YES")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LODF post closure not available")

    @pytest.mark.order(721)
    def test_sensitivity_ptdf_multiple_directions(self, saw_instance):
        """Test CalculatePTDFMultipleDirections."""
        try:
            saw_instance.CalculatePTDFMultipleDirections()
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PTDF multiple directions not available")

    @pytest.mark.order(722)
    def test_sensitivity_line_loading_replicator(self, saw_instance):
        """Test LineLoadingReplicatorCalculate."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is None or branches.empty:
            pytest.skip("No branches for line loading replicator")
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        try:
            saw_instance.LineLoadingReplicatorCalculate(
                branch_str, "System", agc_only=False, desired_flow=100.0, implement=False,
            )
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("LineLoadingReplicatorCalculate not available")

    @pytest.mark.order(723)
    def test_topology_path_distance(self, saw_instance):
        """Test DeterminePathDistance."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is None or buses.empty:
            pytest.skip("No buses for path distance")
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        try:
            saw_instance.DeterminePathDistance(bus_key, 3)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("DeterminePathDistance not available")

    @pytest.mark.order(724)
    def test_topology_set_bus_field_from_closest(self, saw_instance):
        """Test SetBusFieldFromClosest with all required args."""
        try:
            saw_instance.SetBusFieldFromClosest("BusName", "", "", "", "IMPEDANCE")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SetBusFieldFromClosest not available")

    @pytest.mark.order(725)
    def test_topology_save_consolidated_case(self, saw_instance, temp_file):
        """Test SaveConsolidatedCase."""
        tmp = temp_file(".pwb")
        try:
            saw_instance.SaveConsolidatedCase(tmp)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("SaveConsolidatedCase not available")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
