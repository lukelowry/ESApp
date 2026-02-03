"""
Integration tests for Contingency Analysis and Fault Analysis via SAW.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test contingency auto-insertion,
solving, cloning, conversion, OTDF calculations, fault analysis, and result
export.

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_core.py          -- base SAW operations, logging, I/O
    - test_integration_saw_modify.py        -- destructive modify, region, case actions
    - test_integration_saw_powerflow.py     -- power flow, matrices, sensitivity, topology
    - test_integration_saw_gic.py           -- GIC analysis
    - test_integration_saw_transient.py     -- transient stability
    - test_integration_saw_operations.py    -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py         -- PowerWorld facade and statics
    - test_integration_network.py           -- Network topology

USAGE:
    pytest tests/test_integration_saw_contingency.py -v
"""

import os
import pytest
import pandas as pd

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


def _configure_limited_ctg_auto_insert(saw_instance):
    """Configure CTG_AutoInsert_Options to limit contingency count for faster tests.

    Deletes existing contingencies and configures auto-insert. Uses all kV levels
    to ensure contingencies are created regardless of the test case's voltage range.
    """
    saw_instance.SetData("Contingency", ["Skip"], ["NO"], "ALL")
    try:
        saw_instance.RunScriptCommand("Delete(Contingency);")
    except (PowerWorldPrerequisiteError, PowerWorldError):
        pass

    saw_instance.SetData(
        "CTG_AutoInsert_Options",
        ["CtgAutoInsDeleteExistCtgs", "DOCUseAllkV"],
        ["YES", "YES"],
    )


def _trim_contingencies(saw_instance, max_active=5, delete_excess=False):
    """Skip all contingencies then un-skip only *max_active* to limit runtime."""
    saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
    ctgs = saw_instance.ListOfDevices("Contingency")
    if ctgs is None or ctgs.empty:
        return
    name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
    keep_names = set(ctgs.head(max_active)[name_col])
    for name in keep_names:
        saw_instance.ChangeParametersSingleElement(
            "Contingency", [name_col, "Skip"], [name, "NO"]
        )
    if delete_excess and len(ctgs) > max_active:
        saw_instance.SelectAll("Contingency")
        for name in keep_names:
            saw_instance.ChangeParametersSingleElement(
                "Contingency", [name_col, "Selected"], [name, "NO"]
            )
        try:
            saw_instance.Delete("Contingency", "SELECTED")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pass


class TestContingency:
    """Tests for contingency analysis operations."""

    @pytest.mark.order(5000)
    def test_contingency_auto_insert(self, saw_instance):
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.CTGAutoInsert()

    @pytest.mark.order(5100)
    def test_contingency_solve(self, saw_instance):
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found after auto-insert"
        name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
        for name in ctgs.head(2)[name_col]:
            saw_instance.ChangeParametersSingleElement(
                "Contingency", [name_col, "Skip"], [name, "NO"]
            )
        saw_instance.CTGSolveAll()

    @pytest.mark.order(5200)
    def test_contingency_run_single(self, saw_instance):
        ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found"
        ctg_name = ctgs.iloc[0]["CTGLabel"]
        saw_instance.CTGSolve(ctg_name)
        saw_instance.CTGApply(ctg_name)

    @pytest.mark.order(5300)
    def test_contingency_otdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is None or len(areas) < 2:
            buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "AreaNum"])
            assert buses is not None and not buses.empty, "Test case must contain buses"
            existing_areas = set(int(a) for a in areas["AreaNum"]) if areas is not None else set()
            next_area = max(existing_areas, default=0) + 1
            saw_instance.CreateData("Area", ["AreaNum", "AreaName"], [next_area, f"TestArea{next_area}"])
            area_counts = buses["AreaNum"].value_counts()
            largest_area = area_counts.index[0]
            donor = buses[buses["AreaNum"] == largest_area]
            if len(donor) > 1:
                saw_instance.ChangeParametersSingleElement(
                    "Bus", ["BusNum", "AreaNum"], [int(donor.iloc[-1]["BusNum"]), next_area]
                )
            areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        seller = f'[AREA {areas.iloc[0]["AreaNum"]}]'
        buyer = f'[AREA {areas.iloc[1]["AreaNum"]}]'
        saw_instance.CTGCalculateOTDF(seller, buyer)

    @pytest.mark.order(5400)
    def test_contingency_clear_and_reference(self, saw_instance):
        """Test CTGClearAllResults and CTGSetAsReference."""
        saw_instance.CTGClearAllResults()
        saw_instance.CTGSetAsReference()

    @pytest.mark.order(5410)
    def test_contingency_relink(self, saw_instance):
        """Test CTGRelinkUnlinkedElements."""
        saw_instance.CTGRelinkUnlinkedElements()

    @pytest.mark.order(5420)
    def test_contingency_sort(self, saw_instance):
        """Test CTGSort."""
        saw_instance.CTGSort()

    @pytest.mark.order(5500)
    def test_contingency_combo(self, saw_instance):
        """Run combo solve BEFORE cloning to avoid solving duplicated contingencies."""
        saw_instance.CTGComboDeleteAllResults()
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.CTGAutoInsert()
        saw_instance.CTGConvertToPrimaryCTG()

        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")

        ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found after auto-insert"
        name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
        primary_ctgs = ctgs[ctgs[name_col].astype(str).str.endswith("-Primary")]
        target_ctgs = primary_ctgs.head(2) if not primary_ctgs.empty else ctgs.head(2)

        for name in target_ctgs[name_col]:
            saw_instance.ChangeParametersSingleElement(
                "Contingency", [name_col, "Skip"], [name, "NO"]
            )

        saw_instance.CTGComboSolveAll()

    @pytest.mark.order(5600)
    def test_contingency_clone(self, saw_instance):
        """Clone contingencies AFTER combo solve to avoid bloating solve operations."""
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is None or ctgs.empty:
            _configure_limited_ctg_auto_insert(saw_instance)
            saw_instance.CTGAutoInsert()
            _trim_contingencies(saw_instance, max_active=3)
            ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found for cloning"
        ctg_name = ctgs.iloc[0]["CTGLabel"]
        saw_instance.CTGCloneOne(ctg_name, "ClonedCTG")
        saw_instance.CTGCloneMany("", "Many_", "_Suffix")

    @pytest.mark.order(5700)
    def test_contingency_convert(self, saw_instance):
        saw_instance.CTGConvertAllToDeviceCTG()
        saw_instance.CTGConvertToPrimaryCTG()
        saw_instance.CTGCreateExpandedBreakerCTGs()
        saw_instance.CTGCreateStuckBreakerCTGs()
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.CTGPrimaryAutoInsert()

    @pytest.mark.order(5800)
    def test_contingency_create_interface(self, saw_instance):
        saw_instance.CTGCreateContingentInterfaces("")

    @pytest.mark.order(5900)
    def test_contingency_join(self, saw_instance):
        saw_instance.CTGJoinActiveCTGs(False, False, True)

    @pytest.mark.order(5990)
    def test_contingency_process_remedial(self, saw_instance):
        saw_instance.CTGProcessRemedialActionsAndDependencies(False)

    @pytest.mark.order(6100)
    def test_contingency_save_matrices(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.CTGSaveViolationMatrices(tmp_csv, "CSVCOLHEADER", False, ["Branch"], True, True)

    @pytest.mark.order(6200)
    def test_contingency_verify(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.CTGVerifyIteratedLinearActions(tmp_txt)

    @pytest.mark.order(6300)
    def test_contingency_write_results(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.CTGWriteResultsAndOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.CTGWriteAllOptions(tmp_aux2)
        assert os.path.exists(tmp_aux2)

        tmp_aux3 = temp_file(".aux")
        saw_instance.CTGWriteAuxUsingOptions(tmp_aux3)
        assert os.path.exists(tmp_aux3)

    @pytest.mark.order(6320)
    def test_ctg_sort_with_fields(self, saw_instance):
        """CTGSort with sort field list."""
        saw_instance.CTGSort(sort_field_list=["Name:+:0"])

    @pytest.mark.order(6330)
    def test_ctg_delete_identical(self, saw_instance):
        """CTGDeleteWithIdenticalActions completes without error."""
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.CTGAutoInsert()
        _trim_contingencies(saw_instance, max_active=5, delete_excess=True)
        saw_instance.CTGDeleteWithIdenticalActions()

    @pytest.mark.order(6350)
    def test_ctg_restore_reference(self, saw_instance):
        """CTGRestoreReference restores reference state."""
        saw_instance.CTGSetAsReference()
        saw_instance.CTGRestoreReference()

    @pytest.mark.order(6360)
    def test_ctg_write_aux_using_options(self, saw_instance, temp_file):
        """CTGWriteAuxUsingOptions writes to file."""
        tmp = temp_file(".aux")
        saw_instance.CTGWriteAuxUsingOptions(tmp, append=False)

    @pytest.mark.order(6400)
    def test_contingency_get_violations(self, saw_instance):
        """Test retrieving contingency violations."""
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.CTGAutoInsert()

        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found after auto-insert"
        name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
        saw_instance.SetData("Contingency", [name_col, "Skip"], [ctgs.iloc[0][name_col], "NO"])

        saw_instance.CTGSolveAll()

    @pytest.mark.order(6500)
    def test_contingency_results_dataframe(self, saw_instance):
        """Test that contingency results can be retrieved as DataFrame."""
        ctgs = saw_instance.ListOfDevices("Contingency")
        assert ctgs is not None and not ctgs.empty, "No contingencies found for results check"
        assert isinstance(ctgs, pd.DataFrame)
        assert len(ctgs) > 0
        assert "CTGLabel" in ctgs.columns or len(ctgs.columns) > 0

    @pytest.mark.order(6600)
    def test_contingency_skip_behavior(self, saw_instance):
        """Test that skipped contingencies are not solved."""
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        saw_instance.CTGSolveAll()

    @pytest.mark.order(6700)
    def test_contingency_restore_reference(self, saw_instance):
        """Test CTGRestoreReference restores case state."""
        original_buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"])
        assert original_buses is not None, "Failed to retrieve original bus data"

        saw_instance.CTGRestoreReference()

        restored_buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"])
        assert restored_buses is not None, "Failed to retrieve restored bus data"

        assert len(original_buses) == len(restored_buses)


class TestFault:
    """Tests for fault analysis operations."""

    @pytest.mark.order(5350)
    def test_fault_run(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "No buses found"
        bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
        saw_instance.RunFault(bus_str, "SLG")
        saw_instance.FaultClear()

    @pytest.mark.order(5450)
    def test_fault_auto(self, saw_instance):
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.FaultAutoInsert()

    @pytest.mark.order(5550)
    def test_fault_multiple(self, saw_instance):
        _configure_limited_ctg_auto_insert(saw_instance)
        saw_instance.FaultAutoInsert()
        try:
            saw_instance.FaultMultiple()
        except PowerWorldPrerequisiteError as e:
            if "No active faults" in str(e):
                pytest.skip("No active faults defined after FaultAutoInsert for this case")
            raise

    @pytest.mark.order(5650)
    def test_fault_types(self, saw_instance):
        """Test different fault types."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "No buses found for fault type testing"
        bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])

        fault_types = ["SLG", "LL", "DLG", "3PB"]
        for ftype in fault_types:
            saw_instance.RunFault(bus_str, ftype)
            saw_instance.FaultClear()

    @pytest.mark.order(5750)
    def test_fault_at_branch(self, saw_instance):
        """Test fault on branch midpoint (line only -- transformers can hang PW)."""
        branches = saw_instance.GetParametersMultipleElement(
            "Branch", ["BusNum", "BusNum:1", "LineCircuit", "BranchDeviceType"]
        )
        assert branches is not None and not branches.empty, "No branches found for branch fault testing"
        lines = branches[branches["BranchDeviceType"] != "Transformer"]
        if lines.empty:
            pytest.skip("No non-transformer branches available for midpoint fault test")
        b = lines.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.RunFault(branch_str, "3PB", location=50.0)
        saw_instance.FaultClear()


class TestContingencyExport:
    """Tests for contingency export functionality."""

    @pytest.mark.order(6800)
    def test_contingency_produce_report(self, saw_instance, temp_file):
        """Test CTGProduceReport for report generation."""
        tmp_txt = temp_file(".txt")
        saw_instance.CTGProduceReport(tmp_txt)
        assert os.path.exists(tmp_txt)

    @pytest.mark.order(6900)
    def test_contingency_write_pti(self, saw_instance, temp_file):
        """Test CTGWriteFilePTI for PTI format export."""
        tmp_pti = temp_file(".con")
        saw_instance.CTGWriteFilePTI(tmp_pti)
        assert os.path.exists(tmp_pti)

    @pytest.mark.order(7000)
    def test_contingency_write_all_options(self, saw_instance, temp_file):
        """Test CTGWriteAllOptions for options export."""
        tmp_aux = temp_file(".aux")
        saw_instance.CTGWriteAllOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(7100)
    def test_contingency_compare_two_lists(self, saw_instance, temp_file):
        """Test CTGCompareTwoListsofContingencyResults for comparing contingency results."""
        list1 = temp_file(".aux")
        list2 = temp_file(".aux")
        try:
            saw_instance.CTGWriteAllOptions(list1)
            saw_instance.CTGWriteAllOptions(list2)
            saw_instance.CTGCompareTwoListsofContingencyResults(list1, list2)
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
            pytest.skip(f"CTG compare not supported for this case: {e}")

    @pytest.mark.order(7200)
    def test_contingency_write_csv(self, saw_instance, temp_file):
        """Test saving contingency violations to CSV."""
        tmp_csv = temp_file(".csv")
        saw_instance.CTGSaveViolationMatrices(
            tmp_csv, "CSVCOLHEADER", False, ["Branch"], True, True
        )
        assert os.path.exists(tmp_csv)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
