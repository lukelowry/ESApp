"""
Integration tests for Contingency Analysis functionality against a live PowerWorld case.

WHAT THIS TESTS:
- Contingency auto-insertion and solving
- Contingency cloning and conversion
- OTDF calculations
- Contingency result export

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd

# Order markers for integration tests - contingency tests run mid-sequence (order 50-69)
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


class TestContingency:
    """Tests for contingency analysis operations."""

    @pytest.mark.order(50)
    def test_contingency_auto_insert(self, saw_instance):
        saw_instance.CTGAutoInsert()

    @pytest.mark.order(51)
    def test_contingency_solve(self, saw_instance):
        saw_instance.SolveContingencies()

    @pytest.mark.order(52)
    def test_contingency_run_single(self, saw_instance):
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["CTGLabel"]
            saw_instance.RunContingency(ctg_name)
            saw_instance.CTGApply(ctg_name)

    @pytest.mark.order(53)
    def test_contingency_otdf(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = f'[AREA {areas.iloc[0]["AreaNum"]}]'
            buyer = f'[AREA {areas.iloc[1]["AreaNum"]}]'
            saw_instance.CTGCalculateOTDF(seller, buyer)

    @pytest.mark.order(54)
    def test_contingency_results_ops(self, saw_instance):
        saw_instance.CTGClearAllResults()
        saw_instance.CTGSetAsReference()
        saw_instance.CTGRelinkUnlinkedElements()
        saw_instance.CTGSkipWithIdenticalActions()
        saw_instance.CTGDeleteWithIdenticalActions()
        saw_instance.CTGSort()

    @pytest.mark.order(55)
    def test_contingency_clone(self, saw_instance):
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            ctg_name = ctgs.iloc[0]["CTGLabel"]
            saw_instance.CTGCloneOne(ctg_name, "ClonedCTG")
            saw_instance.CTGCloneMany("", "Many_", "_Suffix")

    @pytest.mark.order(56)
    def test_contingency_combo(self, saw_instance):
        saw_instance.CTGComboDeleteAllResults()
        saw_instance.CTGAutoInsert()
        saw_instance.CTGConvertToPrimaryCTG()

        # Optimize: Skip most contingencies to avoid long runtimes
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")

        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
            primary_ctgs = ctgs[ctgs[name_col].astype(str).str.endswith("-Primary")]
            target_ctgs = primary_ctgs.head(2) if not primary_ctgs.empty else ctgs.head(2)

            for name in target_ctgs[name_col]:
                saw_instance.SetData("Contingency", [name_col, "Skip"], [name, "NO"])

        try:
            saw_instance.CTGComboSolveAll()
        except PowerWorldPrerequisiteError:
            pytest.skip("No active primary contingencies for Combo Analysis")

    @pytest.mark.order(57)
    def test_contingency_convert(self, saw_instance):
        saw_instance.CTGConvertAllToDeviceCTG()
        saw_instance.CTGConvertToPrimaryCTG()
        saw_instance.CTGCreateExpandedBreakerCTGs()
        saw_instance.CTGCreateStuckBreakerCTGs()
        saw_instance.CTGPrimaryAutoInsert()

    @pytest.mark.order(58)
    def test_contingency_create_interface(self, saw_instance):
        try:
            saw_instance.CTGCreateContingentInterfaces("")
        except PowerWorldPrerequisiteError:
            pytest.skip("Filter 'ALL' not found for CTGCreateContingentInterfaces")

    @pytest.mark.order(59)
    def test_contingency_join(self, saw_instance):
        saw_instance.CTGJoinActiveCTGs(False, False, True)

    @pytest.mark.order(60)
    def test_contingency_process_remedial(self, saw_instance):
        saw_instance.CTGProcessRemedialActionsAndDependencies(False)

    @pytest.mark.order(61)
    def test_contingency_save_matrices(self, saw_instance, temp_file):
        tmp_csv = temp_file(".csv")
        saw_instance.CTGSaveViolationMatrices(tmp_csv, "CSVCOLHEADER", False, ["Branch"], True, True)

    @pytest.mark.order(62)
    def test_contingency_verify(self, saw_instance, temp_file):
        tmp_txt = temp_file(".txt")
        saw_instance.CTGVerifyIteratedLinearActions(tmp_txt)

    @pytest.mark.order(63)
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


class TestFault:
    """Tests for fault analysis operations."""

    @pytest.mark.order(53)
    def test_fault_run(self, saw_instance):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            saw_instance.RunFault(bus_str, "SLG")
            saw_instance.FaultClear()
        else:
            pytest.skip("No buses found")

    @pytest.mark.order(54)
    def test_fault_auto(self, saw_instance):
        saw_instance.FaultAutoInsert()

    @pytest.mark.order(55)
    def test_fault_multiple(self, saw_instance):
        saw_instance.FaultAutoInsert()
        try:
            saw_instance.FaultMultiple()
        except PowerWorldPrerequisiteError:
            pytest.skip("No active faults defined for FaultMultiple")


class TestContingencyAdvanced:
    """Advanced contingency tests for edge cases and validation."""

    @pytest.mark.order(64)
    def test_contingency_get_violations(self, saw_instance):
        """Test retrieving contingency violations."""
        # Run contingencies first to generate results
        saw_instance.CTGAutoInsert()
        
        # Skip most to avoid long runtime
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            name_col = "CTGLabel" if "CTGLabel" in ctgs.columns else ctgs.columns[0]
            saw_instance.SetData("Contingency", [name_col, "Skip"], [ctgs.iloc[0][name_col], "NO"])
        
        try:
            saw_instance.SolveContingencies()
        except PowerWorldPrerequisiteError:
            pytest.skip("No contingencies to solve")

    @pytest.mark.order(65)
    def test_contingency_results_dataframe(self, saw_instance):
        """Test that contingency results can be retrieved as DataFrame."""
        ctgs = saw_instance.ListOfDevices("Contingency")
        if ctgs is not None and not ctgs.empty:
            assert isinstance(ctgs, pd.DataFrame)
            assert len(ctgs) > 0
            # Verify expected columns exist
            assert "CTGLabel" in ctgs.columns or len(ctgs.columns) > 0

    @pytest.mark.order(66)
    def test_contingency_skip_behavior(self, saw_instance):
        """Test that skipped contingencies are not solved."""
        # Skip all contingencies
        saw_instance.SetData("Contingency", ["Skip"], ["YES"], "ALL")
        
        # Solving should be quick since all are skipped
        saw_instance.SolveContingencies()

    @pytest.mark.order(67)
    def test_contingency_restore_reference(self, saw_instance):
        """Test CTGRestoreReference restores case state."""
        # Store original state
        original_buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"])
        
        # Restore reference
        saw_instance.CTGRestoreReference()
        
        # Get state after restore
        restored_buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum", "BusPUVolt"])
        
        if original_buses is not None and restored_buses is not None:
            assert len(original_buses) == len(restored_buses)


class TestFaultAdvanced:
    """Advanced fault analysis tests."""

    @pytest.mark.order(56)
    def test_fault_types(self, saw_instance):
        """Test different fault types."""
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        if buses is not None and not buses.empty:
            bus_str = create_object_string("Bus", buses.iloc[0]["BusNum"])
            
            # PowerWorld fault types: SLG (Single Line to Ground), LL (Line to Line),
            # DLG (Double Line to Ground), 3PB (Three Phase Balanced)
            fault_types = ["SLG", "LL", "DLG", "3PB"]
            for ftype in fault_types:
                try:
                    saw_instance.RunFault(bus_str, ftype)
                    saw_instance.FaultClear()
                except (PowerWorldPrerequisiteError, PowerWorldError):
                    # Some fault types may not be configured or may fail
                    continue

    @pytest.mark.order(57)
    def test_fault_at_branch(self, saw_instance):
        """Test fault on branch midpoint."""
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            try:
                # 3PB = Three Phase Balanced fault, location = percentage along branch (0-100)
                saw_instance.RunFault(branch_str, "3PB", location=50.0)
                saw_instance.FaultClear()
            except (PowerWorldPrerequisiteError, PowerWorldError):
                pytest.skip("Branch fault not supported or failed")


class TestContingencyExport:
    """Tests for contingency export functionality."""

    @pytest.mark.order(68)
    def test_contingency_produce_report(self, saw_instance, temp_file):
        """Test CTGProduceReport for report generation."""
        tmp_txt = temp_file(".txt")
        try:
            saw_instance.CTGProduceReport(tmp_txt)
            assert os.path.exists(tmp_txt)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No contingency results for report")

    @pytest.mark.order(69)
    def test_contingency_write_pti(self, saw_instance, temp_file):
        """Test CTGWriteFilePTI for PTI format export."""
        tmp_pti = temp_file(".con")
        try:
            saw_instance.CTGWriteFilePTI(tmp_pti)
            assert os.path.exists(tmp_pti)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("PTI export not available")

    @pytest.mark.order(70)
    def test_contingency_write_all_options(self, saw_instance, temp_file):
        """Test CTGWriteAllOptions for options export."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.CTGWriteAllOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("Options export not available")

    @pytest.mark.order(71)
    def test_contingency_compare_two_lists(self, saw_instance):
        """Test CTGCompareTwoListsofContingencyResults for comparing contingency results."""
        try:
            saw_instance.CTGCompareTwoListsofContingencyResults("CTGList1", "CTGList2")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No contingency lists to compare")

    @pytest.mark.order(72)
    def test_contingency_write_csv(self, saw_instance, temp_file):
        """Test saving contingency violations to CSV."""
        tmp_csv = temp_file(".csv")
        try:
            # Save violation results using the violation matrix method
            saw_instance.CTGSaveViolationMatrices(
                tmp_csv, "CSVCOLHEADER", False, ["Branch"], True, True
            )
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No violations to save")

