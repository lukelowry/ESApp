"""
Integration tests for GIC, ATC, Transient Stability, and Time Step functionality.

WHAT THIS TESTS:
- GIC (Geomagnetically Induced Current) analysis
- ATC (Available Transfer Capability) analysis
- Transient stability simulations
- Time step simulation operations

DEPENDENCIES:
- PowerWorld Simulator installed and SimAuto registered
- Valid PowerWorld case file configured in tests/config_test.py
"""

import os
import pytest
import pandas as pd
import numpy as np

# Order markers for integration tests - advanced analysis tests (order 73-99)
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


class TestGIC:
    """Tests for GIC (Geomagnetically Induced Current) analysis."""

    @pytest.mark.order(73)
    def test_gic_calculate(self, saw_instance):
        saw_instance.CalculateGIC(1.0, 90.0, False)
        saw_instance.ClearGIC()

    @pytest.mark.order(74)
    def test_gic_save_matrix(self, saw_instance, temp_file):
        tmp_mat = temp_file(".mat")
        tmp_id = temp_file(".txt")
        saw_instance.GICSaveGMatrix(tmp_mat, tmp_id)
        assert os.path.exists(tmp_mat)

    @pytest.mark.order(75)
    def test_gic_setup(self, saw_instance):
        saw_instance.GICSetupTimeVaryingSeries()
        saw_instance.GICShiftOrStretchInputPoints()

    @pytest.mark.order(76)
    def test_gic_time(self, saw_instance):
        saw_instance.GICTimeVaryingCalculate(0.0, False)
        saw_instance.GICTimeVaryingAddTime(10.0)
        saw_instance.GICTimeVaryingDeleteAllTimes()
        saw_instance.GICTimeVaryingEFieldCalculate(0.0, False)
        saw_instance.GICTimeVaryingElectricFieldsDeleteAllTimes()

    @pytest.mark.order(77)
    def test_gic_write(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.GICWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_gmd = temp_file(".gmd")
        saw_instance.GICWriteFilePSLF(tmp_gmd)

        tmp_gic = temp_file(".gic")
        saw_instance.GICWriteFilePTI(tmp_gic)


class TestATC:
    """Tests for ATC (Available Transfer Capability) analysis."""

    @pytest.mark.order(78)
    def test_atc_determine(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            seller = create_object_string("Area", areas.iloc[0]["AreaNum"])
            buyer = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DetermineATC(seller, buyer)
        else:
            pytest.skip("Not enough areas for ATC")

    @pytest.mark.order(79)
    def test_atc_multiple(self, saw_instance):
        areas = saw_instance.GetParametersMultipleElement("Area", ["AreaNum"])
        if areas is not None and len(areas) >= 2:
            s = create_object_string("Area", areas.iloc[0]["AreaNum"])
            b = create_object_string("Area", areas.iloc[1]["AreaNum"])
            saw_instance.DirectionsAutoInsert(s, b)

        try:
            saw_instance.DetermineATCMultipleDirections()
        except PowerWorldPrerequisiteError:
            pytest.skip("No directions defined for ATC")

    @pytest.mark.order(80)
    def test_atc_results(self, saw_instance):
        saw_instance._object_fields["transferlimiter"] = pd.DataFrame({
            "internal_field_name": ["LimitingContingency", "MaxFlow"],
            "field_data_type": ["String", "Real"],
            "key_field": ["", ""],
            "description": ["", ""],
            "display_name": ["", ""]
        }).sort_values(by="internal_field_name")

        saw_instance.GetATCResults(["MaxFlow", "LimitingContingency"])


class TestTransient:
    """Tests for Transient Stability simulations."""

    @pytest.mark.order(81)
    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    @pytest.mark.order(82)
    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(83)
    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        if branches is not None and not branches.empty:
            b = branches.iloc[0]
            branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
            saw_instance.TSCalculateCriticalClearTime(branch_str)

    @pytest.mark.order(84)
    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    @pytest.mark.order(85)
    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)


class TestTimeStep:
    """Tests for Time Step Simulation operations."""

    @pytest.mark.order(86)
    def test_timestep_delete(self, saw_instance):
        saw_instance.TimeStepDeleteAll()

    @pytest.mark.order(87)
    def test_timestep_run(self, saw_instance):
        saw_instance.TimeStepDoRun()
        try:
            saw_instance.TimeStepDoSinglePoint("2025-01-01T10:00:00")
        except PowerWorldPrerequisiteError:
            pass  # Expected if time points not defined
        try:
            saw_instance.TimeStepClearResults()
        except PowerWorldError:
            pass
        saw_instance.TimeStepResetRun()

    @pytest.mark.order(88)
    def test_timestep_save(self, saw_instance, temp_file):
        tmp_pww = temp_file(".pww")
        saw_instance.TimeStepSavePWW(tmp_pww)

        tmp_csv = temp_file(".csv")
        try:
            saw_instance.TimeStepSaveResultsByTypeCSV("Gen", tmp_csv)
        except PowerWorldError:
            pass  # Likely no results

    @pytest.mark.order(89)
    def test_timestep_fields(self, saw_instance):
        saw_instance.TimeStepSaveFieldsSet("Gen", ["GenMW"])
        saw_instance.TimeStepSaveFieldsClear(["Gen"])


class TestPVQV:
    """Tests for PV and QV analysis."""

    @pytest.mark.order(90)
    def test_pv_qv_run(self, saw_instance):
        df = saw_instance.RunQV()
        assert df is not None

    @pytest.mark.order(91)
    def test_pv_clear(self, saw_instance):
        """Test clearing PV analysis results."""
        saw_instance.PVClear()

    @pytest.mark.order(92)
    def test_pv_export(self, saw_instance, temp_file):
        """Test exporting PV analysis results."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.PVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("PV analysis not available or no results")

    @pytest.mark.order(93)
    def test_qv_clear(self, saw_instance):
        """Test clearing QV analysis results."""
        saw_instance.QVDeleteAllResults()

    @pytest.mark.order(94)
    def test_qv_export(self, saw_instance, temp_file):
        """Test exporting QV analysis results."""
        tmp_aux = temp_file(".aux")
        try:
            saw_instance.QVWriteResultsAndOptions(tmp_aux)
            assert os.path.exists(tmp_aux)
        except PowerWorldPrerequisiteError:
            pytest.skip("QV analysis not available or no results")


class TestTransientAdvanced:
    """Additional tests for Transient Stability simulations."""

    @pytest.mark.order(95)
    def test_transient_result_storage_set_all(self, saw_instance):
        """Test TSResultStorageSetAll for all storage modes."""
        # TSResultStorageSetAll(object_type, store_value) - object type first, then bool
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(96)
    def test_transient_clear_playin_signals(self, saw_instance):
        """Test clearing play-in signals."""
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(97)
    def test_transient_get_contingency_results(self, saw_instance):
        """Test getting transient contingency results."""
        # TSGetContingencyResults(CtgName, ObjFieldList, ...) - contingency name first
        # Need an actual contingency name; skip if none available
        try:
            ctgs = saw_instance.ListOfDevices("TSContingency")
            if ctgs is not None and not ctgs.empty:
                ctg_name = ctgs.iloc[0].iloc[0]  # First column, first row
                meta, data = saw_instance.TSGetContingencyResults(ctg_name, ["BusNum", "BusPUVolt"])
                assert meta is None or isinstance(meta, pd.DataFrame)
            else:
                pytest.skip("No transient contingencies defined")
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No transient results available")

    @pytest.mark.order(98)
    def test_transient_validate(self, saw_instance):
        """Test TSValidate for model validation."""
        saw_instance.TSInitialize()
        try:
            saw_instance.TSValidate()
        except PowerWorldPrerequisiteError:
            pytest.skip("Transient validation not available")

    @pytest.mark.order(99)
    def test_transient_auto_correct(self, saw_instance):
        """Test TSAutoCorrect for automatic model corrections."""
        saw_instance.TSInitialize()
        try:
            saw_instance.TSAutoCorrect()
        except PowerWorldPrerequisiteError:
            pytest.skip("Auto-correct not available")

    @pytest.mark.order(100)
    def test_transient_write_results(self, saw_instance, temp_file):
        """Test writing transient results to CSV file."""
        tmp_csv = temp_file(".csv")
        try:
            # TSWriteResultsToCSV(filename, mode, contingencies, plots_fields)
            saw_instance.TSWriteResultsToCSV(tmp_csv, "CSV", ["ALL"], ["GenMW"])
            assert os.path.exists(tmp_csv)
        except (PowerWorldPrerequisiteError, PowerWorldError):
            pytest.skip("No transient results to write")
