"""
Integration tests for Transient Stability via SAW.

These are **integration tests** that require a live connection to PowerWorld
Simulator via the SimAuto COM interface. They test transient stability
initialization, solving, model I/O, result storage, play-in signals,
relay insertion, and file format saves (PTI, GE, BPA).

REQUIREMENTS:
    - PowerWorld Simulator installed with SimAuto COM registered
    - A valid PowerWorld case file path set in ``tests/config_test.py``
      (variable ``SAW_TEST_CASE``) or via the ``SAW_TEST_CASE`` env variable

RELATED TEST FILES:
    - test_integration_saw_core.py          -- base SAW operations, logging, I/O
    - test_integration_saw_modify.py        -- destructive modify, region, case actions
    - test_integration_saw_powerflow.py     -- power flow, matrices, sensitivity, topology
    - test_integration_saw_contingency.py   -- contingency and fault analysis
    - test_integration_saw_gic.py           -- GIC analysis
    - test_integration_saw_operations.py    -- ATC, OPF, PV/QV, time step, weather, scheduled
    - test_integration_workbench.py         -- GridWorkBench facade and statics
    - test_integration_network.py           -- Network topology

USAGE:
    pytest tests/test_integration_saw_transient.py -v
"""

import os
import pytest
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


class TestTransient:
    """SAW-level transient stability commands."""

    @pytest.mark.order(8100)
    def test_transient_initialize(self, saw_instance):
        saw_instance.TSInitialize()

    @pytest.mark.order(8200)
    def test_transient_options(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteOptions(tmp_aux)
        assert os.path.exists(tmp_aux)

    @pytest.mark.order(8300)
    def test_transient_critical_time(self, saw_instance):
        branches = saw_instance.GetParametersMultipleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit"])
        assert branches is not None and not branches.empty, "Test case must contain branches"
        b = branches.iloc[0]
        branch_str = create_object_string("Branch", b["BusNum"], b["BusNum:1"], b["LineCircuit"])
        saw_instance.TSCalculateCriticalClearTime(branch_str)

    @pytest.mark.order(8400)
    def test_transient_playin(self, saw_instance):
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [1.0]])
        saw_instance.TSSetPlayInSignals("TestSignal", times, signals)

    @pytest.mark.order(8450)
    def test_transient_write_results(self, saw_instance, temp_file):
        """Write TS results to CSV after solving a minimal contingency."""
        tmp_csv = temp_file(".csv")
        try:
            saw_instance.RunScriptCommand("Delete(TSContingency);")
        except (PowerWorldError, PowerWorldPrerequisiteError):
            pass
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCTG"])
        ts_ctgs = saw_instance.ListOfDevices("TSContingency")
        if ts_ctgs is None or ts_ctgs.empty:
            pytest.skip("Unable to create TS contingencies for this case")
        name_col = "TSCTGName" if "TSCTGName" in ts_ctgs.columns else ts_ctgs.columns[0]
        ctg_name = str(ts_ctgs.iloc[0][name_col]).strip()
        saw_instance.TSAutoCorrect()
        saw_instance.TSInitialize()
        try:
            saw_instance.TSSolve(ctg_name, start_time=0.0, stop_time=0.1, step_size=0.01)
        except (PowerWorldError, PowerWorldPrerequisiteError) as e:
            pytest.skip(f"TS simulation cannot start for this case: {e}")
        saw_instance.TSGetResults("SINGLE", [ctg_name], ["Gen ALL | GenMW"], filename=tmp_csv)
        assert os.path.exists(tmp_csv)

    @pytest.mark.order(8500)
    def test_transient_save_models(self, saw_instance, temp_file):
        tmp_aux = temp_file(".aux")
        saw_instance.TSWriteModels(tmp_aux)
        assert os.path.exists(tmp_aux)

        tmp_aux2 = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp_aux2, "AUX", "Gen")
        assert os.path.exists(tmp_aux2)

    @pytest.mark.order(9850)
    def test_transient_result_storage(self, saw_instance):
        saw_instance.TSResultStorageSetAll("Gen", True)
        saw_instance.TSResultStorageSetAll("Gen", False)

    @pytest.mark.order(9810)
    def test_transient_clear_playin(self, saw_instance):
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(64000)
    def test_transient_transfer_state(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=True)

    @pytest.mark.order(64100)
    def test_transient_transfer_state_no_mismatch(self, saw_instance):
        saw_instance.TSTransferStateToPowerFlow(calculate_mismatch=False)

    @pytest.mark.order(64200)
    def test_transient_store_response(self, saw_instance):
        saw_instance.TSStoreResponse("Gen", True)
        saw_instance.TSStoreResponse("Gen", False)

    @pytest.mark.order(64250)
    def test_transient_smib_eigenvalues(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSCalculateSMIBEigenValues()

    @pytest.mark.order(64300)
    def test_transient_clear_results(self, saw_instance):
        """Clear TS results: RAM, specific CTG params, and disable storage."""
        saw_instance.TSClearResultsFromRAM()
        saw_instance.TSClearResultsFromRAM(
            ctg_name="ALL",
            clear_summary=True,
            clear_events=False,
            clear_statistics=True,
            clear_time_values=False,
            clear_solution_details=True,
        )
        saw_instance.TSClearResultsFromRAMAndDisableStorage()

    @pytest.mark.order(64350)
    def test_transient_run_until_specified_time(self, saw_instance):
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSInitialize()
        saw_instance.TSRunUntilSpecifiedTime("TestCtg", stop_time=0.1, step_size=0.01)

    @pytest.mark.order(64600)
    def test_transient_clear_all_models(self, saw_instance):
        saw_instance.TSClearAllModels()

    @pytest.mark.order(64800)
    def test_transient_clear_models_for_objects(self, saw_instance):
        saw_instance.TSClearModelsforObjects("Gen")

    @pytest.mark.order(64900)
    def test_transient_disable_machine_model(self, saw_instance):
        saw_instance.TSInitialize()
        saw_instance.TSDisableMachineModelNonZeroDerivative(threshold=0.01)

    @pytest.mark.order(65000)
    def test_transient_auto_insert_dist_relay(self, saw_instance):
        saw_instance.TSAutoInsertDistRelay(
            reach=1.0, add_from=True, add_to=False,
            transfer_trip=True, shape=1, filter_name="ALL",
        )

    @pytest.mark.order(65100)
    def test_transient_auto_insert_zpott(self, saw_instance):
        saw_instance.TSAutoInsertZPOTT(reach=1.0, filter_name="ALL")

    @pytest.mark.order(65200)
    def test_transient_run_result_analyzer(self, saw_instance):
        saw_instance.TSRunResultAnalyzer()

    @pytest.mark.order(65400)
    def test_transient_save_formats(self, saw_instance, temp_file):
        """Save TS models in PTI, GE, and BPA formats."""
        tmp_dyr = temp_file(".dyr")
        saw_instance.TSSavePTI(tmp_dyr)

        tmp_dyd = temp_file(".dyd")
        saw_instance.TSSaveGE(tmp_dyd)

        tmp_bpa = temp_file(".bpa")
        saw_instance.TSSaveBPA(tmp_bpa)

    @pytest.mark.order(65500)
    def test_transient_save_formats_diff(self, saw_instance, temp_file):
        """Save TS models with diff_case_modified_only in PTI, GE, BPA."""
        tmp_dyr = temp_file(".dyr")
        saw_instance.TSSavePTI(tmp_dyr, diff_case_modified_only=True)

        tmp_dyd = temp_file(".dyd")
        saw_instance.TSSaveGE(tmp_dyd, diff_case_modified_only=True)

        tmp_bpa = temp_file(".bpa")
        saw_instance.TSSaveBPA(tmp_bpa, diff_case_modified_only=True)

    @pytest.mark.order(65600)
    def test_transient_write_models_diff(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        saw_instance.TSWriteModels(tmp, diff_case_modified_only=True)

    @pytest.mark.order(65700)
    def test_transient_write_options_custom(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
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

    @pytest.mark.order(65800)
    def test_transient_save_two_bus_equiv(self, saw_instance, temp_file):
        buses = saw_instance.GetParametersMultipleElement("Bus", ["BusNum"])
        assert buses is not None and not buses.empty, "Test case must contain buses"
        bus_key = create_object_string("Bus", buses.iloc[0]["BusNum"])
        tmp = temp_file(".pwb")
        saw_instance.TSSaveTwoBusEquivalent(tmp, bus_key)

    @pytest.mark.order(65900)
    def test_transient_join_active_ctgs(self, saw_instance):
        try:
            saw_instance.TSJoinActiveCTGs(0.1, delete_existing=True, join_with_self=False)
        except PowerWorldError:
            pass

    @pytest.mark.order(66000)
    def test_transient_set_selected_for_refs(self, saw_instance):
        try:
            saw_instance.TSSetSelectedForTransientReferences(
                "ALL", "SET", ["Gen"], ["GENROU"]
            )
        except PowerWorldError:
            pass

    @pytest.mark.order(66100)
    def test_transient_save_dynamic_models_append(self, saw_instance, temp_file):
        tmp = temp_file(".aux")
        saw_instance.TSSaveDynamicModels(tmp, "AUX", "Gen", append=True)

    @pytest.mark.order(66200)
    def test_transient_plot_series_add(self, saw_instance):
        try:
            saw_instance.TSPlotSeriesAdd("TestPlot", 1, 1, "Gen", "GenMW")
        except PowerWorldError:
            pass

    @pytest.mark.order(66300)
    def test_transient_get_vcurve_data(self, saw_instance, temp_file):
        tmp = temp_file(".csv")
        try:
            saw_instance.TSGetVCurveData(tmp, "")
        except PowerWorldError:
            pass

    @pytest.mark.order(77400)
    def test_ts_solve_with_time_params(self, saw_instance):
        """TSSolve with explicit start/stop/step parameters."""
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSInitialize()
        saw_instance.TSSolve(
            "TestCtg", start_time=0.0, stop_time=0.1,
            step_size=0.01, step_in_cycles=False,
        )

    @pytest.mark.order(77500)
    def test_ts_solve_all(self, saw_instance):
        """TSSolveAll completes without error."""
        saw_instance.TSInitialize()
        saw_instance.TSSolveAll()

    @pytest.mark.order(77600)
    def test_ts_clear_results_named_ctg(self, saw_instance):
        """TSClearResultsFromRAM with a specific contingency name."""
        saw_instance.CreateData("TSContingency", ["TSCTGName"], ["TestCtg"])
        saw_instance.TSResultStorageSetAll("ALL", True)
        saw_instance.TSInitialize()
        saw_instance.TSSolve("TestCtg")
        saw_instance.TSClearResultsFromRAM(ctg_name="TestCtg")

    @pytest.mark.order(77700)
    def test_ts_auto_save_plots(self, saw_instance):
        """TSAutoSavePlots completes without error."""
        saw_instance.TSAutoSavePlots(
            plot_names=["TestPlot"],
            ctg_names=["TestCtg"],
            image_type="JPG",
            width=800, height=600, font_scalar=1.0,
            include_case_name=True, include_category=True,
        )

    @pytest.mark.order(77800)
    def test_ts_load_rdb(self, saw_instance, temp_file):
        """TSLoadRDB with a file path."""
        tmp = temp_file(".rdb")
        saw_instance.TSLoadRDB(tmp, "DISTRELAY")

    @pytest.mark.order(77900)
    def test_ts_load_relay_csv(self, saw_instance, temp_file):
        """TSLoadRelayCSV with a file path."""
        tmp = temp_file(".csv")
        with pytest.raises((PowerWorldError, PowerWorldPrerequisiteError)):
            saw_instance.TSLoadRelayCSV(tmp, "DISTRELAY")

    @pytest.mark.order(78000)
    def test_ts_run_until_specified_time_steps(self, saw_instance):
        """TSRunUntilSpecifiedTime with steps_to_do parameter."""
        saw_instance.TSInitialize()
        saw_instance.TSRunUntilSpecifiedTime(
            "TestCtg", stop_time=0.1, step_size=0.01,
            steps_in_cycles=False, reset_start_time=False,
            steps_to_do=5,
        )

    @pytest.mark.order(78100)
    def test_ts_load_pti(self, saw_instance, temp_file):
        """TSLoadPTI with a file path."""
        tmp = temp_file(".dyr")
        saw_instance.TSLoadPTI(tmp)

    @pytest.mark.order(78200)
    def test_ts_load_ge(self, saw_instance, temp_file):
        """TSLoadGE with a file path."""
        tmp = temp_file(".dyd")
        saw_instance.TSLoadGE(tmp)

    @pytest.mark.order(78300)
    def test_ts_load_bpa(self, saw_instance, temp_file):
        """TSLoadBPA with a file path."""
        tmp = temp_file(".bpa")
        saw_instance.TSLoadBPA(tmp)

    @pytest.mark.order(78400)
    def test_ts_clear_playin_signals(self, saw_instance):
        """TSClearPlayInSignals completes without error."""
        saw_instance.TSClearPlayInSignals()

    @pytest.mark.order(78500)
    def test_ts_set_playin_signals_multi_col(self, saw_instance):
        """TSSetPlayInSignals with multiple signal columns."""
        times = np.array([0.0, 0.1, 0.2])
        signals = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
        saw_instance.TSSetPlayInSignals("MultiSignal", times, signals)

    @pytest.mark.order(78600)
    def test_ts_set_playin_signals_validation(self, saw_instance):
        """TSSetPlayInSignals raises ValueError for mismatched dimensions."""
        times = np.array([0.0, 0.1])
        signals = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            saw_instance.TSSetPlayInSignals("Bad", times, signals)

    @pytest.mark.order(78700)
    def test_ts_validate(self, saw_instance):
        """TSValidate completes without error."""
        saw_instance.TSValidate()

    @pytest.mark.order(78800)
    def test_ts_auto_correct(self, saw_instance):
        """TSAutoCorrect runs and may find/fix validation errors."""
        try:
            saw_instance.TSAutoCorrect()
        except PowerWorldError as e:
            assert "Validation Errors were found" in str(e), f"Unexpected error: {e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
