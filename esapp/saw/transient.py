from typing import List, Tuple, Union

import numpy as np
import pandas as pd


class TransientMixin:
    def TSGetContingencyResults(
        self,
        CtgName: str,
        ObjFieldList: List[str],
        StartTime: Union[None, int, float] = None,
        StopTime: Union[None, int, float] = None,
    ) -> Union[Tuple[None, None], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        WARNING: This function should only be used after the simulation
        is run (for example, use this after running script commands
        TSSolveAll or TSSolve).

        The TSGetContingencyResults function is used to read
        transient stability results into an external program (Python)
        using SimAuto.

        `PowerWorld documentation:
        <https://www.powerworld.com/WebHelp/#MainDocumentation_HTML/TSGetContingencyResults%20Function.htm%3FTocPath%3DAutomation%2520Server%2520Add-On%2520(SimAuto)%7CAutomation%2520Server%2520Functions%7C_____49>`__

        Parameters
        ----------
        CtgName : str
            The contingency to obtain results from. Only one
            contingency be obtained at a time.
        ObjFieldList : List[str]
            A list of strings which may contain plots,
            subplots, or individual object/field pairs specifying the
            result variables to obtain.
        StartTime : Union[None, int, float], optional
            The time in seconds in the simulation to begin
            retrieving results. If not specified (None), the start time
            of the simulation is used. Defaults to None.
        StopTime : Union[None, int, float], optional
            The time in seconds in the simulation to stop
            retrieving results. If not specified, the end time of the
            simulation is used. Defaults to None.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame] or Tuple[None, None]
            A tuple containing two DataFrames, "meta" and "data."
            Alternatively, if the given CtgName does not exist, a tuple
            of (None, None) will be returned.
        """
        out = self._call_simauto(
            "TSGetContingencyResults",
            CtgName,
            ObjFieldList,
            str(StartTime),
            str(StopTime),
        )
        # We get (None, (None,)) if the contingency does not exist.
        if out == (None, (None,)):
            return None, None

        assert len(out) == 2, "Unexpected return format from PowerWorld."

        # Extract the meta data.
        meta = pd.DataFrame(
            out[0],
            columns=[
                "ObjectType",
                "PrimaryKey",
                "SecondaryKey",
                "Label",
                "VariableName",
                "ColHeader",
            ],
        )

        # Remove extraneous white space in the strings.
        meta = meta.apply(lambda x: x.str.strip(), axis=0)

        # Extract the data.
        data = pd.DataFrame(out[1])

        # Align column names with meta frame and set time column
        data.rename(columns=lambda x: x - 1, inplace=True)
        data.rename(columns={-1: "time"}, inplace=True)

        # Attempt to convert all columns to numeric.
        data = self._to_numeric(data, errors="ignore")
        meta = self._to_numeric(meta, errors="ignore")

        return meta, data

    def TSTransferStateToPowerFlow(self, calculate_mismatch: bool = False):
        """Transfers the current transient stability state to the power flow.

        After running a transient stability simulation, this allows the
        state of the system at the final time step to be loaded into the
        power flow solver for steady-state analysis.

        This is a wrapper for the ``TSTransferStateToPowerFlow`` script command.
        
        Parameters
        ----------
        calculate_mismatch : bool, optional
            Set to True to calculate power mismatch when transferring. Defaults to False.
        """
        cm = "YES" if calculate_mismatch else "NO"
        self.RunScriptCommand(f"TSTransferStateToPowerFlow({cm});")

    def TSInitialize(self):
        """Initializes the transient stability simulation parameters.

        This command must be called before solving a transient stability
        run. It prepares the simulation engine with the model data.

        This is a wrapper for the ``TSInitialize`` script command.
        """
        try:
            self.RunScriptCommand("TSInitialize()")
        except Exception:
            self.log.warning("Failed to Initialize TS Values")

    def TSResultStorageSetAll(self, object="ALL", value=True):
        """Sets the 'Store results in RAM' flag for all objects of a given type.

        This is a wrapper for the ``TSResultStorageSetAll`` script command.

        Parameters
        ----------
        object : str, optional
            The PowerWorld object type (e.g., "GEN", "BUS", "BRANCH").
            Defaults to "ALL".
        value : bool, optional
            If True, results for this object type will be stored.
            If False, they will not. Defaults to True.
        """
        yn = "YES" if value else "NO"
        self.RunScriptCommand(f"TSResultStorageSetAll({object}, {yn})")

    def TSSolve(self, ctgname: str):
        """Solves a single transient stability contingency.
        
        This is a wrapper for the ``TSSolve`` script command.
        
        Parameters
        ----------
        ctgname : str
            The name of the contingency to solve.
        """
        self.RunScriptCommand(f'TSSolve("{ctgname}")')

    def TSSolveAll(self):
        """Solves all defined transient stability contingencies.
        
        This is a wrapper for the ``TSSolveAll`` script command.
        """
        self.RunScriptCommand("TSSolveAll()")

    def TSStoreResponse(self, object_type: str = "ALL", value: bool = True):
        """Convenience wrapper to toggle transient stability result storage.

        This is a high-level wrapper around ``TSResultStorageSetAll``.
        
        Parameters
        ----------
        object_type : str, optional
            The PowerWorld object type (e.g., "GEN", "BUS", "BRANCH").
            Defaults to "ALL".
        value : bool, optional
            If True, results will be stored. If False, they will not.
            Defaults to True.
        """
        self.TSResultStorageSetAll(object=object_type, value=value)

    def TSClearResultsFromRAM(
        self,
        ctg_name: str = "ALL",
        clear_summary: bool = True,
        clear_events: bool = True,
        clear_statistics: bool = True,
        clear_time_values: bool = True,
        clear_solution_details: bool = True,
    ):
        """Clears all transient stability results from RAM.

        This is useful for managing memory when running many simulations.

        This is a wrapper for the ``TSClearResultsFromRAM`` script command.
        """
        if ctg_name.upper() not in ["ALL", "SELECTED"] and not ctg_name.startswith('"'):
            ctg_name = f'"{ctg_name}"'

        c_sum = "YES" if clear_summary else "NO"
        c_evt = "YES" if clear_events else "NO"
        c_stat = "YES" if clear_statistics else "NO"
        c_time = "YES" if clear_time_values else "NO"
        c_sol = "YES" if clear_solution_details else "NO"
        self.RunScriptCommand(f"TSClearResultsFromRAM({ctg_name},{c_sum},{c_evt},{c_stat},{c_time},{c_sol});")

    def TSClearPlayInSignals(self) -> None:
        """Deletes all defined PlayIn signals.

        This is a wrapper for the ``DELETE(PLAYINSIGNAL)`` script command.
        """
        self.RunScriptCommand("DELETE(PLAYINSIGNAL);")

    def TSSetPlayInSignals(self, name: str, times: np.ndarray, signals: np.ndarray) -> None:
        """Sets PlayIn signals using an AUX file command.

        This method constructs and executes an AUX data block to define
        transient stability play-in signals.

        :param name: Name of the PlayIn Signal configuration.
        :param times: 1D NumPy array of time points.
        :param signals: 2D NumPy array of signal values (rows=time, cols=signals).
        """
        if times.ndim != 1 or signals.ndim != 2 or times.shape[0] != signals.shape[0]:
            raise ValueError("Dimension mismatch in times and signals arrays.")

        # Format Data Header
        header_fields = ["TSName", "TSTime"]
        if signals.shape[1] > 0:
            header_fields.append("TSSignal")
            for i in range(2, signals.shape[1] + 1):
                header_fields.append(f"TSSignal:{i}")

        header = f"DATA (PLAYINSIGNAL, [{', '.join(header_fields)}]){{\n"

        # Format each time record
        body = []
        for t, row in zip(times, signals):
            row_str = "\t".join([f"{d:.6f}" for d in row])
            body.append(f'"{name}"\t{t:.6f}\t{row_str}')

        cmd = header + "\n".join(body) + "\n}\n"

        # Execute
        self.exec_aux(cmd)

    def TSClearResultsFromRAMAndDisableStorage(self) -> None:
        """Disables result storage in RAM and clears any existing results.

        This is a convenience method that calls ``TSResultStorageSetAll(value=False)``
        followed by ``TSClearResultsFromRAM()``.
        """
        self.TSResultStorageSetAll(value=False)
        self.TSClearResultsFromRAM()

    def TSAutoCorrect(self):
        """Runs auto correction of parameters for transient stability."""
        return self.RunScriptCommand("TSAutoCorrect;")

    def TSClearAllModels(self):
        """Clears all transient stability models."""
        return self.RunScriptCommand("TSClearAllModels;")

    def TSValidate(self):
        """Validate transient stability models and input values."""
        return self.RunScriptCommand("TSValidate;")

    def TSWriteOptions(
        self,
        filename: str,
        save_dynamic_model: bool = True,
        save_stability_options: bool = True,
        save_stability_events: bool = True,
        save_results_events: bool = True,
        save_plot_definitions: bool = True,
        save_transient_limit_monitors: bool = True,
        save_result_analyzer_time_window: bool = True,
        key_field: str = "PRIMARY",
    ):
        """Save transient stability option settings to an auxiliary file."""
        opts = [
            "YES" if save_dynamic_model else "NO",
            "YES" if save_stability_options else "NO",
            "YES" if save_stability_events else "NO",
            "YES" if save_results_events else "NO",
            "YES" if save_plot_definitions else "NO",
            "YES" if save_transient_limit_monitors else "NO",
            "YES" if save_result_analyzer_time_window else "NO",
        ]
        opt_str = "[" + ", ".join(opts) + "]"
        return self.RunScriptCommand(f'TSWriteOptions("{filename}", {opt_str}, {key_field});')

    def TSLoadPTI(self, filename: str):
        """Loads transient stability data in the PTI format."""
        return self.RunScriptCommand(f'TSLoadPTI("{filename}");')

    def TSLoadGE(self, filename: str):
        """Loads transient stability data stored in the GE DYD format."""
        return self.RunScriptCommand(f'TSLoadGE("{filename}");')

    def TSLoadBPA(self, filename: str):
        """Loads transient stability data stored in the BPA format."""
        return self.RunScriptCommand(f'TSLoadBPA("{filename}");')

    def TSAutoInsertDistRelay(
        self, reach: float, add_from: bool, add_to: bool, transfer_trip: bool, shape: int, filter_name: str
    ):
        """Inserts DistRelay models on the lines meeting the specified filter."""
        af = "YES" if add_from else "NO"
        at = "YES" if add_to else "NO"
        tt = "YES" if transfer_trip else "NO"
        self.RunScriptCommand(f'TSAutoInsertDistRelay({reach}, {af}, {at}, {tt}, {shape}, "{filter_name}");')

    def TSAutoInsertZPOTT(self, reach: float, filter_name: str):
        """Inserts ZPOTT models on the lines meeting the specified filter."""
        self.RunScriptCommand(f'TSAutoInsertZPOTT({reach}, "{filter_name}");')

    def TSAutoSavePlots(
        self,
        plot_names: List[str],
        ctg_names: List[str],
        image_type: str = "JPG",
        width: int = 800,
        height: int = 600,
        font_scalar: float = 1.0,
        include_case_name: bool = False,
        include_category: bool = False,
    ):
        """Create and save images of the plots."""
        plots = "[" + ", ".join([f'"{p}"' for p in plot_names]) + "]"
        ctgs = "[" + ", ".join([f'"{c}"' for c in ctg_names]) + "]"
        icn = "YES" if include_case_name else "NO"
        icat = "YES" if include_category else "NO"
        self.RunScriptCommand(
            f"TSAutoSavePlots({plots}, {ctgs}, {image_type}, {width}, {height}, {font_scalar}, {icn}, {icat});"
        )

    def TSCalculateCriticalClearTime(self, element_or_filter: str):
        """Calculate critical clearing time for faults."""
        self.RunScriptCommand(f"TSCalculateCriticalClearTime({element_or_filter});")

    def TSCalculateSMIBEigenValues(self):
        """Calculate single machine infinite bus eigenvalues."""
        self.RunScriptCommand("TSCalculateSMIBEigenValues;")

    def TSClearModelsforObjects(self, object_type: str, filter_name: str = ""):
        """Deletes all transient stability models associated with the objects that meet the filter."""
        self.RunScriptCommand(f'TSClearModelsforObjects({object_type}, "{filter_name}");')

    def TSDisableMachineModelNonZeroDerivative(self, threshold: float = 0.001):
        """Disable machine models with non-zero state derivatives."""
        self.RunScriptCommand(f"TSDisableMachineModelNonZeroDerivative({threshold});")

    def TSGetVCurveData(self, filename: str, filter_name: str):
        """Generates V-curve data for synchronous generators."""
        self.RunScriptCommand(f'TSGetVCurveData("{filename}", "{filter_name}");')

    def TSWriteResultsToCSV(
        self,
        filename: str,
        mode: str,
        contingencies: List[str],
        plots_fields: List[str],
        start_time: float = None,
        end_time: float = None,
    ):
        """Save out results for specific variables to CSV."""
        ctgs = "[" + ", ".join([f'"{c}"' for c in contingencies]) + "]"
        pfs = "[" + ", ".join([f'"{p}"' for p in plots_fields]) + "]"
        time_args = ""
        if start_time is not None and end_time is not None:
            time_args = f", {start_time}, {end_time}"
        self.RunScriptCommand(f'TSGetResults("{filename}", {mode}, {ctgs}, {pfs}{time_args});')

    def TSJoinActiveCTGs(
        self, time_delay: float, delete_existing: bool, join_with_self: bool, filename: str = "", first_ctg: str = "Both"
    ):
        """Joins two lists of TSContingency objects."""
        de = "YES" if delete_existing else "NO"
        jws = "YES" if join_with_self else "NO"
        self.RunScriptCommand(f'TSJoinActiveCTGs({time_delay}, {de}, {jws}, "{filename}", {first_ctg});')

    def TSLoadRDB(self, filename: str, model_type: str, filter_name: str = ""):
        """Loads a SEL RDB file."""
        self.RunScriptCommand(f'TSLoadRDB("{filename}", {model_type}, "{filter_name}");')

    def TSLoadRelayCSV(self, filename: str, model_type: str, filter_name: str = ""):
        """Loads relay data from CSV."""
        self.RunScriptCommand(f'TSLoadRelayCSV("{filename}", {model_type}, "{filter_name}");')

    def TSPlotSeriesAdd(
        self,
        plot_name: str,
        sub_plot_num: int,
        axis_group_num: int,
        object_type: str,
        field_name: str,
        filter_name: str = "",
        attributes: str = "",
    ):
        """Adds one or multiple plot series to a new or existing plot definition."""
        self.RunScriptCommand(
            f'TSPlotSeriesAdd("{plot_name}", {sub_plot_num}, {axis_group_num}, {object_type}, {field_name}, "{filter_name}", "{attributes}");'
        )

    def TSRunResultAnalyzer(self, ctg_name: str = ""):
        """Run the Transient Result Analyzer."""
        self.RunScriptCommand(f'TSRunResultAnalyzer("{ctg_name}");')

    def TSRunUntilSpecifiedTime(
        self,
        ctg_name: str,
        stop_time: float = None,
        step_size: float = None,
        steps_in_cycles: bool = False,
        reset_start_time: bool = False,
        steps_to_do: int = 0,
    ):
        """Allows manual control of the transient stability run."""
        opts = []
        if stop_time is not None:
            opts.append(str(stop_time))
        if step_size is not None:
            opts.append(str(step_size))
        if steps_in_cycles:
            opts.append("YES")
        else:
            opts.append("NO")
        if reset_start_time:
            opts.append("YES")
        else:
            opts.append("NO")
        if steps_to_do > 0:
            opts.append(str(steps_to_do))

        opt_str = "[" + ", ".join(opts) + "]"
        self.RunScriptCommand(f'TSRunUntilSpecifiedTime("{ctg_name}", {opt_str});')

    def TSSaveBPA(self, filename: str, diff_case_modified_only: bool = False):
        """Save transient stability data stored in the BPA IPF format."""
        dc = "YES" if diff_case_modified_only else "NO"
        self.RunScriptCommand(f'TSSaveBPA("{filename}", {dc});')

    def TSSaveGE(self, filename: str, diff_case_modified_only: bool = False):
        """Save transient stability data stored in the GE DYD format."""
        dc = "YES" if diff_case_modified_only else "NO"
        self.RunScriptCommand(f'TSSaveGE("{filename}", {dc});')

    def TSSavePTI(self, filename: str, diff_case_modified_only: bool = False):
        """Save transient stability data stored in the PTI DYR format."""
        dc = "YES" if diff_case_modified_only else "NO"
        self.RunScriptCommand(f'TSSavePTI("{filename}", {dc});')

    def TSSaveTwoBusEquivalent(self, filename: str, bus_identifier: str):
        """Save the two bus equivalent model of a specified bus to a PWB file."""
        self.RunScriptCommand(f'TSSaveTwoBusEquivalent("{filename}", {bus_identifier});')

    def TSWriteModels(self, filename: str, diff_case_modified_only: bool = False):
        """Save transient stability dynamic model records only the auxiliary file format."""
        dc = "YES" if diff_case_modified_only else "NO"
        self.RunScriptCommand(f'TSWriteModels("{filename}", {dc});')

    def TSSetSelectedForTransientReferences(
        self, set_what: str, set_how: str, object_types: List[str], model_types: List[str]
    ):
        """Set the Custom Integer field or Selected field for objects referenced in a transient stability model."""
        objs = "[" + ", ".join(object_types) + "]"
        models = "[" + ", ".join(model_types) + "]"
        self.RunScriptCommand(f"TSSetSelectedForTransientReferences({set_what}, {set_how}, {objs}, {models});")

    def TSSaveDynamicModels(
        self, filename: str, file_type: str, object_type: str, filter_name: str = "", append: bool = False
    ):
        """Save dynamics models for specified object types to file."""
        app = "YES" if append else "NO"
        self.RunScriptCommand(
            f'TSSaveDynamicModels("{filename}", {file_type}, {object_type}, "{filter_name}", {app});'
        )