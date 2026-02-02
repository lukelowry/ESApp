from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from scipy.sparse import csr_matrix

from .utils.gic import GIC
from .utils.network import Network
from .utils.dynamics import get_ts_results, process_ts_results
from .indexable import Indexable
from .components import Bus, Branch, Gen, Load, Shunt, Area, Zone, Substation, Contingency, Sim_Solution_Options
from .saw import SAW, create_object_string

import tempfile
import os

class GridWorkBench(Indexable):
    """
    Main entry point for interacting with the PowerWorld grid model.
    """
    def __init__(self, fname: Optional[str] = None):
        """
        Initialize the GridWorkBench.

        Parameters
        ----------
        fname : str, optional
            Path to the PowerWorld case file (.pwb).
        """
        # Applications
        self.network = Network()
        self.gic     = GIC()

        if fname:
            self.fname = fname
            self.open()
        else:
            self.esa = None
            self.fname = None

        # Propagate the esa instance to the applications.
        self.set_esa(self.esa)

    def set_esa(self, esa: Optional[SAW]) -> None:
        """Sets the SAW instance for the workbench and its applications."""
        super().set_esa(esa)
        self.network.set_esa(esa)
        self.gic.set_esa(esa)

    # --- Solver Options (from Statics) ---

    def _set_option(self, key: str, enable: bool) -> None:
        """Set a Sim_Solution_Options boolean flag."""
        self[Sim_Solution_Options, key] = 'YES' if enable else 'NO'

    def set_do_one_iteration(self, enable: bool = True) -> None:
        """Enable/disable single iteration mode for power flow."""
        self._set_option('DoOneIteration', enable)

    def set_max_iterations(self, val: int = 250) -> None:
        """Set maximum number of iterations for power flow convergence."""
        self[Sim_Solution_Options, 'MaxItr'] = val

    def set_disable_angle_rotation(self, enable: bool = True) -> None:
        """Enable/disable angle rotation during power flow."""
        self._set_option('DisableAngleRotation', enable)

    def set_disable_opt_mult(self, enable: bool = True) -> None:
        """Enable/disable optimal multiplier during power flow."""
        self._set_option('DisableOptMult', enable)

    def enable_inner_ss_check(self, enable: bool = True) -> None:
        """Enable/disable inner steady-state contingency check."""
        self._set_option('SSContPFInnerLoop', enable)

    def disable_gen_mvr_check(self, enable: bool = True) -> None:
        """Enable/disable generator MVAR limit checking."""
        self._set_option('DisableGenMVRCheck', enable)

    def enable_inner_check_gen_vars(self, enable: bool = True) -> None:
        """Enable/disable inner loop generator VAR checking."""
        self._set_option('ChkVars', enable)

    def enable_inner_backoff_gen_vars(self, enable: bool = True) -> None:
        """Enable/disable inner loop generator VAR backoff."""
        self._set_option('ChkVars:1', enable)

    # --- Bus Voltage & Analysis ---

    def voltage(
        self,
        complex: bool = True,
        pu: bool = True,
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Retrieve bus voltages.

        Parameters
        ----------
        complex : bool, default True
            If True, return complex voltage. Else (magnitude, angle_rad).
        pu : bool, default True
            If True, per-unit. Else kV.

        Returns
        -------
        pd.Series or tuple of pd.Series
        """
        fields = ["BusPUVolt", "BusAngle"] if pu else ["BusKVVolt", "BusAngle"]
        df = self[Bus, fields]
        mag = df[fields[0]]
        ang = df['BusAngle'] * np.pi / 180.0
        if complex:
            return mag * np.exp(1j * ang)
        return mag, ang

    def set_voltages(self, V: np.ndarray) -> None:
        """
        Set bus voltages from a complex vector.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector (per-unit).
        """
        V_df = np.vstack([np.abs(V), np.angle(V, deg=True)]).T
        self[Bus, ["BusPUVolt", "BusAngle"]] = V_df

    def violations(self, v_min: float = 0.9, v_max: float = 1.1) -> DataFrame:
        """
        Return bus voltage violations.

        Parameters
        ----------
        v_min : float, default 0.9
            Low voltage threshold (pu).
        v_max : float, default 1.1
            High voltage threshold (pu).

        Returns
        -------
        DataFrame
            Columns 'Low' and 'High' with violating bus voltages.
        """
        v = self.voltage(complex=False, pu=True)[0]
        low = v[v < v_min]
        high = v[v > v_max]
        return DataFrame({'Low': low, 'High': high})

    def mismatch(self, asComplex: bool = False):
        """
        Return bus power mismatches.

        Parameters
        ----------
        asComplex : bool, default False
            If True, return P + jQ as complex Series.

        Returns
        -------
        tuple of pd.Series or pd.Series
            (P, Q) mismatches or complex S = P + jQ.
        """
        df = self[Bus, ["BusMismatchP", "BusMismatchQ"]]
        P = df['BusMismatchP']
        Q = df['BusMismatchQ']
        if asComplex:
            return P + 1j * Q
        return P, Q

    def netinj(self, asComplex: bool = False):
        """
        Sum of all generator, load, bus shunt, and switched shunt P and Q.

        Parameters
        ----------
        asComplex : bool, default False
            If True, return P + jQ as complex array.

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            (P, Q) or complex S = P + jQ.
        """
        df = self[Bus, ['BusNetMW', 'BusNetMVR']]
        P = df['BusNetMW'].to_numpy()
        Q = df['BusNetMVR'].to_numpy()
        if asComplex:
            return P + 1j * Q
        return P, Q

    # --- Matrix Retrieval ---

    def ybus(self, dense: bool = False):
        """
        Return the Y-Bus matrix.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
        """
        return self.esa.get_ybus(dense)

    def branch_admittance(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Compute branch admittance matrices Yf and Yt.

        Returns
        -------
        tuple of csr_matrix
            (Yf, Yt) branch-bus admittance matrices.
        """
        bus_df = self[Bus, ["BusNum"]]
        branch_df = self[Branch, [
            "BusNum", "BusNum:1", "LineCircuit",
            "LineR", "LineX", "LineC",
        ]]

        nb = len(bus_df)
        nl = len(branch_df)

        R = pd.to_numeric(branch_df["LineR"], errors="coerce").fillna(0).to_numpy(dtype=float)
        X = pd.to_numeric(branch_df["LineX"], errors="coerce").fillna(0).to_numpy(dtype=float)
        Z = R + 1j * X
        # Replace zero impedance with a small value (short circuit approximation)
        zero_mask = np.abs(Z) < 1e-20
        Z = np.where(zero_mask, 1e-12, Z)
        Ys = 1 / Z
        Bc = pd.to_numeric(branch_df["LineC"], errors="coerce").fillna(0).to_numpy(dtype=float)
        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt
        Yft = -Ys
        Ytf = -Ys

        bus_to_idx = {bus: idx for idx, bus in enumerate(bus_df["BusNum"])}
        f = np.array([bus_to_idx[b] for b in branch_df["BusNum"]])
        t = np.array([bus_to_idx[b] for b in branch_df["BusNum:1"]])

        i = np.r_[range(nl), range(nl)]
        Yf = csr_matrix(
            (np.hstack([Yff.reshape(-1), Yft.reshape(-1)]),
             (i, np.hstack([f, t]))),
            (nl, nb),
        )
        Yt = csr_matrix(
            (np.hstack([Ytf.reshape(-1), Ytt.reshape(-1)]),
             (i, np.hstack([f, t]))),
            (nl, nb),
        )
        return Yf, Yt

    def jacobian(self, dense: bool = False, form: str = 'R'):
        """
        Get the power flow Jacobian matrix.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.
        form : str, default 'R'
            Coordinate form: 'R' (rectangular), 'P' (polar), 'DC' (B').

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix
        """
        return self.esa.get_jacobian(dense, form=form)

    def jacobian_with_ids(self, dense: bool = False, form: str = 'R'):
        """
        Get the Jacobian matrix with row/column ID labels.

        Parameters
        ----------
        dense : bool, default False
            If True, return dense array. Else sparse CSR.
        form : str, default 'R'
            Coordinate form: 'R' (rectangular), 'P' (polar), 'DC' (B').

        Returns
        -------
        tuple
            (matrix, row_ids) where row_ids describes each row/column.
        """
        return self.esa.get_jacobian_with_ids(dense, form=form)

    # --- Network Delegation ---

    def busmap(self):
        """Bus number to matrix index mapping. Delegates to ``network.busmap()``."""
        return self.network.busmap()

    def buscoords(self, astuple=True):
        """Bus coordinates from substation data. Delegates to ``network.buscoords()``."""
        return self.network.buscoords(astuple)

    # --- Simulation Control ---

    def pflow(self, getvolts: bool = True, method: str = "POLARNEWT") -> Optional[Union[pd.Series, Tuple[pd.Series, pd.Series]]]:
        """
        Solve Power Flow.

        Parameters
        ----------
        getvolts : bool, optional
            Return voltages after solving. Defaults to True.
        method : str, optional
            Solution method. Defaults to "POLARNEWT".

        Returns
        -------
        pd.Series or tuple or None
        """
        self.esa.SolvePowerFlow(method)
        if getvolts:
            return self.voltage()

    def ts_solve(self, ctgs: Union[str, List[str]], fields: List[str]) -> Tuple[DataFrame, DataFrame]:
        """
        Run transient stability simulation for the specified contingencies.

        Handles auto-correction, initialization, solving each contingency,
        result retrieval, processing, and concatenation.

        Parameters
        ----------
        ctgs : Union[str, List[str]]
            A single contingency name or a list of names.
        fields : List[str]
            Retrieval field strings (e.g., from TSWatch.prepare).

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            (Metadata, Time-Series Data).
        """
        logger = logging.getLogger(__name__)
        ctgs_list = [ctgs] if isinstance(ctgs, str) else list(ctgs)

        if not fields:
            logger.warning("No fields provided. Simulation will run but no results will be retrieved.")

        self.esa.TSAutoCorrect()
        self.esa.TSInitialize()

        all_meta_frames = []
        all_data_frames = {}

        for ctg in ctgs_list:
            logger.info(f"Solving contingency: {ctg}")
            self.esa.TSSolve(ctg)
            meta, df = get_ts_results(self.esa, ctg, fields)

            if meta is None or df is None or df.empty:
                logger.warning(f"No results returned for contingency: {ctg}")
                continue

            meta, df = process_ts_results(meta, df, ctg)
            if not df.empty:
                all_data_frames[ctg] = df
                all_meta_frames.append(meta)

        if not all_meta_frames:
            return DataFrame(), DataFrame()

        final_meta = concat(all_meta_frames, axis=0, ignore_index=True).set_index('ColHeader')
        final_data = concat(all_data_frames.values(), axis=1, keys=all_data_frames.keys()).sort_index(axis=1)

        return final_meta, final_data

    def flatstart(self) -> None:
        """Resets the case to a flat start (1.0 pu voltage, 0.0 angle)."""
        self.esa.ResetToFlatStart()

    def save(self, filename: Optional[str] = None) -> None:
        """Saves the case to the specified filename."""
        self.esa.SaveCase(filename)

    def command(self, script: str):
        """Executes a raw script command string."""
        return self.esa.RunScriptCommand(script)

    def log(self, message: str):
        """Adds a message to the PowerWorld log."""
        self.esa.LogAdd(message)

    def print_log(self, clear: bool = False, new_only: bool = False):
        """
        Prints the PowerWorld Message Log to the console.

        Parameters
        ----------
        clear : bool, optional
            If True, clears the log after printing. Defaults to False.
        new_only : bool, optional
            If True, only prints new entries. Defaults to False.

        Returns
        -------
        str
            The log contents.
        """
        if not hasattr(self, "_log_last_position"):
            self._log_last_position = 0

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            self.esa.LogSave(tmp_path, append=False)
            with open(tmp_path, "r") as f:
                content = f.read()
        finally:
            os.unlink(tmp_path)

        if new_only:
            output = content[self._log_last_position:]
        else:
            output = content

        self._log_last_position = len(content)

        if output.strip():
            print(output)

        if clear:
            self.esa.LogClear()
            self._log_last_position = 0

        return output

    def close(self) -> None:
        """Closes the current case."""
        self.esa.CloseCase()

    def edit_mode(self) -> None:
        """Enter PowerWorld into EDIT mode."""
        self.esa.EnterMode("EDIT")

    def run_mode(self) -> None:
        """Enter PowerWorld into RUN mode."""
        self.esa.EnterMode("RUN")

    # --- Data Retrieval ---

    def generations(self):
        """Returns a DataFrame of generator outputs (MW, Mvar) and status."""
        return self[Gen, ["GenMW", "GenMVR", "GenStatus"]]

    def loads(self):
        """Returns a DataFrame of load demands (MW, Mvar) and status."""
        return self[Load, ["LoadMW", "LoadMVR", "LoadStatus"]]

    def shunts(self):
        """Returns a DataFrame of switched shunt outputs (MW, Mvar) and status."""
        return self[Shunt, ["ShuntMW", "ShuntMVR", "ShuntStatus"]]

    def lines(self):
        """Returns all transmission lines."""
        branches = self[Branch, :]
        return branches[branches["BranchDeviceType"] == "Line"]

    def transformers(self):
        """Returns all transformers."""
        branches = self[Branch, :]
        return branches[branches["BranchDeviceType"] == "Transformer"]

    def areas(self):
        """Returns all areas."""
        return self[Area, :]

    def zones(self):
        """Returns all zones."""
        return self[Zone, :]

    # --- Modification ---

    def open_branch(self, bus1, bus2, ckt='1'):
        """Opens a branch."""
        self.esa.ChangeParametersSingleElement(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"],
            [bus1, bus2, ckt, "Open"],
        )

    def close_branch(self, bus1, bus2, ckt='1'):
        """Closes a branch."""
        self.esa.ChangeParametersSingleElement(
            "Branch",
            ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"],
            [bus1, bus2, ckt, "Closed"],
        )

    def path_distance(self, start_element_str):
        """Calculates distance from a starting element to all buses."""
        return self.esa.DeterminePathDistance(start_element_str)

    # --- Difference Flows ---

    def set_as_base_case(self):
        """Sets the current case as the base case for difference flows."""
        self.esa.DiffCaseSetAsBase()

    def diff_mode(self, mode="DIFFERENCE"):
        """Sets the difference mode (PRESENT, BASE, DIFFERENCE, CHANGE)."""
        self.esa.DiffCaseMode(mode)

    def shortest_path(self, start_bus, end_bus):
        """Determines the shortest path between two buses."""
        start_str = create_object_string("Bus", start_bus)
        end_str = create_object_string("Bus", end_bus)
        return self.esa.DetermineShortestPath(start_str, end_str)
