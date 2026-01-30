from typing import Optional, Tuple, Union

from .apps.gic import GIC
from .apps.network import Network
from .apps.dynamics import Dynamics
from .apps.static import Statics
from .indexable import Indexable
from .components import Bus, Branch, Gen, Load, Shunt, Area, Zone, Substation, Contingency
from .saw import SAW, create_object_string

import pandas as pd
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
        self.dyn     = Dynamics()
        self.statics = Statics()

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
        self.dyn.set_esa(esa)
        self.statics.set_esa(esa)

    # --- Delegation to Statics ---

    def voltage(self, complex: bool = True, pu: bool = True) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """Retrieves bus voltages. Delegates to ``statics.voltage()``."""
        return self.statics.voltage(complex=complex, pu=pu)

    def set_voltages(self, V):
        """Sets bus voltages from a complex vector. Delegates to ``statics.set_voltages()``."""
        self.statics.set_voltages(V)

    def violations(self, v_min=0.9, v_max=1.1):
        """Returns bus voltage violations. Delegates to ``statics.violations()``."""
        return self.statics.violations(v_min, v_max)

    def mismatch(self, asComplex=False):
        """Returns bus mismatches. Delegates to ``statics.mismatch()``."""
        return self.statics.mismatch(asComplex)

    def netinj(self, asComplex=False):
        """Net injection at each bus. Delegates to ``statics.netinj()``."""
        return self.statics.netinj(asComplex)

    def ybus(self, dense=False):
        """Returns the Y-Bus Matrix. Delegates to ``statics.ybus()``."""
        return self.statics.ybus(dense)

    def branch_admittance(self):
        """Branch admittance matrices Yf and Yt. Delegates to ``statics.branch_admittance()``."""
        return self.statics.branch_admittance()

    def jacobian(self, dense=False, form='R'):
        """Power flow Jacobian matrix. Delegates to ``statics.jacobian()``."""
        return self.statics.jacobian(dense, form=form)

    # --- Delegation to Network ---

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

    def set_gen(self, bus, id, mw=None, mvar=None, status=None):
        """Sets generator parameters."""
        param_map = {"GenMW": mw, "GenMVR": mvar, "GenStatus": status}
        params = {k: v for k, v in param_map.items() if v is not None}
        if params:
            fields = ["BusNum", "GenID"] + list(params.keys())
            values = [bus, id] + list(params.values())
            self.esa.ChangeParametersSingleElement("Gen", fields, values)

    def set_load(self, bus, id, mw=None, mvar=None, status=None):
        """Sets load parameters."""
        param_map = {"LoadMW": mw, "LoadMVR": mvar, "LoadStatus": status}
        params = {k: v for k, v in param_map.items() if v is not None}
        if params:
            fields = ["BusNum", "LoadID"] + list(params.keys())
            values = [bus, id] + list(params.values())
            self.esa.ChangeParametersSingleElement("Load", fields, values)

    def scale_load(self, factor):
        """Scales system load by a factor."""
        self.esa.Scale("LOAD", "FACTOR", [factor], "SYSTEM")

    def scale_gen(self, factor):
        """Scales system generation by a factor."""
        self.esa.Scale("GEN", "FACTOR", [factor], "SYSTEM")

    def path_distance(self, start_element_str):
        """Calculates distance from a starting element to all buses."""
        return self.esa.DeterminePathDistance(start_element_str)

    def network_cut(self, bus_on_side, branch_filter="SELECTED"):
        """Selects objects on one side of a network cut."""
        self.esa.SetSelectedFromNetworkCut(
            True, bus_on_side,
            branch_filter=branch_filter,
            objects_to_select=["Bus", "Gen", "Load"],
        )

    # --- Difference Flows ---

    def set_as_base_case(self):
        """Sets the current case as the base case for difference flows."""
        self.esa.DiffCaseSetAsBase()

    def diff_mode(self, mode="DIFFERENCE"):
        """Sets the difference mode (PRESENT, BASE, DIFFERENCE, CHANGE)."""
        self.esa.DiffCaseMode(mode)

    def islands(self):
        """Returns information about islands."""
        return self.esa.DetermineBranchesThatCreateIslands()

    # --- Sensitivity & Faults ---

    def ptdf(self, seller, buyer, method='DC'):
        """Calculates PTDF between seller and buyer."""
        return self.esa.CalculatePTDF(seller, buyer, method)

    def lodf(self, branch, method='DC'):
        """Calculates LODF for a branch."""
        return self.esa.CalculateLODF(branch, method)

    def fault(self, bus_num, fault_type='SLG', r=0.0, x=0.0):
        """Runs a fault at a specified bus number."""
        return self.esa.RunFault(
            create_object_string("Bus", bus_num), fault_type, r, x,
        )

    def shortest_path(self, start_bus, end_bus):
        """Determines the shortest path between two buses."""
        start_str = create_object_string("Bus", start_bus)
        end_str = create_object_string("Bus", end_bus)
        return self.esa.DetermineShortestPath(start_str, end_str)

    # --- Advanced Analysis ---

    def solve_opf(self):
        """Solves Primal LP OPF."""
        return self.esa.SolvePrimalLP()
