import numpy as np
from pandas import DataFrame, Series, concat
from .grid.components import Bus, Branch, Gen, Load, Shunt, Area, Zone, Substation, InjectionGroup, Interface, Contingency
from .indextool import IndexTool

class Adapter:
    """
    A convenient adapter for common PowerWorld operations, providing a pythonic
    interface to the underlying SAW functionality.
    """
    def __init__(self, io: IndexTool):
        """
        Initialize the Adapter.

        Parameters
        ----------
        io : IndexTool
            The IndexTool instance to use for I/O.
        """
        self.io = io
        self.esa = io.esa

    # --- Simulation Control ---

    def solve(self):
        """
        Solves the AC Power Flow.
        """
        self.io.pflow()

    def reset(self):
        """
        Resets the case to a flat start (1.0 pu voltage, 0.0 angle).
        """
        self.esa.ResetToFlatStart()

    def save(self, filename=None):
        """
        Saves the case to the specified filename, or overwrites current if None.

        Parameters
        ----------
        filename : str, optional
            The path to save the case to.
        """
        self.esa.SaveCase(filename)

    def command(self, script: str):
        """
        Executes a raw script command string.

        Parameters
        ----------
        script : str
            The PowerWorld script command.

        Returns
        -------
        str
            The result of the command.
        """
        return self.esa.RunScriptCommand(script)

    def log(self, message: str):
        """
        Adds a message to the PowerWorld log.

        Parameters
        ----------
        message : str
            The message to log.
        """
        self.esa.LogAdd(message)

    def close(self):
        """
        Closes the current case.
        """
        self.esa.CloseCase()

    def mode(self, mode: str):
        """
        Enters RUN or EDIT mode.

        Parameters
        ----------
        mode : str
            The mode to enter ('RUN' or 'EDIT').
        """
        self.esa.EnterMode(mode)

    # --- File Operations ---

    def load_aux(self, filename: str):
        """
        Loads an auxiliary file.

        Parameters
        ----------
        filename : str
            The path to the .aux file.
        """
        self.esa.LoadAux(filename)
    
    def load_script(self, filename: str):
        """
        Loads and runs a script file.

        Parameters
        ----------
        filename : str
            The path to the script file.
        """
        self.esa.LoadScript(filename)

    def voltages(self, pu=True, complex=True):
        """
        Retrieves bus voltages.

        Parameters
        ----------
        pu : bool, optional
            If True, returns per-unit voltages. Else kV. Defaults to True.
        complex : bool, optional
            If True, returns complex numbers. Else tuple of (mag, angle_rad). Defaults to True.

        Returns
        -------
        Union[pd.Series, Tuple[pd.Series, pd.Series]]
            The voltage data.
        """
        fields = ['BusPUVolt', 'BusAngle'] if pu else ['BusKVVolt', 'BusAngle']
        df = self.io[Bus, fields]
        
        mag = df[fields[0]]
        ang = df['BusAngle'] * np.pi / 180.0

        if complex:
            return mag * np.exp(1j * ang)
        return mag, ang

    def generations(self):
        """
        Returns a DataFrame of generator outputs (MW, Mvar) and status.

        Returns
        -------
        pd.DataFrame
            Generator data.
        """
        return self.io[Gen, ['GenMW', 'GenMVR', 'GenStatus']]

    def loads(self):
        """
        Returns a DataFrame of load demands (MW, Mvar) and status.

        Returns
        -------
        pd.DataFrame
            Load data.
        """
        return self.io[Load, ['LoadMW', 'LoadMVR', 'LoadStatus']]

    def shunts(self):
        """
        Returns a DataFrame of switched shunt outputs (MW, Mvar) and status.

        Returns
        -------
        pd.DataFrame
            Shunt data.
        """
        return self.io[Shunt, ['ShuntMW', 'ShuntMVR', 'ShuntStatus']]

    def lines(self):
        """
        Returns all transmission lines.

        Returns
        -------
        pd.DataFrame
            Line data.
        """
        branches = self.io[Branch, :]
        return branches[branches['BranchDeviceType'] == 'Line']

    def transformers(self):
        """
        Returns all transformers.

        Returns
        -------
        pd.DataFrame
            Transformer data.
        """
        branches = self.io[Branch, :]
        return branches[branches['BranchDeviceType'] == 'Transformer']

    def areas(self):
        """
        Returns all areas.

        Returns
        -------
        pd.DataFrame
            Area data.
        """
        return self.io[Area, :]

    def zones(self):
        """
        Returns all zones.

        Returns
        -------
        pd.DataFrame
            Zone data.
        """
        return self.io[Zone, :]

    def get_fields(self, obj_type):
        """
        Returns a DataFrame describing the fields for a given object type.

        Parameters
        ----------
        obj_type : str
            The PowerWorld object type.

        Returns
        -------
        pd.DataFrame
            Field information.
        """
        return self.esa.GetFieldList(obj_type)

    # --- Modification ---

    def set_voltages(self, V):
        """
        Sets bus voltages from a complex vector.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector.
        """
        V_df = np.vstack([np.abs(V), np.angle(V, deg=True)]).T
        self.io[Bus, ['BusPUVolt', 'BusAngle']] = V_df

    def open_branch(self, bus1, bus2, ckt='1'):
        """
        Opens a branch.

        Parameters
        ----------
        bus1 : int
            From bus number.
        bus2 : int
            To bus number.
        ckt : str, optional
            Circuit ID. Defaults to '1'.
        """
        self.esa.ChangeParametersSingleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"], [bus1, bus2, ckt, "Open"])

    def close_branch(self, bus1, bus2, ckt='1'):
        """
        Closes a branch.

        Parameters
        ----------
        bus1 : int
            From bus number.
        bus2 : int
            To bus number.
        ckt : str, optional
            Circuit ID. Defaults to '1'.
        """
        self.esa.ChangeParametersSingleElement("Branch", ["BusNum", "BusNum:1", "LineCircuit", "LineStatus"], [bus1, bus2, ckt, "Closed"])

    def set_gen(self, bus, id, mw=None, mvar=None, status=None):
        """
        Sets generator parameters.

        Parameters
        ----------
        bus : int
            Bus number.
        id : str
            Generator ID.
        mw : float, optional
            MW output.
        mvar : float, optional
            Mvar output.
        status : str, optional
            Status ('Closed' or 'Open').
        """
        params = []
        values = []
        if mw is not None:
            params.append("GenMW")
            values.append(mw)
        if mvar is not None:
            params.append("GenMVR")
            values.append(mvar)
        if status is not None:
            params.append("GenStatus")
            values.append(status)
        
        if params:
            self.esa.ChangeParametersSingleElement("Gen", ["BusNum", "GenID"] + params, [bus, id] + values)

    def set_load(self, bus, id, mw=None, mvar=None, status=None):
        """
        Sets load parameters.

        Parameters
        ----------
        bus : int
            Bus number.
        id : str
            Load ID.
        mw : float, optional
            MW demand.
        mvar : float, optional
            Mvar demand.
        status : str, optional
            Status ('Closed' or 'Open').
        """
        params = []
        values = []
        if mw is not None:
            params.append("LoadMW")
            values.append(mw)
        if mvar is not None:
            params.append("LoadMVR")
            values.append(mvar)
        if status is not None:
            params.append("LoadStatus")
            values.append(status)
        
        if params:
            self.esa.ChangeParametersSingleElement("Load", ["BusNum", "LoadID"] + params, [bus, id] + values)

    def scale_load(self, factor):
        """
        Scales system load by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor.
        """
        self.esa.Scale("LOAD", "FACTOR", [factor], "SYSTEM")

    def scale_gen(self, factor):
        """
        Scales system generation by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor.
        """
        self.esa.Scale("GEN", "FACTOR", [factor], "SYSTEM")

    def create(self, obj_type, **kwargs):
        """
        Creates an object with specified parameters.
        Example: adapter.create('Load', BusNum=1, LoadID='1', LoadMW=10)

        Parameters
        ----------
        obj_type : str
            The PowerWorld object type.
        **kwargs
            Field names and values.
        """
        fields = list(kwargs.keys())
        values = list(kwargs.values())
        self.esa.CreateData(obj_type, fields, values)

    def delete(self, obj_type, filter_name=""):
        """
        Deletes objects of a given type, optionally matching a filter.

        Parameters
        ----------
        obj_type : str
            The PowerWorld object type.
        filter_name : str, optional
            The filter to apply.
        """
        self.esa.Delete(obj_type, filter_name)

    def select(self, obj_type, filter_name=""):
        """
        Sets the Selected field to YES for objects matching the filter.

        Parameters
        ----------
        obj_type : str
            The PowerWorld object type.
        filter_name : str, optional
            The filter to apply.
        """
        self.esa.SelectAll(obj_type, filter_name)

    def unselect(self, obj_type, filter_name=""):
        """
        Sets the Selected field to NO for objects matching the filter.

        Parameters
        ----------
        obj_type : str
            The PowerWorld object type.
        filter_name : str, optional
            The filter to apply.
        """
        self.esa.UnSelectAll(obj_type, filter_name)

    # --- Advanced Topology & Switching ---

    def energize(self, obj_type, identifier, close_breakers=True):
        """
        Energizes a specific object by closing breakers.

        Parameters
        ----------
        obj_type : str
            Object type (e.g. 'Bus', 'Gen', 'Load').
        identifier : str
            Identifier string (e.g. '[1]', '[1 "1"]').
        close_breakers : bool, optional
            Whether to close breakers. Defaults to True.
        """
        self.esa.CloseWithBreakers(obj_type, identifier, only_specified=False, close_normally_closed=True)

    def deenergize(self, obj_type, identifier):
        """
        De-energizes a specific object by opening breakers.

        Parameters
        ----------
        obj_type : str
            Object type (e.g. 'Bus', 'Gen', 'Load').
        identifier : str
            Identifier string (e.g. '[1]', '[1 "1"]').
        """
        self.esa.OpenWithBreakers(obj_type, identifier)

    def radial_paths(self):
        """
        Identifies radial paths in the network.
        """
        self.esa.FindRadialBusPaths()

    def path_distance(self, start_element_str):
        """
        Calculates distance from a starting element to all buses.

        Parameters
        ----------
        start_element_str : str
            e.g. '[BUS 1]' or '[AREA "Top"]'.

        Returns
        -------
        pd.DataFrame
            Distance data.
        """
        return self.esa.DeterminePathDistance(start_element_str)

    def network_cut(self, bus_on_side, branch_filter="SELECTED"):
        """
        Selects objects on one side of a network cut defined by selected branches.

        Parameters
        ----------
        bus_on_side : str
            Bus identifier string (e.g. '[BUS 1]') on the desired side.
        branch_filter : str, optional
            Filter for branches defining the cut. Defaults to "SELECTED".
        """
        self.esa.SetSelectedFromNetworkCut(True, bus_on_side, branch_filter=branch_filter, objects_to_select=["Bus", "Gen", "Load"])

    def isolate_zone(self, zone_num):
        """
        Opens all tie-lines connecting the specified zone to other zones.

        Parameters
        ----------
        zone_num : int
            The zone number to isolate.
        """
        # Retrieve branch connectivity and zone information
        # Note: 'BusZone' refers to From Bus Zone, 'BusZone:1' refers to To Bus Zone in PowerWorld
        branches = self.io[Branch, ['BusNum', 'BusNum:1', 'LineCircuit', 'BusZone', 'BusZone:1']]
        
        # Filter for tie-lines where one end is in the zone and the other is not
        ties = branches[
            ((branches['BusZone'] == zone_num) & (branches['BusZone:1'] != zone_num)) |
            ((branches['BusZone'] != zone_num) & (branches['BusZone:1'] == zone_num))
        ]
        
        for _, row in ties.iterrows():
            self.open_branch(row['BusNum'], row['BusNum:1'], row['LineCircuit'])

    # --- Validation & Comparison ---

    def find_violations(self, v_min=0.95, v_max=1.05, branch_max_pct=100.0):
        """
        Finds bus voltage and branch flow violations.

        Parameters
        ----------
        v_min : float, optional
            Minimum per-unit voltage threshold. Defaults to 0.95.
        v_max : float, optional
            Maximum per-unit voltage threshold. Defaults to 1.05.
        branch_max_pct : float, optional
            Branch loading percentage threshold. Defaults to 100.0.

        Returns
        -------
        dict
            Dictionary with 'bus_low', 'bus_high', 'branch_overload' DataFrames.
        """
        # Bus Violations
        buses = self.io[Bus, ['BusNum', 'BusName', 'BusPUVolt']]
        low = buses[buses['BusPUVolt'] < v_min]
        high = buses[buses['BusPUVolt'] > v_max]
        
        # Branch Violations
        branches = self.io[Branch, ['BusNum', 'BusNum:1', 'LineCircuit', 'LineMVA', 'LineLimit']]
        # Filter branches with valid limits to avoid division by zero or misleading results
        branches = branches[branches['LineLimit'] > 0]
        overloaded = branches[branches['LineMVA'] > (branches['LineLimit'] * (branch_max_pct / 100.0))]
        
        return {'bus_low': low, 'bus_high': high, 'branch_overload': overloaded}

    # --- Difference Flows ---

    def set_as_base_case(self):
        """
        Sets the currently open case as the base case for difference flows.
        """
        self.esa.DiffCaseSetAsBase()

    def diff_mode(self, mode="DIFFERENCE"):
        """
        Sets the difference mode (PRESENT, BASE, DIFFERENCE, CHANGE).

        Parameters
        ----------
        mode : str, optional
            The mode to set. Defaults to "DIFFERENCE".
        """
        self.esa.DiffCaseMode(mode)

    def compare_case(self, other_case_path, output_aux):
        """
        Compares the current case (set as base) with another case file.
        Generates an AUX file with the differences.

        Parameters
        ----------
        other_case_path : str
            Path to the case to compare against.
        output_aux : str
            Path to the output .aux file.
        """
        self.esa.DiffCaseSetAsBase()
        self.esa.OpenCase(other_case_path)
        self.esa.DiffCaseRefresh()
        self.esa.DiffCaseWriteCompleteModel(output_aux)

    # --- Analysis ---

    def run_contingency(self, name):
        """Runs a single contingency."""
        self.esa.RunContingency(name)

    def solve_contingencies(self):
        """Solves all defined contingencies."""
        self.esa.SolveContingencies()
    
    def auto_insert_contingencies(self):
        """Auto-inserts contingencies based on current options."""
        self.esa.CTGAutoInsert()

    def violations(self, v_min=0.9, v_max=1.1):
        """Returns a DataFrame of bus voltage violations."""
        v = self.voltages(pu=True, complex=False)[0]
        low = v[v < v_min]
        high = v[v > v_max]
        return DataFrame({'Low': low, 'High': high})

    def mismatches(self):
        """Returns bus mismatches."""
        return self.esa.GetBusMismatches()

    def islands(self):
        """Returns information about islands."""
        return self.esa.DetermineBranchesThatCreateIslands()

    def save_image(self, filename, oneline_name, image_type="JPG"):
        """Exports the oneline diagram to an image."""
        self.esa.ExportOneline(filename, oneline_name, image_type)

    def refresh_onelines(self):
        """Relinks all open oneline diagrams."""
        self.esa.RelinkAllOpenOnelines()

    # --- Sensitivity & Faults ---

    def ptdf(self, seller, buyer, method='DC'):
        """
        Calculates PTDF between seller and buyer.
        Seller/Buyer can be strings like '[AREA "Top"]' or '[BUS 1]'.
        """
        return self.esa.CalculatePTDF(seller, buyer, method)
    
    def lodf(self, branch, method='DC'):
        """
        Calculates LODF for a branch.
        Branch should be a string identifier like '[BRANCH 1 2 1]'.
        """
        return self.esa.CalculateLODF(branch, method)

    def fault(self, bus_num, fault_type='SLG', r=0.0, x=0.0):
        """Runs a fault at a specified bus number."""
        return self.esa.RunFault(f'[BUS {bus_num}]', fault_type, r, x)
    
    def clear_fault(self):
        """Clears the currently applied fault."""
        self.esa.FaultClear()

    def shortest_path(self, start_bus, end_bus):
        """
        Determines the shortest path between two buses.
        Returns a DataFrame of the path.
        """
        return self.esa.DetermineShortestPath(f'[BUS {start_bus}]', f'[BUS {end_bus}]')

    # --- Advanced Analysis ---

    def run_pv(self, source, sink):
        """Runs PV analysis between source and sink injection groups."""
        self.esa.RunPV(source, sink)

    def run_qv(self, filename=None):
        """Runs QV analysis."""
        return self.esa.RunQV(filename)
    
    def calculate_atc(self, seller, buyer):
        """Calculates Available Transfer Capability."""
        return self.esa.DetermineATC(seller, buyer)
    
    def calculate_gic(self, max_field, direction):
        """Calculates GIC with specified field (V/km) and direction (degrees)."""
        return self.esa.CalculateGIC(max_field, direction)
    
    def solve_opf(self):
        """Solves Primal LP OPF."""
        return self.esa.SolvePrimalLP()
