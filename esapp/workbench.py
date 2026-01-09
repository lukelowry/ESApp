from .apps.gic import GIC
from .apps.network import Network
from .apps.modes import ForcedOscillation
from .indexable import Indexable
from .grid import Bus, Branch, Gen, Load, Shunt, Area, Zone, Substation
from .saw import create_object_string

import numpy as np
from numpy import any as np_any
from pandas import DataFrame

class GridWorkBench(Indexable):
    """
    Main entry point for interacting with the PowerWorld grid model.
    """
    def __init__(self, fname=None):
        """
        Initialize the GridWorkBench.

        Parameters
        ----------
        fname : str, optional
            Path to the PowerWorld case file (.pwb).

        Examples
        --------
        >>> wb = GridWorkBench("case.pwb")
        """
        # Applications
        self.network = Network()
        self.gic     = GIC()
        self.modes   = ForcedOscillation()

        #self.dyn = Dynamics(self.esa)
        #self.statics = Statics(self.esa)

        # State chain for iterative solvers
        self._state_chain_idx = -1
        self._state_chain_max = 2

        # ZIP load dispatch dataframe (initialized on first use)
        self._dispatch_pq = None

        if fname:
            # Required to set to use IndexTool
            self.fname = fname
            # Sets the global esa object
            self.open()
        else:
            self.esa = None
            self.fname = None

        # Propagate the esa instance to the applications.
        self.set_esa(self.esa)

    def set_esa(self, esa):
        """Sets the SAW instance for the workbench and its applications."""
        super().set_esa(esa)
        self.network.set_esa(esa)
        self.gic.set_esa(esa)
        self.modes.set_esa(esa)

    def voltage(self, asComplex=True):
        """
        The vector of voltages in PowerWorld.

        Parameters
        ----------
        asComplex : bool, optional
            Whether to return complex values. Defaults to True.

        Returns
        -------
        pd.Series or tuple
            Series of complex values if asComplex=True, 
            else tuple of (Vmag, Angle in Radians).

        Examples
        --------
        >>> V = wb.voltage()
        >>> V_mag, V_ang = wb.voltage(asComplex=False)
        """
        v_df = self[Bus, ["BusPUVolt", "BusAngle"]] 

        vmag = v_df['BusPUVolt']
        rad = v_df['BusAngle']*np.pi/180

        if asComplex:
            return vmag * np.exp(1j * rad)
        
        return vmag, rad

    # --- Simulation Control ---

    def pflow(self, getvolts=True) -> DataFrame:
        """
        Solve Power Flow in external system.
        By default bus voltages will be returned.

        Parameters
        ----------
        getvolts : bool, optional
            Flag to indicate the voltages should be returned after power flow, 
            defaults to True.

        Returns
        -------
        pd.DataFrame or None
            Dataframe of bus number and voltage if requested.

        Examples
        --------
        >>> wb.pflow()
        """
        # Solve Power Flow through External Tool
        self.esa.SolvePowerFlow()

        # Request Voltages if needed
        if getvolts:
            return self.voltage()


    def reset(self):
        """
        Resets the case to a flat start (1.0 pu voltage, 0.0 angle).

        Examples
        --------
        >>> wb.reset()
        """
        self.esa.ResetToFlatStart()

    def save(self, filename=None):
        """
        Saves the case to the specified filename, or overwrites current if None.

        Parameters
        ----------
        filename : str, optional
            The path to save the case to.

        Examples
        --------
        >>> wb.save("case_modified.pwb")
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

        Examples
        --------
        >>> wb.command("SolvePowerFlow;")
        """
        return self.esa.RunScriptCommand(script)

    def log(self, message: str):
        """
        Adds a message to the PowerWorld log.

        Parameters
        ----------
        message : str
            The message to log.

        Examples
        --------
        >>> wb.log("Starting analysis...")
        """
        self.esa.LogAdd(message)

    def close(self):
        """
        Closes the current case.

        Examples
        --------
        >>> wb.close()
        """
        self.esa.CloseCase()

    def mode(self, mode: str):
        """
        Enters RUN or EDIT mode.

        Parameters
        ----------
        mode : str
            The mode to enter ('RUN' or 'EDIT').

        Examples
        --------
        >>> wb.mode("EDIT")
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

        Examples
        --------
        >>> wb.load_aux("data.aux")
        """
        self.esa.LoadAux(filename)
    
    def load_script(self, filename: str):
        """
        Loads and runs a script file.

        Parameters
        ----------
        filename : str
            The path to the script file.

        Examples
        --------
        >>> wb.load_script("run.pws")
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

        Examples
        --------
        >>> v_complex = wb.voltages()
        """
        fields = ["BusPUVolt", "BusAngle"] if pu else ["BusKVVolt", "BusAngle"]
        df = self[Bus, fields]
        
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

        Examples
        --------
        >>> gens = wb.generations()
        """
        return self[Gen, ["GenMW", "GenMVR", "GenStatus"]]

    def loads(self):
        """
        Returns a DataFrame of load demands (MW, Mvar) and status.

        Returns
        -------
        pd.DataFrame
            Load data.

        Examples
        --------
        >>> loads = wb.loads()
        """
        return self[Load, ["LoadMW", "LoadMVR", "LoadStatus"]]

    def shunts(self):
        """
        Returns a DataFrame of switched shunt outputs (MW, Mvar) and status.

        Returns
        -------
        pd.DataFrame
            Shunt data.

        Examples
        --------
        >>> shunts = wb.shunts()
        """
        return self[Shunt, ["ShuntMW", "ShuntMVR", "ShuntStatus"]]

    def lines(self):
        """
        Returns all transmission lines.

        Returns
        -------
        pd.DataFrame
            Line data.

        Examples
        --------
        >>> lines = wb.lines()
        """
        branches = self[Branch, :]
        return branches[branches["BranchDeviceType"] == "Line"]

    def transformers(self):
        """
        Returns all transformers.

        Returns
        -------
        pd.DataFrame
            Transformer data.

        Examples
        --------
        >>> xformers = wb.transformers()
        """
        branches = self[Branch, :]
        return branches[branches["BranchDeviceType"] == "Transformer"]

    def areas(self):
        """
        Returns all areas.

        Returns
        -------
        pd.DataFrame
            Area data.

        Examples
        --------
        >>> areas = wb.areas()
        """
        return self[Area, :]

    def zones(self):
        """
        Returns all zones.

        Returns
        -------
        pd.DataFrame
            Zone data.

        Examples
        --------
        >>> zones = wb.zones()
        """
        return self[Zone, :]

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

        Examples
        --------
        >>> fields = wb.get_fields("Bus")
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

        Examples
        --------
        >>> V_new = np.ones(len(wb.buses)) * 1.05
        >>> wb.set_voltages(V_new)
        """
        V_df = np.vstack([np.abs(V), np.angle(V, deg=True)]).T
        self[Bus, ["BusPUVolt", "BusAngle"]] = V_df

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

        Examples
        --------
        >>> wb.open_branch(1, 2, "1")
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

        Examples
        --------
        >>> wb.close_branch(1, 2, "1")
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

        Examples
        --------
        >>> wb.set_gen(bus=10, id="1", mw=150.0)
        """
        param_map = {"GenMW": mw, "GenMVR": mvar, "GenStatus": status}
        params = {k: v for k, v in param_map.items() if v is not None}
        
        if params:
            fields = ["BusNum", "GenID"] + list(params.keys())
            values = [bus, id] + list(params.values())
            self.esa.ChangeParametersSingleElement("Gen", fields, values)

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

        Examples
        --------
        >>> wb.set_load(bus=5, id="1", mw=50.0)
        """
        param_map = {"LoadMW": mw, "LoadMVR": mvar, "LoadStatus": status}
        params = {k: v for k, v in param_map.items() if v is not None}
        
        if params:
            fields = ["BusNum", "LoadID"] + list(params.keys())
            values = [bus, id] + list(params.values())
            self.esa.ChangeParametersSingleElement("Load", fields, values)

    def scale_load(self, factor):
        """
        Scales system load by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor.

        Examples
        --------
        >>> wb.scale_load(1.1)  # Increase load by 10%
        """
        self.esa.Scale("LOAD", "FACTOR", [factor], "SYSTEM")

    def scale_gen(self, factor):
        """
        Scales system generation by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor.

        Examples
        --------
        >>> wb.scale_gen(1.1)  # Increase generation by 10%
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

        Examples
        --------
        >>> wb.create("Load", BusNum=1, LoadID="1", LoadMW=10)
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

        Examples
        --------
        >>> wb.delete("Gen", filter_name="AreaNum = 1")
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

        Examples
        --------
        >>> wb.select("Bus", filter_name="BusPUVolt < 0.95")
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

        Examples
        --------
        >>> wb.unselect("Bus")
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

        Examples
        --------
        >>> wb.energize("Bus", "[1]")
        """
        self.esa.CloseWithBreakers(obj_type, identifier)

    def deenergize(self, obj_type, identifier):
        """
        De-energizes a specific object by opening breakers.

        Parameters
        ----------
        obj_type : str
            Object type (e.g. 'Bus', 'Gen', 'Load').
        identifier : str
            Identifier string (e.g. '[1]', '[1 "1"]').
      
        Examples
        --------
        >>> wb.deenergize("Bus", "[1]")
        """
        self.esa.OpenWithBreakers(obj_type, identifier)

    def radial_paths(self):
        """
        Identifies radial paths in the network.

        Examples
        --------
        >>> wb.radial_paths()
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

        Examples
        --------
        >>> dists = wb.path_distance("[BUS 1]")
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

        Examples
        --------
        >>> wb.network_cut("[BUS 1]")
        """
        self.esa.SetSelectedFromNetworkCut(True, bus_on_side, branch_filter=branch_filter, objects_to_select=["Bus", "Gen", "Load"])

    # --- Difference Flows ---

    def set_as_base_case(self):
        """
        Sets the currently open case as the base case for difference flows.

        Examples
        --------
        >>> wb.set_as_base_case()
        """
        self.esa.DiffCaseSetAsBase()

    def diff_mode(self, mode="DIFFERENCE"):
        """
        Sets the difference mode (PRESENT, BASE, DIFFERENCE, CHANGE).

        Parameters
        ----------
        mode : str, optional
            The mode to set. Defaults to "DIFFERENCE".

        Examples
        --------
        >>> wb.diff_mode("DIFFERENCE")
        """
        self.esa.DiffCaseMode(mode)

    # --- Analysis ---

    def run_contingency(self, name):
        """
        Runs a single contingency.

        Examples
        --------
        >>> wb.run_contingency("Line 1-2 Out")
        """
        self.esa.RunContingency(name)

    def solve_contingencies(self):
        """
        Solves all defined contingencies.

        Examples
        --------
        >>> wb.solve_contingencies()
        """
        self.esa.SolveContingencies()
    
    def auto_insert_contingencies(self):
        """
        Auto-inserts contingencies based on current options.

        Examples
        --------
        >>> wb.auto_insert_contingencies()
        """
        self.esa.CTGAutoInsert()

    def violations(self, v_min=0.9, v_max=1.1):
        """
        Returns a DataFrame of bus voltage violations.

        Examples
        --------
        >>> v_viols = wb.violations(v_min=0.95, v_max=1.05)
        >>> print(v_viols.head())
        """
        v = self.voltages(pu=True, complex=False)[0]
        low = v[v < v_min]
        high = v[v > v_max]
        return DataFrame({'Low': low, 'High': high})

    def mismatches(self):
        """Returns bus mismatches."""
        """
        Returns bus mismatches.

        Examples
        --------
        >>> mm = wb.mismatches()
        """
        return self.esa.GetBusMismatches()

    def islands(self):
        """
        Returns information about islands.

        Examples
        --------
        >>> islands = wb.islands()
        """
        return self.esa.DetermineBranchesThatCreateIslands()

    def refresh_onelines(self):
        """
        Relinks all open oneline diagrams.

        Examples
        --------
        >>> wb.refresh_onelines()
        """
        self.esa.RelinkAllOpenOnelines()

    # --- Sensitivity & Faults ---

    def ptdf(self, seller, buyer, method='DC'):
        """
        Calculates PTDF between seller and buyer.

        Parameters
        ----------
        seller : str
            Seller identifier (e.g. '[AREA "Top"]' or '[BUS 1]').
        buyer : str
            Buyer identifier (e.g. '[AREA "Bottom"]' or '[BUS 2]').
        method : str, optional
            Calculation method ('DC', etc.). Defaults to 'DC'.

        Returns
        -------
        pd.DataFrame
            PTDF results.

        Examples
        --------
        >>> ptdf = wb.ptdf("[AREA 1]", "[AREA 2]")
        """
        return self.esa.CalculatePTDF(seller, buyer, method)
    
    def lodf(self, branch, method='DC'):
        """
        Calculates LODF for a branch.
        
        Parameters
        ----------
        branch : str
            Branch identifier string like '[BRANCH 1 2 1]'.
        method : str, optional
            Calculation method. Defaults to 'DC'.

        Returns
        -------
        pd.DataFrame
            LODF results.

        Examples
        --------
        >>> lodf = wb.lodf("[BRANCH 1 2 1]")
        """
        return self.esa.CalculateLODF(branch, method)

    def fault(self, bus_num, fault_type='SLG', r=0.0, x=0.0):
        """
        Runs a fault at a specified bus number.

        Parameters
        ----------
        bus_num : int
            The bus number to fault.
        fault_type : str, optional
            Type of fault (e.g. 'SLG', '3PB'). Defaults to 'SLG'.
        r : float, optional
            Fault resistance. Defaults to 0.0.
        x : float, optional
            Fault reactance. Defaults to 0.0.

        Returns
        -------
        str
            Result string from SimAuto.

        Examples
        --------
        >>> wb.fault(bus_num=5, fault_type="SLG")
        """
        return self.esa.RunFault(create_object_string("Bus", bus_num), fault_type, r, x)
    
    def clear_fault(self):
        """
        Clears the currently applied fault.

        Examples
        --------
        >>> wb.clear_fault()
        """
        self.esa.FaultClear()

    def shortest_path(self, start_bus, end_bus):
        """
        Determines the shortest path between two buses.

        Parameters
        ----------
        start_bus : int
            Starting bus number.
        end_bus : int
            Ending bus number.

        Returns
        -------
        pd.DataFrame
            DataFrame describing the path.

        Examples
        --------
        >>> path = wb.shortest_path(1, 10)
        """
        start_str = create_object_string("Bus", start_bus)
        end_str = create_object_string("Bus", end_bus)
        return self.esa.DetermineShortestPath(start_str, end_str)

    # --- Advanced Analysis ---


    def calculate_gic(self, max_field, direction):
        """
        Calculates GIC with specified field (V/km) and direction (degrees).

        Parameters
        ----------
        max_field : float
            Maximum electric field in V/km.
        direction : float
            Direction of the field in degrees.

        Returns
        -------
        str
            Result string.

        Examples
        --------
        >>> wb.calculate_gic(max_field=1.0, direction=90.0)
        """
        return self.esa.CalculateGIC(max_field, direction)
    
    def solve_opf(self):
        """
        Solves Primal LP OPF.

        Returns
        -------
        str
            Result string.

        Examples
        --------
        >>> wb.solve_opf()
        """
        return self.esa.SolvePrimalLP()

    def ybus(self, dense=False):
        """
        Returns the Y-Bus Matrix.

        Parameters
        ----------
        dense : bool, optional
            Whether to return a dense array. Defaults to False (sparse).

        Returns
        -------
        Union[np.ndarray, csr_matrix]
            The Y-Bus matrix.

        Examples
        --------
        >>> Y = wb.ybus()
        """
        return self.esa.get_ybus(dense)
        
    ''' LOCATION FUNCTIONS '''

    def busmap(self):
        """
        Returns a Pandas Series indexed by BusNum to the positional value of each bus
        in matricies like the Y-Bus, Incidence Matrix, Etc.

        Returns
        -------
        pd.Series
            Series mapping BusNum to index.

        Examples
        --------
        >>> mapping = wb.busmap()
        """
        return self.network.busmap()
    
    
    def buscoords(self, astuple=True):
        """
        Retrive dataframe of bus latitude and longitude coordinates based on substation data.

        Parameters
        ----------
        astuple : bool, optional
            Whether to return as a tuple of (Lon, Lat). Defaults to True.

        Returns
        -------
        pd.DataFrame or tuple
            Coordinates data.

        Examples
        --------
        >>> lon, lat = wb.buscoords()
        """
        A, S = self[Bus, "SubNum"],  self[Substation, ["Longitude", "Latitude"]]
        LL = A.merge(S, on='SubNum') 
        if astuple:
            return LL['Longitude'], LL['Latitude']
        return LL
    
    def write_voltage(self,V):
        """
        Given Complex 1-D vector write to PowerWorld.

        Parameters
        ----------
        V : np.ndarray
            Complex voltage vector.

        Examples
        --------
        >>> V_new = np.ones(len(wb.buses)) * 1.05
        >>> wb.write_voltage(V_new)
        """
        V_df =  np.vstack([np.abs(V), np.angle(V,deg=True)]).T

        self[Bus, ["BusPUVolt", "BusAngle"]] = V_df

    # --- Generator Limit Checking ---

    def gens_above_pmax(self, p=None, is_closed=None, tol=0.001):
        """
        Check if any closed generators are outside active power limits.

        Parameters
        ----------
        p : pd.Series, optional
            Generator MW output. If None, retrieves from case.
        is_closed : pd.Series, optional
            Boolean series of generator status. If None, retrieves from case.
        tol : float, optional
            Tolerance for limit checking. Defaults to 0.001.

        Returns
        -------
        bool
            True if any closed generators violate P limits.

        Examples
        --------
        >>> if wb.gens_above_pmax():
        ...     print("Generator P limit violation detected")
        """
        return self._check_gen_limits('GenMW', 'GenMWMax', 'GenMWMin', p, is_closed, tol)

    def gens_above_qmax(self, q=None, is_closed=None, tol=0.001):
        """
        Check if any closed generators are outside reactive power limits.

        Parameters
        ----------
        q : pd.Series, optional
            Generator Mvar output. If None, retrieves from case.
        is_closed : pd.Series, optional
            Boolean series of generator status. If None, retrieves from case.
        tol : float, optional
            Tolerance for limit checking. Defaults to 0.001.

        Returns
        -------
        bool
            True if any closed generators violate Q limits.

        Examples
        --------
        >>> if wb.gens_above_qmax():
        ...     print("Generator Q limit violation detected")
        """
        return self._check_gen_limits('GenMVR', 'GenMVRMax', 'GenMVRMin', q, is_closed, tol)

    # --- State Chain Management ---

    def _check_gen_limits(self, value_col, max_col, min_col, value=None, is_closed=None, tol=0.001):
        """
        Helper method to check if generators exceed limits.

        Parameters
        ----------
        value_col : str
            Column name for current value (e.g., 'GenMW' or 'GenMVR').
        max_col : str
            Column name for maximum limit.
        min_col : str
            Column name for minimum limit.
        value : pd.Series, optional
            Current values. If None, retrieves from case.
        is_closed : pd.Series, optional
            Boolean series of generator status. If None, retrieves from case.
        tol : float, optional
            Tolerance for limit checking. Defaults to 0.001.

        Returns
        -------
        bool
            True if any closed generators violate limits.
        """
        gens = self[Gen, [value_col, max_col, min_col, 'GenStatus']]
        
        value = gens[value_col] if value is None else value
        is_closed = (gens['GenStatus'] == 'Closed') if is_closed is None else is_closed
        
        violation = is_closed & ((value > gens[max_col] + tol) | (value < gens[min_col] - tol))
        return np_any(violation)

    # --- GIC Functions ---

    def gic_storm(self, max_field: float, direction: float, solve_pf=True):
        """
        Configure a synthetic GIC storm with uniform electric field.

        Parameters
        ----------
        max_field : float
            Maximum electric field magnitude in Volts/km.
        direction : float
            Storm direction in degrees (0-360).
        solve_pf : bool, optional
            Whether to include results in power flow. Defaults to True.

        Examples
        --------
        >>> wb.gic_storm(max_field=1.0, direction=90.0)
        """
        yn = "YES" if solve_pf else "NO"
        self.esa.RunScriptCommand(f"GICCalculate({max_field}, {direction}, {yn})")

    def gic_clear(self):
        """
        Clear manual GIC calculations from PowerWorld.

        Examples
        --------
        >>> wb.gic_clear()
        """
        self.esa.RunScriptCommand("GICClear;")

    def gic_load_b3d(self, file_type: str, filename: str, setup_on_load=True):
        """
        Load a B3D file containing electric field data for GIC analysis.

        Parameters
        ----------
        file_type : str
            The type of B3D file.
        filename : str
            Path to the B3D file.
        setup_on_load : bool, optional
            Whether to configure GIC settings on load. Defaults to True.

        Examples
        --------
        >>> wb.gic_load_b3d("STORM", "storm_data.b3d")
        """
        yn = "YES" if setup_on_load else "NO"
        self.esa.RunScriptCommand(f"GICLoad3DEfield({file_type}, {filename}, {yn})")