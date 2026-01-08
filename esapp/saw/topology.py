import os
import tempfile
from pathlib import Path
import pandas as pd


class TopologyMixin:

    def DeterminePathDistance(
        self,
        start: str,
        BranchDistMeas: str = "X",
        BranchFilter: str = "ALL",
        BusField="CustomFloat:1",
    ) -> pd.DataFrame:
        """Calculate a distance measure at each bus in the entire model."""
        original = self.pw_order
        self.pw_order = True
        statement = f"DeterminePathDistance({start}, {BranchDistMeas}, {BranchFilter}, {BusField});"
        self.RunScriptCommand(statement)
        key = self.get_key_field_list("Bus")
        df = self.GetParametersMultipleElement("Bus", key + [BusField])
        df.rename(columns={BusField: BranchDistMeas}, inplace=True)
        df["BusNum"] = df["BusNum"].astype(int)
        df[BranchDistMeas] = df[BranchDistMeas].astype(float)
        self.pw_order = original
        return df

    def DetermineBranchesThatCreateIslands(
        self, Filter: str = "ALL", StoreBuses: str = "YES", SetSelectedOnLines: str = "NO"
    ) -> pd.DataFrame:
        """Determine the branches whose outage results in island formation."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            filename = Path(tmp.name).as_posix()
        
        try:
            statement = f'DetermineBranchesThatCreateIslands({Filter},{StoreBuses},"{filename}",{SetSelectedOnLines},CSV);'
            self.RunScriptCommand(statement)
            return pd.read_csv(filename, header=0)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def DetermineShortestPath(
        self, start: str, end: str, BranchDistanceMeasure: str = "X", BranchFilter: str = "ALL"
    ) -> pd.DataFrame:
        """Calculate the shortest path between a starting group and an ending group."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            filename = Path(tmp.name).as_posix()
            
        try:
            statement = f'DetermineShortestPath({start}, {end}, {BranchDistanceMeasure}, {BranchFilter}, "{filename}");'
            self.RunScriptCommand(statement)
            df = pd.read_csv(
                filename, header=None, sep=r'\s+', names=["BusNum", BranchDistanceMeasure, "BusName"]
            )
            df["BusNum"] = df["BusNum"].astype(int)
            return df
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def DoFacilityAnalysis(self, filename: str, set_selected: bool = False):
        """Determine the branches that would isolate the Facility from the External region.

        This command assumes the user has set options in the Select Bus Dialog in the Simulator Tool dialog
        (or via other automation means) before calling this.
        
        Parameters
        ----------
        filename : str
            The auxiliary file to which the results will be written.
        set_selected : bool, optional
            If True, sets the Selected field to YES for branches in the minimum cut. Defaults to False.
        """
        yn = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'DoFacilityAnalysis("{filename}", {yn});')

    def FindRadialBusPaths(
        self,
        ignore_status: bool = False,
        treat_parallel_as_not_radial: bool = False,
        bus_or_superbus: str = "BUS",
    ):
        """Calculate series paths of buses or superbuses that are radial.

        Populates fields: Radial Path End Number, Radial Path Index, Radial Path Length.
        """
        ign = "YES" if ignore_status else "NO"
        treat = "YES" if treat_parallel_as_not_radial else "NO"
        return self.RunScriptCommand(f"FindRadialBusPaths({ign}, {treat}, {bus_or_superbus});")

    def SetBusFieldFromClosest(self, variable_name: str, bus_filter_set_to: str, bus_filter_from_these: str, branch_filter_traverse: str, branch_dist_meas: str):
        """Set buses field values equal to the closest bus's value."""
        return self.RunScriptCommand(
            f'SetBusFieldFromClosest("{variable_name}", "{bus_filter_set_to}", "{bus_filter_from_these}", {branch_filter_traverse}, {branch_dist_meas});'
        )

    def SetSelectedFromNetworkCut(
        self,
        set_how: bool,
        bus_on_cut_side: str,
        branch_filter: str = "",
        interface_filter: str = "",
        dc_line_filter: str = "",
        energized: bool = True,
        num_tiers: int = 0,
        initialize_selected: bool = True,
        objects_to_select: list = None,
        use_area_zone: bool = False,
        use_kv: bool = False,
        min_kv: float = 0.0,
        max_kv: float = 9999.0,
        lower_min_kv: float = 0.0,
        lower_max_kv: float = 9999.0,
    ):
        """Set the Selected field of specified object types if they are on the specified side of a network cut."""
        sh = "YES" if set_how else "NO"
        en = "YES" if energized else "NO"
        init = "YES" if initialize_selected else "NO"
        uaz = "YES" if use_area_zone else "NO"
        ukv = "YES" if use_kv else "NO"

        objs = ""
        if objects_to_select:
            objs = "[" + ", ".join(objects_to_select) + "]"

        bf = f'"{branch_filter}"' if branch_filter and branch_filter not in ["SELECTED", "AREAZONE", "ALL"] else branch_filter
        inf = f'"{interface_filter}"' if interface_filter and interface_filter not in ["SELECTED", "AREAZONE", "ALL"] else interface_filter
        dcf = f'"{dc_line_filter}"' if dc_line_filter and dc_line_filter not in ["SELECTED", "AREAZONE", "ALL"] else dc_line_filter

        cmd = (
            f"SetSelectedFromNetworkCut({sh}, {bus_on_cut_side}, {bf}, {inf}, "
            f"{dcf}, {en}, {num_tiers}, {init}, {objs}, {uaz}, {ukv}, "
            f"{min_kv}, {max_kv}, {lower_min_kv}, {lower_max_kv});"
        )
        return self.RunScriptCommand(cmd)

    def CreateNewAreasFromIslands(self):
        """Create permanent areas that match the area Simulator creates temporarily while solving."""
        return self.RunScriptCommand("CreateNewAreasFromIslands;")

    def ExpandAllBusTopology(self):
        """Expand the topology around all buses."""
        return self.RunScriptCommand("ExpandAllBusTopology;")

    def ExpandBusTopology(self, bus_identifier: str, topology_type: str):
        """Expand the topology around the specified bus."""
        return self.RunScriptCommand(f'ExpandBusTopology({bus_identifier}, {topology_type});')

    def SaveConsolidatedCase(self, filename: str, filetype: str = "PWB", bus_format: str = "Number", truncate_ctg_labels: bool = False, add_comments: bool = False):
        """Saves the full topology model into a consolidated case."""
        tcl = "YES" if truncate_ctg_labels else "NO"
        ac = "YES" if add_comments else "NO"
        return self.RunScriptCommand(f'SaveConsolidatedCase("{filename}", {filetype}, [{bus_format}, {tcl}, {ac}]);')

    def CloseWithBreakers(self, object_type: str, filter_val: str, only_specified: bool = False, switching_types: list = None, close_normally_closed: bool = False):
        """Energize objects by closing breakers."""
        only = "YES" if only_specified else "NO"
        cnc = "YES" if close_normally_closed else "NO"
        sw_types = '["Breaker"]'
        if switching_types:
            sw_types = "[" + ", ".join([f'"{t}"' for t in switching_types]) + "]"
        
        return self.RunScriptCommand(f'CloseWithBreakers({object_type}, {filter_val}, {only}, {sw_types}, {cnc});')

    def OpenWithBreakers(self, object_type: str, filter_val: str, switching_types: list = None, open_normally_open: bool = False):
        """Disconnect objects by opening breakers."""
        ono = "YES" if open_normally_open else "NO"
        sw_types = '["Breaker"]'
        if switching_types:
            sw_types = "[" + ", ".join([f'"{t}"' for t in switching_types]) + "]"
        return self.RunScriptCommand(f'OpenWithBreakers({object_type}, {filter_val}, {sw_types}, {ono});')