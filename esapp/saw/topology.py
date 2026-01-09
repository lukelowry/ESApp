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
        """
        Calculate a distance measure at each bus in the entire model.

        Parameters
        ----------
        start : str
            The starting element identifier (e.g. '[BUS 1]').
        BranchDistMeas : str, optional
            The branch field to use as the distance measure. Defaults to "X".
        BranchFilter : str, optional
            Filter to apply to branches. Defaults to "ALL".
        BusField : str, optional
            The bus field to store the distance in temporarily. Defaults to "CustomFloat:1".

        Returns
        -------
        pd.DataFrame
            DataFrame containing BusNum and the calculated distance.
        """
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
        """
        Determine the branches whose outage results in island formation.

        Parameters
        ----------
        Filter : str, optional
            Filter to apply to branches. Defaults to "ALL".
        StoreBuses : str, optional
            Whether to store bus information. Defaults to "YES".
        SetSelectedOnLines : str, optional
            Whether to set the Selected field on lines. Defaults to "NO".

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results.
        """
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
        """
        Calculate the shortest path between a starting group and an ending group.

        Parameters
        ----------
        start : str
            The starting element identifier.
        end : str
            The ending element identifier.
        BranchDistanceMeasure : str, optional
            The branch field to use as distance. Defaults to "X".
        BranchFilter : str, optional
            Filter to apply to branches. Defaults to "ALL".

        Returns
        -------
        pd.DataFrame
            DataFrame describing the shortest path.
        """
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
        
        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        yn = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'DoFacilityAnalysis("{filename}", {yn});')

    def FindRadialBusPaths(
        self,
        ignore_status: bool = False,
        treat_parallel_as_not_radial: bool = False,
        bus_or_superbus: str = "BUS",
    ):
        """
        Calculate series paths of buses or superbuses that are radial.
        
        Populates fields: Radial Path End Number, Radial Path Index, Radial Path Length.

        Parameters
        ----------
        ignore_status : bool, optional
            If True, ignores element status. Defaults to False.
        treat_parallel_as_not_radial : bool, optional
            If True, treats parallel lines as not radial. Defaults to False.
        bus_or_superbus : str, optional
            "BUS" or "SUPERBUS". Defaults to "BUS".

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        ign = "YES" if ignore_status else "NO"
        treat = "YES" if treat_parallel_as_not_radial else "NO"
        return self.RunScriptCommand(f"FindRadialBusPaths({ign}, {treat}, {bus_or_superbus});")

    def SetBusFieldFromClosest(self, variable_name: str, bus_filter_set_to: str, bus_filter_from_these: str, branch_filter_traverse: str, branch_dist_meas: str):
        """
        Set buses field values equal to the closest bus's value.

        Parameters
        ----------
        variable_name : str
            The variable to set.
        bus_filter_set_to : str
            Filter for buses to set.
        bus_filter_from_these : str
            Filter for source buses.
        branch_filter_traverse : str
            Filter for branches to traverse.
        branch_dist_meas : str
            Distance measure.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
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
        """
        Set the Selected field of specified object types if they are on the specified side of a network cut.

        Parameters
        ----------
        set_how : bool
            How to set the field (True for YES, False for NO).
        bus_on_cut_side : str
            Identifier for a bus on the desired side.
        branch_filter : str, optional
            Filter for branches defining the cut.
        interface_filter : str, optional
            Filter for interfaces defining the cut.
        dc_line_filter : str, optional
            Filter for DC lines defining the cut.
        energized : bool, optional
            If True, only considers energized elements. Defaults to True.
        num_tiers : int, optional
            Number of tiers to traverse. Defaults to 0.
        initialize_selected : bool, optional
            If True, initializes Selected field before setting. Defaults to True.
        objects_to_select : list, optional
            List of object types to select.
        use_area_zone : bool, optional
            If True, uses Area/Zone filters. Defaults to False.
        use_kv : bool, optional
            If True, uses kV limits. Defaults to False.
        min_kv : float, optional
            Minimum kV. Defaults to 0.0.
        max_kv : float, optional
            Maximum kV. Defaults to 9999.0.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
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
        """
        Create permanent areas that match the area Simulator creates temporarily while solving.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("CreateNewAreasFromIslands;")

    def ExpandAllBusTopology(self):
        """
        Expand the topology around all buses.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand("ExpandAllBusTopology;")

    def ExpandBusTopology(self, bus_identifier: str, topology_type: str):
        """
        Expand the topology around the specified bus.

        Parameters
        ----------
        bus_identifier : str
            The bus identifier.
        topology_type : str
            The type of topology expansion.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        return self.RunScriptCommand(f'ExpandBusTopology({bus_identifier}, {topology_type});')

    def SaveConsolidatedCase(self, filename: str, filetype: str = "PWB", bus_format: str = "Number", truncate_ctg_labels: bool = False, add_comments: bool = False):
        """
        Saves the full topology model into a consolidated case.

        Parameters
        ----------
        filename : str
            The file path to save.
        filetype : str, optional
            The file type ("PWB", "AUX"). Defaults to "PWB".
        bus_format : str, optional
            Bus format ("Number", "Name"). Defaults to "Number".
        truncate_ctg_labels : bool, optional
            If True, truncates contingency labels. Defaults to False.
        add_comments : bool, optional
            If True, adds comments. Defaults to False.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        tcl = "YES" if truncate_ctg_labels else "NO"
        ac = "YES" if add_comments else "NO"
        return self.RunScriptCommand(f'SaveConsolidatedCase("{filename}", {filetype}, [{bus_format}, {tcl}, {ac}]);')

    def CloseWithBreakers(self, object_type: str, filter_val: str, only_specified: bool = False, switching_types: list = None, close_normally_closed: bool = False):
        """
        Energize objects by closing breakers.

        Parameters
        ----------
        object_type : str
            The type of object to energize.
        filter_val : str
            Filter or identifier for the object.
        only_specified : bool, optional
            If True, only closes specified breakers. Defaults to False.
        switching_types : list, optional
            List of switching device types to use. Defaults to None (Breakers).
        close_normally_closed : bool, optional
            If True, closes normally closed breakers. Defaults to False.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        only = "YES" if only_specified else "NO"
        cnc = "YES" if close_normally_closed else "NO"
        sw_types = '["Breaker"]'
        if switching_types:
            sw_types = "[" + ", ".join([f'"{t}"' for t in switching_types]) + "]"
        
        # This command has a unique syntax where the object type is the first argument
        # and the second argument is an identifier with keys *only*, not the full object string.
        # This block handles cases where a full object string (e.g., from create_object_string)
        # is passed as filter_val.
        processed_val = filter_val
        prefix_to_check = f"[{object_type.upper()} "
        if filter_val.strip().upper().startswith(prefix_to_check):
            # It's a full object string, extract just the keys part.
            keys_part = filter_val.strip()[len(prefix_to_check):-1].strip()
            processed_val = f"[{keys_part}]"

        return self.RunScriptCommand(f'CloseWithBreakers({object_type}, {processed_val}, {only}, {sw_types}, {cnc});')

    def OpenWithBreakers(self, object_type: str, filter_val: str, switching_types: list = None, open_normally_open: bool = False):
        """
        Disconnect objects by opening breakers.

        Parameters
        ----------
        object_type : str
            The type of object to disconnect.
        filter_val : str
            Filter or identifier for the object.
        switching_types : list, optional
            List of switching device types to use. Defaults to None (Breakers).
        open_normally_open : bool, optional
            If True, opens normally open breakers. Defaults to False.

        Returns
        -------
        str
            The response from the PowerWorld script command.
        """
        ono = "YES" if open_normally_open else "NO"
        sw_types = '["Breaker"]'
        if switching_types:
            sw_types = "[" + ", ".join([f'"{t}"' for t in switching_types]) + "]"

        # This command has a unique syntax where the object type is the first argument
        # and the second argument is an identifier with keys *only*, not the full object string.
        # This block handles cases where a full object string (e.g., from create_object_string)
        # is passed as filter_val.
        processed_val = filter_val
        prefix_to_check = f"[{object_type.upper()} "
        if filter_val.strip().upper().startswith(prefix_to_check):
            keys_part = filter_val.strip()[len(prefix_to_check):-1].strip()
            processed_val = f"[{keys_part}]"

        return self.RunScriptCommand(f'OpenWithBreakers({object_type}, {processed_val}, {sw_types}, {ono});')