import os
from pathlib import Path
import pandas as pd

from ._enums import YesNo, format_filter
from ._helpers import format_list, get_temp_filepath, pack_args


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
        args = pack_args(start, BranchDistMeas, BranchFilter, BusField)
        self.RunScriptCommand(f"DeterminePathDistance({args});")

    def DetermineBranchesThatCreateIslands(
        self, Filter: str = "ALL", StoreBuses: bool = True, SetSelectedOnLines: bool = False
    ) -> pd.DataFrame:
        """
        Determine the branches whose outage results in island formation.

        Parameters
        ----------
        Filter : str, optional
            Filter to apply to branches. Defaults to "ALL".
        StoreBuses : bool, optional
            Whether to store bus information. Defaults to True.
        SetSelectedOnLines : bool, optional
            Whether to set the Selected field on lines. Defaults to False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results.
        """
        filename = get_temp_filepath(".csv")

        sb = YesNo.from_bool(StoreBuses)
        ssl = YesNo.from_bool(SetSelectedOnLines)
        try:
            args = pack_args(Filter, sb, f'"{filename}"', ssl, "CSV")
            statement = f"DetermineBranchesThatCreateIslands({args});"
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
        filename = get_temp_filepath(".txt")
            
        try:
            args = pack_args(start, end, BranchDistanceMeasure, BranchFilter, f'"{filename}"')
            statement = f"DetermineShortestPath({args});"
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
        yn = YesNo.from_bool(set_selected)
        args = pack_args(f'"{filename}"', yn)
        return self.RunScriptCommand(f"DoFacilityAnalysis({args});")

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
        ign = YesNo.from_bool(ignore_status)
        treat = YesNo.from_bool(treat_parallel_as_not_radial)
        args = pack_args(ign, treat, bus_or_superbus)
        return self.RunScriptCommand(f"FindRadialBusPaths({args});")

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
        args = pack_args(f'"{variable_name}"', f'"{bus_filter_set_to}"', f'"{bus_filter_from_these}"', branch_filter_traverse, branch_dist_meas)
        return self.RunScriptCommand(f"SetBusFieldFromClosest({args});")

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
        sh = YesNo.from_bool(set_how)
        en = YesNo.from_bool(energized)
        init = YesNo.from_bool(initialize_selected)
        uaz = YesNo.from_bool(use_area_zone)
        ukv = YesNo.from_bool(use_kv)

        objs = format_list(objects_to_select) if objects_to_select else ""

        bf = format_filter(branch_filter)
        inf = format_filter(interface_filter)
        dcf = format_filter(dc_line_filter)

        args = pack_args(sh, bus_on_cut_side, bf, inf, dcf, en, num_tiers, init, objs, uaz, ukv, min_kv, max_kv, lower_min_kv, lower_max_kv)
        cmd = f"SetSelectedFromNetworkCut({args});"
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
        args = pack_args(bus_identifier, topology_type)
        return self.RunScriptCommand(f"ExpandBusTopology({args});")

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
        tcl = YesNo.from_bool(truncate_ctg_labels)
        ac = YesNo.from_bool(add_comments)
        args = pack_args(f'"{filename}"', filetype, f'[{bus_format}, {tcl}, {ac}]')
        return self.RunScriptCommand(f"SaveConsolidatedCase({args});")

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
        only = YesNo.from_bool(only_specified)
        cnc = YesNo.from_bool(close_normally_closed)
        sw_types = format_list(switching_types, quote_items=True) if switching_types else '["Breaker"]'

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

        args = pack_args(object_type, processed_val, only, sw_types, cnc)
        return self.RunScriptCommand(f"CloseWithBreakers({args});")

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
        ono = YesNo.from_bool(open_normally_open)
        sw_types = format_list(switching_types, quote_items=True) if switching_types else '["Breaker"]'

        # This command has a unique syntax where the object type is the first argument
        # and the second argument is an identifier with keys *only*, not the full object string.
        # This block handles cases where a full object string (e.g., from create_object_string)
        # is passed as filter_val.
        processed_val = filter_val
        prefix_to_check = f"[{object_type.upper()} "
        if filter_val.strip().upper().startswith(prefix_to_check):
            keys_part = filter_val.strip()[len(prefix_to_check):-1].strip()
            processed_val = f"[{keys_part}]"

        args = pack_args(object_type, processed_val, sw_types, ono)
        return self.RunScriptCommand(f"OpenWithBreakers({args});")