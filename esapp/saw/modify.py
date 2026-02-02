"""Modify Case Objects specific functions."""
from typing import List

from ._enums import (
    YesNo,
    format_filter,
    format_filter_selected_only,
    format_filter_areazone,
)


class ModifyMixin:
    """Mixin for modifying case objects."""

    def AutoInsertTieLineTransactions(self):
        """Deletes existing MW transactions and creates new ones based on tie-line flows.

        This action automatically generates transactions to represent power transfers
        across tie-lines between areas.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("AutoInsertTieLineTransactions")

    def BranchMVALimitReorder(self, filter_name: str = "", limits: List[str] = None):
        """Modifies MVA limits for branches, allowing reordering or setting specific limits.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to branches. Defaults to an empty string (all branches).
        limits : List[str], optional
            A list of 15 strings representing the MVA limits (A through O) to apply.
            If None, default limits are used.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        if limits is None:
            limits = []
        # Pad limits to 15 entries (A through O)
        while len(limits) < 15:
            limits.append("")

        filt = f'"{filter_name}"' if filter_name else ""
        lim_str = ", ".join(limits)
        return self._run_script("BranchMVALimitReorder", filt, lim_str)

    def CalculateRXBGFromLengthConfigCondType(self, filter_name: str = ""):
        """Recalculates R, X, G, B parameters for transmission lines using the TransLineCalc tool.

        This is typically used when line length, configuration, or conductor type changes.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to transmission lines. Defaults to an empty string (all lines).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., TransLineCalc not registered).
        """
        filt = format_filter_selected_only(filter_name)
        return self._run_script("CalculateRXBGFromLengthConfigCondType", filt)

    def ChangeSystemMVABase(self, new_base: float):
        """Changes the system MVA base.

        This action rescales all per-unit values in the case to reflect the new MVA base.

        Parameters
        ----------
        new_base : float
            The new system MVA base value.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("ChangeSystemMVABase", new_base)

    def ClearSmallIslands(self):
        """Identifies the largest island in the system and de-energizes all other smaller islands.

        This is useful for simplifying the system or isolating the main connected component.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("ClearSmallIslands")

    def CreateLineDeriveExisting(
        self, from_bus: int, to_bus: int, circuit: str, new_length: float, branch_id: str, existing_length: float = None, zero_g: bool = False
    ):
        """Creates a new branch derived from an existing one with scaled impedance.

        This allows for quickly modeling new lines based on the characteristics of an
        already defined line, adjusting for length.

        Parameters
        ----------
        from_bus : int
            The bus number of the 'From' end of the new branch.
        to_bus : int
            The bus number of the 'To' end of the new branch.
        circuit : str
            The circuit identifier for the new branch.
        new_length : float
            The length of the new branch.
        branch_id : str
            The ID of an existing branch to derive parameters from.
        existing_length : float, optional
            The length of the existing branch. If None, PowerWorld will attempt to
            determine it.
        zero_g : bool, optional
            If True, sets the shunt conductance (G) of the new branch to zero.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        el = str(existing_length) if existing_length is not None else ""
        zg = YesNo.from_bool(zero_g)
        return self._run_script("CreateLineDeriveExisting", from_bus, to_bus, f'"{circuit}"', new_length, branch_id, el, zg)

    def DirectionsAutoInsert(self, source: str, sink: str, delete_existing: bool = True, use_area_zone_filters: bool = False):
        """Auto-inserts directions to the case for transfer analysis.

        Parameters
        ----------
        source : str
            The source object type (e.g., 'AREA', 'BUS', 'ZONE').
        sink : str
            The sink object type (e.g., 'AREA', 'BUS', 'ZONE').
        delete_existing : bool, optional
            If True, deletes existing directions before inserting new ones. Defaults to True.
        use_area_zone_filters : bool, optional
            If True, uses Area/Zone filters for auto-insertion. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        de = YesNo.from_bool(delete_existing)
        uaz = YesNo.from_bool(use_area_zone_filters)
        return self._run_script("DirectionsAutoInsert", source, sink, de, uaz)

    def DirectionsAutoInsertReference(self, source_type: str, reference_object: str, delete_existing: bool = True, source_filter: str = "", opposite_direction: bool = False):
        """Auto-inserts directions from multiple source objects to the same ReferenceObject.

        Parameters
        ----------
        source_type : str
            The type of source objects (e.g., "BUS", "AREA", "ZONE").
        reference_object : str
            The reference object string (e.g., '[BUS 100]', '[AREA "LoadZone"]').
        delete_existing : bool, optional
            If True, deletes existing directions before inserting new ones. Defaults to True.
        source_filter : str, optional
            A PowerWorld filter name to apply to the source objects. Defaults to an empty string (all).
        opposite_direction : bool, optional
            If True, inserts directions in the opposite sense (from reference to sources).
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        de = YesNo.from_bool(delete_existing)
        filt = f'"{source_filter}"' if source_filter else '""'
        od = YesNo.from_bool(opposite_direction)
        return self._run_script("DirectionsAutoInsertReference", source_type, f'"{reference_object}"', de, filt, od)

    def InitializeGenMvarLimits(self):
        """Initializes all generators to be marked as at Mvar limits or not.

        This action typically resets the Mvar limit status of generators based
        on their current operating point and defined limits.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("InitializeGenMvarLimits")

    def InjectionGroupsAutoInsert(self):
        """Inserts injection groups according to the IG_AutoInsert_Options.

        Injection groups are used for various analyses, such as PV/QV curves.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("InjectionGroupsAutoInsert")

    def InjectionGroupCreate(self, name: str, object_type: str, initial_value: float, filter_name: str, append: bool = True):
        """Creates or modifies an injection group.

        Parameters
        ----------
        name : str
            The name of the injection group.
        object_type : str
            The type of objects to include in the group (e.g., "Gen", "Load", "Bus").
        initial_value : float
            The initial value for the injection group.
        filter_name : str
            A PowerWorld filter name to select objects for the group.
        append : bool, optional
            If True, appends to an existing group; if False, creates a new one or overwrites.
            Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = YesNo.from_bool(append)
        filt = format_filter(filter_name)
        return self._run_script("InjectionGroupCreate", f'"{name}"', object_type, initial_value, filt, app)

    def InjectionGroupRemoveDuplicates(self, preference_filter: str = ""):
        """Removes duplicate injection groups.

        Parameters
        ----------
        preference_filter : str, optional
            A PowerWorld filter name to specify which groups to prioritize if duplicates exist.
            Defaults to an empty string.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{preference_filter}"' if preference_filter else ""
        return self._run_script("InjectionGroupRemoveDuplicates", filt)

    def InterfacesAutoInsert(self, type_: str, delete_existing: bool = True, use_filters: bool = False, prefix: str = "", limits: str = "AUTO"):
        """Auto-inserts interfaces based on specified criteria.

        Interfaces are used to monitor power flow across boundaries.

        Parameters
        ----------
        type_ : str
            The type of interface to auto-insert (e.g., "AREA", "ZONE", "OWNER").
        delete_existing : bool, optional
            If True, deletes existing interfaces before auto-inserting new ones.
            Defaults to True.
        use_filters : bool, optional
            If True, uses Area/Zone filters for auto-insertion. Defaults to False.
        prefix : str, optional
            A prefix to add to the names of the auto-inserted interfaces. Defaults to "".
        limits : str, optional
            How to set interface limits ("AUTO", "NONE", or a specific value). Defaults to "AUTO".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        de = YesNo.from_bool(delete_existing)
        uf = YesNo.from_bool(use_filters)
        return self._run_script("InterfacesAutoInsert", type_, de, uf, f'"{prefix}"', limits)

    def InterfaceAddElementsFromContingency(self, interface_name: str, contingency_name: str):
        """Adds elements from a contingency to an existing interface.

        This can be used to define an interface that monitors elements affected
        by a specific contingency.

        Parameters
        ----------
        interface_name : str
            The name of the target interface.
        contingency_name : str
            The name of the contingency from which to add elements.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("InterfaceAddElementsFromContingency", f'"{interface_name}"', f'"{contingency_name}"')

    def InterfaceFlatten(self, interface_name: str):
        """Flattens an interface.

        Flattening an interface replaces its complex definition (e.g., based on areas)
        with a direct list of its constituent branches and transformers.

        Parameters
        ----------
        interface_name : str
            The name of the interface to flatten.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("InterfaceFlatten", f'"{interface_name}"')

    def InterfaceFlattenFilter(self, filter_name: str):
        """Flattens interfaces that meet a specified filter.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to interfaces. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("InterfaceFlattenFilter", filt)

    def InterfaceModifyIsolatedElements(self, filter_name: str = ""):
        """Modifies isolated elements within interfaces.

        This action typically removes or adjusts elements in an interface that
        become isolated from the main system.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to interfaces. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self._run_script("InterfaceModifyIsolatedElements", filt)

    def InterfaceRemoveDuplicates(self, preference_filter: str = ""):
        """Removes duplicate interfaces.

        Parameters
        ----------
        preference_filter : str, optional
            A PowerWorld filter name to specify which interfaces to prioritize if duplicates exist.
            Defaults to an empty string.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = f'"{preference_filter}"' if preference_filter else ""
        return self._run_script("InterfaceRemoveDuplicates", filt)

    def InterfaceCreate(self, name: str, delete_existing: bool, object_type: str, filter_name: str):
        """Creates or modifies an interface with elements of a single object type.

        Parameters
        ----------
        name : str
            The name of the interface to create or modify.
        delete_existing : bool
            If True, deletes any existing interface with the same name before creating.
        object_type : str
            The type of objects to include in the interface (e.g., "Branch", "Transformer").
        filter_name : str
            A PowerWorld filter name to select objects for the interface.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        de = YesNo.from_bool(delete_existing)
        return self._run_script("InterfaceCreate", f'"{name}"', de, object_type, f'"{filter_name}"')

    def MergeBuses(self, element: str, filter_name: str = ""):
        """Merges buses based on specified criteria.

        Parameters
        ----------
        element : str
            The element defining the buses to merge (e.g., "[BUS 1]", "SELECTED").
        filter_name : str, optional
            A PowerWorld filter name to apply. Defaults to an empty string.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter_areazone(filter_name)
        return self._run_script("MergeBuses", element, filt)

    def MergeLineTerminals(self, filter_name: str = "SELECTED"):
        """Merges line terminals.

        This action is typically used to simplify the representation of line connections.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to line terminals. Defaults to "SELECTED".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter_selected_only(filter_name)
        return self._run_script("MergeLineTerminals", filt)

    def MergeMSLineSections(self, filter_name: str = "SELECTED"):
        """Eliminates multi-section line records by merging them into single lines.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to multisection lines. Defaults to "SELECTED".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter_selected_only(filter_name)
        return self._run_script("MergeMSLineSections", filt)

    def Move(self, element_a: str, destination: str, how_much: float = 100.0, abort_on_error: bool = True):
        """Moves a generator, load, transmission line, or switched shunt.

        This action transfers a specified percentage of an element's value (e.g., MW/Mvar)
        from one location to another.

        Parameters
        ----------
        element_a : str
            The source element string (e.g., '[GEN 1]', '[LOAD 2]').
        destination : str
            The destination object string (e.g., '[BUS 10]', '[AREA "ZoneA"]').
        how_much : float, optional
            The percentage (0-100) of the element's value to move. Defaults to 100.0.
        abort_on_error : bool, optional
            If True, aborts the operation if an error occurs. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        abort = YesNo.from_bool(abort_on_error)
        return self._run_script("Move", element_a, destination, how_much, abort)

    def ReassignIDs(self, object_type: str, field: str, filter_name: str = "", use_right: bool = False):
        """Sets IDs of specified objects to the first/last two characters of a specified field.

        This is a utility for re-identifying objects based on their attributes.

        Parameters
        ----------
        object_type : str
            The type of object to reassign IDs for (e.g., "Load", "Gen").
        field : str
            The field whose characters will be used for the new ID (e.g., "BusName").
        filter_name : str, optional
            A PowerWorld filter name to apply to objects. Defaults to an empty string (all).
        use_right : bool, optional
            If True, uses the last two characters of the field; otherwise, uses the first two.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        ur = YesNo.from_bool(use_right)
        filt = format_filter(filter_name)
        return self._run_script("ReassignIDs", object_type, field, filt, ur)

    def Remove3WXformerContainer(self, filter_name: str = ""):
        """Deletes three-winding transformer container objects, leaving their internal two-winding transformers.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to 3-winding transformers. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter(filter_name)
        return self._run_script("Remove3WXformerContainer", filt)

    def RenameInjectionGroup(self, old_name: str, new_name: str):
        """Renames an injection group.

        Parameters
        ----------
        old_name : str
            The current name of the injection group.
        new_name : str
            The new name for the injection group.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., group not found, new name already exists).
        """
        return self._run_script("RenameInjectionGroup", f'"{old_name}"', f'"{new_name}"')

    def RotateBusAnglesInIsland(self, bus_key: str, value: float):
        """Rotates bus angles in an island by a specified value.

        This action adjusts the phase angles of all buses within the island
        containing the specified `bus_key`.

        Parameters
        ----------
        bus_key : str
            The key of a bus within the target island (e.g., '[BUS 1]').
        value : float
            The angle in degrees by which to rotate the bus angles.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("RotateBusAnglesInIsland", bus_key, value)

    def SetGenPMaxFromReactiveCapabilityCurve(self, filter_name: str = ""):
        """Changes generator maximum MW output (PMax) based on its reactive capability curve.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to generators. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter_areazone(filter_name)
        return self._run_script("SetGenPMaxFromReactiveCapabilityCurve", filt)

    def SetParticipationFactors(self, method: str, constant_value: float, object_str: str):
        """Modifies generator participation factors.

        Participation factors determine how generators respond to changes in system load
        or frequency.

        Parameters
        ----------
        method : str
            The method for setting participation factors (e.g., "CONSTANT", "PROPORTIONAL").
        constant_value : float
            A constant value to apply if `method` is "CONSTANT".
        object_str : str
            The object string defining the scope (e.g., "SYSTEM", '[AREA "Top"]').

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("SetParticipationFactors", method, constant_value, object_str)

    def SetScheduledVoltageForABus(self, bus_id: str, voltage: float):
        """Sets the stored scheduled voltage for a specific bus.

        Parameters
        ----------
        bus_id : str
            The bus identifier string (e.g., '[BUS 1]').
        voltage : float
            The new scheduled voltage in per unit.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("SetScheduledVoltageForABus", bus_id, voltage)

    def SetInterfaceLimitToMonitoredElementLimitSum(self, filter_name: str = "ALL"):
        """Sets interface limits to the sum of its monitored element limits.

        This action is useful for ensuring interface limits are consistent with
        the underlying component limits.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to apply to interfaces. Defaults to "ALL".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter(filter_name)
        return self._run_script("SetInterfaceLimitToMonitoredElementLimitSum", filt)

    def SplitBus(
        self,
        element: str,
        new_bus_number: int,
        insert_tie: bool = True,
        line_open: bool = False,
        branch_device_type: str = "Line",
    ):
        """Splits a bus into two, optionally inserting a tie line between them.

        This action is used to model substation reconfigurations or new bus connections.

        Parameters
        ----------
        element : str
            The bus identifier string to split (e.g., '[BUS 1]').
        new_bus_number : int
            The bus number for the newly created bus.
        insert_tie : bool, optional
            If True, inserts a tie line between the original and new bus. Defaults to True.
        line_open : bool, optional
            If True, the inserted tie line (if any) is initially open. Defaults to False.
        branch_device_type : str, optional
            The type of branch device to use for the tie line (e.g., "Line", "Breaker").
            Defaults to "Line".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        tie = YesNo.from_bool(insert_tie)
        open_line = YesNo.from_bool(line_open)
        new_bus_number = int(new_bus_number.iloc[0]) if hasattr(new_bus_number, 'iloc') else int(new_bus_number)
        return self._run_script("SplitBus", element, new_bus_number, tie, open_line, f'"{branch_device_type}"')

    def SuperAreaAddAreas(self, name: str, filter_name: str):
        """Adds areas to a Super Area.

        Super Areas are collections of regular areas, useful for hierarchical modeling.

        Parameters
        ----------
        name : str
            The name of the Super Area.
        filter_name : str
            A PowerWorld filter name to select areas to add.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter(filter_name)
        return self._run_script("SuperAreaAddAreas", f'"{name}"', filt)

    def SuperAreaRemoveAreas(self, name: str, filter_name: str):
        """Removes areas from a Super Area.

        Parameters
        ----------
        name : str
            The name of the Super Area.
        filter_name : str
            A PowerWorld filter name to select areas to remove.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        filt = format_filter(filter_name)
        return self._run_script("SuperAreaRemoveAreas", f'"{name}"', filt)

    def TapTransmissionLine(
        self,
        element: str,
        pos_along_line: float,
        new_bus_number: int,
        shunt_model: str = "CAPACITANCE",
        treat_as_ms_line: bool = False,
        update_onelines: bool = False,
        new_bus_name: str = "",
    ):
        """Taps a transmission line at a specified position to insert a new bus.

        This action is used to model new substations or connections along an existing line.

        Parameters
        ----------
        element : str
            The transmission line identifier string (e.g., ``[BRANCH 1 2 1]``).
        pos_along_line : float
            The position along the line (0-100%) where the tap is made.
        new_bus_number : int
            The bus number for the newly created tapped bus.
        shunt_model : str, optional
            The shunt model to use for the new bus ("CAPACITANCE", "INDUCTANCE", etc.).
            Defaults to "CAPACITANCE".
        treat_as_ms_line : bool, optional
            If True, treats the tapped line as a multisection line. Defaults to False.
        update_onelines : bool, optional
            If True, attempts to update any open oneline diagrams to reflect the change.
            Defaults to False.
        new_bus_name : str, optional
            A name for the newly created bus. Defaults to an empty string.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        ms = YesNo.from_bool(treat_as_ms_line)
        uo = YesNo.from_bool(update_onelines)
        new_bus_number = int(new_bus_number.iloc[0]) if hasattr(new_bus_number, 'iloc') else int(new_bus_number)
        return self._run_script("TapTransmissionLine", element, pos_along_line, new_bus_number, shunt_model, ms, uo, new_bus_name)
