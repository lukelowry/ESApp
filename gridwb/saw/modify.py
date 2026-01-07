"""Modify Case Objects specific functions."""
from typing import List


class ModifyMixin:
    """Mixin for modifying case objects."""

    def AutoInsertTieLineTransactions(self):
        """Deletes existing MW transactions and creates new ones based on tie-line flows."""
        return self.RunScriptCommand("AutoInsertTieLineTransactions;")

    def BranchMVALimitReorder(self, filter_name: str = "", limits: List[str] = None):
        """Modifies MVA limits for branches."""
        if limits is None:
            limits = []
        # Pad limits to 15 entries (A through O)
        while len(limits) < 15:
            limits.append("")
        
        filt = f'"{filter_name}"' if filter_name else ""
        lim_str = ", ".join(limits)
        return self.RunScriptCommand(f"BranchMVALimitReorder({filt}, {lim_str});")

    def CalculateRXBGFromLengthConfigCondType(self, filter_name: str = ""):
        """Recalculates R, X, G, B using TransLineCalc tool."""
        filt = f'"{filter_name}"' if filter_name and filter_name != "SELECTED" else filter_name
        return self.RunScriptCommand(f"CalculateRXBGFromLengthConfigCondType({filt});")

    def ChangeSystemMVABase(self, new_base: float):
        """Changes the system MVA base."""
        return self.RunScriptCommand(f"ChangeSystemMVABase({new_base});")

    def ClearSmallIslands(self):
        """Identifies the largest island and de-energizes all other islands."""
        return self.RunScriptCommand("ClearSmallIslands;")

    def CreateLineDeriveExisting(
        self, from_bus: int, to_bus: int, circuit: str, new_length: float, branch_id: str, existing_length: float = None, zero_g: bool = False
    ):
        """Creates a new branch derived from an existing one with scaled impedance."""
        el = str(existing_length) if existing_length is not None else ""
        zg = "YES" if zero_g else "NO"
        return self.RunScriptCommand(
            f'CreateLineDeriveExisting({from_bus}, {to_bus}, "{circuit}", {new_length}, {branch_id}, {el}, {zg});'
        )

    def DirectionsAutoInsert(self, source: str, sink: str, delete_existing: bool = True, use_area_zone_filters: bool = False):
        """Auto-inserts directions to the case."""
        de = "YES" if delete_existing else "NO"
        uaz = "YES" if use_area_zone_filters else "NO"
        return self.RunScriptCommand(f"DirectionsAutoInsert({source}, {sink}, {de}, {uaz});")

    def DirectionsAutoInsertReference(self, source_type: str, reference_object: str, delete_existing: bool = True, source_filter: str = "", opposite_direction: bool = False):
        """Auto-inserts directions from multiple source objects to the same ReferenceObject."""
        de = "YES" if delete_existing else "NO"
        filt = f'"{source_filter}"' if source_filter else '""'
        od = "YES" if opposite_direction else "NO"
        return self.RunScriptCommand(f'DirectionsAutoInsertReference({source_type}, "{reference_object}", {de}, {filt}, {od});')

    def InitializeGenMvarLimits(self):
        """Initializes all generators to be marked as at Mvar limits or not."""
        return self.RunScriptCommand("InitializeGenMvarLimits;")

    def InjectionGroupsAutoInsert(self):
        """Inserts injection groups according to IG_AutoInsert_Options."""
        return self.RunScriptCommand("InjectionGroupsAutoInsert;")

    def InjectionGroupCreate(self, name: str, object_type: str, initial_value: float, filter_name: str, append: bool = True):
        """Creates or modifies an injection group."""
        app = "YES" if append else "NO"
        filt = f'"{filter_name}"'
        return self.RunScriptCommand(f'InjectionGroupCreate("{name}", {object_type}, {initial_value}, {filt}, {app});')

    def InjectionGroupRemoveDuplicates(self, preference_filter: str = ""):
        """Removes duplicate injection groups."""
        filt = f'"{preference_filter}"' if preference_filter else ""
        return self.RunScriptCommand(f'InjectionGroupRemoveDuplicates({filt});')

    def InterfacesAutoInsert(self, type_: str, delete_existing: bool = True, use_filters: bool = False, prefix: str = "", limits: str = "AUTO"):
        """Auto-inserts interfaces."""
        de = "YES" if delete_existing else "NO"
        uf = "YES" if use_filters else "NO"
        return self.RunScriptCommand(f'InterfacesAutoInsert({type_}, {de}, {uf}, "{prefix}", {limits});')

    def InterfaceAddElementsFromContingency(self, interface_name: str, contingency_name: str):
        """Adds elements from a contingency to an interface."""
        return self.RunScriptCommand(f'InterfaceAddElementsFromContingency("{interface_name}", "{contingency_name}");')

    def InterfaceFlatten(self, interface_name: str):
        """Flattens an interface."""
        return self.RunScriptCommand(f'InterfaceFlatten("{interface_name}");')

    def InterfaceFlattenFilter(self, filter_name: str):
        """Flattens interfaces meeting a filter."""
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'InterfaceFlattenFilter({filt});')

    def InterfaceModifyIsolatedElements(self, filter_name: str = ""):
        """Modifies isolated elements in interfaces."""
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'InterfaceModifyIsolatedElements({filt});')

    def InterfaceRemoveDuplicates(self, preference_filter: str = ""):
        """Removes duplicate interfaces."""
        filt = f'"{preference_filter}"' if preference_filter else ""
        return self.RunScriptCommand(f'InterfaceRemoveDuplicates({filt});')

    def InterfaceCreate(self, name: str, delete_existing: bool, object_type: str, filter_name: str):
        """Creates or modifies an interface with elements of a single object type."""
        de = "YES" if delete_existing else "NO"
        return self.RunScriptCommand(f'InterfaceCreate("{name}", {de}, {object_type}, "{filter_name}");')

    def MergeBuses(self, element: str, filter_name: str = ""):
        """Merge buses."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"MergeBuses({element}, {filt});")

    def MergeLineTerminals(self, filter_name: str = "SELECTED"):
        """Merges line terminals."""
        filt = f'"{filter_name}"' if filter_name and filter_name != "SELECTED" else filter_name
        return self.RunScriptCommand(f"MergeLineTerminals({filt});")

    def MergeMSLineSections(self, filter_name: str = "SELECTED"):
        """Eliminates multi-section line records."""
        filt = f'"{filter_name}"' if filter_name and filter_name != "SELECTED" else filter_name
        return self.RunScriptCommand(f"MergeMSLineSections({filt});")

    def Move(self, element_a: str, destination: str, how_much: float = 100.0, abort_on_error: bool = True):
        """Moves a generator, load, transmission line, or switched shunt."""
        abort = "YES" if abort_on_error else "NO"
        return self.RunScriptCommand(f"Move({element_a}, {destination}, {how_much}, {abort});")

    def ReassignIDs(self, object_type: str, field: str, filter_name: str = "", use_right: bool = False):
        """Sets IDs of specified objects to the first/last two characters of a specified field."""
        ur = "YES" if use_right else "NO"
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f"ReassignIDs({object_type}, {field}, {filt}, {ur});")

    def Remove3WXformerContainer(self, filter_name: str = ""):
        """Deletes three-winding transformers leaving internal two-winding transformers."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f"Remove3WXformerContainer({filt});")

    def RenameInjectionGroup(self, old_name: str, new_name: str):
        """Renames an injection group."""
        return self.RunScriptCommand(f'RenameInjectionGroup("{old_name}", "{new_name}");')

    def RotateBusAnglesInIsland(self, bus_key: str, value: float):
        """Rotates angles in an island."""
        return self.RunScriptCommand(f"RotateBusAnglesInIsland({bus_key}, {value});")

    def SetGenPMaxFromReactiveCapabilityCurve(self, filter_name: str = ""):
        """Changes Max MW output based on capability curve."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE"] else filter_name
        return self.RunScriptCommand(f"SetGenPMaxFromReactiveCapabilityCurve({filt});")

    def SetParticipationFactors(self, method: str, constant_value: float, object_str: str):
        """Modifies generator participation factors."""
        return self.RunScriptCommand(f"SetParticipationFactors({method}, {constant_value}, {object_str});")

    def SetScheduledVoltageForABus(self, bus_id: str, voltage: float):
        """Sets the stored scheduled voltage for a bus."""
        return self.RunScriptCommand(f"SetScheduledVoltageForABus({bus_id}, {voltage});")

    def SetInterfaceLimitToMonitoredElementLimitSum(self, filter_name: str = "ALL"):
        """Sets interface limits to sum of monitored element limits."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f"SetInterfaceLimitToMonitoredElementLimitSum({filt});")

    def SplitBus(
        self,
        element: str,
        new_bus_number: int,
        insert_tie: bool = True,
        line_open: bool = False,
        branch_device_type: str = "Line",
    ):
        """Splits a bus."""
        tie = "YES" if insert_tie else "NO"
        open_line = "YES" if line_open else "NO"
        return self.RunScriptCommand(
            f'SplitBus({element}, {new_bus_number}, {tie}, {open_line}, "{branch_device_type}");'
        )

    def SuperAreaAddAreas(self, name: str, filter_name: str):
        """Adds areas to a Super Area."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f'SuperAreaAddAreas("{name}", {filt});')

    def SuperAreaRemoveAreas(self, name: str, filter_name: str):
        """Removes areas from a Super Area."""
        filt = f'"{filter_name}"' if filter_name and filter_name not in ["SELECTED", "AREAZONE", "ALL"] else filter_name
        return self.RunScriptCommand(f'SuperAreaRemoveAreas("{name}", {filt});')

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
        """Taps a transmission line."""
        ms = "YES" if treat_as_ms_line else "NO"
        uo = "YES" if update_onelines else "NO"
        return self.RunScriptCommand(
            f'TapTransmissionLine({element}, {pos_along_line}, {new_bus_number}, {shunt_model}, {ms}, {uo}, "{new_bus_name}");'
        )