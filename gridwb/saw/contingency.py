"""Contingency analysis specific functions."""
from typing import List


class ContingencyMixin:
    """Mixin for contingency analysis functions."""

    def RunContingency(self, ctg_name: str):
        """Runs a single defined contingency.

        This is a wrapper for the ``CTGSolve`` script command.

        :param ctg_name: Name of the contingency to run.
        """
        return self.RunScriptCommand(f'CTGSolve("{ctg_name}");')

    def SolveContingencies(self):
        """Solves all defined contingencies.

        This is a wrapper for the ``CTGSolveAll`` script command.
        """
        return self.RunScriptCommand("CTGSolveAll(NO, YES);")

    def CTGAutoInsert(self):
        """Auto insert contingencies based on Ctg_AutoInsert_Options."""
        return self.RunScriptCommand("CTGAutoInsert;")

    def CTGWriteResultsAndOptions(
        self,
        filename: str,
        options: list = None,
        key_field: str = "PRIMARY",
        use_data_section: bool = False,
        use_concise: bool = False,
        use_object_ids: str = "NO",
        use_selected_data_maintainer: bool = False,
        save_dependencies: bool = False,
        use_area_zone_filters: bool = False,
    ):
        """Writes out all information related to contingency analysis to an auxiliary file."""
        opts_str = ""
        if options:
            opts_str = "[" + ", ".join(options) + "]"

        uds = "YES" if use_data_section else "NO"
        uc = "YES" if use_concise else "NO"
        usdm = "YES" if use_selected_data_maintainer else "NO"
        sd = "YES" if save_dependencies else "NO"
        uazf = "YES" if use_area_zone_filters else "NO"

        cmd = f'CTGWriteResultsAndOptions("{filename}", {opts_str}, {key_field}, {uds}, {uc}, {use_object_ids}, {usdm}, {sd}, {uazf});'
        return self.RunScriptCommand(cmd)

    def CTGApply(self, contingency_name: str):
        """Call this action to apply the actions in a contingency without solving the power flow."""
        return self.RunScriptCommand(f'CTGApply("{contingency_name}");')

    def CTGCalculateOTDF(self, seller: str, buyer: str, linear_method: str = "DC"):
        """Computes OTDFs using the specified linear method.

        :param seller: The seller (source) (e.g. '[AREA "Top"]').
        :param buyer: The buyer (sink) (e.g. '[BUS 7]').
        :param linear_method: Linear method (AC, DC, DCPS). Defaults to DC.
        """
        return self.RunScriptCommand(f'CTGCalculateOTDF({seller}, {buyer}, {linear_method});')

    def CTGClearAllResults(self):
        """Deletes all contingency violations and any contingency comparison results."""
        return self.RunScriptCommand("CTGClearAllResults;")

    def CTGSetAsReference(self):
        """Sets the present system state as the reference for contingency analysis."""
        return self.RunScriptCommand("CTGSetAsReference;")

    def CTGProduceReport(self, filename: str):
        """Produces a text-based contingency analysis report."""
        return self.RunScriptCommand(f'CTGProduceReport("{filename}");')

    def CTGWriteFilePTI(self, filename: str, bus_format: str = "Name12", truncate_labels: bool = True, filter_name: str = "", append: bool = False):
        """Write contingencies to file in the PTI CON format."""
        trunc = "YES" if truncate_labels else "NO"
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CTGWriteFilePTI("{filename}", {bus_format}, {trunc}, "{filter_name}", {app});')

    def CTGCloneMany(self, filter_name: str = "", prefix: str = "", suffix: str = "", set_selected: bool = False):
        """Creates copies of any contingencies returned by the filter."""
        sel = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'CTGCloneMany("{filter_name}", "{prefix}", "{suffix}", {sel});')

    def CTGCloneOne(
        self, ctg_name: str, new_ctg_name: str = "", prefix: str = "", suffix: str = "", set_selected: bool = False
    ):
        """Creates a copy of a single existing contingency."""
        sel = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'CTGCloneOne("{ctg_name}", "{new_ctg_name}", "{prefix}", "{suffix}", {sel});')

    def CTGComboDeleteAllResults(self):
        """Deletes all results that are associated with contingency combination analysis."""
        return self.RunScriptCommand("CTGComboDeleteAllResults;")

    def CTGComboSolveAll(self, do_distributed: bool = False, clear_all_results: bool = True):
        """Run contingency combination analysis for all primary and regular/secondary contingencies."""
        dist = "YES" if do_distributed else "NO"
        clear = "YES" if clear_all_results else "NO"
        return self.RunScriptCommand(f"CTGComboSolveAll({dist}, {clear});")

    def CTGCompareTwoListsofContingencyResults(self, controlling: str, comparison: str):
        """Compares two different contingency result lists."""
        return self.RunScriptCommand(f'CTGCompareTwoListsofContingencyResults({controlling}, {comparison});')

    def CTGConvertAllToDeviceCTG(self, keep_original_if_empty: bool = False):
        """Convert breaker/disconnect contingencies to device outages."""
        keep = "YES" if keep_original_if_empty else "NO"
        return self.RunScriptCommand(f"CTGConvertAllToDeviceCTG({keep});")

    def CTGConvertToPrimaryCTG(
        self, filter_name: str = "", keep_original: bool = True, prefix: str = "", suffix: str = "-Primary"
    ):
        """Converts regular/secondary contingencies to Primary contingencies."""
        keep = "YES" if keep_original else "NO"
        return self.RunScriptCommand(f'CTGConvertToPrimaryCTG("{filter_name}", {keep}, "{prefix}", "{suffix}");')

    def CTGCreateContingentInterfaces(self, filter_name: str, max_option: str = ""):
        """Creates an interface based on contingency violations."""
        return self.RunScriptCommand(f'CTGCreateContingentInterfaces("{filter_name}", {max_option});')

    def CTGCreateExpandedBreakerCTGs(self):
        """Convert Open/Close with Breakers actions into OPEN/CLOSE actions on explicit breakers."""
        return self.RunScriptCommand("CTGCreateExpandedBreakerCTGs;")

    def CTGCreateStuckBreakerCTGs(
        self,
        filter_name: str = "",
        allow_duplicates: bool = False,
        prefix_name: str = "",
        include_ctg_label: bool = True,
        branch_field_name: str = "",
        suffix_name: str = "STK",
        prefix_comment: str = "",
        branch_field_comment: str = "",
        suffix_comment: str = "",
    ):
        """Creates new contingencies from contingencies that have explicit breaker outages defined."""
        dup = "YES" if allow_duplicates else "NO"
        inc = "YES" if include_ctg_label else "NO"
        return self.RunScriptCommand(
            f'CTGCreateStuckBreakerCTGs("{filter_name}", {dup}, "{prefix_name}", {inc}, "{branch_field_name}", '
            f'"{suffix_name}", "{prefix_comment}", "{branch_field_comment}", "{suffix_comment}");'
        )

    def CTGDeleteWithIdenticalActions(self):
        """Deletes contingencies that have identical actions."""
        return self.RunScriptCommand("CTGDeleteWithIdenticalActions;")

    def CTGJoinActiveCTGs(
        self, insert_solve_pf: bool, delete_existing: bool, join_with_self: bool, filename: str = ""
    ):
        """Creates new contingencies that are a join of the current contingency list."""
        ispf = "YES" if insert_solve_pf else "NO"
        de = "YES" if delete_existing else "NO"
        jws = "YES" if join_with_self else "NO"
        return self.RunScriptCommand(f'CTGJoinActiveCTGs({ispf}, {de}, {jws}, "{filename}");')

    def CTGPrimaryAutoInsert(self):
        """Auto insert Primary Contingencies."""
        return self.RunScriptCommand("CTGPrimaryAutoInsert;")

    def CTGProcessRemedialActionsAndDependencies(self, do_delete: bool, filter_name: str = ""):
        """Process Remedial Actions and dependencies."""
        delete = "YES" if do_delete else "NO"
        return self.RunScriptCommand(f'CTGProcessRemedialActionsAndDependencies({delete}, "{filter_name}");')

    def CTGReadFilePSLF(self, filename: str):
        """Load a file in the PSLF OTG format and create contingencies."""
        return self.RunScriptCommand(f'CTGReadFilePSLF("{filename}");')

    def CTGReadFilePTI(self, filename: str):
        """Load a file in the PTI CON format and create contingencies."""
        return self.RunScriptCommand(f'CTGReadFilePTI("{filename}");')

    def CTGRelinkUnlinkedElements(self):
        """Attempt to relink unlinked elements in the contingency records."""
        return self.RunScriptCommand("CTGRelinkUnlinkedElements;")

    def CTGSaveViolationMatrices(
        self,
        filename: str,
        filetype: str,
        use_percentage: bool,
        object_types_to_report: List[str],
        save_contingency: bool,
        save_objects: bool,
        field_list_object_type: str = "",
        field_list: List[str] = None,
        include_unsolvable_ctgs: bool = False,
    ):
        """Save contingency violations in a matrix format."""
        if field_list is None:
            field_list = []
        perc = "YES" if use_percentage else "NO"
        objs = "[" + ", ".join(object_types_to_report) + "]"
        sc = "YES" if save_contingency else "NO"
        so = "YES" if save_objects else "NO"
        fields = "[" + ", ".join(field_list) + "]"
        unsolv = "YES" if include_unsolvable_ctgs else "NO"

        return self.RunScriptCommand(
            f'CTGSaveViolationMatrices("{filename}", {filetype}, {perc}, {objs}, {sc}, {so}, '
            f'{field_list_object_type}, {fields}, {unsolv});'
        )

    def CTGSkipWithIdenticalActions(self):
        """Contingencies that have identical actions will have their Skip field set to YES."""
        return self.RunScriptCommand("CTGSkipWithIdenticalActions;")

    def CTGSort(self, sort_field_list: List[str] = None):
        """Sorts the contingencies stored in Simulator's internal data structure."""
        if sort_field_list is None:
            sort_field_list = []
        sort = "[" + ", ".join(sort_field_list) + "]"
        return self.RunScriptCommand(f"CTGSort({sort});")

    def CTGVerifyIteratedLinearActions(self, filename: str):
        """Creates a text file that contains validation information."""
        return self.RunScriptCommand(f'CTGVerifyIteratedLinearActions("{filename}");')

    def CTGWriteAllOptions(
        self,
        filename: str,
        key_field: str = "PRIMARY",
        use_selected_data_maintainer: bool = False,
        save_dependencies: bool = False,
        use_area_zone_filters: bool = False,
    ):
        """Writes out all information related to contingency analysis using concise variable names."""
        return self.CTGWriteResultsAndOptions(
            filename, [], key_field, True, True, "YES_MS_3W", use_selected_data_maintainer, save_dependencies, use_area_zone_filters
        )

    def CTGWriteAuxUsingOptions(self, filename: str, append: bool = True):
        """Writes out information related to contingency analysis as an auxiliary file."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CTGWriteAuxUsingOptions("{filename}", {app});')