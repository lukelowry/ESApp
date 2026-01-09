"""Contingency analysis specific functions."""
from typing import List


class ContingencyMixin:
    """Mixin for contingency analysis functions."""

    def RunContingency(self, ctg_name: str):
        """Runs a single defined contingency.

        This method is a wrapper for the `CTGSolve` script command, which
        executes the actions defined in a specific contingency and solves
        the power flow.

        Parameters
        ----------
        ctg_name : str
            The name of the contingency to run.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., contingency not found, power flow divergence).
        """
        return self.RunScriptCommand(f'CTGSolve("{ctg_name}");')

    def SolveContingencies(self):
        """Solves all defined contingencies in the PowerWorld case.

        This method is a wrapper for the `CTGSolveAll` script command, which
        iterates through all active contingencies, applies their actions, and
        solves the power flow for each.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or any contingency solution diverges.
        """
        return self.RunScriptCommand("CTGSolveAll(NO, YES);")

    def CTGAutoInsert(self):
        """Auto-inserts contingencies based on the `Ctg_AutoInsert_Options` configured in PowerWorld.

        This typically generates N-1 contingencies for lines, transformers, and generators.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Writes out all information related to contingency analysis to an auxiliary file.

        This method provides a comprehensive way to export contingency definitions,
        results, and options for documentation or further processing.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file where the information will be written.
        options : List[str], optional
            A list of specific options to include in the output. Defaults to None (all options).
        key_field : str, optional
            Identifier to use for the data ("PRIMARY", "SECONDARY", "LABEL"). Defaults to "PRIMARY".
        use_data_section : bool, optional
            If True, includes a data section in the auxiliary file. Defaults to False.
        use_concise : bool, optional
            If True, uses concise variable names in the output. Defaults to False.
        use_object_ids : str, optional
            Specifies how object IDs are handled ("NO", "YES_MS_3W"). Defaults to "NO".
        use_selected_data_maintainer : bool, optional
            If True, uses the selected data maintainer. Defaults to False.
        save_dependencies : bool, optional
            If True, saves contingency dependencies. Defaults to False.
        use_area_zone_filters : bool, optional
            If True, applies Area/Zone filters. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Applies the actions defined in a contingency without solving the power flow.

        This can be useful for inspecting the network topology changes caused by a
        contingency before running a full power flow solution.

        Parameters
        ----------
        contingency_name : str
            The name of the contingency whose actions are to be applied.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., contingency not found).
        """
        return self.RunScriptCommand(f'CTGApply("{contingency_name}");')

    def CTGCalculateOTDF(self, seller: str, buyer: str, linear_method: str = "DC"):
        """Computes OTDFs using the specified linear method.

        OTDFs quantify the impact of an outage on power transfers between a seller and buyer.

        Parameters
        ----------
        seller : str
            The seller (source) object string (e.g., '[AREA "Top"]', '[BUS 7]').
        buyer : str
            The buyer (sink) object string (e.g., '[AREA "Bottom"]', '[BUS 8]').
        linear_method : str, optional
            The linear method to use for calculation ("AC", "DC", "DCPS"). Defaults to "DC".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CTGCalculateOTDF({seller}, {buyer}, {linear_method});')

    def CTGClearAllResults(self):
        """Deletes all contingency violations and any contingency comparison results from memory.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("CTGClearAllResults;")

    def CTGSetAsReference(self):
        """Sets the present system state as the reference for contingency analysis.

        This baseline state is used for comparison when evaluating contingency impacts.

        Returns
        -------
        None
        """
        return self.RunScriptCommand("CTGSetAsReference;")

    def CTGProduceReport(self, filename: str):
        """Produces a text-based contingency analysis report.

        Parameters
        ----------
        filename : str
            The path to the file where the report will be saved.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CTGProduceReport("{filename}");')

    def CTGWriteFilePTI(self, filename: str, bus_format: str = "Name12", truncate_labels: bool = True, filter_name: str = "", append: bool = False):
        """Writes contingencies to a file in the PTI CON format.

        Parameters
        ----------
        filename : str
            The path to the output file.
        bus_format : str, optional
            The format for bus names ("Name12", "Number", etc.). Defaults to "Name12".
        truncate_labels : bool, optional
            If True, truncates contingency labels. Defaults to True.
        filter_name : str, optional
            A PowerWorld filter name to apply to contingencies. Defaults to an empty string (all).
        append : bool, optional
            If True, appends to the file if it exists. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        trunc = "YES" if truncate_labels else "NO"
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CTGWriteFilePTI("{filename}", {bus_format}, {trunc}, "{filter_name}", {app});')

    def CTGCloneMany(self, filter_name: str = "", prefix: str = "", suffix: str = "", set_selected: bool = False):
        """Creates copies of multiple contingencies based on a filter.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to select contingencies to clone. Defaults to an empty string (all).
        prefix : str, optional
            A prefix to add to the names of the cloned contingencies. Defaults to "".
        suffix : str, optional
            A suffix to add to the names of the cloned contingencies. Defaults to "".
        set_selected : bool, optional
            If True, sets the 'Selected' field of the cloned contingencies to YES. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        sel = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'CTGCloneMany("{filter_name}", "{prefix}", "{suffix}", {sel});')

    def CTGCloneOne(
        self, ctg_name: str, new_ctg_name: str = "", prefix: str = "", suffix: str = "", set_selected: bool = False
    ):
        """Creates a copy of a single existing contingency.

        Parameters
        ----------
        ctg_name : str
            The name of the contingency to clone.
        new_ctg_name : str, optional
            The name for the new cloned contingency. If empty, a name is generated.
            Defaults to "".
        prefix : str, optional
            A prefix to add to the new contingency name. Defaults to "".
        suffix : str, optional
            A suffix to add to the new contingency name. Defaults to "".
        set_selected : bool, optional
            If True, sets the 'Selected' field of the cloned contingency to YES. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        sel = "YES" if set_selected else "NO"
        return self.RunScriptCommand(f'CTGCloneOne("{ctg_name}", "{new_ctg_name}", "{prefix}", "{suffix}", {sel});')

    def CTGComboDeleteAllResults(self):
        """Deletes all results associated with contingency combination analysis.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("CTGComboDeleteAllResults;")

    def CTGComboSolveAll(self, do_distributed: bool = False, clear_all_results: bool = True):
        """Runs contingency combination analysis for all primary and regular/secondary contingencies.

        This performs a more complex analysis by considering the combined impact of multiple outages.

        Parameters
        ----------
        do_distributed : bool, optional
            If True, uses distributed processing for the solution. Defaults to False.
        clear_all_results : bool, optional
            If True, clears all previous results before starting. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., no primary contingencies defined).
        """
        dist = "YES" if do_distributed else "NO"
        clear = "YES" if clear_all_results else "NO"
        return self.RunScriptCommand(f"CTGComboSolveAll({dist}, {clear});")

    def CTGCompareTwoListsofContingencyResults(self, controlling: str, comparison: str):
        """Compares two different contingency result lists.

        Parameters
        ----------
        controlling : str
            The name of the controlling contingency result list.
        comparison : str
            The name of the comparison contingency result list.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CTGCompareTwoListsofContingencyResults({controlling}, {comparison});')

    def CTGConvertAllToDeviceCTG(self, keep_original_if_empty: bool = False):
        """Converts breaker/disconnect contingencies to device outages.

        Parameters
        ----------
        keep_original_if_empty : bool, optional
            If True, keeps the original contingency if the converted one is empty.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        keep = "YES" if keep_original_if_empty else "NO"
        return self.RunScriptCommand(f"CTGConvertAllToDeviceCTG({keep});")

    def CTGConvertToPrimaryCTG(
        self, filter_name: str = "", keep_original: bool = True, prefix: str = "", suffix: str = "-Primary"
    ):
        """Converts regular/secondary contingencies to Primary contingencies.

        Primary contingencies are typically used as the first level of outages
        in a combination analysis.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to select contingencies to convert. Defaults to an empty string (all).
        keep_original : bool, optional
            If True, keeps the original contingency after conversion. Defaults to True.
        prefix : str, optional
            A prefix to add to the new primary contingency name. Defaults to "".
        suffix : str, optional
            A suffix to add to the new primary contingency name. Defaults to "-Primary".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        keep = "YES" if keep_original else "NO"
        return self.RunScriptCommand(f'CTGConvertToPrimaryCTG("{filter_name}", {keep}, "{prefix}", "{suffix}");')

    def CTGCreateContingentInterfaces(self, filter_name: str, max_option: str = ""):
        """Creates an interface based on contingency violations.

        This interface can then be used to monitor elements that are frequently
        involved in violations.

        Parameters
        ----------
        filter_name : str
            A PowerWorld filter name to select contingencies whose violations
            will be used to define the interface.
        max_option : str, optional
            An option to specify how to handle multiple violations (e.g., "MAX", "SUM").
            Defaults to "".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f'CTGCreateContingentInterfaces("{filter_name}", {max_option});')

    def CTGCreateExpandedBreakerCTGs(self):
        """Converts 'Open/Close with Breakers' actions in contingencies into explicit OPEN/CLOSE actions on individual breakers.

        This can be useful for more detailed modeling of protection schemes.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Creates new contingencies from existing ones that have explicit breaker outages defined, modeling 'stuck' breakers.

        This is used to simulate scenarios where a breaker fails to operate as intended
        during a contingency.

        Parameters
        ----------
        filter_name : str, optional
            A PowerWorld filter name to select contingencies to process. Defaults to an empty string (all).
        allow_duplicates : bool, optional
            If True, allows creation of duplicate contingencies. Defaults to False.
        prefix_name : str, optional
            A prefix to add to the new contingency name. Defaults to "".
        include_ctg_label : bool, optional
            If True, includes the original contingency label in the new name. Defaults to True.
        branch_field_name : str, optional
            A branch field name to use in the new contingency name. Defaults to "".
        suffix_name : str, optional
            A suffix to add to the new contingency name. Defaults to "STK".
        prefix_comment : str, optional
            A prefix to add to the new contingency comment. Defaults to "".
        branch_field_comment : str, optional
            A branch field name to use in the new contingency comment. Defaults to "".
        suffix_comment : str, optional
            A suffix to add to the new contingency comment. Defaults to "".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        dup = "YES" if allow_duplicates else "NO"
        inc = "YES" if include_ctg_label else "NO"
        return self.RunScriptCommand(
            f'CTGCreateStuckBreakerCTGs("{filter_name}", {dup}, "{prefix_name}", {inc}, "{branch_field_name}", '
            f'"{suffix_name}", "{prefix_comment}", "{branch_field_comment}", "{suffix_comment}");'
        )

    def CTGDeleteWithIdenticalActions(self):
        """Deletes contingencies that have identical actions.

        This helps in reducing redundancy in the contingency list.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("CTGDeleteWithIdenticalActions;")

    def CTGJoinActiveCTGs(
        self, insert_solve_pf: bool, delete_existing: bool, join_with_self: bool, filename: str = ""
    ):
        """Creates new contingencies that are a join of the current active contingency list.

        This allows for creating combined contingencies from existing ones.

        Parameters
        ----------
        insert_solve_pf : bool
            If True, inserts a `SolvePowerFlow` action into the new contingencies.
        delete_existing : bool
            If True, deletes the original contingencies after joining.
        join_with_self : bool
            If True, allows a contingency to be joined with itself (e.g., for N-2 from N-1).
        filename : str, optional
            An optional filename to save the new contingencies to. Defaults to an empty string.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        ispf = "YES" if insert_solve_pf else "NO"
        de = "YES" if delete_existing else "NO"
        jws = "YES" if join_with_self else "NO"
        return self.RunScriptCommand(f'CTGJoinActiveCTGs({ispf}, {de}, {jws}, "{filename}");')

    def CTGPrimaryAutoInsert(self):
        """Auto-inserts Primary Contingencies.

        Primary contingencies are typically used as the first level of outages
        in a combination analysis.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("CTGPrimaryAutoInsert;")

    def CTGProcessRemedialActionsAndDependencies(self, do_delete: bool, filter_name: str = ""):
        """Processes Remedial Actions and their dependencies.

        Remedial actions are corrective measures taken after a contingency occurs.

        Parameters
        ----------
        do_delete : bool
            If True, deletes processed remedial actions.
        filter_name : str, optional
            A PowerWorld filter name to apply to remedial actions. Defaults to an empty string (all).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        delete = "YES" if do_delete else "NO"
        return self.RunScriptCommand(f'CTGProcessRemedialActionsAndDependencies({delete}, "{filter_name}");')

    def CTGReadFilePSLF(self, filename: str):
        """Loads a file in the PSLF OTG format and creates contingencies from it.

        Parameters
        ----------
        filename : str
            The path to the PSLF OTG file.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, invalid format).
        """
        return self.RunScriptCommand(f'CTGReadFilePSLF("{filename}");')

    def CTGReadFilePTI(self, filename: str):
        """Loads a file in the PTI CON format and creates contingencies from it.

        Parameters
        ----------
        filename : str
            The path to the PTI CON file.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., file not found, invalid format).
        """
        return self.RunScriptCommand(f'CTGReadFilePTI("{filename}");')

    def CTGRelinkUnlinkedElements(self):
        """Attempts to relink unlinked elements in the contingency records.

        This action tries to re-establish connections for elements that might
        have become unlinked due to topology changes or data inconsistencies.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
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
        """Saves contingency violations in a matrix format to a file.

        This provides a structured output of which contingencies cause violations
        on which monitored elements.

        Parameters
        ----------
        filename : str
            The path to the output file.
        filetype : str
            The format of the output file (e.g., "CSVCOLHEADER", "AUX").
        use_percentage : bool
            If True, reports violations as a percentage of the limit.
        object_types_to_report : List[str]
            A list of object types for which to report violations (e.g., ["Branch", "Bus"]).
        save_contingency : bool
            If True, saves contingency information (e.g., name, status).
        save_objects : bool
            If True, saves information about the monitored objects.
        field_list_object_type : str, optional
            The object type for which `field_list` applies. Defaults to an empty string.
        field_list : List[str], optional
            A list of specific fields to include for the `field_list_object_type`.
            Defaults to None.
        include_unsolvable_ctgs : bool, optional
            If True, includes information about contingencies that failed to solve.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        ValueError
            If `field_list` is provided without `field_list_object_type`.
        """
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
        """Sets the 'Skip' field to YES for contingencies that have identical actions.

        This helps in avoiding redundant calculations during contingency analysis.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("CTGSkipWithIdenticalActions;")

    def CTGSort(self, sort_field_list: List[str] = None):
        """Sorts the contingencies stored in Simulator's internal data structure.

        Parameters
        ----------
        sort_field_list : List[str], optional
            A list of fields to sort the contingencies by. Defaults to None (no specific sort).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        if sort_field_list is None:
            sort_field_list = []
        sort = "[" + ", ".join(sort_field_list) + "]"
        return self.RunScriptCommand(f"CTGSort({sort});")

    def CTGVerifyIteratedLinearActions(self, filename: str):
        """Creates a text file that contains validation information for iterated linear actions.

        Parameters
        ----------
        filename : str
            The path to the output text file.
        """
        return self.RunScriptCommand(f'CTGVerifyIteratedLinearActions("{filename}");')

    def CTGWriteAllOptions(
        self,
        filename: str,
        key_field: str = "PRIMARY",
        use_selected_data_maintainer: bool = False,
        save_dependencies: bool = False,
        use_area_zone_filters: bool = False,
    ) -> None:
        """Writes out all information related to contingency analysis using concise variable names.

        This is a specialized version of `CTGWriteResultsAndOptions` that uses
        concise variable names and includes data sections by default.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file where the information will be written.
        key_field : str, optional
            Identifier to use for the data ("PRIMARY", "SECONDARY", "LABEL"). Defaults to "PRIMARY".
        use_selected_data_maintainer : bool, optional
            If True, uses the selected data maintainer. Defaults to False.
        save_dependencies : bool, optional
            If True, saves contingency dependencies. Defaults to False.
        use_area_zone_filters : bool, optional
            If True, applies Area/Zone filters. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.CTGWriteResultsAndOptions(
            filename, [], key_field, True, True, "YES_MS_3W", use_selected_data_maintainer, save_dependencies, use_area_zone_filters
        )

    def CTGWriteAuxUsingOptions(self, filename: str, append: bool = True):
        """Writes out information related to contingency analysis as an auxiliary file.

        Parameters
        ----------
        filename : str
            The path to the auxiliary file.
        append : bool, optional
            If True, appends to the file if it exists. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'CTGWriteAuxUsingOptions("{filename}", {app});')

    def CTGRestoreReference(self):
        """Resets the system state to the reference state for contingency analysis.

        Call this action after running contingencies to restore the system to its
        baseline condition. The reference state is set by calling `CTGSetAsReference`.

        This command undoes any changes made by contingency actions (e.g., line
        outages, generator trips) and restores all values to the pre-contingency state.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., no reference state has been set).

        See Also
        --------
        CTGSetAsReference : Sets the current state as the reference.
        CTGApply : Applies contingency actions without solving.
        """
        return self.RunScriptCommand("CTGRestoreReference;")