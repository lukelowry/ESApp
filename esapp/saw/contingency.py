"""Contingency analysis specific functions."""
from typing import List, Union

from ._enums import YesNo, LinearMethod
from ._helpers import format_list


class ContingencyMixin:
    """Mixin for contingency analysis functions."""

    def CTGSolve(self, ctg_name: str):
        """Runs a single defined contingency.

        Executes the actions defined in a specific contingency and solves
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
        return self._run_script("CTGSolve", f'"{ctg_name}"')

    def CTGSolveAll(self, distributed: bool = False, clear_results: bool = True):
        """Solves all contingencies that are not marked to be skipped.

        Iterates through all active contingencies, applies their actions, and
        solves the power flow for each.

        Parameters
        ----------
        distributed : bool, optional
            If True, uses distributed computing for contingency analysis.
            Defaults to False.
        clear_results : bool, optional
            If True, clears all existing contingency results before solving.
            Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails or any contingency solution diverges.
        """
        dist = YesNo.from_bool(distributed)
        clear = YesNo.from_bool(clear_results)
        return self._run_script("CTGSolveAll", dist, clear)

    def CTGAutoInsert(self):
        """Auto-inserts contingencies based on the Ctg_AutoInsert_Options configured in PowerWorld.

        Prior to calling this action, all options for this action must be specified
        in the Ctg_AutoInsert_Options object using the SetData method or DATA sections.
        This typically generates N-1 contingencies for lines, transformers, and generators.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self._run_script("CTGAutoInsert")

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
        opts_str = format_list(options) if options else ""

        uds = YesNo.from_bool(use_data_section)
        uc = YesNo.from_bool(use_concise)
        usdm = YesNo.from_bool(use_selected_data_maintainer)
        sd = YesNo.from_bool(save_dependencies)
        uazf = YesNo.from_bool(use_area_zone_filters)

        return self._run_script("CTGWriteResultsAndOptions", f'"{filename}"', opts_str, key_field, uds, uc, use_object_ids, usdm, sd, uazf)

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
        return self._run_script("CTGApply", f'"{contingency_name}"')

    def CTGCalculateOTDF(self, seller: str, buyer: str, linear_method: Union[LinearMethod, str] = LinearMethod.DC):
        """Computes OTDFs (Outage Transfer Distribution Factors) for contingency violations.

        This action first performs the same action as CalculatePTDF for the specified
        seller and buyer. It then goes through all the violations found by the
        contingency analysis tool and determines the OTDF values for the various
        contingency/violation pairs.

        Parameters
        ----------
        seller : str
            The seller (source) object string (e.g., '[AREA "Top"]', '[BUS 7]').
        buyer : str
            The buyer (sink) object string (e.g., '[AREA "Bottom"]', '[BUS 8]').
        linear_method : Union[LinearMethod, str], optional
            The linear method to use for calculation. Defaults to LinearMethod.DC.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        method = linear_method.value if isinstance(linear_method, LinearMethod) else str(linear_method)
        return self._run_script("CTGCalculateOTDF", seller, buyer, method)

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
        return self._run_script("CTGClearAllResults")

    def CTGSetAsReference(self):
        """Sets the present system state as the reference for contingency analysis.

        This baseline state is used for comparison when evaluating contingency impacts.

        Returns
        -------
        None
        """
        return self._run_script("CTGSetAsReference")

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
        return self._run_script("CTGProduceReport", f'"{filename}"')

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
        trunc = YesNo.from_bool(truncate_labels)
        app = YesNo.from_bool(append)
        return self._run_script("CTGWriteFilePTI", f'"{filename}"', bus_format, trunc, f'"{filter_name}"', app)

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
        sel = YesNo.from_bool(set_selected)
        return self._run_script("CTGCloneMany", f'"{filter_name}"', f'"{prefix}"', f'"{suffix}"', sel)

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
        sel = YesNo.from_bool(set_selected)
        return self._run_script("CTGCloneOne", f'"{ctg_name}"', f'"{new_ctg_name}"', f'"{prefix}"', f'"{suffix}"', sel)

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
        return self._run_script("CTGComboDeleteAllResults")

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
        dist = YesNo.from_bool(do_distributed)
        clear = YesNo.from_bool(clear_all_results)
        return self._run_script("CTGComboSolveAll", dist, clear)

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
        return self._run_script("CTGCompareTwoListsofContingencyResults", f'"{controlling}"', f'"{comparison}"')

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
        keep = YesNo.from_bool(keep_original_if_empty)
        return self._run_script("CTGConvertAllToDeviceCTG", keep)

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
        keep = YesNo.from_bool(keep_original)
        return self._run_script("CTGConvertToPrimaryCTG", f'"{filter_name}"', keep, f'"{prefix}"', f'"{suffix}"')

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
        return self._run_script("CTGCreateContingentInterfaces", f'"{filter_name}"', max_option)

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
        return self._run_script("CTGCreateExpandedBreakerCTGs")

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
        dup = YesNo.from_bool(allow_duplicates)
        inc = YesNo.from_bool(include_ctg_label)
        return self._run_script("CTGCreateStuckBreakerCTGs", f'"{filter_name}"', dup, f'"{prefix_name}"', inc, f'"{branch_field_name}"', f'"{suffix_name}"', f'"{prefix_comment}"', f'"{branch_field_comment}"', f'"{suffix_comment}"')

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
        return self._run_script("CTGDeleteWithIdenticalActions")

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
        ispf = YesNo.from_bool(insert_solve_pf)
        de = YesNo.from_bool(delete_existing)
        jws = YesNo.from_bool(join_with_self)
        return self._run_script("CTGJoinActiveCTGs", ispf, de, jws, f'"{filename}"')

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
        return self._run_script("CTGPrimaryAutoInsert")

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
        delete = YesNo.from_bool(do_delete)
        return self._run_script("CTGProcessRemedialActionsAndDependencies", delete, f'"{filter_name}"')

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
        return self._run_script("CTGReadFilePSLF", f'"{filename}"')

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
        return self._run_script("CTGReadFilePTI", f'"{filename}"')

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
        return self._run_script("CTGRelinkUnlinkedElements")

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
        perc = YesNo.from_bool(use_percentage)
        objs = format_list(object_types_to_report)
        sc = YesNo.from_bool(save_contingency)
        so = YesNo.from_bool(save_objects)
        fields = format_list(field_list)
        unsolv = YesNo.from_bool(include_unsolvable_ctgs)

        return self._run_script("CTGSaveViolationMatrices", f'"{filename}"', filetype, perc, objs, sc, so, field_list_object_type, fields, unsolv)

    def CTGSort(self, sort_field_list: List[str] = None):
        """Sorts the contingencies stored in Simulator's internal data structure.

        This is different than sorting contingencies in case information displays
        in the GUI or sorting data when it is written to an auxiliary file.
        Contingencies are processed in the order in which they are stored in
        the internal data structure, and they are not sorted by default;
        contingencies are added in the order in which they are created.
        This could be significant for other actions like CTGJoinActiveCTGs
        if the goal is to join contingencies alphabetically.

        Parameters
        ----------
        sort_field_list : List[str], optional
            A list of fields to sort by. If None, sorts alphabetically by
            contingency name. Format: ``["fieldname1:+:0", "fieldname2:-:1"]``
            where + is ascending, - is descending, 0 is case insensitive,
            1 is case sensitive.

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
        sort = format_list(sort_field_list)
        return self._run_script("CTGSort", sort)

    def CTGVerifyIteratedLinearActions(self, filename: str):
        """Creates a text file that contains validation information for iterated linear actions.

        Parameters
        ----------
        filename : str
            The path to the output text file.
        """
        return self._run_script("CTGVerifyIteratedLinearActions", f'"{filename}"')

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
        app = YesNo.from_bool(append)
        return self._run_script("CTGWriteAuxUsingOptions", f'"{filename}"', app)

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
        return self._run_script("CTGRestoreReference")
