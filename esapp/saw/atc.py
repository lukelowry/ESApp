"""Available Transfer Capability (ATC) specific functions."""
import pandas as pd
from typing import List


class ATCMixin:
    """Mixin for ATC analysis functions."""

    def DetermineATC(
        self,
        seller: str,
        buyer: str,
        distributed: bool = False,
        multiple_scenarios: bool = False,
    ):
        """Calculates Available Transfer Capability (ATC) between a specified seller and buyer.

        This method initiates an ATC calculation, ramping transfer between the
        seller and buyer until a system limit is reached.

        Parameters
        ----------
        seller : str
            The source object string (e.g., '[AREA "Top"]', '[BUS 1]').
        buyer : str
            The sink object string (e.g., '[AREA "Bottom"]', '[BUS 2]').
        distributed : bool, optional
            If True, uses the distributed ATC solution method. Defaults to False.
        multiple_scenarios : bool, optional
            If True, processes each defined scenario in the case. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., invalid seller/buyer, calculation error).
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermine({seller}, {buyer}, {dist}, {mult});"
        )

    def DetermineATCMultipleDirections(
        self, distributed: bool = False, multiple_scenarios: bool = False
    ):
        """Calculates ATC for all directions defined within the PowerWorld case.

        This method is used when multiple transfer directions have been pre-configured
        in the Simulator.

        Parameters
        ----------
        distributed : bool, optional
            If True, uses the distributed ATC solution method. This requires the
            distributed ATC add-on to be installed. Defaults to False.
        multiple_scenarios : bool, optional
            If True, processes each defined ATC scenario in the case. Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails (e.g., no directions defined, calculation error).
        """
        dist = "YES" if distributed else "NO"
        mult = "YES" if multiple_scenarios else "NO"
        return self.RunScriptCommand(
            f"ATCDetermineMultipleDirections({dist}, {mult});"
        )

    def GetATCResults(self, fields: list = None) -> pd.DataFrame:
        """Retrieves Transfer Limiter results from the case after an ATC calculation.

        This method fetches the detailed results of the ATC analysis, including
        the maximum flow, limiting contingency, and limiting element.

        Parameters
        ----------
        fields : List[str], optional
            A list of internal field names to retrieve for the 'TransferLimiter' object type.
            If None, a default set of common fields is retrieved.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the requested data for 'TransferLimiter' objects.

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        if fields is None:
            fields = [
                "LimitingElement",
                "LimitingContingency",
                "MaxFlow",
                "TransferLimit",
                "LimitUsed",
                "PTDF",
                "OTDF",
            ]

        return self.GetParametersMultipleElement("TransferLimiter", fields)

    def ATCCreateContingentInterfaces(self, filter_name: str = ""):
        """Creates an interface based on Transfer Limiter results from an ATC run.

        Each Transfer Limiter is comprised of a Limiting Element/Contingency pair.
        Each interface is then created with contingent elements from the contingency
        and the Limiting Element included as the monitored element.

        Parameters
        ----------
        filter_name : str, optional
            The name of an Advanced Filter. Only objects of type TransferLimiter
            that meet the named filter will be used to create new interfaces.
            If blank, all transfer limiters will be used. Defaults to "".

        """
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f"ATCCreateContingentInterfaces({filt});")

    def ATCDeleteAllResults(self):
        """Deletes all ATC results including TransferLimiter, ATCExtraMonitor, and ATCFlowValue object types."""
        return self.RunScriptCommand("ATCDeleteAllResults;")

    def ATCDeleteScenarioChangeIndexRange(self, scenario_change_type: str, index_range: List[str]):
        """Deletes entries within an ATC scenario change type by index.

        ATC scenarios are defined by RL (line rating and zone load), G (generator),
        and I (interface rating) changes.

        Parameters
        ----------
        scenario_change_type : str
            "RL", "G", or "I" to indicate the scenario change type to delete.
        index_range : List[str]
            Comma-delimited list of integer ranges (e.g., ["0-2", "5", "7-9"]).
            The indices start at 0.

        """
        ir = "[" + ", ".join(index_range) + "]"
        return self.RunScriptCommand(f"ATCDeleteScenarioChangeIndexRange({scenario_change_type}, {ir});")

    def ATCDetermineATCFor(self, rl: int, g: int, i: int, apply_transfer: bool = False):
        """Determines the ATC for a specific Scenario RL, G, I.

        Parameters
        ----------
        rl : int
            Index for the RL scenario.
        g : int
            Index for the G scenario.
        i : int
            Index for the I scenario.
        apply_transfer : bool, optional
            If True, leaves the system state at the transfer level that was determined.
            Defaults to False.

        """
        at = "YES" if apply_transfer else "NO"
        return self.RunScriptCommand(f"ATCDetermineATCFor({rl}, {g}, {i}, {at});")

    def ATCDetermineMultipleDirectionsATCFor(self, rl: int, g: int, i: int):
        """Determines the ATC for Scenario RL, G, I for all defined directions."""
        return self.RunScriptCommand(f"ATCDetermineMultipleDirectionsATCFor({rl}, {g}, {i});")

    def ATCIncreaseTransferBy(self, amount: float):
        """Increases the transfer between the seller and buyer by a specified amount."""
        return self.RunScriptCommand(f"ATCIncreaseTransferBy({amount});")

    def ATCRestoreInitialState(self):
        """Restores the initial state for the ATC tool.

        Call this action to restore the system to its state before any ATC
        calculations were performed.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("ATCRestoreInitialState;")

    def ATCSetAsReference(self):
        """Sets the present system state as the reference state for ATC analysis.

        This baseline state is used as the starting point for ATC calculations.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand("ATCSetAsReference;")

    def ATCTakeMeToScenario(self, rl: int, g: int, i: int):
        """Sets the present case according to the scenarios along the RL, G, and I axes.

        All three parameters must be specified as integers indicating the index
        of the respective scenario. Indices start at 0.

        Parameters
        ----------
        rl : int
            Index of the RL (line rating and zone load) scenario.
        g : int
            Index of the G (generator) scenario.
        i : int
            Index of the I (interface rating) scenario.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        return self.RunScriptCommand(f"ATCTakeMeToScenario({rl}, {g}, {i});")

    def ATCDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to ATC analysis to an auxiliary file.

        Saves the same information as the ATCWriteResultsAndOptions script command.
        The auxiliary file is formatted using the concise format for DATA section
        headers and variable names. Data is written using DATA sections instead of
        SUBDATA sections.

        Note: This command was named ATCWriteAllOptions prior to December 2021.

        Parameters
        ----------
        filename : str
            Name of the auxiliary file to save.
        append : bool, optional
            If True, appends results to existing file. If False, overwrites. Defaults to True.
        key_field : str, optional
            Identifier to use for the data ("PRIMARY", "SECONDARY", "LABEL").
            Defaults to "PRIMARY".

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'ATCDataWriteOptionsAndResults("{filename}", {app}, {key_field});')

    def ATCWriteAllOptions(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to ATC analysis (deprecated name).

        .. deprecated::
            Use `ATCDataWriteOptionsAndResults` instead. This method was renamed
            in the December 9, 2021 patch of Simulator 22.

        Parameters
        ----------
        filename : str
            Name of the auxiliary file to save.
        append : bool, optional
            If True, appends results to existing file. Defaults to True.
        key_field : str, optional
            Identifier to use for the data. Defaults to "PRIMARY".

        Returns
        -------
        None
        """
        return self.ATCDataWriteOptionsAndResults(filename, append, key_field)

    def ATCWriteResultsAndOptions(self, filename: str, append: bool = True):
        """Writes out all information related to ATC analysis to an auxiliary file.

        This includes Contingency Definitions, Remedial Action Definitions, Limit
        Monitoring Settings, Solution Options, ATC Options, ATC results, as well as
        any Model Criteria that are used by the Contingency and Remedial Action
        Definitions.

        Parameters
        ----------
        filename : str
            Name of the auxiliary file to save.
        append : bool, optional
            If True, appends results to existing file. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'ATCWriteResultsAndOptions("{filename}", {app});')

    def ATCWriteScenarioLog(self, filename: str, append: bool = False, filter_name: str = ""):
        """Writes out detailed log information for ATC Multiple Scenarios to a text file.

        If no scenarios have been defined, no file will be created; this is not
        treated as a fatal error.

        Parameters
        ----------
        filename : str
            Name of log file.
        append : bool, optional
            If True, appends to existing file. Defaults to False.
        filter_name : str, optional
            Filter name. Only scenarios meeting the filter will be written.
            Defaults to "" (all scenarios).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = "YES" if append else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'ATCWriteScenarioLog("{filename}", {app}, {filt});')

    def ATCWriteScenarioMinMax(
        self,
        filename: str,
        filetype: str = "CSV",
        append: bool = False,
        fieldlist: List[str] = None,
        operation: str = "MIN",
        operation_field: str = "MaxFlow",
        group_scenario: bool = True,
    ):
        """Writes out TransferLimiter results from multiple scenario ATC calculations.

        The results are grouped based on the input parameters, and the minimum,
        maximum, or minimum and maximum limiter from each group is written to file.

        Parameters
        ----------
        filename : str
            Name of output file.
        filetype : str, optional
            Output format: "AUX", "AUXCSV", "CSV", "CSVNOHEADER", "CSVCOLHEADER".
            Defaults to "CSV".
        append : bool, optional
            If True, appends to existing file. Defaults to False.
        fieldlist : List[str], optional
            List of fields to save. Defaults to None.
        operation : str, optional
            Operation to perform on each grouping: "MIN", "MAX", or "MINMAX".
            Defaults to "MIN".
        operation_field : str, optional
            Field to use for the min/max operation. Defaults to "MaxFlow".
        group_scenario : bool, optional
            If True, groups by scenario. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        app = "YES" if append else "NO"
        gs = "YES" if group_scenario else "NO"
        fields = ""
        if fieldlist:
            fields = "[" + ", ".join(fieldlist) + "]"
        else:
            fields = "[]"
        return self.RunScriptCommand(
            f'ATCWriteScenarioMinMax("{filename}", {filetype}, {app}, {fields}, {operation}, {operation_field}, {gs});'
        )

    def ATCWriteToExcel(self, worksheet_name: str, fieldlist: List[str] = None):
        """Sends ATC analysis results to an Excel spreadsheet for Multiple Scenarios ATC analysis.

        Parameters
        ----------
        worksheet_name : str
            Name of the Excel worksheet.
        fieldlist : List[str], optional
            List of fields to include. Defaults to None (all fields).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = ""
        if fieldlist:
            fields = ", [" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'ATCWriteToExcel("{worksheet_name}"{fields});')

    def ATCWriteToText(self, filename: str, filetype: str = "TAB", fieldlist: List[str] = None):
        """Writes Multiple Scenario ATC analysis results to text files.

        Parameters
        ----------
        filename : str
            Base name of the output file.
        filetype : str, optional
            Output format: "TAB" or "CSV". Defaults to "TAB".
        fieldlist : List[str], optional
            List of fields to include. Defaults to None (all fields).

        Returns
        -------
        None

        Raises
        ------
        PowerWorldError
            If the SimAuto call fails.
        """
        fields = ""
        if fieldlist:
            fields = ", [" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'ATCWriteToText("{filename}", {filetype}{fields});')
