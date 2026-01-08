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
        """Restores the initial state for the ATC tool."""
        return self.RunScriptCommand("ATCRestoreInitialState;")

    def ATCSetAsReference(self):
        """Sets the present system state as the reference state for ATC analysis."""
        return self.RunScriptCommand("ATCSetAsReference;")

    def ATCTakeMeToScenario(self, rl: int, g: int, i: int):
        """Sets the present case according to the scenarios along the RL, G, and I axes."""
        return self.RunScriptCommand(f"ATCTakeMeToScenario({rl}, {g}, {i});")

    def ATCDataWriteOptionsAndResults(self, filename: str, append: bool = True, key_field: str = "PRIMARY"):
        """Writes out all information related to ATC analysis to an auxiliary file."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'ATCDataWriteOptionsAndResults("{filename}", {app}, {key_field});')

    def ATCWriteResultsAndOptions(self, filename: str, append: bool = True):
        """Writes out all information related to ATC analysis to an auxiliary file."""
        app = "YES" if append else "NO"
        return self.RunScriptCommand(f'ATCWriteResultsAndOptions("{filename}", {app});')

    def ATCWriteScenarioLog(self, filename: str, append: bool = False, filter_name: str = ""):
        """Writes out detailed log information for ATC Multiple Scenarios to a text file."""
        app = "YES" if append else "NO"
        filt = f'"{filter_name}"' if filter_name else ""
        return self.RunScriptCommand(f'ATCWriteScenarioLog("{filename}", {app}, {filt});')

    def ATCWriteToExcel(self, worksheet_name: str, fieldlist: List[str] = None):
        """Sends ATC analysis results to an Excel spreadsheet for Multiple Scenarios ATC analysis."""
        fields = ""
        if fieldlist:
            fields = ", [" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'ATCWriteToExcel("{worksheet_name}"{fields});')

    def ATCWriteToText(self, filename: str, filetype: str = "TAB", fieldlist: List[str] = None):
        """Writes Multiple Scenario ATC analysis results to text files."""
        fields = ""
        if fieldlist:
            fields = ", [" + ", ".join(fieldlist) + "]"
        return self.RunScriptCommand(f'ATCWriteToText("{filename}", {filetype}{fields});')
